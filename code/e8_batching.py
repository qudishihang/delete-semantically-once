"""E8 -- Cost-aware planning: how long should the writer wait to batch requests?

Group commit trades latency for work.  If the writer drains immediately, concurrent
requests that arrive a millisecond later form a second group and pay a second pass
over the retained data.  If it waits for a short window, it amortises the pass over
more requests but delays every one of them.  This experiment measures the frontier
with the synthetic backend (isolating protocol behaviour) and with the real backend
(isolating the cost of a retained-data pass).
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time

import numpy as np

from su import config as C
from su import harness as H
from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import DeleteService

MODES = ("semantic", "su")
WINDOWS = (0.0, 0.002, 0.005, 0.010, 0.025, 0.050)


def rids(n, off=0):
    return [record_identity("bench", f"r/{off+i}", version_digest(b"x")) for i in range(n)]


def run(mode, window, conc, size, sleep, workdir, rate=0.0):
    be = SyntheticBackend(os.path.join(workdir, "art"), sleep_s=sleep)
    svc = DeleteService(os.path.join(workdir, "svc.sqlite"), be, mode=mode,
                        batch_window_s=window)
    svc.bootstrap()
    reqs = [(f"k{i}", rids(size, off=i * size)) for i in range(conc)]
    lat, errs = [], []
    bar = threading.Barrier(conc)

    use_barrier = rate <= 0

    def work(k, r):
        try:
            if use_barrier:
                bar.wait(timeout=120)
            t0 = time.perf_counter()
            svc.delete(k, "t", r)
            lat.append(time.perf_counter() - t0)
        except Exception as e:      # pragma: no cover
            errs.append(repr(e))

    def delayed(delay, k, r):
        time.sleep(delay)
        work(k, r)
    rng = np.random.default_rng(7)
    delays = (rng.exponential(1.0 / rate, size=conc) if rate > 0 else np.zeros(conc))
    ts = [threading.Thread(target=delayed, args=(delays[i], k, r))
          for i, (k, r) in enumerate(reqs)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    st = svc.stats()
    out = dict(mode=mode, window_s=window, concurrency=conc, request_size=size,
               sleep_s=sleep, physical_runs=st["physical_runs"],
               versions=st["versions"] - 1, processed=st["processed"],
               keys_per_version=st["keys_in_versions"] / max(1, st["versions"] - 1),
               errors=len(errs), **H.summarise_latencies(lat))
    svc.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e8_batching_window.csv")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--concurrency", default="16,32")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rate", type=float, default=0.0,
                    help="staggered arrivals: Poisson rate (requests/second)")
    args = ap.parse_args()
    rows = []
    for conc in [int(c) for c in args.concurrency.split(",")]:
        for mode in MODES:
            for w in WINDOWS:
                for rep in range(args.reps):
                    with tempfile.TemporaryDirectory() as td:
                        r = run(mode, w, conc, args.size, args.sleep, td, args.rate)
                    r["rep"] = rep
                    rows.append(r)
                sub = [r for r in rows if r["mode"] == mode and r["window_s"] == w
                       and r["concurrency"] == conc]
                print(f"  c={conc:2d} {mode:9s} w={w*1000:5.1f}ms "
                      f"runs={np.mean([s['physical_runs'] for s in sub]):5.1f} "
                      f"p50={np.mean([s['p50_ms'] for s in sub]):7.1f}ms "
                      f"p95={np.mean([s['p95_ms'] for s in sub]):7.1f}ms", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
