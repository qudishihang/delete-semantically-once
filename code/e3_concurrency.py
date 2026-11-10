"""E3 -- Concurrency: correctness and throughput under parallel duplicate/overlapping
deletion requests.

The synthetic backend models an unlearning job of fixed duration, so this
experiment isolates the *protocol* rather than the ML cost.  A real-backend spot
check is produced by ``--backend torch``.
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

MODES = ("stateless", "key", "key_atomic", "semantic", "su_opt", "su_nogroup", "su")
SCENARIOS = ("identical", "overlap50", "disjoint")


def rids(n, offset=0):
    return [record_identity("bench", f"r/{offset+i}", version_digest(b"x")) for i in range(n)]


def build(c: int, scenario: str, size: int):
    core = rids(size)
    if scenario == "identical":
        return [("inter", "tenant", core) for _ in range(c)]
    if scenario == "overlap50":
        shared = core[: size // 2]
        out = []
        for i in range(c):
            own = rids(size // 2, offset=10_000 + i * size)
            out.append((f"o{i}", "tenant", shared + own))
        return out
    if scenario == "disjoint":
        out = []
        for i in range(c):
            out.append((f"d{i}", "tenant", rids(size, offset=i * size)))
        return out
    raise ValueError(scenario)


def run_one(mode: str, c: int, scenario: str, size: int, sleep_s: float, workdir: str):
    be = SyntheticBackend(os.path.join(workdir, "art"), sleep_s=sleep_s)
    db = os.path.join(workdir, "svc.sqlite")
    svc = DeleteService(db, be, mode=mode)
    svc.bootstrap()
    reqs = build(c, scenario, size)
    lat, errs = [], []
    bar = threading.Barrier(c)

    def work(key, payload):
        try:
            bar.wait(timeout=60)
            t0 = time.perf_counter()
            svc.delete(key, "tenant", payload)
            lat.append(time.perf_counter() - t0)
        except Exception as e:      # pragma: no cover
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(k, r)) for k, _, r in reqs]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    wall = time.time() - t0
    st = svc.stats()
    lost = svc.lost_effects({k: r for k, _, r in reqs})
    committed = st["requests"].get("committed", 0)
    out = dict(mode=mode, concurrency=c, scenario=scenario, request_size=size,
               sleep_s=sleep_s, n_requests=c,
               physical_runs=st["physical_runs"], versions=st["versions"] - 1,
               processed=st["processed"], cost_examples=st["cost_examples"],
               wall_s=wall, throughput_rps=c / wall if wall > 0 else float("nan"),
               **H.summarise_latencies(lat),
               errors=len(errs),
               committed_requests=committed,
               keys_in_versions=st["keys_in_versions"],
               stale_commits=st["stale_commits"],
               keys_per_version=(st["keys_in_versions"] / max(1, st["versions"] - 1)),
               lost_effects=None if not svc.tracks_processed else (sum(lost.values()) if lost else 0))
    svc.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e3_concurrency.csv")
    ap.add_argument("--sleep", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    rows = []
    for scenario in SCENARIOS:
        for c in [int(x) for x in args.concurrency.split(",")]:
            for mode in MODES:
                for rep in range(args.reps):
                    with tempfile.TemporaryDirectory() as td:
                        r = run_one(mode, c, scenario, args.size, args.sleep, td)
                    r["rep"] = rep
                    rows.append(r)
                sub = [r for r in rows if r["scenario"] == scenario and r["concurrency"] == c
                       and r["mode"] == mode]
                print(f"  {scenario:10s} c={c:2d} {mode:9s} "
                      f"runs={np.mean([s['physical_runs'] for s in sub]):5.1f} "
                      f"vers={np.mean([s['versions'] for s in sub]):5.1f} "
                      f"thr={np.mean([s['throughput_rps'] for s in sub]):6.1f}/s "
                      f"p95={np.mean([s.get('p95_ms', 0) for s in sub]):7.1f}ms", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
