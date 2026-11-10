"""E13 -- Adversarial request sequences: can a client force unbounded recomputation?

Semantic idempotence implies a work bound: a record identity is executed at most once,
so for *any* request sequence the deletion work is bounded by the union of the
requested effects, however many requests carry it.  A service that executes every
request has no such bound.

Six sequences are designed to maximise recomputation if the service is naive:

  progressive       a 1000-record window that slides by one record per request
  partition_bomb    ten disjoint slices submitted in a cycle
  dual_cover        two 50%-overlapping sets, alternated
  superset_ladder   a fixed core plus a growing tail
  random_retry      i.i.d. random subsets
  concurrent_flood  32 identical requests at once

For each we report the number of requests that genuinely carry new records
(`jobs_min`, the minimum any correct service must execute), the jobs actually
executed, and two amplifications:

  job_amp   = jobs / jobs_min            -- how many retain passes were wasted
  delta_amp = example-visits / union     -- how many times a record was forgotten
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import threading
import time

import numpy as np

from su import config as C
from su import harness as H
from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import DeleteService

MODES = ("stateless", "key", "semantic", "su")
EPOCHS = 4
BIG_REQUEST = 5000


def universe(n, seed=0):
    """Record universe; seed 0 is the published order, larger seeds shuffle it so that
    windows, slices and retries land on a different subset of record identities."""
    order = list(range(n))
    if seed:
        np.random.default_rng(10_000 + seed).shuffle(order)
    return [record_identity("adv", f"u/{i}", version_digest(b"a")) for i in order]


def jobs_min(reqs):
    seen = set()
    n = 0
    for _, r in reqs:
        if any(x not in seen for x in r):
            n += 1
        seen.update(r)
    return n


def sequences(U, n_requests=100, window=1000, seed=0):
    out = {}
    out["progressive"] = [(f"p{i}", U[i:i + window]) for i in range(n_requests)]
    slices = 10
    step = len(U) // slices
    out["partition_bomb"] = [
        (f"q{i}", U[(i % slices) * step:((i % slices) + 1) * step]) for i in range(n_requests)]
    a = U[: len(U) // 2]
    b = U[len(U) // 4: 3 * len(U) // 4]
    out["dual_cover"] = [(f"r{i}", (a if i % 2 == 0 else b)) for i in range(n_requests)]
    core = U[:window]
    out["superset_ladder"] = [
        (f"s{i}", core + U[window:window + i]) for i in range(n_requests)]
    rng = np.random.default_rng(seed)
    out["random_retry"] = [
        (f"t{i}", [U[j] for j in rng.choice(len(U), size=2000, replace=False)])
        for i in range(n_requests)]
    return out


def run_sequence(mode, reqs, union_size, workdir, concurrent_last=False):
    be = SyntheticBackend(os.path.join(workdir, "art"), sleep_s=0.002)
    svc = DeleteService(os.path.join(workdir, "svc.sqlite"), be, mode=mode,
                        epochs=EPOCHS)
    svc.bootstrap()
    t0 = time.perf_counter()
    if concurrent_last:
        bar = threading.Barrier(len(reqs))
        ts = [threading.Thread(target=lambda k=k, r=r: (bar.wait(timeout=120),
                                                        svc.delete(k, "t", r)))
              for k, r in reqs]
        [t.start() for t in ts]
        [t.join() for t in ts]
    else:
        for k, r in reqs:
            svc.delete(k, "t", r)
    wall = time.perf_counter() - t0
    st = svc.stats()
    out = dict(mode=mode, n_requests=len(reqs), union=union_size,
               jobs=st["physical_runs"], versions=st["versions"] - 1,
               cost_examples=st["cost_examples"],
               delta_amp=st["cost_examples"] / max(1, union_size * EPOCHS),
               wall_s=wall)
    svc.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", type=int, default=20000)
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--window", type=int, default=1000)
    ap.add_argument("--seeds", default="0",
                    help="comma-separated universe seeds; 0 is the published run")
    ap.add_argument("--out", default="e13_adversarial.csv")
    args = ap.parse_args()
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        U = universe(args.universe, seed)
        seqs = sequences(U, args.requests, args.window, seed)
        seqs["concurrent_flood"] = [(f"f{i}", U[:1000]) for i in range(32)]
        for name, reqs in seqs.items():
            jmin = jobs_min(reqs)
            union = len({x for _, r in reqs for x in r})
            for mode in MODES:
                td = tempfile.mkdtemp()
                try:
                    r = run_sequence(mode, reqs, union, td,
                                     concurrent_last=(name == "concurrent_flood"))
                finally:
                    shutil.rmtree(td, ignore_errors=True)
                r.update(sequence=name, jobs_min=jmin, seed=seed,
                         job_amp=r["jobs"] / max(1, jmin))
                rows.append(r)
                print(f"  seed={seed} {name:16s} {mode:9s} jobs={r['jobs']:4d} "
                      f"(min {jmin:3d}) job_amp={r['job_amp']:6.2f} "
                      f"delta_amp={r['delta_amp']:6.2f} wall={r['wall_s']:6.2f}s",
                      flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
