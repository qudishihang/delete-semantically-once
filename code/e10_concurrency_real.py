"""E10 -- 32-way concurrency with the *real* unlearning backend (not the synthetic
executor): what does semantic idempotence buy in wall-clock time?

Thirty-two clients submit disjoint deletion requests simultaneously through a
thread pool of 8 workers.  We report

  * unlearning jobs executed and committed model versions,
  * total *job time* (the sum of executor wall-clock, i.e. the work the service
    had to buy) and end-to-end wall-clock,
  * per-request latency percentiles,
  * the quality of the final model (test accuracy and JS to the exact-retraining
    oracle), so that a "cheap" service cannot win by doing less work than asked.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from su import config as C
from su import harness as H
from su import metrics as MET
from su.service import DeleteService
from su.world import World, request_indices

torch.set_num_threads(int(C.NT))
MODES = ("stateless", "semantic", "su")


def run_one(w: World, mode: str, n_requests: int, per_request: int, epochs: int,
            workers: int, oracle_probs) -> dict:
    rng = np.random.default_rng(909)
    perm = rng.permutation(w.n)
    groups = [np.sort(perm[i * per_request:(i + 1) * per_request])
              for i in range(n_requests)]
    rids = [w.rids_of(g) for g in groups]
    td = tempfile.mkdtemp()
    try:
        be = w.backend(method="neggradplus", epochs=epochs,
                       artifact_subdir=f"e10/{mode}/c{n_requests}")
        svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                            method="neggradplus", epochs=epochs)
        svc.bootstrap()
        lat, errs = [], []
        lock = threading.Lock()

        def work(i):
            t0 = time.perf_counter()
            try:
                svc.delete(f"k{i}", "tenant", rids[i])
                dt = time.perf_counter() - t0
                with lock:
                    lat.append(dt)
            except Exception as e:      # pragma: no cover
                with lock:
                    errs.append(repr(e))

        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, range(n_requests)))
        wall = time.perf_counter() - t0

        st = svc.stats()
        model = w.model(svc.current_artifact())
        p = MET.probs_of(model, w.Fte)
        res = dict(
            mode=mode, n_requests=n_requests, per_request=per_request,
            workers=workers, epochs=epochs,
            physical_runs=st["physical_runs"], versions=st["versions"] - 1,
            stale_commits=st["stale_commits"],
            job_seconds=float(st["exec_wall_s"]), wall_seconds=wall,
            cost_examples=st["cost_examples"],
            errors=len(errs), processed=st["processed"],
            test_acc=MET.accuracy_of_probs(p, w.yte),
            js_to_oracle=MET.js_divergence(p, oracle_probs),
            p50_ms=1000 * float(np.percentile(lat, 50)) if lat else None,
            p95_ms=1000 * float(np.percentile(lat, 95)) if lat else None,
            max_ms=1000 * float(np.max(lat)) if lat else None,
        )
        svc.close()
        return res
    finally:
        import shutil
        shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--per-request", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="e10_concurrency_real.csv")
    args = ap.parse_args()

    w = World.build(args.dataset, args.arch, args.seed)
    oracle_probs = w.probs(w.retrain_path)
    rows = []
    for mode in MODES:
        t0 = time.time()
        r = run_one(w, mode, args.requests, args.per_request, args.epochs,
                    args.workers, oracle_probs)
        r["total_s"] = time.time() - t0
        rows.append(r)
        print(f"  {mode:9s} jobs={r['physical_runs']:3d} vers={r['versions']:3d} "
              f"stale={r['stale_commits']:3d} job_time={r['job_seconds']:7.1f}s "
              f"wall={r['wall_seconds']:6.1f}s p95={r['p95_ms']:8.1f}ms "
              f"acc={r['test_acc']:.4f} js_oracle={r['js_to_oracle']:.4f}", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
