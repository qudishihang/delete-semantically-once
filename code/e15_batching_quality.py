"""E15 -- does group commit preserve per-region forgetting?

Group commit merges several requests into one unlearning job.  This script asks whether
the merged execution still forgets the *requested records*, per request, not just on
average.

Arms (same 32 disjoint deletions of 200 records each):
  sequential   one job per request (group commit off)
  batched      through the service, group commit on
  oracle       exact retraining on D \\ H

Reported per arm: jobs, executor examples, wall-clock, full-test accuracy, JS to the
oracle, and the *distribution* of per-request forgetting: for each deleted set, the
accuracy of the model on those 200 records (lower = better forgotten) and the gap to the
oracle's accuracy on the same 200 records.  A mean can hide requests that are not
forgotten, so we print min / quartiles / max as well.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np

from su import config as C
from su import harness as H
from su import metrics as MET
from su.service import DeleteService
from su.world import World


def mia_advantage(p_deleted: np.ndarray, p_heldout: np.ndarray) -> float:
    s = np.concatenate([p_deleted.max(axis=1), p_heldout.max(axis=1)])
    y = np.concatenate([np.ones(len(p_deleted)), np.zeros(len(p_heldout))])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(2 * auc - 1)


def per_request_acc(m, F, y, groups):
    """Accuracy of model `m` on each request's own deleted records."""
    out = []
    for g in groups:
        p = MET.probs_of(m, F[g])
        out.append(MET.accuracy_of_probs(p, y[g]))
    return np.array(out)


def describe(name, vals):
    q = np.percentile(vals, [0, 25, 50, 75, 100])
    print(f"  {name:26s} min={q[0]:.3f} p25={q[1]:.3f} med={q[2]:.3f} "
          f"p75={q[3]:.3f} max={q[4]:.3f} mean={vals.mean():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--ue-epochs", type=int, default=10)
    ap.add_argument("--retrain-epochs", type=int, default=C.BASE_EPOCHS)
    ap.add_argument("--out", default="e15_batching_quality.csv")
    ap.add_argument("--arms", default="sequential,batched",
                    help="comma-separated subset of {sequential,batched}")
    ap.add_argument("--concurrent", action="store_true",
                    help="submit the requests as a burst (needed for group commit)")
    args = ap.parse_args()

    w = World.build(args.dataset, args.arch, args.seed)
    rng = np.random.default_rng(2024)
    perm = rng.permutation(w.n)
    groups = [np.sort(perm[i * args.size:(i + 1) * args.size])
              for i in range(args.requests)]
    rids = [w.rids_of(g) for g in groups]
    hall = np.unique(np.concatenate(groups))
    hold = np.arange(0, len(w.yte), 2)
    y_del = w.ytr[hall]
    print(f"  setup: {args.requests} requests x {args.size} records = {len(hall)} "
          f"deleted ({100.0*len(hall)/w.n:.1f}% of {w.n})", flush=True)

    t0 = time.time()
    oracle_path = w.oracle_for(np.setdiff1d(np.arange(w.n), hall), "e15")
    m_oracle = w.model(oracle_path)
    p_oracle = w.probs(oracle_path)
    print(f"  oracle ready in {time.time()-t0:.0f}s", flush=True)
    acc_oracle = per_request_acc(m_oracle, w.Ftr, w.ytr, groups)
    describe("oracle per-request forget acc", acc_oracle)

    rows = []
    wanted = {a.strip() for a in args.arms.split(",")}
    for mode, label in (("su_nogroup", "sequential"), ("su", "batched")):
        if label not in wanted:
            continue
        td = tempfile.mkdtemp()
        try:
            be = w.backend(method="neggradplus", epochs=args.ue_epochs,
                           artifact_subdir=f"e15/{mode}")
            svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                                method="neggradplus", epochs=args.ue_epochs)
            svc.bootstrap()
            t1 = time.time()
            if args.concurrent:
                # a burst: all requests arrive together, so group commit can merge them
                import threading
                bar = threading.Barrier(len(rids))
                errs = []

                def one(i, r):
                    try:
                        bar.wait(timeout=120)
                        svc.delete(f"k{i}", "tenant", r)
                    except BaseException as exc:      # noqa: BLE001
                        errs.append(repr(exc))

                ts = [threading.Thread(target=one, args=(i, r))
                      for i, r in enumerate(rids)]
                [t.start() for t in ts]
                [t.join() for t in ts]
                if errs:
                    print("  errors:", errs[:3], flush=True)
            else:
                for i, r in enumerate(rids):
                    svc.delete(f"k{i}", "tenant", r)
            wall = time.time() - t1
            st = svc.stats()
            m = w.model(svc.current_artifact())
            p = MET.probs_of(m, w.Fte)
            acc = per_request_acc(m, w.Ftr, w.ytr, groups)
            gap = acc - acc_oracle
            describe(f"{label} per-request forget acc", acc)
            describe(f"{label} gap to oracle", gap)
            for i, (a, g) in enumerate(zip(acc, gap)):
                rows.append(dict(arm=label, request=i, forget_acc=float(a),
                                 gap_to_oracle=float(g)))
            r = dict(arm=label, jobs=st["physical_runs"], versions=st["versions"] - 1,
                     cost_examples=st["cost_examples"], wall_s=wall,
                     test_acc=MET.accuracy_of_probs(p, w.yte),
                     js_to_oracle=MET.js_divergence(p, p_oracle),
                     forg_mia=mia_advantage(MET.probs_of(m, w.Ftr[hall]), p[hold]))
            rows.append(dict(r))
            print(f"  {label:11s} jobs={r['jobs']:3d} examples={r['cost_examples']:9d} "
                  f"wall={wall:6.0f}s test={r['test_acc']:.4f} "
                  f"js={r['js_to_oracle']:.4f} mia={r['forg_mia']:+.4f}", flush=True)
            H.write_csv(H.result_path(f"e15_{label}.csv"), rows)
            svc.close()
        finally:
            shutil.rmtree(td, ignore_errors=True)

    rows.append(dict(arm="oracle", jobs=1, cost_examples=0, wall_s=0.0,
                     test_acc=MET.accuracy_of_probs(p_oracle, w.yte),
                     js_to_oracle=0.0,
                     forg_mia=mia_advantage(MET.probs_of(m_oracle, w.Ftr[hall]), p_oracle[hold])))
    p_or = MET.probs_of(m_oracle, w.Fte)
    rows[-1]["test_acc"] = MET.accuracy_of_probs(p_or, w.yte)
    print(f"  {'oracle':11s} test={rows[-1]['test_acc']:.4f} "
          f"mia={rows[-1]['forg_mia']:+.4f}", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
