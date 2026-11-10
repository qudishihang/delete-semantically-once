"""E14 -- Full recomputation as the exactness/cost reference.

Three strategies face the same five-request workload: each request repeats a fixed
2,000-record payload and adds 400 new records.

  stateless   execute the operator on the whole payload every time
  delta       the proposed service: execute each record at most once
  recompute   retrain from scratch on D \\ H after every request (exact by definition)

Reported: distance to the retraining oracle, utility, jobs, retained-data passes and
wall-clock.  The point is a three-point cost/quality frontier: exactness costs about
twice the retained-data work, the delta service is close to exact at the cheaper
point, and the stateless service pays the same as the delta service while drifting
furthest from the target.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np
import torch

from su import config as C
from su import harness as H
from su import metrics as MET
from su import platform as P
from su.service import DeleteService
from su.world import World


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--requests", type=int, default=5)
    ap.add_argument("--core", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--pattern", default="grow", choices=("grow", "repeat"),
                    help="grow: repeated core plus new records; "
                         "repeat: the same payload every time")
    ap.add_argument("--ue-epochs", type=int, default=10)
    ap.add_argument("--retrain-epochs", type=int, default=C.BASE_EPOCHS)
    ap.add_argument("--out", default="e14_recompute.csv")
    args = ap.parse_args()

    w = World.build(args.dataset, args.arch, args.seed)
    rng = np.random.default_rng(31337)
    perm = rng.permutation(w.n)
    core = np.sort(perm[: args.core])
    if args.pattern == "grow":
        chunks = [np.sort(perm[args.core + i * args.chunk:
                               args.core + (i + 1) * args.chunk])
                  for i in range(args.requests)]
    else:
        chunks = [np.zeros(0, dtype=int) for _ in range(args.requests)]
    req_idx = [np.union1d(core, chunks[i]) for i in range(args.requests)]
    req_rids = [w.rids_of(g) for g in req_idx]

    retain_visits = {"stateless": 0, "su": 0, "recompute": 0}
    hset: set[int] = set()
    for g in req_idx:
        new = set(g.tolist()) - hset
        hset |= set(g.tolist())
        keep = w.n - len(hset)
        # the stateless service executes the operator once per request; the
        # service executes it only when the requested effect is not yet covered
        retain_visits["stateless"] += args.ue_epochs * keep
        if new:
            retain_visits["su"] += args.ue_epochs * keep
        retain_visits["recompute"] += args.retrain_epochs * keep
    keep_all = np.setdiff1d(np.arange(w.n), np.array(sorted(hset)))
    oracle_path = w.oracle_for(keep_all, "e14")
    P_oracle = w.probs(oracle_path)
    P_base = w.probs(w.base_path)
    print(f"  oracle acc {MET.accuracy(w.model(oracle_path), w.Fte, w.yte):.4f}",
          flush=True)

    rows = []
    for mode, label in (("stateless", "stateless"), ("su", "delta")):
        td = tempfile.mkdtemp()
        try:
            be = w.backend(method="neggradplus", epochs=args.ue_epochs,
                           artifact_subdir=f"e14/{mode}")
            svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                                method="neggradplus", epochs=args.ue_epochs)
            svc.bootstrap()
            t0 = time.time()
            for i, rids in enumerate(req_rids):
                svc.delete(f"k{i}", "tenant", rids)
            wall = time.time() - t0
            st = svc.stats()
            p = MET.probs_of(w.model(svc.current_artifact()), w.Fte)
            rows.append(dict(
                strategy=label, pattern=args.pattern,
                requests=args.requests, jobs=st["physical_runs"],
                versions=st["versions"] - 1, cost_examples=st["cost_examples"],
                retain_visits=retain_visits[mode], wall_s=wall,
                test_acc=MET.accuracy_of_probs(p, w.yte),
                js_to_oracle=MET.js_divergence(p, P_oracle),
                js_to_base=MET.js_divergence(p, P_base)))
            print(f"  {label:10s} jobs={st['physical_runs']} "
                  f"retain={retain_visits[mode]:,} wall={wall:.1f}s "
                  f"acc={rows[-1]['test_acc']:.4f} "
                  f"js={rows[-1]['js_to_oracle']:.4f}", flush=True)
            svc.close()
        finally:
            shutil.rmtree(td, ignore_errors=True)

    t0 = time.time()
    m = None
    hset = set()
    for g in req_idx:
        hset |= set(g.tolist())
        keep = np.setdiff1d(np.arange(w.n), np.array(sorted(hset)))
        m = P.train_head(w.Ftr[keep], w.ytr[keep], w.arch, w.n_classes,
                         epochs=args.retrain_epochs, lr=C.BASE_LR,
                         weight_decay=C.BASE_WD, seed=w.seed, verbose=False)
    wall = time.time() - t0
    p = MET.probs_of(m, w.Fte)
    rows.append(dict(
        strategy="recompute", pattern=args.pattern, requests=args.requests,
        jobs=args.requests,
        versions=args.requests, cost_examples=retain_visits["recompute"],
        retain_visits=retain_visits["recompute"], wall_s=wall,
        test_acc=MET.accuracy_of_probs(p, w.yte),
        js_to_oracle=MET.js_divergence(p, P_oracle),
        js_to_base=MET.js_divergence(p, P_base)))
    print(f"  recompute  jobs={args.requests} retain={retain_visits['recompute']:,} "
          f"wall={wall:.1f}s acc={rows[-1]['test_acc']:.4f} "
          f"js={rows[-1]['js_to_oracle']:.4f}", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
