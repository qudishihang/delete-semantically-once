"""E7 -- Corrected duplicate-amplification statistics.

For n independent unlearning trajectories and replay counts k = 1..4 it reports

  * V_stoch^LOO, V_dup^LOO and DAR^LOO (exactly 1 under perfect idempotence),
  * A_paired = V_paired / V_stoch^LOO (exactly 0 under perfect idempotence),
  * the old, miscalibrated pairwise ratio and its n/(n-1) bias,
  * a bootstrap standard error and interval with the unlearning trajectory as the
    resampling unit and test examples as a nested level,
  * the one-sided bootstrap probability that paired replay drift stays inside the
    single-request noise floor,
  * the k=1 effectiveness measurements (forget accuracy, membership-inference
    advantage) next to the exact-retraining reference, so that request completion
    and unlearning effectiveness can be reported separately.

Trajectory endpoints are cached, so the statistics can be recomputed without
re-running the models.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from su import config as C
from su import harness as H
from su import metrics as MET
from su import stats as ST
from su.methods import UnlearnConfig, run as run_method
from su.world import World, request_indices

torch.set_num_threads(int(C.NT))
ENDPOINT_DIR = os.path.join(C.ARTIFACTS, "endpoints")
REF_DIR = os.path.join(C.ARTIFACTS, "endpoint_refs")


def _paths(tag, method, i, k):
    base = os.path.join(ENDPOINT_DIR, f"{tag}_{method}_r{i}_k{k}")
    return base + ".pt", base + ".npy", base + ".json"


def trajectories(w: World, method: str, epochs: int, n_runs: int, A_idx: np.ndarray,
                 ks=(1, 2, 3, 4), tag: str = ""):
    os.makedirs(ENDPOINT_DIR, exist_ok=True)
    rng = np.random.default_rng(1234 + w.seed)
    base = w.model(w.base_path)
    keep = np.setdiff1d(np.arange(w.n), A_idx)
    retain = torch.tensor(keep, dtype=torch.long)
    forget = torch.tensor(np.sort(A_idx), dtype=torch.long)
    out = {"probs": {k: [] for k in ks}, "forget_acc": {k: [] for k in ks},
           "mia": {k: [] for k in ks}, "test_acc": {k: [] for k in ks}}
    for i in range(n_runs):
        runs_seed = int(rng.integers(0, 2**31 - 1))
        cur = base
        for k in ks:
            pt, npyp, jsonp = _paths(tag, method, i, k)
            if os.path.exists(npyp) and os.path.exists(jsonp):
                p = np.load(npyp)
                meta = json.load(open(jsonp))
                cur = None
            elif os.path.exists(pt):
                # endpoint checkpoint already on disk: recompute measurements only
                cur = w.model(pt)
                p = MET.probs_of(cur, w.Fte)
                mia = MET.mia_advantage(cur, w.Ftr[forget], w.ytr[forget],
                                        w.Fte[:4000], w.yte[:4000])
                meta = dict(forget_acc=MET.accuracy(cur, w.Ftr[forget], w.ytr[forget]),
                            mia_advantage=mia["mia_advantage"], mia_auc=mia["mia_auc"],
                            test_acc=MET.accuracy(cur, w.Fte, w.yte))
                np.save(npyp, p.astype(np.float32))
                json.dump(meta, open(jsonp, "w"))
            else:
                cfg = UnlearnConfig(method=method, epochs=epochs, lr=C.UE_LR,
                                    weight_decay=C.UE_WD, beta=C.UE_BETA,
                                    batch_size=256, seed=runs_seed + k)
                cur = run_method(cur, w.Ftr, w.ytr, retain, forget, cfg)
                p = MET.probs_of(cur, w.Fte)
                mia = MET.mia_advantage(cur, w.Ftr[forget], w.ytr[forget],
                                        w.Fte[:4000], w.yte[:4000])
                meta = dict(forget_acc=MET.accuracy(cur, w.Ftr[forget], w.ytr[forget]),
                            mia_advantage=mia["mia_advantage"], mia_auc=mia["mia_auc"],
                            test_acc=MET.accuracy(cur, w.Fte, w.yte))
                torch.save(cur.state_dict(), pt)
                np.save(npyp, p.astype(np.float32))
                json.dump(meta, open(jsonp, "w"))
            out["probs"][k].append(p)
            out["forget_acc"][k].append(float(meta["forget_acc"]))
            out["mia"][k].append(float(meta["mia_advantage"]))
            out["test_acc"][k].append(float(meta["test_acc"]))
            if cur is None and k < max(ks):
                if not os.path.exists(_paths(tag, method, i, k + 1)[0]):
                    cur = w.model(pt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="neggradplus,finetune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--runs", type=int, default=6)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="e7_stats.csv")
    args = ap.parse_args()

    rows, ref_rows = [], []
    for seed in [int(s) for s in args.seeds.split(",")]:
        w = World.build(args.dataset, args.arch, seed)
        A_idx = np.sort(request_indices(w.n, w.request_ratio, w.request_seed))
        keep = np.setdiff1d(np.arange(w.n), A_idx)
        oracle_path = w.oracle_for(keep, "e7")
        tag = f"{w.dataset}_{w.arch}_s{seed}"
        forget = torch.tensor(A_idx, dtype=torch.long)

        os.makedirs(REF_DIR, exist_ok=True)
        for name, path in (("base", w.base_path), ("retrainA", oracle_path)):
            f = os.path.join(REF_DIR, f"{tag}_{name}.json")
            if os.path.exists(f):
                meta = json.load(open(f))
            else:
                m = w.model(path)
                mia = MET.mia_advantage(m, w.Ftr[forget], w.ytr[forget],
                                        w.Fte[:4000], w.yte[:4000])
                meta = dict(forget_acc=MET.accuracy(m, w.Ftr[forget], w.ytr[forget]),
                            mia_advantage=mia["mia_advantage"], mia_auc=mia["mia_auc"],
                            test_acc=MET.accuracy(m, w.Fte, w.yte))
                json.dump(meta, open(f, "w"))
            ref_rows.append(dict(dataset=w.dataset, arch=w.arch, seed=seed,
                                 reference=name, **meta))

        for method in args.methods.split(","):
            with H.Timer(f"seed{seed}/{method} trajectories"):
                T = trajectories(w, method, args.epochs, args.runs, A_idx, tag=tag)
            for k in (2, 3, 4):
                s = ST.summarise(T["probs"][1], T["probs"][k], n_boot=args.boot)
                rows.append(dict(dataset=w.dataset, arch=w.arch, seed=seed, method=method,
                                 epochs=args.epochs, k=k, runs=args.runs, **s,
                                 forget_acc_k1=float(np.mean(T["forget_acc"][1])),
                                 forget_acc_k=float(np.mean(T["forget_acc"][k])),
                                 mia_k1=float(np.mean(T["mia"][1])),
                                 mia_k=float(np.mean(T["mia"][k])),
                                 test_acc_k1=float(np.mean(T["test_acc"][1])),
                                 test_acc_k=float(np.mean(T["test_acc"][k]))))
                print(f"    seed{seed} {method:12s} k={k} DAR={s['dar_loo']:.3f}"
                      f"+-{s['dar_loo_se']:.3f} A_p={s['a_paired']:.3f}"
                      f" old={s['dar_old']:.3f} p(A_p<=1)={s['p_paired_exceeds_stoch']:.4f}"
                      f" acc={rows[-1]['test_acc_k1']:.4f}->{rows[-1]['test_acc_k']:.4f}",
                      flush=True)
    H.write_csv(H.result_path(args.out), rows)
    H.write_csv(H.result_path("e7_reference.csv"), ref_rows)


if __name__ == "__main__":
    main()
