"""E6 -- What actually controls replay sensitivity?

Three orthogonal sweeps, all on the image platform, each with n independent
unlearning trajectories so that the calibrated estimators (DAR_LOO, A_paired)
can be compared across cells:

  operators     fine-tune, fine-tune+L1, noisy fine-tune, NegGrad+ (beta sweep),
                SalUn
  budget        invocation length in epochs
  request       request size, and *structure* of the deletion set
                (random / class-concentrated / subgroup)
  heterogeneity five disjoint deletion sets of the same size: are some requests
                intrinsically less idempotent than others?

For every cell we report the calibrated statistics *and* the one-shot quality of
the endpoint, because idempotence is only one axis of interest: a method that is
idempotent but useless is not a solution.
"""
from __future__ import annotations

import argparse
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


def trajectories(w: World, A_idx, method, epochs, n_runs, cfg_over=None, seed0=0, K=2):
    base = w.model(w.base_path)
    keep = np.setdiff1d(np.arange(w.n), A_idx)
    retain = torch.tensor(keep, dtype=torch.long)
    forget = torch.tensor(np.sort(A_idx), dtype=torch.long)
    rng = np.random.default_rng(1000 + seed0)
    P1, P2 = [], []
    acc1, acc2, f1, f2, mia1, mia2 = [], [], [], [], [], []
    for i in range(n_runs):
        s = int(rng.integers(0, 2**31 - 1))
        over = dict(cfg_over or {})
        cfg = UnlearnConfig(method=method, epochs=epochs, lr=over.get("lr", C.UE_LR),
                            weight_decay=C.UE_WD, beta=over.get("beta", C.UE_BETA),
                            l1_lambda=over.get("l1", 0.0), noise_sigma=over.get("sigma", 0.0),
                            batch_size=256, seed=s)
        m1 = run_method(base, w.Ftr, w.ytr, retain, forget, cfg)
        cfg2 = UnlearnConfig(**{**cfg.__dict__, "seed": s + 1})
        m2 = run_method(m1, w.Ftr, w.ytr, retain, forget, cfg2)
        p1, p2 = MET.probs_of(m1, w.Fte), MET.probs_of(m2, w.Fte)
        P1.append(p1)
        P2.append(p2)
        acc1.append(MET.accuracy(m1, w.Fte, w.yte))
        acc2.append(MET.accuracy(m2, w.Fte, w.yte))
        f1.append(MET.accuracy(m1, w.Ftr[forget], w.ytr[forget]))
        f2.append(MET.accuracy(m2, w.Ftr[forget], w.ytr[forget]))
        mia1.append(MET.mia_advantage(m1, w.Ftr[forget], w.ytr[forget], w.Fte[:3000],
                                      w.yte[:3000])["mia_advantage"])
        mia2.append(MET.mia_advantage(m2, w.Ftr[forget], w.ytr[forget], w.Fte[:3000],
                                      w.yte[:3000])["mia_advantage"])
    dar = ST.compute_dar(P1, P2)
    pr = w.probs(w.retrain_path)
    return dict(dar_loo=dar.dar_loo, a_paired=dar.a_paired, v_stoch=dar.v_stoch_loo,
                v_paired=dar.v_paired,
                acc_k1=float(np.mean(acc1)), acc_k2=float(np.mean(acc2)),
                forget_acc_k1=float(np.mean(f1)), forget_acc_k2=float(np.mean(f2)),
                mia_k1=float(np.mean(mia1)), mia_k2=float(np.mean(mia2)),
                js_to_retrain_k1=float(np.mean([MET.js_divergence(p, pr) for p in P1])),
                js_to_retrain_k2=float(np.mean([MET.js_divergence(p, pr) for p in P2])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--sweeps", default="operators,budget,request,heterogeneity")
    ap.add_argument("--out", default="e6_sensitivity.csv")
    args = ap.parse_args()

    w = World.build(args.dataset, args.arch, args.seed)
    A_default = np.sort(request_indices(w.n, w.request_ratio, w.request_seed))
    rows = []

    def flush():
        if rows:
            H.write_csv(H.result_path(args.out), rows)

    if "operators" in args.sweeps:
        cells = [
            ("finetune", 10, {}),
            ("finetune_l1", 10, {"l1": 1e-2}),
            ("finetune_noisy", 10, {"sigma": 0.05}),
            ("neggradplus", 10, {"beta": 0.90}),
            ("neggradplus", 10, {"beta": 0.95}),
            ("neggradplus", 10, {"beta": 0.99}),
            ("salun", 10, {}),
        ]
        for method, ep, over in cells:
            r = trajectories(w, A_default, method, ep, args.runs, over)
            rows.append(dict(sweep="operator", method=method, epochs=ep, seed=args.seed,
                             param=over.get("beta", over.get("l1", over.get("sigma", ""))),
                             n_request=len(A_default), **r))
            flush()
            print(f"  operator {method:15s} p={rows[-1]['param']!s:6s} "
                  f"DAR={r['dar_loo']:.2f} A_p={r['a_paired']:.2f} "
                  f"acc={r['acc_k1']:.4f}->{r['acc_k2']:.4f} "
                  f"forget={r['forget_acc_k1']:.3f}->{r['forget_acc_k2']:.3f}", flush=True)

    if "budget" in args.sweeps:
        for method in ("neggradplus", "finetune"):
            for ep in (1, 2, 5, 10):
                if method == "finetune" and ep == 10:
                    continue
                r = trajectories(w, A_default, method, ep, args.runs)
                rows.append(dict(sweep="budget", method=method, epochs=ep, seed=args.seed,
                                 param=ep, n_request=len(A_default), **r))
                flush()
                print(f"  budget {method:12s} ep={ep:2d} DAR={r['dar_loo']:.2f} "
                      f"A_p={r['a_paired']:.2f} acc={r['acc_k1']:.4f}->{r['acc_k2']:.4f}",
                      flush=True)

    if "request" in args.sweeps:
        for ratio in (0.05, 0.10, 0.20):
            idx = np.sort(request_indices(w.n, ratio, 31))
            r = trajectories(w, idx, "neggradplus", 10, args.runs)
            rows.append(dict(sweep="request_size", method="neggradplus", epochs=10,
                             seed=args.seed, param=ratio, n_request=len(idx), **r))
            flush()
            print(f"  request size {ratio:.3f} DAR={r['dar_loo']:.2f} "
                  f"A_p={r['a_paired']:.2f} acc={r['acc_k1']:.4f}->{r['acc_k2']:.4f}",
                  flush=True)
    if "structure" in args.sweeps:
        labels = w.ytr.numpy()
        structures = [
            ("random", None, "random"),
            ("class_0_1", [0, 1], "class"),
            ("class_0", [0], "class"),
        ]
        for name, cls, kind in structures:
            if kind == "random":
                idx = np.sort(request_indices(w.n, 0.10, 41))
            else:
                cand = np.where(np.isin(labels, cls))[0]
                rng = np.random.default_rng(41)
                k = int(round(0.10 * w.n))
                idx = np.sort(rng.choice(cand, size=min(k, len(cand)), replace=False))
            r = trajectories(w, idx, "neggradplus", 10, args.runs)
            rows.append(dict(sweep="request_structure", method="neggradplus", epochs=10,
                             seed=args.seed, param=name, n_request=len(idx), **r))
            flush()
            print(f"  structure {name:12s} |A|={len(idx)} DAR={r['dar_loo']:.2f} "
                  f"A_p={r['a_paired']:.2f} acc={r['acc_k1']:.4f}->{r['acc_k2']:.4f}",
                  flush=True)

    if "heterogeneity" in args.sweeps:
        perm = np.random.default_rng(77).permutation(w.n)
        chunk = len(A_default)
        for j in range(4):
            idx = np.sort(perm[j * chunk:(j + 1) * chunk])
            r = trajectories(w, idx, "neggradplus", 10, args.runs)
            rows.append(dict(sweep="heterogeneity", method="neggradplus", epochs=10,
                             seed=args.seed, param=f"set{j}", n_request=len(idx), **r))
            flush()
            print(f"  delete-set {j}: DAR={r['dar_loo']:.2f} A_p={r['a_paired']:.2f} "
                  f"acc={r['acc_k1']:.4f}->{r['acc_k2']:.4f}", flush=True)

    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
