"""E0b -- calibration of the unlearning budget on the new platform.

We need a configuration in which (a) a single request does something meaningful
(forget accuracy moves towards the exact-retraining oracle) and (b) replay is
observable.  This script sweeps beta, epochs and the operator and prints the
one-shot and replay behaviour, so the default configuration is chosen from data
rather than folklore.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--cells", default="neggradplus:10:0.5,neggradplus:10:0.9,neggradplus:10:0.95,finetune:10:0.95,finetune:20:0.95")
    args = ap.parse_args()

    w = World.build(args.dataset, args.arch, args.seed)
    A = np.sort(request_indices(w.n, w.request_ratio, w.request_seed))
    retain = torch.tensor(np.setdiff1d(np.arange(w.n), A), dtype=torch.long)
    forget = torch.tensor(A, dtype=torch.long)
    base = w.model(w.base_path)
    oracle_path = w.oracle_for(np.setdiff1d(np.arange(w.n), A), "e0b")
    oracle = w.model(oracle_path)
    pr = w.probs(oracle_path)
    # reference quantities
    forget_acc_oracle = MET.accuracy(oracle, w.Ftr[forget], w.ytr[forget])
    forget_acc_base = MET.accuracy(base, w.Ftr[forget], w.ytr[forget])
    acc_base = MET.accuracy(base, w.Fte, w.yte)
    acc_oracle = MET.accuracy(oracle, w.Fte, w.yte)
    print(f"  base: test {acc_base:.4f} forget {forget_acc_base:.4f}")
    print(f"  oracle(retrain on D\\A): test {acc_oracle:.4f} forget {forget_acc_oracle:.4f}")
    rows = [dict(cell="reference", test_acc=acc_base, forget_acc=forget_acc_base,
                 js_to_oracle=MET.js_divergence(w.probs(w.base_path), pr)),
            dict(cell="oracle", test_acc=acc_oracle, forget_acc=forget_acc_oracle,
                 js_to_oracle=0.0)]

    for cell in args.cells.split(","):
        method, ep, beta = cell.split(":")
        ep, beta = int(ep), float(beta)
        P1, P2, P3 = [], [], []
        accs, faccs, js1, js2 = [], [], [], []
        for i in range(args.runs):
            cfg = UnlearnConfig(method=method, epochs=ep, lr=C.UE_LR, weight_decay=C.UE_WD,
                                beta=beta, batch_size=256, seed=100 + i)
            m1 = run_method(base, w.Ftr, w.ytr, retain, forget, cfg)
            m2 = run_method(m1, w.Ftr, w.ytr, retain, forget,
                            UnlearnConfig(**{**cfg.__dict__, "seed": 200 + i}))
            m3 = run_method(m2, w.Ftr, w.ytr, retain, forget,
                            UnlearnConfig(**{**cfg.__dict__, "seed": 300 + i}))
            p1, p2, p3 = (MET.probs_of(m, w.Fte) for m in (m1, m2, m3))
            P1.append(p1)
            P2.append(p2)
            P3.append(p3)
            accs.append(MET.accuracy(m1, w.Fte, w.yte))
            faccs.append(MET.accuracy(m1, w.Ftr[forget], w.ytr[forget]))
            js1.append(MET.js_divergence(p1, pr))
            js2.append(MET.js_divergence(p2, pr))
        dar2 = ST.compute_dar(P1, P2)
        dar3 = ST.compute_dar(P1, P3)
        r = dict(cell=cell, method=method, epochs=ep, beta=beta,
                 test_acc=float(np.mean(accs)), forget_acc=float(np.mean(faccs)),
                 js_to_oracle_k1=float(np.mean(js1)), js_to_oracle_k2=float(np.mean(js2)),
                 dar_loo_k2=dar2.dar_loo, a_paired_k2=dar2.a_paired,
                 dar_loo_k3=dar3.dar_loo, a_paired_k3=dar3.a_paired,
                 v_stoch=dar2.v_stoch_loo)
        rows.append(r)
        print(f"  {cell:26s} test={r['test_acc']:.4f} forget={r['forget_acc']:.4f} "
              f"js_oracle={r['js_to_oracle_k1']:.4f}->{r['js_to_oracle_k2']:.4f} "
              f"DAR2={dar2.dar_loo:.2f} Ap2={dar2.a_paired:.2f} "
              f"DAR3={dar3.dar_loo:.2f}", flush=True)
    H.write_csv(H.result_path("e0b_calibration.csv"), rows)


if __name__ == "__main__":
    main()
