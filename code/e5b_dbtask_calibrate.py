"""E5b -- calibrate the unlearning operator for the cardinality-estimation backend.

NegGrad+ applies gradient ascent, and for an MSE-regression head the ascent
direction is unbounded, so an uncalibrated configuration destroys the estimator in
a single invocation.  We sweep the step budget and step size and pick the setting in
which (a) one invocation leaves Q-error near the base model and (b) replay is
visible.  This is reported in the paper as a limitation of gradient-ascent
unlearning for regression targets, not hidden.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from su import config as C
from su import harness as H
from su import metrics as MET
from su import platform as P
from su.dbtask import DBWorkload, build
from su.methods import UnlearnConfig, run as run_method

torch.set_num_threads(int(C.NT))


def make_model(dim, hidden=256):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def train_ce(X, y, seed, epochs=60, lr=1e-3, wd=1e-5, bs=256):
    import torch.nn as nn
    torch.manual_seed(seed)
    m = make_model(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            nn.functional.mse_loss(m(X[idx]).squeeze(-1), y[idx]).backward()
            opt.step()
        sched.step()
    m.eval()
    return m


def metrics(m, Xte, card):
    with torch.no_grad():
        est = np.expm1(m(Xte).squeeze(-1).numpy().astype(np.float64))
    s = MET.q_error_stats(est, card)
    return {k: s[k] for k in ("qerr_median", "qerr_geomean", "qerr_p95")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e5b_calibration.csv")
    ap.add_argument("--cell", default="neggradplus:3:1e-4,neggradplus:3:1e-3,finetune:3:1e-3,finetune:10:1e-3,finetune:20:1e-3")
    args = ap.parse_args()
    data = build(DBWorkload())
    Xtr = torch.from_numpy(data["Xtr"]).float()
    ytr = torch.from_numpy(data["ytr"]).float()
    Xte = torch.from_numpy(data["Xte"]).float()
    card = data["card_te"]
    tenant = data["tenant"]
    base = train_ce(Xtr, ytr, 0)
    rows = [dict(cell="base", **metrics(base, Xte, card))]
    print("  base", rows[-1], flush=True)
    t = 0
    mask = tenant == t
    keep = torch.from_numpy(np.where(~mask)[0])
    forget = torch.from_numpy(np.where(mask)[0])
    oracle = train_ce(Xtr[keep], ytr[keep], 0)
    rows.append(dict(cell="oracle", **metrics(oracle, Xte, card)))
    print("  oracle", rows[-1], flush=True)
    for cell in args.cell.split(","):
        method, ep, lr = cell.split(":")
        ep, lr = int(ep), float(lr)
        cur = make_model(Xtr.shape[1])
        cur.load_state_dict(base.state_dict())
        for k in range(1, 4):
            cfg = UnlearnConfig(method=method, epochs=ep, lr=lr, weight_decay=1e-5,
                                beta=0.95, batch_size=256, seed=k, loss="mse", grad_clip=1.0)
            cur = run_method(cur, Xtr, ytr, keep, forget, cfg)
            r = dict(cell=cell, k=k, **metrics(cur, Xte, card))
            rows.append(r)
            print(f"  {cell:22s} k={k} {r}", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
