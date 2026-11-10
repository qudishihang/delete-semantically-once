"""E9 -- full-network replication on a *converged* convolutional network.

Earlier we ran this experiment with a 12-epoch network, where the retained-data
term still dominates an invocation, and replay drift stayed inside the noise floor.
This version trains the same architecture to convergence (many more epochs) so the
endpoint of the first invocation is close to a fixed point of the retain objective;
if the fixed-point argument is right, replay should now visibly move the model.

Reports calibrated statistics for k = 2..K.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from su import config as C
from su import harness as H
from su import metrics as MET
from su import platform as P
from su import stats as ST
from su.methods import UnlearnConfig, run as run_method

torch.set_num_threads(int(C.NT))
CKPT = os.path.join(P.ARTIFACTS, "cnn")


class CNN(nn.Module):
    def __init__(self, n_classes=10, width=32):
        super().__init__()
        w = width
        self.c1 = nn.Conv2d(3, w, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(w)
        self.c2 = nn.Conv2d(w, 2 * w, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(2 * w)
        self.c3 = nn.Conv2d(2 * w, 4 * w, 3, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(4 * w)
        self.fc = nn.Linear(4 * w, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        x = F.max_pool2d(F.relu(self.b3(self.c3(x))), 2)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def train_cnn(X, y, seed, epochs, tag, lr=0.05, wd=5e-4, bs=256):
    path = os.path.join(CKPT, f"{tag}.pt")
    if os.path.exists(path):
        m = CNN()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        return m
    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(seed)
    m = CNN()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        t0 = time.time()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"    epoch {ep} loss {tot/n:.4f} ({time.time()-t0:.0f}s)", flush=True)
    m.eval()
    torch.save(m.state_dict(), path)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--kmax", type=int, default=3)
    ap.add_argument("--base-epochs", type=int, default=80)
    ap.add_argument("--ue-epochs", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=0.10)
    ap.add_argument("--with-oracle", action="store_true")
    ap.add_argument("--tag", default="conv")
    ap.add_argument("--out", default="e9_fullnet_conv.csv")
    args = ap.parse_args()

    ds = P.load_image_dataset("cifar10")
    mean = ds.Xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = ds.Xtr.std(dim=(0, 2, 3), keepdim=True)
    Xtr = (ds.Xtr - mean) / std
    Xte = (ds.Xte - mean) / std
    rng = np.random.default_rng(C.REQUEST_SEED)
    A = np.sort(rng.choice(len(Xtr), size=int(args.ratio * len(Xtr)), replace=False))
    keep = np.setdiff1d(np.arange(len(Xtr)), A)
    tag = f"s{args.seed}"

    t0 = time.time()
    base = train_cnn(Xtr, ds.ytr, args.seed, args.base_epochs, tag=f"{args.tag}_base_{tag}")
    acc_b = MET.accuracy(base, Xte, ds.yte)
    acc_fb = MET.accuracy(base, Xtr[A], ds.ytr[A])
    print(f"  base trained in {time.time()-t0:.0f}s: test acc {acc_b:.4f} "
          f"forget acc {acc_fb:.4f}", flush=True)

    retain = torch.tensor(keep, dtype=torch.long)
    forget = torch.tensor(A, dtype=torch.long)
    probs = {k: [] for k in range(1, args.kmax + 1)}
    accs = {k: [] for k in range(1, args.kmax + 1)}
    fgas = {k: [] for k in range(1, args.kmax + 1)}
    rows = []
    rng2 = np.random.default_rng(4242 + args.seed)
    for i in range(args.runs):
        s = int(rng2.integers(0, 2**31 - 1))
        cur = base
        rec = {"run": i}
        t0 = time.time()
        for k in range(1, args.kmax + 1):
            cfg = UnlearnConfig(method="neggradplus", epochs=args.ue_epochs, lr=0.01,
                                weight_decay=5e-4, beta=0.95, batch_size=256, seed=s + k)
            cur = run_method(cur, Xtr, ds.ytr, retain, forget, cfg)
            probs[k].append(MET.probs_of(cur, Xte))
            accs[k].append(MET.accuracy(cur, Xte, ds.yte))
            fgas[k].append(MET.accuracy(cur, Xtr[A], ds.ytr[A]))
            rec[f"acc_k{k}"] = accs[k][-1]
            rec[f"forget_k{k}"] = fgas[k][-1]
        rec["seconds"] = time.time() - t0
        rows.append(rec)
        print(f"    run {i}: " + " ".join(
            f"acc{k}={rec[f'acc_k{k}']:.4f}" for k in range(1, args.kmax + 1)) + " "
            + " ".join(f"forget{k}={rec[f'forget_k{k}']:.4f}"
                       for k in range(1, args.kmax + 1))
            + f" ({rec['seconds']:.0f}s)", flush=True)

    out = dict(dataset="cifar10", model="cnn4conv", seed=args.seed, runs=args.runs,
               base_epochs=args.base_epochs, ue_epochs=args.ue_epochs,
               base_acc=acc_b, base_forget_acc=acc_fb)
    for k in range(1, args.kmax + 1):
        out[f"mean_acc_k{k}"] = float(np.mean(accs[k]))
        out[f"mean_forget_k{k}"] = float(np.mean(fgas[k]))
    for k in range(2, args.kmax + 1):
        s = ST.summarise(probs[1], probs[k], n_boot=600)
        for kk, v in s.items():
            out[f"k{k}_{kk}"] = v
        print(f"  k={k}: DAR_LOO={s['dar_loo']:.3f}+-{s['dar_loo_se']:.3f} "
              f"A_paired={s['a_paired']:.3f} "
              f"p(A_p<=1)={s['p_paired_exceeds_stoch']:.4f}", flush=True)

    H.write_json(H.result_path(args.out.replace(".csv", ".json")), out)

    if args.with_oracle:
        t0 = time.time()
        oracle = train_cnn(Xtr[keep], ds.ytr[keep], args.seed, args.base_epochs,
                           tag=f"{args.tag}_oracle_{tag}")
        out["oracle_acc"] = MET.accuracy(oracle, Xte, ds.yte)
        out["oracle_forget_acc"] = MET.accuracy(oracle, Xtr[A], ds.ytr[A])
        print(f"  oracle trained in {time.time()-t0:.0f}s: acc {out['oracle_acc']:.4f} "
              f"forget {out['oracle_forget_acc']:.4f}", flush=True)
        H.write_json(H.result_path(args.out.replace(".csv", ".json")), out)

    H.write_csv(H.result_path(args.out), [out] + [{**r, **{k: "" for k in out}}
                                                  for r in rows])


if __name__ == "__main__":
    main()
