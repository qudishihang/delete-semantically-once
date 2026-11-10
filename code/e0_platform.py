"""E0 -- platform construction.

Trains, for each (dataset, architecture, training seed):
  * the base model theta_0 on the full training pool D
  * the exact retraining oracle Retrain-A on D \\ A
and records plain utility so that every later comparison has a reference point.

Artifacts (all content-addressed by name):
  artifacts/checkpoints/{ds}_{arch}_s{seed}_base.pt
  artifacts/checkpoints/{ds}_{arch}_s{seed}_retrainA.pt
  results/e0_platform.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import numpy as np
import torch

from su import platform as P
from su import metrics as MET
from su import config as C

torch.set_num_threads(int(C.NT))


def request_indices(n: int, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(round(ratio * n))
    return np.sort(rng.choice(n, size=k, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="cifar10")
    ap.add_argument("--archs", default="mlp")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()

    rows = []
    for ds_name in args.datasets.split(","):
        t0 = time.time()
        ds = P.load_image_dataset(ds_name)
        print(f"[{ds_name}] loaded {len(ds)} train / {len(ds.Xte)} test in {time.time()-t0:.0f}s", flush=True)
        Ftr, Fte = P.extract_features(ds, tag=C.FEATURE_TAG)
        print(f"[{ds_name}] features {tuple(Ftr.shape)} {tuple(Fte.shape)}", flush=True)

        for arch in args.archs.split(","):
            for seed in [int(s) for s in args.seeds.split(",")]:
                base_name = f"{ds_name}_{arch}_s{seed}_base"
                retr_name = f"{ds_name}_{arch}_s{seed}_retrainA"
                t = time.time()
                base = P.train_head(Ftr, ds.ytr, arch, ds.n_classes, epochs=C.BASE_EPOCHS,
                                    lr=C.BASE_LR, weight_decay=C.BASE_WD, seed=seed, verbose=False)
                _, d0 = P.save_state(base, base_name)
                acc = MET.accuracy(base, Fte, ds.yte)
                print(f"  {base_name}: test acc {acc:.4f} ({time.time()-t:.0f}s)", flush=True)

                A = request_indices(len(Ftr), C.REQUEST_RATIO, C.REQUEST_SEED)
                mask = np.ones(len(Ftr), dtype=bool)
                mask[A] = False
                R = np.where(mask)[0]
                t = time.time()
                retr = P.train_head(Ftr[R], ds.ytr[R], arch, ds.n_classes, epochs=C.BASE_EPOCHS,
                                    lr=C.BASE_LR, weight_decay=C.BASE_WD, seed=seed, verbose=False)
                _, d1 = P.save_state(retr, retr_name)
                acc_r = MET.accuracy(retr, Fte, ds.yte)
                acc_A = MET.accuracy(retr, Ftr[A], ds.ytr[A]) if len(A) else float("nan")
                print(f"  {retr_name}: test acc {acc_r:.4f} forget-acc {acc_A:.4f} "
                      f"({time.time()-t:.0f}s)", flush=True)

                rows.append(dict(dataset=ds_name, arch=arch, seed=seed,
                                 n_train=len(Ftr), n_request=len(A),
                                 request_ratio=C.REQUEST_RATIO,
                                 base_test_acc=acc, retrainA_test_acc=acc_r,
                                 retrainA_forget_acc=acc_A,
                                 base_digest=d0, retrainA_digest=d1,
                                 base_path=os.path.join(P.CKPT, base_name + ".pt"),
                                 retrainA_path=os.path.join(P.CKPT, retr_name + ".pt")))

    out = os.path.join(C.RESULTS, "e0_platform.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
