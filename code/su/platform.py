"""Data platform: frozen-encoder features, base training, retrain oracles, artifact store.

Design rationale
----------------
ICDE-style evaluation needs many repeated unlearning trajectories (methods x
request structures x seeds x replay counts x concurrency).  Full deep-network
retraining on CPU is not feasible at that scale, so the platform uses the
*last-layer unlearning* regime: a frozen encoder produces 512-d features and the
deleted unit of influence lives in a small head that is actually optimised.

This is not a toy: last-layer / linear-probe unlearning is one of the standard
deployment patterns for deletion services, and it preserves the property the
paper is about -- a finite-step optimisation operator exposed as a model-update
procedure.  A full-CNN replication config is provided for the subset of
experiments where wall-clock permits.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
CODE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../code/su
ROOT = os.path.dirname(CODE_DIR)                                # .../code
ROOT = os.path.dirname(ROOT)                                    # .../icde2027
DATA = os.path.join(ROOT, "data")
ARTIFACTS = os.path.join(ROOT, "artifacts")
RESULTS = os.path.join(ROOT, "results")
FEATURES = os.path.join(ARTIFACTS, "features")
CKPT = os.path.join(ARTIFACTS, "checkpoints")
for _d in (DATA, ARTIFACTS, RESULTS, FEATURES, CKPT):
    os.makedirs(_d, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------- #
# dataset loading
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    name: str
    Xtr: torch.Tensor      # (N,3,32,32) float in [0,1]
    ytr: torch.Tensor
    Xte: torch.Tensor
    yte: torch.Tensor
    n_classes: int

    def __len__(self):
        return len(self.Xtr)


def load_image_dataset(name: str = "cifar10") -> Dataset:
    import torchvision
    from torchvision import transforms
    tf = transforms.ToTensor()
    if name == "cifar10":
        cls, nc = torchvision.datasets.CIFAR10, 10
    elif name == "cifar100":
        cls, nc = torchvision.datasets.CIFAR100, 100
    else:
        raise ValueError(name)
    tr = cls(DATA, train=True, download=True, transform=tf)
    te = cls(DATA, train=False, download=True, transform=tf)
    Xtr = torch.stack([x for x, _ in tr])
    ytr = torch.tensor([y for _, y in tr])
    Xte = torch.stack([x for x, _ in te])
    yte = torch.tensor([y for _, y in te])
    return Dataset(name, Xtr, ytr, Xte, yte, nc)


# --------------------------------------------------------------------------- #
# encoders
# --------------------------------------------------------------------------- #
class FrozenEncoder(nn.Module):
    """ImageNet ResNet-18 penultimate features for small images."""

    def __init__(self, img_size: int = 128, dim: int = 512):
        super().__init__()
        import torchvision.models as M
        m = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Identity()
        self.m = m
        self.img_size = img_size
        self.dim = dim
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=self.img_size, mode="bilinear", align_corners=False)
        return self.m((x - self.mean) / self.std)


@torch.no_grad()
def extract_features(ds: Dataset, tag: str = "resnet18_128", img_size: int = 128,
                     batch_size: int = 128, rebuild: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (train, test) features once and cache them under the artifact store."""
    ftr = os.path.join(FEATURES, f"{ds.name}_{tag}_train.npy")
    fte = os.path.join(FEATURES, f"{ds.name}_{tag}_test.npy")
    if os.path.exists(ftr) and os.path.exists(fte) and not rebuild:
        return torch.from_numpy(np.load(ftr)), torch.from_numpy(np.load(fte))
    enc = FrozenEncoder(img_size=img_size)
    enc.eval()

    def run(X):
        out = []
        t0 = time.time()
        for i in range(0, len(X), batch_size):
            out.append(enc(X[i:i + batch_size]))
            if i % (batch_size * 20) == 0:
                print(f"    [{tag}] {i}/{len(X)} {time.time()-t0:.0f}s", flush=True)
        return torch.cat(out).numpy().astype(np.float32)

    a = run(ds.Xtr)
    b = run(ds.Xte)
    np.save(ftr, a)
    np.save(fte, b)
    return torch.from_numpy(a), torch.from_numpy(b)


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    def __init__(self, dim: int, n_classes: int, hidden: int = 0):
        super().__init__()
        if hidden:
            self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_classes))
        else:
            self.net = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.net(x)


def make_head(arch: str, dim: int, n_classes: int) -> nn.Module:
    if arch == "linear":
        return LinearHead(dim, n_classes, hidden=0)
    if arch == "mlp":
        return LinearHead(dim, n_classes, hidden=256)
    raise ValueError(arch)


def train_head(X, y, arch: str, n_classes: int, epochs: int = 20, lr: float = 0.05,
               momentum: float = 0.9, weight_decay: float = 5e-4, batch_size: int = 256,
               seed: int = 0, model: Optional[nn.Module] = None,
               verbose: bool = False) -> nn.Module:
    set_seed(seed)
    m = model if model is not None else make_head(arch, X.shape[1], n_classes)
    m.train()
    opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    gen = torch.Generator().manual_seed(seed)
    n = len(X)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(X[idx]), y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"    epoch {ep} loss {tot/n:.4f}", flush=True)
    m.eval()
    return m


# --------------------------------------------------------------------------- #
# content-addressed artifact store
# --------------------------------------------------------------------------- #
def state_digest(model: nn.Module) -> str:
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def save_state(model: nn.Module, name: str) -> tuple[str, str]:
    path = os.path.join(CKPT, f"{name}.pt")
    torch.save(model.state_dict(), path)
    return path, state_digest(model)


def load_state(path: str, arch: str, dim: int, n_classes: int) -> nn.Module:
    m = make_head(arch, dim, n_classes)
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def load_cached_state(name: str) -> Optional[str]:
    p = os.path.join(CKPT, f"{name}.pt")
    return p if os.path.exists(p) else None


def write_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
