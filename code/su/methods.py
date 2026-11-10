"""Approximate unlearning operators.

Every operator is a *finite-step model-update procedure* with the signature

    run(model, X, y, retain_idx, forget_idx, cfg) -> model'

which is exactly the stateless interface that the original ICLR submission
criticises.  The service layer is responsible for the deletion *semantics*;
these operators are deliberately kept stateless so that the two concerns can be
measured separately.

Supported operators
-------------------
retrain        : exact reference (handled by the platform, not here)
finetune       : further SGD on the retain set only
finetune_l1    : finetune + L1 parameter sparsity penalty (SPARSITY-style)
finetune_noisy : finetune with per-gradient Gaussian noise
neggradplus    : beta * CE(retain) - (1-beta) * CE(forget)   (gradient ascent)
salun          : SalUn-style saliency-masked forget update

An "epoch" is one pass over the retain set.  When an operator needs a forgetting
term, a fresh forget batch is drawn *with replacement* at every step, which keeps
the constant forgetting pressure that makes repeated replay non-trivial.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UnlearnConfig:
    method: str = "neggradplus"
    epochs: int = 10
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 256
    beta: float = 0.95                # neggradplus / gen retain weight
    l1_lambda: float = 0.0            # sparsity penalty for finetune_l1
    noise_sigma: float = 0.0          # gradient noise for finetune_noisy
    saliency_coef: float = 1.0        # salun mask scale
    mask_fraction: float = 0.5        # salun mask keep fraction
    grad_clip: float = 5.0
    seed: int = 0
    loss: str = "ce"                  # "ce" for classification, "mse" for regression
    forget_mode: str = "matched"      # "matched": same-size forget batch every step
                                      # "proportional": one pass over the forget set per epoch
    extra: dict = field(default_factory=dict)

    def copy(self, **kw) -> "UnlearnConfig":
        d = dict(self.__dict__)
        d.update(kw)
        return UnlearnConfig(**d)


def _needs_forget(method: str) -> bool:
    return method in ("neggradplus", "salun")


def run(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
        retain_idx: torch.Tensor, forget_idx: torch.Tensor,
        cfg: UnlearnConfig) -> nn.Module:
    """Apply one finite-step unlearning invocation and return the updated model."""
    if cfg.method == "retrain":
        raise ValueError("retrain is handled by the platform, not by run()")

    out = copy.deepcopy(model)
    out.train()
    gen = torch.Generator().manual_seed(int(cfg.seed))
    opt = torch.optim.SGD(out.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)

    n_ret = int(len(retain_idx))
    n_for = int(len(forget_idx))
    if n_for == 0:
        # A correct service never reaches this point: Delta = empty means no-op.
        # Defining it as a no-op keeps the "empty delta" ablation well defined.
        out.eval()
        return out

    mask = None
    if cfg.method == "salun":
        mask = _saliency_mask(out, X, y, forget_idx, cfg)

    steps = math.ceil(n_ret / cfg.batch_size)
    for _ in range(cfg.epochs):
        perm = torch.randperm(n_ret, generator=gen)
        fperm = torch.randperm(n_for, generator=gen) if cfg.forget_mode == "proportional" \
            else None
        fpos = 0
        for s in range(steps):
            ri = retain_idx[perm[s * cfg.batch_size:(s + 1) * cfg.batch_size]]
            opt.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=X.device)
            if cfg.method in ("finetune", "finetune_l1", "finetune_noisy", "salun", "neggradplus"):
                loss = loss + _ce_batch(out, X, y, ri, cfg.loss)
            if cfg.method == "neggradplus":
                # J_A(theta) = beta * L_retain(theta) - (1-beta) * L_forget(theta)
                fi = _forget_batch(forget_idx, fperm, fpos, len(ri), n_for, cfg, gen)
                if cfg.forget_mode == "proportional":
                    fpos += len(ri)
                if fi is not None:
                    loss = cfg.beta * loss - (1.0 - cfg.beta) * _ce_batch(out, X, y, fi, cfg.loss)
                else:
                    loss = cfg.beta * loss
            elif cfg.method == "salun":
                fi = _forget_batch(forget_idx, fperm, fpos, len(ri), n_for, cfg, gen)
                if cfg.forget_mode == "proportional":
                    fpos += len(ri)
                if fi is not None:
                    loss = loss - 0.5 * _ce_batch(out, X, y, fi, cfg.loss)
            if cfg.method == "finetune_l1" and cfg.l1_lambda > 0:
                loss = loss + cfg.l1_lambda * sum(p.abs().sum() for p in out.parameters()
                                                  if p.requires_grad)
            if not torch.isfinite(loss):
                break
            loss.backward()
            if cfg.method == "finetune_noisy" and cfg.noise_sigma > 0:
                with torch.no_grad():
                    for p in out.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad) * cfg.noise_sigma)
            if cfg.grad_clip and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(out.parameters(), cfg.grad_clip)
            if mask is not None:
                for name, p in out.named_parameters():
                    if name in mask and p.grad is not None:
                        p.grad.mul_(mask[name])
            opt.step()
    out.eval()
    return out


def _ce_batch(model, X, y, idx, kind="ce"):
    out = model(X[idx])
    if kind == "mse":
        return F.mse_loss(out.squeeze(-1), y[idx])
    return F.cross_entropy(out, y[idx])


def _forget_batch(forget_idx, fperm, fpos, batch, n_for, cfg, gen):
    """Forget minibatch under one of the two sampling conventions."""
    if n_for == 0:
        return None
    if cfg.forget_mode == "proportional":
        if fpos >= n_for:
            return None
        take = min(batch, n_for - fpos)
        return forget_idx[fperm[fpos:fpos + take]]
    return forget_idx[torch.randint(0, n_for, (batch,), generator=gen)]


def _saliency_mask(model, X, y, forget_idx, cfg) -> dict:
    """Gradient-magnitude saliency mask computed on the forget set (SalUn-lite)."""
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    params = [p for p in model.parameters() if p.requires_grad]
    take = forget_idx[: min(len(forget_idx), 4096)]
    grads = torch.autograd.grad(F.cross_entropy(model(X[take]), y[take]), params,
                                allow_unused=True)
    mask = {}
    for name, g in zip(names, grads):
        if g is None:
            continue
        if cfg.mask_fraction >= 1.0:
            mask[name] = torch.ones_like(g) * cfg.saliency_coef
            continue
        thr = torch.quantile(g.abs().flatten().float(), 1.0 - cfg.mask_fraction) \
            if g.numel() > 1 else g.abs().max()
        mask[name] = (g.abs() >= thr).float() * cfg.saliency_coef
    return mask


def example_cost(n_retain: int, n_forget: int, cfg: UnlearnConfig) -> int:
    """Logical work of one invocation, in example-visits (the planner's cost unit)."""
    steps = math.ceil(n_retain / cfg.batch_size)
    per_step = min(n_retain, cfg.batch_size) + (min(n_forget, cfg.batch_size)
                                                if _needs_forget(cfg.method) else 0)
    return int(steps * per_step * cfg.epochs)
