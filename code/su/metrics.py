"""Metrics: predictive divergence, utility, membership inference, Q-error."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# predictive divergence
# --------------------------------------------------------------------------- #
def logits_of(model, X, bs: int = 8192) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(model(X[i:i + bs]).float())
    return torch.cat(out)


def probs_of(model, X, bs: int = 8192) -> np.ndarray:
    logits = logits_of(model, X, bs)
    return F.softmax(logits, dim=1).numpy()


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean per-example symmetric Jensen-Shannon divergence (natural log).

    The distance is bounded by ln 2; identical distributions give exactly 0.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return float(np.mean(0.5 * kl_pm + 0.5 * kl_qm))


def js_between_probs(p: np.ndarray, q: np.ndarray) -> float:
    return js_divergence(p, q)


def per_example_js(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p) - np.log(m)), axis=1)
    kl_qm = np.sum(q * (np.log(q) - np.log(m)), axis=1)
    return 0.5 * kl_pm + 0.5 * kl_qm


# --------------------------------------------------------------------------- #
# utility
# --------------------------------------------------------------------------- #
def accuracy(model, X, y) -> float:
    pred = logits_of(model, X).argmax(1)
    return float((pred == y).float().mean())


def accuracy_of_probs(p: np.ndarray, y) -> float:
    return float((p.argmax(1) == np.asarray(y)).mean())


def per_example_ce(model, X, y, bs: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(F.cross_entropy(model(X[i:i + bs]), y[i:i + bs], reduction="none"))
    return torch.cat(out).numpy()


# --------------------------------------------------------------------------- #
# membership inference (loss based, the standard unlearning audit)
# --------------------------------------------------------------------------- #
def _auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUC via rank statistics (Mann-Whitney U)."""
    order = np.argsort(np.concatenate([scores_pos, scores_neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    all_scores = np.concatenate([scores_pos, scores_neg])
    srt = np.sort(all_scores)
    # tie correction
    uniq, inv, cnt = np.unique(all_scores, return_inverse=True, return_counts=True)
    if np.any(cnt > 1):
        # recompute with average ranks
        avg = np.zeros(len(uniq))
        start = 1
        for i, c in enumerate(cnt):
            avg[i] = (start + start + c - 1) / 2.0
            start += c
        ranks = avg[inv]
    r_pos = ranks[: len(scores_pos)].sum()
    n_pos, n_neg = len(scores_pos), len(scores_neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def mia_advantage(model, X_forget, y_forget, X_ref, y_ref) -> dict:
    """Loss-based membership inference: forget examples (members) vs held-out refs.

    score = -loss (lower loss -> more likely member).  advantage = 2*AUC-1.
    """
    lf = per_example_ce(model, X_forget, y_forget)
    lr = per_example_ce(model, X_ref, y_ref)
    auc = _auc(-lf, -lr)
    return {"mia_auc": auc, "mia_advantage": 2.0 * auc - 1.0,
            "forget_loss": float(lf.mean()), "ref_loss": float(lr.mean())}


# --------------------------------------------------------------------------- #
# cardinality estimation error
# --------------------------------------------------------------------------- #
def q_error(pred: np.ndarray, true: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """Per-query Q-error with additive smoothing so zero cardinalities are finite."""
    p = np.maximum(pred.astype(np.float64), eps)
    t = np.maximum(true.astype(np.float64), eps)
    return np.maximum(p / t, t / p)


def q_error_stats(pred: np.ndarray, true: np.ndarray) -> dict:
    q = q_error(pred, true)
    return {
        "qerr_median": float(np.median(q)),
        "qerr_mean": float(np.mean(q)),
        "qerr_geomean": float(np.exp(np.mean(np.log(q)))),
        "qerr_p90": float(np.percentile(q, 90)),
        "qerr_p95": float(np.percentile(q, 95)),
        "qerr_p99": float(np.percentile(q, 99)),
        "qerr_max": float(np.max(q)),
        "frac_within_2": float(np.mean(q <= 2.0)),
        "frac_within_10": float(np.mean(q <= 10.0)),
    }
