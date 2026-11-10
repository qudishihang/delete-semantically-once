"""Statistics for replay-induced drift.

The earlier formulation

    V_dup = mean_{i,j} d(theta2_i, theta1_j)          (all n^2 pairs)
    V_stoch = mean_{i!=j} d(theta1_i, theta1_j)       (n(n-1) pairs)

is not calibrated: under perfect idempotence (theta2_i == theta1_i for all i) the
ratio equals n/(n-1) -- 1.25 for n=5 -- rather than 1.  This module defines the
corrected estimators and an inference scheme whose resampling unit is the
independent unlearning trajectory, with test examples as a nested level.

    V_stoch^LOO = 1/(n(n-1)) sum_{i != j} d(theta1_i, theta1_j)
    V_dup^LOO   = 1/(n(n-1)) sum_{i != j} d(theta2_i, theta1_j)
    DAR^LOO     = V_dup^LOO / V_stoch^LOO     == 1 under perfect idempotence
    V_paired    = 1/n sum_i d(theta2_i, theta1_i) == 0 under perfect idempotence
    A_paired    = V_paired / V_stoch^LOO      (0 = idempotent,
                                               1 = a replay drifts as much as an
                                                   independent re-run of the request)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .metrics import js_divergence, per_example_js


# --------------------------------------------------------------------------- #
# pairwise distances
# --------------------------------------------------------------------------- #
def pairwise_js(P: Sequence[np.ndarray], Q: Sequence[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(P), len(Q)))
    for i, p in enumerate(P):
        for j, q in enumerate(Q):
            out[i, j] = js_divergence(p, q)
    return out


def pairwise_js_pec(P: Sequence[np.ndarray], Q: Sequence[np.ndarray]) -> np.ndarray:
    """Per-example JS for every pair: array (len(P), len(Q), n_examples)."""
    n_ex = P[0].shape[0]
    out = np.zeros((len(P), len(Q), n_ex))
    for i, p in enumerate(P):
        for j, q in enumerate(Q):
            out[i, j] = per_example_js(p, q)
    return out


@dataclass
class DarResult:
    n: int
    v_stoch_loo: float
    v_dup_loo: float
    dar_loo: float
    v_paired: float
    a_paired: float
    dar_old: float
    dar_old_limit: float
    n_examples: int


def _stats_from_pec(M11, M21, runs, ex, n_examples):
    """Compute the estimators from pre-computed per-example pair matrices."""
    r = M11[np.ix_(runs, runs)][:, :, ex]
    c = M21[np.ix_(runs, runs)][:, :, ex]
    k = len(runs)
    iu = np.triu_indices(k, k=1)
    v_stoch = float(r[iu].mean())
    v_dup = float((c[iu].mean() + c.transpose(1, 0, 2)[iu].mean()) / 2)
    v_paired = float(np.mean([c[i, i].mean() for i in range(k)]))
    v_old = float(c.mean())          # old definition: all k^2 pairs, i == j included
    return v_stoch, v_dup, v_paired, v_old


def compute_dar(P1: Sequence[np.ndarray], P2: Sequence[np.ndarray]) -> DarResult:
    n = len(P1)
    assert len(P2) == n and n >= 2
    M11 = pairwise_js(P1, P1)
    M21 = pairwise_js(P2, P1)
    iu = np.triu_indices(n, k=1)
    v_stoch = float((M11[iu].mean() + M11.T[iu].mean()) / 2)
    v_dup = float((M21[iu].mean() + M21.T[iu].mean()) / 2)
    v_paired = float(np.mean([js_divergence(P2[i], P1[i]) for i in range(n)]))
    # NOTE on the limit: with V_dup averaged over all n^2 pairs (i == j included)
    # and V_stoch over i != j, perfect idempotence gives V_dup = (1 - 1/n) V_stoch,
    # so the old ratio reads 1 - 1/n.  (Under the alternative convention in which
    # V_stoch is also an all-pairs average, the limit is n/(n-1).)
    return DarResult(n=n, v_stoch_loo=v_stoch, v_dup_loo=v_dup,
                     dar_loo=v_dup / v_stoch if v_stoch > 0 else float("nan"),
                     v_paired=v_paired,
                     a_paired=v_paired / v_stoch if v_stoch > 0 else float("nan"),
                     dar_old=float(M21.mean()) / v_stoch if v_stoch > 0 else float("nan"),
                     dar_old_limit=1.0 - 1.0 / n,
                     n_examples=int(P1[0].shape[0]))


# --------------------------------------------------------------------------- #
# inference: nested (run-level, example-level) bootstrap
# --------------------------------------------------------------------------- #
def bootstrap_dar(P1: Sequence[np.ndarray], P2: Sequence[np.ndarray],
                  n_boot: int = 1000, seed: int = 0,
                  resample_examples: bool = True) -> dict:
    """Bootstrap over unlearning trajectories, with nested example resampling.

    Returns the bootstrap standard error and a percentile interval for every
    estimator, plus a one-sided p-value for H0: V_paired <= V_stoch (equivalently
    A_paired <= 1), i.e. "replay drift stays within the single-request noise floor".
    """
    rng = np.random.default_rng(seed)
    n = len(P1)
    nex = P1[0].shape[0]
    M11 = pairwise_js_pec(P1, P1)
    M21 = pairwise_js_pec(P2, P1)
    rec = {k: [] for k in ("dar_loo", "v_stoch_loo", "v_dup_loo", "v_paired", "a_paired")}
    for _ in range(n_boot):
        runs = rng.integers(0, n, size=n)
        # a bootstrap sample needs at least two distinct trajectories
        if len(set(runs.tolist())) < 2:
            continue
        ex = (rng.integers(0, nex, size=nex) if resample_examples else np.arange(nex))
        vs, vd, vp, _ = _stats_from_pec(M11, M21, runs, ex, nex)
        if vs <= 0:
            continue
        rec["v_stoch_loo"].append(vs)
        rec["v_dup_loo"].append(vd)
        rec["v_paired"].append(vp)
        rec["dar_loo"].append(vd / vs)
        rec["a_paired"].append(vp / vs)
    out = {}
    for k, v in rec.items():
        v = np.asarray(v)
        out[k] = dict(mean=float(v.mean()), se=float(v.std(ddof=1)),
                      lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5)))
    excess = np.asarray(rec["v_paired"]) - np.asarray(rec["v_stoch_loo"])
    out["p_paired_exceeds_stoch"] = float(np.mean(excess <= 0)) if len(excess) else float("nan")
    out["p_dar_le_one"] = float(np.mean(np.asarray(rec["dar_loo"]) <= 1.0))
    out["n_boot"] = int(len(excess))
    return out


def summarise(P1: Sequence[np.ndarray], P2: Sequence[np.ndarray], n_boot: int = 1000,
              seed: int = 0) -> dict:
    r = compute_dar(P1, P2)
    b = bootstrap_dar(P1, P2, n_boot=n_boot, seed=seed)
    return {
        "v_stoch_loo": r.v_stoch_loo, "v_dup_loo": r.v_dup_loo,
        "dar_loo": r.dar_loo, "dar_loo_se": b["dar_loo"]["se"],
        "dar_loo_lo": b["dar_loo"]["lo"], "dar_loo_hi": b["dar_loo"]["hi"],
        "v_paired": r.v_paired, "a_paired": r.a_paired,
        "a_paired_se": b["a_paired"]["se"],
        "a_paired_lo": b["a_paired"]["lo"], "a_paired_hi": b["a_paired"]["hi"],
        "v_dup_minus_stoch": r.v_dup_loo - r.v_stoch_loo,
        "p_paired_exceeds_stoch": b["p_paired_exceeds_stoch"],
        "p_dar_le_one": b["p_dar_le_one"],
        "dar_old": r.dar_old, "dar_old_limit": r.dar_old_limit,
        "n_runs": r.n, "n_examples": r.n_examples, "n_boot": b["n_boot"],
    }


# --------------------------------------------------------------------------- #
# clustered bootstrap across training seeds
# --------------------------------------------------------------------------- #
def clustered_bootstrap(values: dict, n_boot: int = 5000, seed: int = 0) -> dict:
    """Bootstrap a scalar estimator over independent clusters (e.g. training seeds)."""
    rng = np.random.default_rng(seed)
    clusters = list(values)
    arrs = [np.asarray(values[c], dtype=float) for c in clusters]
    k = len(clusters)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, size=k)
        out.append(float(np.concatenate([arrs[i] for i in pick]).mean()))
    out = np.asarray(out)
    return {"mean": float(np.concatenate(arrs).mean()),
            "se": float(out.std(ddof=1)),
            "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "n_clusters": k}


# --------------------------------------------------------------------------- #
# per-group drift
# --------------------------------------------------------------------------- #
def group_drift(P1: np.ndarray, P2: np.ndarray, groups: np.ndarray) -> dict:
    d = per_example_js(P2, P1)
    return {int(g): {"mean_js": float(d[groups == g].mean()), "n": int((groups == g).sum())}
            for g in np.unique(groups)}
