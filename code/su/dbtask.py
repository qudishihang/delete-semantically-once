"""A learned-database downstream task: cardinality estimation under record deletion.

The service is backend agnostic, so the same protocol that protects an image
classifier also has to protect a learned component of a database.  We build a
controlled workload in the style of query-driven cardinality estimation:

  * two tables, ``orders`` (fact) and ``customers`` (dimension);
  * a workload of parametrised range/equality queries whose true cardinality is
    computed exactly over the materialised join;
  * every query record belongs to a tenant, and each tenant's queries concentrate
    on a *region of query space* (a country plus an amount/day band).  A tenant
    deletion request therefore removes a specific region from the estimator's
    training data, which is the interesting case for a deletion service: it has
    a local, measurable effect rather than a uniform thinning of the workload.

The learned estimator is a small MLP regressing log1p(cardinality); the metric is
Q-error, the standard cardinality-estimation error measure.  Everything is
generated locally and deterministically, so the task is fully reproducible
offline.  It is a *controlled* workload that mirrors the query families of
query-driven cardinality-estimation benchmarks, not a re-run of CEBench/DMV.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from . import platform as P

DB_PATH = os.path.join(P.ARTIFACTS, "dbtask_v2.npz")


@dataclass
class DBWorkload:
    n_orders: int = 30000
    n_customers: int = 4000
    n_train: int = 30000
    n_test: int = 12000
    n_tenants: int = 10
    seed: int = 0


def _gen_tables(cfg: DBWorkload, rng):
    n_c = cfg.n_customers
    age = rng.integers(18, 81, size=n_c).astype(np.float32)
    country = rng.integers(0, 10, size=n_c)
    n_o = cfg.n_orders
    o_cid = rng.integers(0, n_c, size=n_o)
    amount = np.exp(rng.normal(3.2, 1.0, size=n_o)).astype(np.float32)
    amount = amount * (1.0 + 0.18 * country[o_cid])
    day = rng.integers(0, 730, size=n_o).astype(np.float32)
    T = dict(o_cid=o_cid, amount=amount, day=day,
             age=age[o_cid], country=country[o_cid], n_customers=n_c)
    return T


def _tenant_profiles(n_t: int):
    """One (country, amount band, day band) region per tenant."""
    prof = []
    for t in range(n_t):
        prof.append(dict(
            country=t % 10,
            a_center=30 + 25 * t, a_half=45 + 5 * (t % 3),
            d_center=36 + 68 * t % 700, d_half=120 + 20 * (t % 4),
        ))
    return prof


def _sample_queries(n: int, cfg: DBWorkload, rng, profiles, tenant_of_query: bool = True):
    pred = np.zeros((n, 8), dtype=np.float32)
    country = np.zeros(n, dtype=np.int64)
    ten = rng.integers(0, cfg.n_tenants, size=n) if tenant_of_query else np.full(n, -1)
    for i in range(n):
        t = ten[i] if ten[i] >= 0 else int(rng.integers(0, cfg.n_tenants))
        p = profiles[t]
        ac = p["a_center"] + rng.normal(0, 12)
        aw = max(4.0, rng.normal(p["a_half"], 12))
        dc = (p["d_center"] + rng.normal(0, 40)) % 730
        dw = max(10.0, rng.normal(p["d_half"], 30))
        a_lo, a_hi = max(0.0, ac - aw), max(ac + aw, ac - aw + 2)
        d_lo = max(0.0, dc - dw)
        d_hi = min(730.0, dc + dw)
        if d_hi - d_lo < 4:
            d_hi = min(730.0, d_lo + 4)
        g_lo = float(rng.integers(18, 70))
        g_hi = g_lo + float(rng.integers(8, 40))
        c_lo = float(rng.integers(0, cfg.n_customers - 200))
        c_hi = c_lo + float(rng.integers(150, 2500))
        pred[i] = [a_lo, a_hi, d_lo, d_hi, g_lo, g_hi, c_lo, c_hi]
        country[i] = p["country"] if rng.random() < 0.85 else int(rng.integers(-1, 10))
    return pred, country, ten


def _featurise(pred: np.ndarray, country: np.ndarray, cfg: DBWorkload) -> np.ndarray:
    a_lo, a_hi, d_lo, d_hi, g_lo, g_hi, c_lo, c_hi = pred.T
    def norm(x, lo, hi):
        return (x - lo) / (hi - lo)
    feats = [
        norm(a_lo, 0, 300), norm(a_hi, 0, 300), np.log1p(np.maximum(a_hi - a_lo, 0)) / 6,
        d_lo / 730, d_hi / 730, (d_hi - d_lo) / 730,
        norm(g_lo, 18, 81), norm(g_hi, 18, 81), (g_hi - g_lo) / 63,
        c_lo / cfg.n_customers, c_hi / cfg.n_customers, (c_hi - c_lo) / cfg.n_customers,
    ]
    onehot = np.zeros((len(pred), 11), dtype=np.float32)
    onehot[np.arange(len(pred)), country + 1] = 1.0
    return np.concatenate([np.stack(feats, axis=1).astype(np.float32), onehot], axis=1)


def _true_cardinality(pred: np.ndarray, country: np.ndarray, T: dict) -> np.ndarray:
    out = np.zeros(len(pred), dtype=np.float64)
    a_lo, a_hi, d_lo, d_hi, g_lo, g_hi, c_lo, c_hi = pred.T
    amt, day, age, cid, ctry = T["amount"], T["day"], T["age"], T["o_cid"], T["country"]
    for i in range(len(pred)):
        m = ((amt >= a_lo[i]) & (amt <= a_hi[i]) & (day >= d_lo[i]) & (day <= d_hi[i])
             & (age >= g_lo[i]) & (age <= g_hi[i]) & (cid >= c_lo[i]) & (cid <= c_hi[i]))
        if country[i] >= 0:
            m &= (ctry == country[i])
        out[i] = m.sum()
    return out


def build(cfg: DBWorkload = DBWorkload(), force: bool = False) -> dict:
    if os.path.exists(DB_PATH) and not force:
        z = np.load(DB_PATH)
        if int(z["n_train"]) == cfg.n_train and int(z["seed"]) == cfg.seed:
            return {k: z[k] for k in z.files}
    rng = np.random.default_rng(cfg.seed)
    T = _gen_tables(cfg, rng)
    profiles = _tenant_profiles(cfg.n_tenants)
    n_all = cfg.n_train + cfg.n_test
    pred, country, ten = _sample_queries(n_all, cfg, rng, profiles)
    card = _true_cardinality(pred, country, T)
    X = _featurise(pred, country, cfg)
    n = cfg.n_train
    out = dict(
        Xtr=X[:n], ytr=np.log1p(card[:n]).astype(np.float32), card_tr=card[:n],
        tenant=ten[:n], pred_tr=pred[:n], country_tr=country[:n],
        Xte=X[n:], yte=np.log1p(card[n:]).astype(np.float32),
        card_te=card[n:], pred_te=pred[n:], country_te=country[n:],
        tenant_te=ten[n:],
        n_train=np.array(n), n_tenants=np.array(cfg.n_tenants), seed=np.array(cfg.seed),
        n_orders=np.array(cfg.n_orders), n_customers=np.array(cfg.n_customers),
    )
    np.savez_compressed(DB_PATH, **out)
    return out
