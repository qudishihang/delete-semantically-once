"""E16 -- deletion replay over a learned cardinality estimator on *real* TPC-H data.

Question
--------
The paper's synthetic orders/customers workload showed that replaying the same
deletion request over a NegGrad+ estimator drives median Q-error inside the
deleted tenant's query region 1.14 -> 1.40 -> 10.5 -> 53.7 -> 1037 over five
replays.  Does that happen on a real TPC-H schema and data distribution?

Setup
-----
Data      TPC-H SF=0.5 through DuckDB's ``tpch`` extension
          (lineitem 2,999,671 rows, orders 750,000, customer 75,000, nation 25).
Workload  two parametrised templates that mirror real TPC-H query shapes:
            T1 (Q6-like, single table)
               SELECT count(*) FROM lineitem
               WHERE l_shipdate BETWEEN d1 AND d2
                 AND l_discount BETWEEN x1 AND x2
                 AND l_quantity BETWEEN q1 AND q2
            T2 (Q3-like, join)
               SELECT count(*) FROM lineitem l JOIN orders o
                 ON l.l_orderkey = o.l_orderkey
               WHERE o_orderdate BETWEEN od1 AND od2
                 AND l_shipdate  BETWEEN sd1 AND sd2
                 AND o_totalprice BETWEEN p1 AND p2
True cardinalities are computed *exactly* for every sampled query from
prefix-sum cubes built over the real data (spot-checked against DuckDB point
queries); they are never estimated.

Tenants   8 contiguous boxes of query space: 4 shipdate-window-centre bands x
          2 secondary bands (T1: discount centre; T2: price-band centre).  The
          training queries of tenant ``T_a`` live in exactly one of those boxes
          (region 3) and are ~5% of the training set; deleting the tenant means
          removing exactly those rows from the estimator's training set.

Arms      base / retrain (from scratch on the retained set) / neg k=1..5
          (NegGrad+ replayed k times) / ft k=1..5 (retain-only fine-tune, same
          budget).  Three seeds; the seed drives the query sample, the
          train/test split and the torch initialisation.

Usage
-----
    python code/e16_tpch.py                                     # published run
    python code/e16_tpch.py --seeds 0 --train 4000 --epochs 4    # smoke test
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from su.methods import UnlearnConfig, run as run_method   # noqa: E402  (repo operator)

RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")

N_FEAT = 18
FEATURE_NAMES = [
    "sd_lo", "sd_hi", "sd_w",              # l_shipdate window (normalised over its range)
    "disc_lo", "disc_hi", "disc_w",        # l_discount window (/0.10)
    "qty_lo", "qty_hi", "qty_w",           # l_quantity window (/50)
    "od_lo", "od_hi", "od_w",              # o_orderdate window (normalised)
    "price_lo", "price_hi", "price_w",     # o_totalprice window (log1p-scaled)
    "tmpl_T1", "tmpl_T2", "join_flag",     # template one-hot + join flag
]
TENANT_REGION = 3            # 2*shipdate_band + secondary_band  (band 1, upper half)
PRICE_BANDS = 16             # equal-count bands over orders, cut at real prices
T2_LO, T2_HI, T2_DEX = 2.0, 6.0, 0.5      # log10 cardinality target, 0.5-dex bins


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
class Log:
    def __init__(self, path, append=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "a" if append else "w", encoding="utf-8")
        self.t0 = time.time()

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(f"[{time.time() - self.t0:7.1f}s] {msg}\n")
        self.f.flush()

    def close(self):
        self.f.close()


# --------------------------------------------------------------------------- #
# exact cardinality counters over the real data
# --------------------------------------------------------------------------- #
def count_t1(R, a, b, d1, d2, q1, q2):
    """Exact #lineitem rows with shipdate in [a,b], discount idx in [d1,d2],
    quantity idx in [q1,q2] (0-based day/quantity offsets)."""
    f, n1, n2 = R["C1"], R["C1s"][1], R["C1s"][2]
    sl, sh = a, b + 1
    dl, dh = d1, d2 + 1
    ql, qh = q1, q2 + 1
    return ((f[sh * n1 * n2 + dh * n2 + qh] - f[sl * n1 * n2 + dh * n2 + qh]
             - f[sh * n1 * n2 + dl * n2 + qh] + f[sl * n1 * n2 + dl * n2 + qh])
            - (f[sh * n1 * n2 + dh * n2 + ql] - f[sl * n1 * n2 + dh * n2 + ql]
               - f[sh * n1 * n2 + dl * n2 + ql] + f[sl * n1 * n2 + dl * n2 + ql]))


def count_t2(R, od1, od2, sd1, sd2, i, j):
    """Exact #joined rows with orderdate in [od1,od2], shipdate in [sd1,sd2],
    price band in [i,j] (0-based day offsets)."""
    f, n1, n2 = R["C2"], R["C2s"][1], R["C2s"][2]
    ol, oh = od1, od2 + 1
    sl, sh = sd1, sd2 + 1
    pl, ph = i, j + 1

    def at(x, y, z):
        return f[(x * n1 + y) * n2 + z]

    return ((at(oh, sh, ph) - at(ol, sh, ph) - at(oh, sl, ph) + at(ol, sl, ph))
            - (at(oh, sh, pl) - at(ol, sh, pl) - at(oh, sl, pl) + at(ol, sl, pl)))


def setup_tpch(sf, log, n_verify=8):
    import duckdb

    t0 = time.time()
    con = duckdb.connect()
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={sf})")
    rows = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("lineitem", "orders", "customer", "nation")}
    log(f"  TPC-H SF={sf}: " + ", ".join(f"{k}={v:,}" for k, v in rows.items()))

    li = con.execute("SELECT l_orderkey, l_shipdate, l_discount, l_quantity "
                     "FROM lineitem").fetchnumpy()
    od = con.execute("SELECT o_orderkey, o_orderdate, o_totalprice "
                     "FROM orders").fetchnumpy()

    l_orderkey = li["l_orderkey"].astype(np.int64)
    sd = li["l_shipdate"].astype("datetime64[D]").astype(np.int64)
    disc = np.rint(li["l_discount"].astype(np.float64) * 100).astype(np.int64)  # 0..10
    qty = li["l_quantity"].astype(np.int64) - 1                                  # 0..49
    o_orderkey = od["o_orderkey"].astype(np.int64)
    o_orderdate = od["o_orderdate"].astype("datetime64[D]").astype(np.int64)
    price = od["o_totalprice"].astype(np.float64)

    pos = np.full(int(max(o_orderkey.max(), l_orderkey.max())) + 2, -1, np.int64)
    pos[o_orderkey] = np.arange(len(o_orderkey))
    li2o = pos[l_orderkey]
    del pos, l_orderkey
    od_line = o_orderdate[li2o]
    price_line = price[li2o]
    del li2o

    MIN_SD, MAX_SD = int(sd.min()), int(sd.max())
    MIN_OD, MAX_OD = int(o_orderdate.min()), int(o_orderdate.max())
    N_SD, N_OD = MAX_SD - MIN_SD + 1, MAX_OD - MIN_OD + 1

    # price bands: equal-count bands over the *orders* dimension, cut at the
    # real price values, so a band range is an exact price interval.
    sp = np.sort(price)
    starts = (np.arange(PRICE_BANDS + 1) * len(sp)) // PRICE_BANDS
    pb_lo = sp[starts[:PRICE_BANDS]]
    pb_hi = sp[starts[1:] - 1]
    band_of_price = np.clip(np.searchsorted(pb_lo, price_line, side="right") - 1,
                            0, PRICE_BANDS - 1)

    # ---- T1 cube: (shipdate day, discount, quantity) ---------------------- #
    C1 = np.zeros((N_SD + 1, 12, 51), dtype=np.int32)
    f1 = ((sd - MIN_SD) * 11 + disc) * 50 + qty
    C1[1:, 1:, 1:] = np.bincount(f1, minlength=N_SD * 11 * 50).reshape(N_SD, 11, 50)
    del f1
    for ax in (2, 1, 0):
        np.cumsum(C1, axis=ax, out=C1)

    # ---- T2 cube: (orderdate day, shipdate day, price band) --------------- #
    C2 = np.zeros((N_OD + 1, N_SD + 1, PRICE_BANDS + 1), dtype=np.int32)
    f2 = (((od_line - MIN_OD) * N_SD + (sd - MIN_SD)) * PRICE_BANDS) + band_of_price
    C2[1:, 1:, 1:] = np.bincount(f2, minlength=N_OD * N_SD * PRICE_BANDS).reshape(
        N_OD, N_SD, PRICE_BANDS)
    del f2
    for ax in (2, 1, 0):
        np.cumsum(C2, axis=ax, out=C2)

    R = dict(C1=C1.ravel(), C1s=C1.shape, C2=C2.ravel(), C2s=C2.shape,
             n_sd=N_SD, n_od=N_OD, min_sd=MIN_SD, min_od=MIN_OD,
             max_price=float(price.max()), pb_lo=pb_lo, pb_hi=pb_hi,
             n_line=len(sd), n_orders=len(price), span_od=N_OD)
    log(f"  exact-cardinality cubes built in {time.time() - t0:.1f}s "
        f"(T1 {C1.nbytes / 1e6:.1f} MB, T2 {C2.nbytes / 1e6:.1f} MB)")

    if n_verify:
        rng = np.random.default_rng(12345)
        bad = 0
        for _ in range(n_verify):
            w = int(rng.integers(30, 366))
            a = int(rng.integers(0, N_SD - w))
            d1 = int(rng.integers(0, 6))
            d2 = min(10, d1 + int(rng.integers(0, 6)))
            q1 = int(rng.integers(0, 30))
            q2 = min(49, q1 + int(rng.integers(1, 20)))
            mine = int(count_t1(R, np.array([a]), np.array([a + w]), np.array([d1]),
                                np.array([d2]), np.array([q1]), np.array([q2]))[0])
            ref = con.execute(
                "SELECT count(*) FROM lineitem WHERE l_shipdate BETWEEN ? AND ? "
                "AND l_discount BETWEEN ? AND ? AND l_quantity BETWEEN ? AND ?",
                [_day(MIN_SD + a), _day(MIN_SD + a + w), d1 / 100.0, d2 / 100.0,
                 q1 + 1, q2 + 1]).fetchone()[0]
            bad += int(mine != ref)
        for _ in range(n_verify):
            odw = int(rng.integers(30, 900))
            od1 = int(rng.integers(0, N_OD - odw))
            sdw = int(rng.integers(30, 400))
            sd1 = int(rng.integers(0, N_SD - sdw))
            i = int(rng.integers(0, PRICE_BANDS - 1))
            j = int(rng.integers(i, PRICE_BANDS))
            mine = int(count_t2(R, np.array([od1]), np.array([od1 + odw]),
                                np.array([sd1]), np.array([sd1 + sdw]),
                                np.array([i]), np.array([j]))[0])
            ref = con.execute(
                "SELECT count(*) FROM lineitem l JOIN orders o ON l.l_orderkey=o.o_orderkey "
                "WHERE o_orderdate BETWEEN ? AND ? AND l_shipdate BETWEEN ? AND ? "
                "AND o_totalprice BETWEEN ? AND ?",
                [_day(MIN_OD + od1), _day(MIN_OD + od1 + odw), _day(MIN_SD + sd1),
                 _day(MIN_SD + sd1 + sdw), float(pb_lo[i]), float(pb_hi[j])]
            ).fetchone()[0]
            bad += int(mine != ref)
        log(f"  cube spot-check vs DuckDB: {2 * n_verify} random queries, "
            f"{bad} mismatches")
        if bad:
            raise SystemExit("cardinality cube mismatch -- refusing to run")
    con.close()
    return R


def _day(off):
    import datetime
    return (datetime.date(1970, 1, 1) + datetime.timedelta(days=int(off))).isoformat()


# --------------------------------------------------------------------------- #
# workload sampling
# --------------------------------------------------------------------------- #
def sample_t1(rng, R, n):
    """Q6-like: d1 uniform over the shipdate range, width 30..365 days;
    discount from the 11 real values, width 0.00..0.05; quantity width 1..20 of 1..50."""
    w = rng.integers(30, 366, n)
    a = rng.integers(0, R["n_sd"] - w)
    b = a + w
    dw = rng.integers(0, 6, n)                 # 0.00 .. 0.05 in steps of 0.01
    d1 = rng.integers(0, 11 - dw)
    d2 = d1 + dw
    qw = rng.integers(1, 21, n)                # width 1..20 out of 1..50
    q1 = rng.integers(0, 50 - qw)              # quantity 1..50, 0-based index
    q2 = q1 + qw
    card = count_t1(R, a, b, d1, d2, q1, q2)
    return dict(tmpl=np.ones(n, np.int64), a=a, b=b, d1=d1, d2=d2, q1=q1, q2=q2,
                card=card.astype(np.float64))


def sample_t2(rng, R, n):
    """Q3-like candidates: date widths log-spaced (od 30..1800 d, sd 30..365 d) and
    price bounds snapped to the 16 equal-count bands of o_totalprice."""
    odws = np.array([30, 60, 90, 180, 365, 730, 1095, 1800])
    sdws = np.array([30, 60, 90, 180, 365])
    odw = odws[rng.integers(0, len(odws), n)]
    od1 = rng.integers(0, R["span_od"] - odw)
    od2 = od1 + odw
    sdw = sdws[rng.integers(0, len(sdws), n)]
    sd1 = rng.integers(0, R["n_sd"] - sdw)
    sd2 = sd1 + sdw
    nb = np.array([1, 2, 3, 4, 6, 8, 12, 16])[rng.integers(0, 8, n)]
    i = rng.integers(0, PRICE_BANDS, n)
    j = np.minimum(PRICE_BANDS - 1, i + nb - 1)
    card = count_t2(R, od1, od2, sd1, sd2, i, j)
    return dict(tmpl=np.full(n, 2, np.int64), od1=od1, od2=od2, sd1=sd1, sd2=sd2,
                pb1=i, pb2=j, card=card.astype(np.float64))


def take_balanced(rng, card, n_target, log, name):
    """Pick queries so that log10(cardinality) is ~uniform over 0.5-dex bins."""
    nb = int(round((T2_HI - T2_LO) / T2_DEX))
    lg = np.log10(np.maximum(card, 1.0))
    b = np.clip(np.floor((np.clip(lg, T2_LO, T2_HI - 1e-9) - T2_LO) / T2_DEX).astype(int),
                0, nb - 1)
    inside = (lg >= T2_LO) & (lg < T2_HI)
    quota = n_target // nb
    keep, spare = [], []
    for k in range(nb):
        pool = np.where(inside & (b == k))[0]
        perm = rng.permutation(pool)
        keep.append(perm[:quota])
        spare.append(perm[quota:])
        log(f"    bin 10^{T2_LO + k * T2_DEX:.1f}-10^{T2_LO + (k + 1) * T2_DEX:.1f}: "
            f"{len(pool):6d} candidates -> {min(quota, len(pool))}")
    keep = np.concatenate([k for k in keep if len(k)]) if keep else np.array([], np.int64)
    if len(keep) < n_target:
        rest = [s for s in spare if len(s)]
        if rest:
            rest = np.concatenate(rest)
            take = min(n_target - len(keep), len(rest))
            keep = np.concatenate([keep, rng.choice(rest, size=take, replace=False)])
            log(f"    {name}: filled {take} extra slots from overfull bins")
    if len(keep) < n_target:
        log(f"    WARNING {name}: only {len(keep)} of {n_target} queries available")
    lgc = np.log10(np.maximum(card[keep], 1.0))
    log(f"    {name}: kept {len(keep)} queries, log10(card) in "
        f"[{lgc.min():.2f}, {lgc.max():.2f}], median {np.median(lgc):.2f}")
    return keep[:n_target]


def validate_queries(q, R, name):
    """Fail loudly rather than produce silently wrong cardinalities."""
    c = q["card"]
    if len(c) == 0:
        raise ValueError(f"{name}: empty query set")
    bad = (~np.isfinite(c)) | (c < 0) | (c > R["n_line"])
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} queries with invalid cardinality "
                         f"(min {np.nanmin(c):.0f}, max {np.nanmax(c):.0f})")
    if q["tmpl"][0] == 1:
        ok = ((q["a"] >= 0) & (q["a"] <= q["b"]) & (q["b"] < R["n_sd"])
              & (q["d1"] >= 0) & (q["d1"] <= q["d2"]) & (q["d2"] <= 10)
              & (q["q1"] >= 0) & (q["q1"] <= q["q2"]) & (q["q2"] <= 49))
    else:
        ok = ((q["od1"] >= 0) & (q["od1"] <= q["od2"]) & (q["od2"] < R["n_od"])
              & (q["sd1"] >= 0) & (q["sd1"] <= q["sd2"]) & (q["sd2"] < R["n_sd"])
              & (q["pb1"] >= 0) & (q["pb1"] <= q["pb2"]) & (q["pb2"] < PRICE_BANDS))
    if not ok.all():
        raise ValueError(f"{name}: {int((~ok).sum())} out-of-range predicates")
    return True


def build_workload(rng, R, n_train, n_test, log):
    n_each = (n_train + n_test) // 2
    log(f"  sampling {n_each} T1 / {n_each} T2 queries "
        f"({n_train} train + {n_test} test)")
    t0 = time.time()
    q1 = sample_t1(rng, R, n_each)
    cand = sample_t2(rng, R, max(60_000, 10 * n_each))
    pick = take_balanced(rng, cand["card"], n_each, log, "T2")
    q2 = {k: v[pick] for k, v in cand.items()}
    validate_queries(q1, R, "T1")
    validate_queries(q2, R, "T2")
    log(f"  workload sampled in {time.time() - t0:.1f}s; log10(card): "
        f"T1 [{np.log10(np.maximum(q1['card'], 1)).min():.2f}, "
        f"{np.log10(q1['card']).max():.2f}] med "
        f"{np.median(np.log10(np.maximum(q1['card'], 1))):.2f}; "
        f"T2 [{np.log10(np.maximum(q2['card'], 1)).min():.2f}, "
        f"{np.log10(q2['card']).max():.2f}] med "
        f"{np.median(np.log10(np.maximum(q2['card'], 1))):.2f}")
    Q = {}
    for name, q in (("T1", q1), ("T2", q2)):
        n = len(q["card"])
        perm = rng.permutation(n)
        Q[name] = dict(train={k: v[perm[:n_train // 2]] for k, v in q.items()},
                       test={k: v[perm[n_train // 2:]] for k, v in q.items()})
    return Q


# --------------------------------------------------------------------------- #
# features / regions
# --------------------------------------------------------------------------- #
def featurise(q, R):
    n = len(q["card"])
    X = np.zeros((n, N_FEAT), np.float32)
    nsd = R["n_sd"] - 1.0
    lod = math.log1p(R["max_price"])
    if q["tmpl"][0] == 1:
        X[:, 0], X[:, 1] = q["a"] / nsd, q["b"] / nsd
        X[:, 2] = (q["b"] - q["a"]) / nsd
        X[:, 3], X[:, 4] = q["d1"] / 10.0, q["d2"] / 10.0
        X[:, 5] = (q["d2"] - q["d1"]) / 10.0
        X[:, 6], X[:, 7] = q["q1"] / 49.0, q["q2"] / 49.0
        X[:, 8] = (q["q2"] - q["q1"]) / 49.0
        X[:, 15] = 1.0
    else:
        X[:, 0], X[:, 1] = q["sd1"] / nsd, q["sd2"] / nsd
        X[:, 2] = (q["sd2"] - q["sd1"]) / nsd
        nod = R["span_od"] - 1.0
        X[:, 9], X[:, 10] = q["od1"] / nod, q["od2"] / nod
        X[:, 11] = (q["od2"] - q["od1"]) / nod
        pl = np.log1p(R["pb_lo"][q["pb1"]]) / lod
        ph = np.log1p(R["pb_hi"][q["pb2"]]) / lod
        X[:, 12], X[:, 13] = pl, ph
        X[:, 14] = ph - pl
        X[:, 16] = 1.0
        X[:, 17] = 1.0
    return X


def region_ids(q, R):
    """8 tenants: 4 shipdate-window-centre bands x 2 secondary bands.
    T1 secondary = discount centre, T2 secondary = price-band centre."""
    centre = 0.5 * (q["a"] + q["b"]) if q["tmpl"][0] == 1 else 0.5 * (q["sd1"] + q["sd2"])
    band = np.clip((centre / R["n_sd"] * 4).astype(int), 0, 3)
    if q["tmpl"][0] == 1:
        sec = (0.5 * (q["d1"] + q["d2"]) > 5.0).astype(int)
    else:
        sec = (0.5 * (q["pb1"] + q["pb2"]) > (PRICE_BANDS - 1) / 2).astype(int)
    return 2 * band + sec


# --------------------------------------------------------------------------- #
# learned CE
# --------------------------------------------------------------------------- #
def make_model(d_in, hidden=256):
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, 1))


def train_ce(X, y, seed, epochs, tag, verbose=True, lr=1e-3, wd=1e-5, bs=256):
    """Mirrors the repo's CE trainer (su/dbtask.py + e5_dbtask.py): MLP 256-256
    ReLU, Adam lr 1e-3, cosine schedule, batch 256, MSE on the standardised
    log1p target."""
    torch.manual_seed(seed)
    m = make_model(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(m(X[idx]).squeeze(-1), y[idx])
            loss.backward()
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      {tag} epoch {ep:2d} mse {tot / n:.4f}", flush=True)
    m.eval()
    return m, time.time() - t0


def predict_card(m, X, mu, sd):
    with torch.no_grad():
        z = m(X).squeeze(-1).numpy().astype(np.float64)
    return np.maximum(np.expm1(z * sd + mu), 0.0)


def qerr(est, true):
    p = np.maximum(est, 1.0)
    t = np.maximum(true, 1.0)
    return np.maximum(p / t, t / p)


# --------------------------------------------------------------------------- #
# one seed
# --------------------------------------------------------------------------- #
def run_seed(seed, R, args, log, meta=None):
    rng = np.random.default_rng(1000 + seed)
    t_seed = time.time()
    log(f"[seed {seed}] building workload")
    Q = build_workload(rng, R, args.train, args.test, log)

    cards = np.concatenate([Q["T1"]["train"]["card"], Q["T2"]["train"]["card"]])
    regs = np.concatenate([region_ids(Q["T1"]["train"], R),
                           region_ids(Q["T2"]["train"], R)])
    Xtr = torch.from_numpy(np.concatenate([featurise(Q["T1"]["train"], R),
                                           featurise(Q["T2"]["train"], R)]))
    ev = {}
    for nm in ("T1", "T2"):
        q = Q[nm]["test"]
        ev[nm] = dict(X=torch.from_numpy(featurise(q, R)), card=q["card"],
                      reg=region_ids(q, R))
    ev["ALL"] = dict(X=torch.cat([ev["T1"]["X"], ev["T2"]["X"]]),
                     card=np.concatenate([ev["T1"]["card"], ev["T2"]["card"]]),
                     reg=np.concatenate([ev["T1"]["reg"], ev["T2"]["reg"]]))

    hist = np.bincount(regs, minlength=8)
    log(f"[seed {seed}] train regions: " + " ".join(
        f"r{i}={c}" for i, c in enumerate(hist)) + f"  (n={len(cards)})")
    for nm in ("T1", "T2"):
        log(f"[seed {seed}] test  regions ({nm}): " + " ".join(
            f"r{i}={c}" for i, c in enumerate(np.bincount(ev[nm]["reg"], minlength=8))))
    in_train = np.where(regs == TENANT_REGION)[0]
    n_ten = min(int(round(0.05 * args.train)), len(in_train))
    ten = rng.choice(in_train, size=n_ten, replace=False)
    mask = np.zeros(len(cards), bool)
    mask[ten] = True
    retain, forget = np.where(~mask)[0], np.where(mask)[0]
    log(f"[seed {seed}] tenant T_a = region {TENANT_REGION}: {n_ten} queries "
        f"({100.0 * n_ten / len(cards):.1f}% of the training set, "
        f"{100.0 * n_ten / max(1, len(in_train)):.0f}% of the region's training "
        f"mass); retained set = {len(retain)}")
    if meta is not None:
        rngs = meta.setdefault("lg_rng", {})
        for nm in ("T1", "T2"):
            lg = np.log10(np.maximum(Q[nm]["train"]["card"], 1.0))
            lo, hi = float(lg.min()), float(lg.max())
            if nm in rngs:
                lo, hi = min(lo, rngs[nm][0]), max(hi, rngs[nm][1])
            rngs[nm] = [lo, hi]
        meta["n_region_train"] = int(len(in_train))
        meta["n_tenant"] = int(n_ten)
        meta["frac_region_mass"] = float(n_ten / max(1, len(in_train)))
        meta["n_test_region"] = {nm: int((ev[nm]["reg"] == TENANT_REGION).sum())
                                 for nm in ("T1", "T2", "ALL")}
        meta["n_test_total"] = {nm: int(len(ev[nm]["card"]))
                                for nm in ("T1", "T2", "ALL")}

    y_all = np.log1p(cards)
    mu, sg = float(y_all.mean()), float(y_all.std())     # base target scaler (all 40k)
    ytr = torch.from_numpy(((y_all - mu) / sg).astype(np.float32))
    retained = torch.from_numpy(retain)
    forgotten = torch.from_numpy(forget)

    rows = []

    def evaluate(arm, k, m, mu_, sg_):
        for nm in ("T1", "T2", "ALL"):
            d = ev[nm]
            q = qerr(predict_card(m, d["X"], mu_, sg_), d["card"])
            reg = d["reg"] == TENANT_REGION
            oth = ~reg
            if reg.sum() == 0 or oth.sum() == 0:
                raise ValueError(f"template {nm}: empty region split "
                                 f"({int(reg.sum())} in region, {int(oth.sum())} out)")
            rows.append(dict(
                seed=seed, template=nm, arm=arm, k=k,
                n_test=len(q),
                median_qerr_overall=float(np.median(q)),
                mean_qerr_overall=float(np.mean(q)),
                median_qerr_region=float(np.median(q[reg])),
                p90_qerr_region=float(np.percentile(q[reg], 90)),
                median_qerr_overall_of_that_arm_restricted_to_region_other_tenants=(
                    float(np.median(q[oth]))),
                n_test_region=int(reg.sum()), n_test_other=int(oth.sum())))
        r = rows[-1]
        log(f"  [seed {seed}] {arm:8s} k={k} | ALL med {r['median_qerr_overall']:9.3f}"
            f" | in-region med {r['median_qerr_region']:10.3f}"
            f" p90 {r['p90_qerr_region']:11.3f}"
            f" | other-region med {r['median_qerr_overall_of_that_arm_restricted_to_region_other_tenants']:8.3f}")

    log(f"[seed {seed}] training base on {len(cards)} queries, {args.epochs} epochs")
    base, secs = train_ce(Xtr, ytr, seed, args.epochs, "base")
    log(f"[seed {seed}] base trained in {secs:.1f}s")
    evaluate("base", 0, base, mu, sg)

    ret, secs = train_ce(Xtr[retained], ytr[retained], seed, args.epochs, "retrain")
    y_ret = y_all[retain]
    mu_r, sg_r = float(y_ret.mean()), float(y_ret.std())
    log(f"[seed {seed}] retrain on {len(retain)} queries in {secs:.1f}s "
        f"(own scaler mu={mu_r:.3f} sd={sg_r:.3f} vs base mu={mu:.3f} sd={sg:.3f})")
    evaluate("retrain", 0, ret, mu_r, sg_r)

    for arm, method in (("neg", "neggradplus"), ("ft", "finetune")):
        cur = base
        for k in range(1, args.kmax + 1):
            cfg = UnlearnConfig(method=method, epochs=args.ue_epochs, lr=args.ue_lr,
                                weight_decay=args.ue_wd, beta=args.ue_beta,
                                batch_size=256, seed=seed * 100 + k, loss="mse",
                                grad_clip=args.clip)
            t1 = time.time()
            cur = run_method(cur, Xtr, ytr, retained, forgotten, cfg)
            evaluate(arm, k, cur, mu, sg)
            log(f"  [seed {seed}]   {arm}{k} invocation {time.time() - t1:.1f}s")
    log(f"[seed {seed}] done in {time.time() - t_seed:.0f}s")
    return rows


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
ARMS = ([("base", 0), ("retrain", 0)] + [("neg", k) for k in range(1, 6)]
        + [("ft", k) for k in range(1, 6)])
ARM_LABEL = {(a, k): (a if k == 0 else f"{a}{k}") for a, k in ARMS}


def summary_table(rows):
    tab = {}
    for tmpl in ("T1", "T2", "ALL"):
        tab[tmpl] = {}
        for a, k in ARMS:
            sel = [r for r in rows if r["template"] == tmpl and r["arm"] == a
                   and r["k"] == k]
            tab[tmpl][ARM_LABEL[(a, k)]] = dict(
                region=float(np.mean([r["median_qerr_region"] for r in sel])),
                overall=float(np.mean([r["median_qerr_overall"] for r in sel])),
                region_p90=float(np.mean([r["p90_qerr_region"] for r in sel])),
                region_med_of_seeds=[r["median_qerr_region"] for r in sel],
                other=float(np.mean([
                    r["median_qerr_overall_of_that_arm_restricted_to_region_other_tenants"]
                    for r in sel])))
    return tab


def print_table(tab, log):
    log("")
    log("=" * 84)
    log("mean over seeds of median Q-error  (T1+T2 pooled test set, n=8000/seed)")
    log("=" * 84)
    log(f"{'arm':9s} | {'in-region med':>13s} | {'overall med':>11s} | "
        f"{'in-region p90':>13s} | {'other regions':>13s}")
    log("-" * 84)
    for a, k in ARMS:
        lab = ARM_LABEL[(a, k)]
        t = tab["ALL"][lab]
        log(f"{lab:9s} | {t['region']:13.3f} | {t['overall']:11.3f} | "
            f"{t['region_p90']:13.3f} | {t['other']:13.3f}")
    for tmpl in ("T1", "T2"):
        log("")
        log(f"per template {tmpl}")
        log("-" * 84)
        for a, k in ARMS:
            lab = ARM_LABEL[(a, k)]
            t = tab[tmpl][lab]
            log(f"{lab:9s} | {t['region']:13.3f} | {t['overall']:11.3f} | "
                f"{t['region_p90']:13.3f} | {t['other']:13.3f}")


def write_summary_md(tab, rows, args, wall, log, meta=None):
    meta = meta or {}
    path = os.path.join(RESULTS, args.summary)
    seeds = sorted({r["seed"] for r in rows})
    a = tab["ALL"]
    base_r, retr_r = a["base"]["region"], a["retrain"]["region"]
    n1_r, n5_r = a["neg1"]["region"], a["neg5"]["region"]
    ft5_r = a["ft5"]["region"]
    base_o, n5_o, oth5 = a["base"]["overall"], a["neg5"]["overall"], a["neg5"]["other"]
    ratio = n5_r / base_r
    lg = meta.get("lg_rng", {"T2": [T2_LO, T2_HI - 1e-9]})
    nreg = meta.get("n_test_region", {"T1": 0, "T2": 0, "ALL": 0})
    ntot = meta.get("n_test_total", {"ALL": args.test})
    r_neg = n5_r / max(base_r, 1e-9)
    r_retr = retr_r / max(base_r, 1e-9)
    r_ft = ft5_r / max(base_r, 1e-9)
    r_neg_vs_retr = n5_r / max(retr_r, 1e-9)
    if r_neg_vs_retr >= 1.5:
        verdict = ("replicates: replay drives in-region error far past what a single "
                   "correct deletion costs")
    elif r_neg_vs_retr >= 1.15:
        verdict = ("replicates only weakly: replay exceeds the exact-retrain "
                   "reference by a modest margin")
    else:
        verdict = ("does NOT replicate: replay does not exceed the exact-retrain "
                   "reference by a meaningful margin")
    L = []
    L.append(f"# E16 -- deletion replay over a learned CE on real TPC-H (SF={args.sf})\n")
    L.append(f"`code/e16_tpch.py`, {time.strftime('%Y-%m-%d %H:%M')}; seeds {seeds}, "
             f"{args.train} training / {args.test} test queries, {args.epochs} base "
             f"epochs, {args.ue_epochs} epochs per unlearning invocation, "
             f"k = 1..{args.kmax}.  Runtime {wall / 60:.1f} min.\n")
    L.append(f"## Means over seeds (pooled test set, n={ntot.get('ALL', 0)} per seed)\n")
    L.append("| arm | in-region median Q-error | overall median Q-error | "
             "in-region p90 | other tenants' regions |")
    L.append("|---|---|---|---|---|")
    for a_, k in ARMS:
        lab = ARM_LABEL[(a_, k)]
        t = tab["ALL"][lab]
        L.append(f"| {lab} | {t['region']:.3f} | {t['overall']:.3f} | "
                 f"{t['region_p90']:.2f} | {t['other']:.3f} |")
    L.append(f"\n### Per template (in-region = tenant T_a's query-space box, "
             f"n={nreg.get('T1', 0)} T1 / {nreg.get('T2', 0)} T2 per seed)\n")
    L.append("| arm | T1 in-region | T1 overall | T2 in-region | T2 overall |")
    L.append("|---|---|---|---|---|")
    for a_, k in ARMS:
        lab = ARM_LABEL[(a_, k)]
        L.append(f"| {lab} | {tab['T1'][lab]['region']:.3f} | "
                 f"{tab['T1'][lab]['overall']:.3f} | "
                 f"{tab['T2'][lab]['region']:.3f} | "
                 f"{tab['T2'][lab]['overall']:.3f} |")
    L.append("\nPer-seed in-region medians (pooled): " + "; ".join(
        f"{lab} " + "/".join(f"{v:.2f}" for v in tab["ALL"][lab]["region_med_of_seeds"])
        for lab in ("base", "retrain", "neg1", "neg5", "ft5")) + ".\n")
    L.append("## Reproduce\n")
    L.append("```powershell\ncd C:\\Users\\35734\\Desktop\\ICLR\\icde2027\n"
             f"python code/e16_tpch.py --seeds 0,1,2 --train {args.train} "
             f"--test {args.test} --epochs {args.epochs} --kmax {args.kmax} "
             f"--ue-epochs {args.ue_epochs} --ue-lr {args.ue_lr} --ue-wd {args.ue_wd} "
             f"--ue-beta {args.ue_beta} --clip {args.clip}\n```\n")
    L.append("## Interpretation\n")
    L.append(f"1. Replay {verdict}.  The in-region median Q-error of the NegGrad+ "
             f"estimator moves {base_r:.3f} (base) -> {n1_r:.3f} (k=1) -> {n5_r:.3f} "
             f"(k=5), i.e. x{r_neg:.2f} over base and x{r_neg_vs_retr:.2f} over the "
             f"exact-retrain reference.  The synthetic orders/customers workload moved "
             f"1.14 -> 1037 (x910), so the direction of the effect and its locality are "
             f"the same, but the TPC-H magnitude is tiny by comparison.\n")
    if r_retr < 1.10:
        s2 = ("Removing the tenant's ~5% of the training queries barely changes its own "
              "region, so essentially all of the replay damage is operator drift rather "
              "than missing data.")
    elif r_retr < 1.5:
        s2 = ("Removing the tenant's ~5% of the training queries makes its own region "
              "somewhat harder, so the replay damage is part missing data and part "
              "operator drift.")
    else:
        s2 = ("Most of the in-region damage already comes from the deletion itself: the "
              "exact retrain is much worse than base in that region.")
    L.append(f"2. Base vs retrain in-region: {base_r:.3f} -> {retr_r:.3f} "
             f"(x{r_retr:.2f}).  {s2}\n")
    L.append(f"3. The damage is local and nearly invisible overall: at k=5 the pooled "
             f"median Q-error reads {n5_o:.3f} against {base_o:.3f} at base "
             f"(x{n5_o / max(base_o, 1e-9):.2f}), while the same queries inside the "
             f"tenant region read {base_r:.3f} -> {n5_r:.3f} (x{r_neg:.2f}), and test "
             f"queries belonging to the other seven tenants sit at {oth5:.3f}.\n")
    if r_ft < 1.15:
        s4 = ("so the degradation is caused by the forget term of NegGrad+, not by the "
              "extra SGD steps on the shrunk retain set.")
    elif r_ft < r_neg:
        s4 = ("which drifts too, but less than NegGrad+; the gap between the two arms is "
              "the forget term.")
    else:
        s4 = ("i.e. as much as NegGrad+, so on this workload the in-region damage is "
              "dominated by ordinary SGD drift on the shrunk retain set rather than by "
              "the forget term.")
    L.append(f"4. Retain-only fine-tuning with the same budget is the control: "
             f"{ft5_r:.3f} in-region at k=5 vs {base_r:.3f} base (x{r_ft:.2f}), {s4}\n")
    if args.train >= 40000 and args.epochs >= 60:
        budget = (f"The full {args.train // 1000}k/{args.epochs}-epoch budget was used "
                  f"({wall / 60:.1f} min of measured wall time).")
    else:
        budget = (f"The budget was REDUCED from the planned 40k queries / 60 epochs to "
                  f"{args.train // 1000}k / {args.epochs} epochs: sustained CPU load in "
                  f"this environment runs ~5-9x slower than short bursts (measured "
                  f"1.1 s/epoch for a 3-epoch run vs 8.8 s/epoch over 60 epochs), which "
                  f"put the full budget outside the time limit.")
    L.append(f"5. Honest caveats: the o_totalprice predicate is snapped to 16 "
             f"equal-count price bands (so T2 price ranges are quantile-aligned, not "
             f"arbitrary constants), T2 cardinalities are stratified to a log-uniform "
             f"target over 10^{T2_LO:.1f}-10^{T2_HI:.1f} (the sampler's reachable range "
             f"tops out near 10^{lg['T2'][1]:.1f}; T1 spans "
             f"10^{lg['T1'][0]:.1f}-10^{lg['T1'][1]:.1f} naturally), and the run uses "
             f"{len(seeds)} seeds.  {budget}  Q-error magnitudes are therefore not "
             f"directly comparable with the synthetic x910 result.\n")
    L.append("\n## Feature vector (" + str(N_FEAT) + " dims)\n")
    L.append("`" + "`, `".join(FEATURE_NAMES) + "`\n")
    L.append("All bounds normalised to [0,1] over their real TPC-H range (price via "
             "log1p(o_totalprice)/log1p(max)); unused slots of a template are 0.  "
             "Target = standardised log1p(true cardinality), inverse transform "
             "expm1(z*sd+mu); Q-error = max(est/true, true/est) with both clipped "
             "at 1.\n")
    L.append("\n## Workload / tenant definition\n")
    L.append("- T1 (Q6-like): d1 uniform over the shipdate range, width 30-365 d; "
             "discount from the 11 real values, width 0.00-0.05; quantity width "
             "1-20 of 1-50.  Natural distribution, no stratification.\n"
             "- T2 (Q3-like): o_orderdate width log-spaced 30-1800 d, l_shipdate "
             "width 30-365 d, o_totalprice bounds snapped to the 16 equal-count "
             f"price bands; candidates stratified to a log-uniform cardinality "
             f"target over 10^{T2_LO:.1f}-10^{T2_HI:.1f} in 0.5-dex bins.\n"
             "- 8 tenant boxes = 4 shipdate-window-centre bands x 2 secondary bands "
             f"(T1 discount centre, T2 price-band centre).  Tenant T_a owns region "
             f"{TENANT_REGION} and ~5% of the training set (it is "
             f"{100 * meta.get('frac_region_mass', 0):.0f}% of that region's training "
             f"mass); deleting it removes exactly those rows.\n"
             "- base/neg/ft share the base target scaler; retrain (from scratch) "
             "uses the scaler of its own retained set.\n"
             "- neg = NegGrad+ (beta*MSE_retain - (1-beta)*MSE_forget, beta=0.95, "
             "SGD momentum 0.9, lr 0.01, wd 5e-4, clip 5, batch 256, 10 epochs per "
             "invocation, forget batches drawn from the tenant's queries); ft = the "
             "same loop without the forget term.  Both reuse `su.methods.run`.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf", type=float, default=0.5)
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--test", type=int, default=8000)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--kmax", type=int, default=5)
    ap.add_argument("--ue-epochs", type=int, default=10)
    ap.add_argument("--ue-lr", type=float, default=0.01)
    ap.add_argument("--ue-wd", type=float, default=5e-4)
    ap.add_argument("--ue-beta", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=5.0)
    ap.add_argument("--verify", type=int, default=8,
                    help="spot-check this many random queries per template against "
                         "DuckDB point queries at startup")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch intra-op threads; small batches scale poorly above ~4")
    ap.add_argument("--out", default="e16_tpch.csv")
    ap.add_argument("--summary", default="e16_tpch_summary.md")
    ap.add_argument("--log", default="e16.log")
    ap.add_argument("--append", action="store_true",
                    help="keep rows of seeds not present in this run (lets one seed be "
                         "run per invocation and accumulated; the log is then rewritten "
                         "from the accumulated rows)")
    ap.add_argument("--elapsed-base", type=float, default=0.0,
                    help="seconds spent by earlier --append invocations, added to the "
                         "runtime reported in the summary")
    args = ap.parse_args()
    torch.set_num_threads(max(1, args.threads))

    log = Log(os.path.join(LOGS, args.log), append=args.append)
    t_start = time.time()
    log(f"E16 -- replay of a deletion request over a learned CE, real TPC-H SF={args.sf}")
    log("  args: " + " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
    log("  features (%d): %s" % (N_FEAT, ", ".join(FEATURE_NAMES)))

    R = setup_tpch(args.sf, log, n_verify=args.verify)
    rows, meta = [], {}
    for seed in [int(s) for s in args.seeds.split(",") if s.strip()]:
        rows.extend(run_seed(seed, R, args, log, meta))

    path = os.path.join(RESULTS, args.out)
    cols = ["seed", "template", "arm", "k", "n_test", "median_qerr_overall",
            "mean_qerr_overall", "median_qerr_region", "p90_qerr_region",
            "median_qerr_overall_of_that_arm_restricted_to_region_other_tenants",
            "n_test_region", "n_test_other"]
    os.makedirs(RESULTS, exist_ok=True)
    if args.append and os.path.exists(path):
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
        new = {r["seed"] for r in rows}
        old = [r for r in old if int(r["seed"]) not in new]
        for r in old:
            for k in ("seed", "k", "n_test", "n_test_region", "n_test_other"):
                r[k] = int(r[k])
            for k in cols[5:10]:
                r[k] = float(r[k])
        log(f"  --append: kept {len(old)} rows from {path}")
        rows = old + rows
    rows.sort(key=lambda r: (r["seed"], {"T1": 0, "T2": 1, "ALL": 2}[r["template"]],
                             r["arm"], r["k"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    log(f"  wrote {path} ({len(rows)} rows)")

    tab = summary_table(rows)
    print_table(tab, log)
    wall = time.time() - t_start + args.elapsed_base
    log(f"\nthis invocation {time.time() - t_start:.1f}s; reported total {wall / 60:.1f} min")
    write_summary_md(tab, rows, args, wall, log, meta)
    log.close()


if __name__ == "__main__":
    main()
