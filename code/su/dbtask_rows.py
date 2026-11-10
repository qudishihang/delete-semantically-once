"""Row-deletion cardinality estimation: workload v3.

Deletion units are table rows, and the tenant owns a *subset* of the rows in an
amount band (half of them).  That is deliberate: if a tenant owned every row of a
region, the correct post-deletion model would have no mass there and Q-error would
saturate, which would hide over-forgetting.  With a half-deleted band the exact
retraining target is a moderate, finite change (the band's mass halves), so both
under- and over-forgetting are visible.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import platform as P

DATA_PATH = os.path.join(P.ARTIFACTS, "dbrows_v3.npz")

N_COUNTRY = 10
N_AMOUNT = 32
N_DAY = 32
BAND = 4
N_TENANTS = N_AMOUNT // BAND          # 8 tenants, one amount band each


@dataclass
class RowWorkload:
    n_rows: int = 30000
    n_queries: int = 4000
    per_tenant: int = 300
    seed: int = 0


def build(cfg: RowWorkload = RowWorkload(), force: bool = False) -> dict:
    if os.path.exists(DATA_PATH) and not force:
        z = np.load(DATA_PATH)
        if int(z["n_rows"]) == cfg.n_rows and int(z["seed"]) == cfg.seed:
            return {k: z[k] for k in z.files}
    rng = np.random.default_rng(cfg.seed)
    country = rng.integers(0, N_COUNTRY, size=cfg.n_rows)
    amount_val = np.exp(rng.normal(3.2, 0.9, size=cfg.n_rows)) * (1 + 0.2 * country)
    day_val = rng.integers(0, 730, size=cfg.n_rows).astype(np.float64)
    a_edges = np.quantile(amount_val, np.linspace(0, 1, N_AMOUNT + 1))
    a_edges[0] -= 1e-9
    a_edges[-1] += 1e-9
    a_bin = np.clip(np.digitize(amount_val, a_edges) - 1, 0, N_AMOUNT - 1)
    d_bin = np.clip((day_val / 730.0 * N_DAY).astype(int), 0, N_DAY - 1)

    band = a_bin // BAND
    half = (np.arange(cfg.n_rows) % 2 == 0)
    tenant = np.where(half, band, -1)        # tenant t owns half of band t's rows

    def true_counts(qq):
        out = np.zeros(len(qq), dtype=np.float64)
        for i, (alo, ahi, dlo, dhi, cc) in enumerate(qq):
            m = (a_bin >= alo) & (a_bin < ahi) & (d_bin >= dlo) & (d_bin < dhi)
            if cc >= 0:
                m &= (country == cc)
            out[i] = m.sum()
        return out

    nq = cfg.n_queries
    a_lo = rng.integers(0, N_AMOUNT, size=nq)
    a_hi = np.minimum(N_AMOUNT, a_lo + rng.integers(1, 12, size=nq))
    d_lo = rng.integers(0, N_DAY, size=nq)
    d_hi = np.minimum(N_DAY, d_lo + rng.integers(2, 16, size=nq))
    c = rng.integers(-1, N_COUNTRY, size=nq)
    q = np.stack([a_lo, a_hi, d_lo, d_hi, c], axis=1)

    qt = []
    for t in range(N_TENANTS):
        # queries confined to the tenant's amount band
        lo = rng.integers(t * BAND, (t + 1) * BAND, size=cfg.per_tenant)
        hi = np.minimum(N_AMOUNT, lo + rng.integers(1, BAND + 1, size=cfg.per_tenant))
        dlo = rng.integers(0, N_DAY, size=cfg.per_tenant)
        dhi = np.minimum(N_DAY, dlo + rng.integers(2, 16, size=cfg.per_tenant))
        cc = rng.integers(-1, N_COUNTRY, size=cfg.per_tenant)
        qt.append(np.stack([lo, hi, dlo, dhi, cc], axis=1))
    ct = [true_counts(x) for x in qt]

    h = nq // 2
    out = dict(
        country=country, a_bin=a_bin, d_bin=d_bin, tenant=tenant,
        q_tr=q[:h], card_tr=true_counts(q[:h]),
        q_te=q[h:], card_te=true_counts(q[h:]),
        q_tenant=np.stack(qt), card_tenant=np.stack(ct),
        n_rows=np.array(cfg.n_rows), seed=np.array(cfg.seed),
        n_country=np.array(N_COUNTRY), n_amount=np.array(N_AMOUNT),
        n_day=np.array(N_DAY), n_tenants=np.array(N_TENANTS), band=np.array(BAND),
    )
    np.savez_compressed(DATA_PATH, **out)
    return out


class RowDensity(nn.Module):
    """p(c) p(a|c) p(d|a,c)."""

    def __init__(self, n_country=N_COUNTRY, n_amount=N_AMOUNT, n_day=N_DAY, emb=16):
        super().__init__()
        self.nc, self.na, self.nd = n_country, n_amount, n_day
        self.emb_c = nn.Embedding(n_country, emb)
        self.emb_a = nn.Embedding(n_amount, emb)
        self.logit_c = nn.Parameter(torch.zeros(1, n_country))
        self.head_a = nn.Linear(emb, n_amount)
        self.head_d = nn.Linear(2 * emb, n_day)

    def log_probs(self, c, a, d):
        lc = self.logit_c.expand(len(c), -1)
        ec, ea = self.emb_c(c), self.emb_a(a)
        la = self.head_a(ec)
        ld = self.head_d(torch.cat([ec, ea], dim=1))
        return (F.cross_entropy(lc, c) + F.cross_entropy(la, a)
                + F.cross_entropy(ld, d))

    @torch.no_grad()
    def log_joint(self):
        log_pc = F.log_softmax(self.logit_c, -1)[0]
        c = torch.arange(self.nc)
        la = self.head_a(self.emb_c(c))
        log_pa = F.log_softmax(la, -1)
        ca = torch.arange(self.nc).repeat_interleave(self.na)
        aa = torch.arange(self.na).repeat(self.nc)
        ld = self.head_d(torch.cat([self.emb_c(ca), self.emb_a(aa)], dim=1))
        log_pd = F.log_softmax(ld.reshape(self.nc, self.na, self.nd), -1)
        return log_pc[:, None, None] + log_pa[:, :, None] + log_pd

    @torch.no_grad()
    def selectivity_from_joint(self, log_joint, queries):
        p = log_joint.exp().numpy()
        total = p.sum(axis=0)

        def csum2(arr):
            return arr.cumsum(axis=0).cumsum(axis=1)

        cs_total = csum2(total)
        cs_c = [csum2(p[c]) for c in range(self.nc)]

        def box(cs, a0, a1, d0, d1):
            if a1 <= a0 or d1 <= d0:
                return 0.0
            v = cs[a1 - 1, d1 - 1]
            if a0 > 0:
                v -= cs[a0 - 1, d1 - 1]
            if d0 > 0:
                v -= cs[a1 - 1, d0 - 1]
            if a0 > 0 and d0 > 0:
                v += cs[a0 - 1, d0 - 1]
            return float(v)

        out = np.zeros(len(queries))
        for i, (alo, ahi, dlo, dhi, cc) in enumerate(queries):
            alo, ahi = int(alo), int(ahi)
            dlo, dhi = int(dlo), int(dhi)
            cs = cs_c[int(cc)] if cc >= 0 else cs_total
            out[i] = box(cs, alo, ahi, dlo, dhi)
        return out


def train_row_model(country, a_bin, d_bin, seed=0, epochs=60, lr=0.01, bs=512,
                    model=None):
    torch.manual_seed(seed)
    m = model if model is not None else RowDensity()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    n = len(country)
    ct = torch.from_numpy(country).long()
    at = torch.from_numpy(a_bin).long()
    dt = torch.from_numpy(d_bin).long()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = m.log_probs(ct[idx], at[idx], dt[idx])
            loss.backward()
            opt.step()
    m.eval()
    return m


def estimate_rows(model, queries, n_rows):
    lj = model.log_joint()
    return model.selectivity_from_joint(lj, queries) * n_rows


def unlearn_row_model(model, retain_idx, forget_idx, method="neggradplus", epochs=10,
                      lr=2e-3, beta=0.95, seed=0, bs=512, data=None):
    import copy

    m = copy.deepcopy(model)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    ct = torch.from_numpy(data["country"]).long()
    at = torch.from_numpy(data["a_bin"]).long()
    dt = torch.from_numpy(data["d_bin"]).long()
    n_ret, n_for = len(retain_idx), len(forget_idx)
    if n_for == 0:
        m.eval()
        return m
    for _ in range(epochs):
        perm = retain_idx[torch.randperm(n_ret, generator=gen)]
        for i in range(0, n_ret, bs):
            ri = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = m.log_probs(ct[ri], at[ri], dt[ri])
            if method == "neggradplus":
                fi = forget_idx[torch.randint(0, n_for, (len(ri),), generator=gen)]
                loss = beta * loss - (1.0 - beta) * m.log_probs(ct[fi], at[fi], dt[fi])
            loss.backward()
            opt.step()
    m.eval()
    return m
