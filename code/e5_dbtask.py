"""E5 -- The learned-database task: cardinality estimation under record deletion.

Questions
---------
1. Does a committed deletion request actually remove the tenant's influence, and
   how far is the unlearned estimator from an exact retrain (measured by Q-error
   overall and inside the deleted tenant's query region)?
2. What does *replaying* that request do to Q-error -- the DB-level consequence of
   a non-idempotent operator?
3. Does the service protocol preserve both, end to end, on a workload whose
   per-tenant requests overlap?
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from su import config as C
from su import harness as H
from su import metrics as MET
from su import platform as P
from su.backends import CEBackend
from su.dbtask import DBWorkload, build
from su.identity import record_identity, version_digest
from su.methods import UnlearnConfig, run as run_method
from su.service import DeleteService

torch.set_num_threads(int(C.NT))


def make_model(dim, hidden=256):
    return nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def train_ce(X, y, seed, epochs=60, lr=1e-3, wd=1e-5, bs=256, log=False):
    torch.manual_seed(seed)
    m = make_model(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    n = len(X)
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
        if log and (ep % 20 == 0 or ep == epochs - 1):
            print(f"      ce epoch {ep} mse {tot/n:.4f}", flush=True)
    m.eval()
    return m


def predict(m, X):
    with torch.no_grad():
        return np.expm1(m(X).squeeze(-1).numpy().astype(np.float64))


def qstats(est, true, mask=None):
    if mask is not None:
        est, true = est[mask], true[mask]
    if len(est) == 0:
        return {}
    s = MET.q_error_stats(est, true)
    return {k: s[k] for k in ("qerr_median", "qerr_geomean", "qerr_p95", "frac_within_2")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e5_dbtask.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--tenants", default="0,1,2")
    args = ap.parse_args()

    data = build(DBWorkload())
    Xtr = torch.from_numpy(data["Xtr"]).float()
    ytr = torch.from_numpy(data["ytr"]).float()
    Xte = torch.from_numpy(data["Xte"]).float()
    card_te = data["card_te"]
    tenant = data["tenant"]
    ten_te = data["tenant_te"]
    print(f"  workload: {len(Xtr)} train queries, {len(Xte)} test queries, "
          f"{data['n_tenants']} tenants", flush=True)

    base = train_ce(Xtr, ytr, seed=0, log=True)
    base_path = os.path.join(P.CKPT, "ce_base_v2.pt")
    torch.save(base.state_dict(), base_path)
    rows = []
    est = predict(base, Xte)
    all_mask = np.ones(len(Xte), dtype=bool)
    rows.append(dict(experiment="reference", condition="base", k=0,
                     **qstats(est, card_te), **{f"all_{k}": v for k, v in
                                                qstats(est, card_te).items()}))
    print(f"  base Q-error: median={np.median(MET.q_error(est, card_te)):.3f} "
          f"gmean={float(np.exp(np.mean(np.log(np.maximum(est,1)/np.maximum(card_te,1))))):.3f}",
          flush=True)

    for t in [int(s) for s in args.tenants.split(",")]:
        in_region = ten_te == t
        mask = tenant == t
        keep = torch.from_numpy(np.where(~mask)[0])
        oracle = train_ce(Xtr[keep], ytr[keep], seed=0)
        est_o = predict(oracle, Xte)
        o_all = qstats(est_o, card_te)
        o_in = qstats(est_o, card_te, in_region)
        b_all = qstats(est, card_te)
        b_in = qstats(est, card_te, in_region)
        rows.append(dict(experiment="reference", condition=f"base_in_region{t}",
                         tenant=t, k=0, region="in", **b_in))
        rows.append(dict(experiment="reference", condition=f"retrain_tenant{t}",
                         tenant=t, k=0, region="all", **o_all))
        rows.append(dict(experiment="reference", condition=f"retrain_tenant{t}",
                         tenant=t, k=0, region="in", **o_in))
        print(f"  tenant {t}: base in-region median={b_in['qerr_median']:.3f} -> "
              f"retrain in-region median={o_in['qerr_median']:.3f} "
              f"(n={int(in_region.sum())})", flush=True)

        forget = torch.from_numpy(np.where(mask)[0])
        for method in ("finetune", "neggradplus"):
            cur = make_model(Xtr.shape[1])
            cur.load_state_dict(base.state_dict())
            for k in range(1, args.replays + 1):
                cfg = UnlearnConfig(method=method, epochs=args.epochs, lr=args.lr,
                                    weight_decay=1e-5, beta=0.95, batch_size=256,
                                    seed=k, loss="mse", grad_clip=args.clip)
                cur = run_method(cur, Xtr, ytr, keep, forget, cfg)
                est_k = predict(cur, Xte)
                rows.append(dict(experiment="replay", condition=method, tenant=t, k=k,
                                 region="all", **qstats(est_k, card_te)))
                rows.append(dict(experiment="replay", condition=method, tenant=t, k=k,
                                 region="in", **qstats(est_k, card_te, in_region)))
                print(f"    {method:12s} tenant{t} k={k} all_med="
                      f"{qstats(est_k, card_te)['qerr_median']:.3f} in_med="
                      f"{qstats(est_k, card_te, in_region)['qerr_median']:.3f}", flush=True)

        # ---- service-level run -------------------------------------------- #
        rid2idx = {record_identity("dbworkload", f"q/{i}", version_digest(str(i).encode())): i
                   for i in range(len(Xtr))}
        rid_of = {i: r for r, i in rid2idx.items()}
        A = [rid_of[i] for i in np.where(mask)[0]]
        B = [rid_of[i] for i in np.where(tenant == (t + 1) % 10)[0]]
        for mode in ("stateless", "key", "semantic", "su"):
            with tempfile.TemporaryDirectory() as td:
                be = CEBackend(os.path.join(P.ARTIFACTS, "versions", f"e5/{mode}"),
                               rid2idx, base_path, Xtr, ytr, Xte,
                               torch.from_numpy(data["yte"]).float(), card_te,
                               lr=args.lr, grad_clip=args.clip)
                svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                                    method="neggradplus", epochs=args.epochs)
                svc.bootstrap()
                for r in range(3):                       # three retries of the same effect
                    svc.delete(f"tenant{t}-retry{r}", f"tenant{t}", A)
                svc.delete(f"tenant{t}-new", f"tenant{t}", A + B)
                st = svc.stats()
                m = be._model(svc.current_artifact())
                est_s = predict(m, Xte)
                rows.append(dict(experiment="service", condition=mode, tenant=t, k=3,
                                 physical_runs=st["physical_runs"],
                                 versions=st["versions"] - 1,
                                 region="all", **qstats(est_s, card_te)))
                rows.append(dict(experiment="service", condition=mode, tenant=t, k=3,
                                 physical_runs=st["physical_runs"],
                                 versions=st["versions"] - 1,
                                 region="in", **qstats(est_s, card_te, in_region)))
                print(f"    service {mode:9s} tenant{t}: runs={st['physical_runs']} "
                      f"vers={st['versions']-1} all_med="
                      f"{qstats(est_s, card_te)['qerr_median']:.3f} in_med="
                      f"{qstats(est_s, card_te, in_region)['qerr_median']:.3f}", flush=True)
                svc.close()
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
