"""E11 -- row-deletion cardinality estimation.

The deletion unit is a table row: a tenant owns every row of one country, and the
service is asked to remove that tenant.  The learned component is the Naru-style row
density model of ``su/dbtask_rows.py``; its retraining target is well defined because
deleting rows only removes training examples (no labels move).

We report, for three tenants:
  * Q-error on the whole test workload and on the tenant's own query region,
  * the exact-retraining reference for both,
  * the effect of replaying the same deletion with NegGrad+ and with fine-tuning,
  * the service-level comparison (three retries plus one overlapping request).
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch

from su import config as C
from su import harness as H
from su import metrics as MET
from su import platform as P
from su.dbtask_rows import (RowDensity, build, estimate_rows, train_row_model,
                            unlearn_row_model)
from su.identity import record_identity, version_digest
from su.service import Backend, DeleteService, UnlearnOutcome, UnlearnTask

torch.set_num_threads(int(C.NT))


class RowCEBackend(Backend):
    """Service backend around the row-density estimator."""

    name = "rowce"

    def __init__(self, data, rid2idx, base_path, artifact_dir, epochs=5, lr=1e-3,
                 beta=0.95):
        self.data = data
        self.rid2idx = rid2idx
        self.base_path = base_path
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.epochs, self.lr, self.beta = epochs, lr, beta
        self.n_rows = len(data["country"])

    def _model(self, path):
        m = RowDensity()
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        return m

    def initial_artifact(self):
        return UnlearnOutcome(self.base_path, "rowce-base", 0, 0.0)

    def unlearn(self, base_artifact, task: UnlearnTask):
        import time
        t0 = time.time()
        m = self._model(base_artifact)
        forget = np.array(sorted({self.rid2idx[r] for r in task.scope
                                  if r in self.rid2idx}), dtype=np.int64)
        excluded = {self.rid2idx[r] for r in task.processed if r in self.rid2idx}
        excluded.update(forget.tolist())
        keep = np.setdiff1d(np.arange(self.n_rows), np.array(sorted(excluded)))
        out = unlearn_row_model(m, torch.from_numpy(keep),
                                torch.from_numpy(np.sort(forget)),
                                epochs=self.epochs, lr=self.lr, beta=self.beta,
                                seed=task.seed, data=self.data)
        tag = f"{int(time.time()*1e6)}"
        path = os.path.join(self.artifact_dir, f"v_{tag}.pt")
        torch.save(out.state_dict(), path)
        return UnlearnOutcome(path, tag, cost_examples=len(keep) * self.epochs,
                              wall_time_s=time.time() - t0)


def qstats(model, queries, cards, n_rows):
    est = estimate_rows(model, queries, n_rows)
    s = MET.q_error_stats(est, cards)
    return {k: s[k] for k in ("qerr_median", "qerr_geomean", "qerr_p95",
                              "frac_within_2")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", default="0,1,2")
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="e11_dbrows.csv")
    args = ap.parse_args()

    data = build()
    n_rows = len(data["country"])
    ct = torch.from_numpy(data["country"]).long()
    at = torch.from_numpy(data["a_bin"]).long()
    dt = torch.from_numpy(data["d_bin"]).long()

    base = train_row_model(data["country"], data["a_bin"], data["d_bin"],
                           seed=0, epochs=60)
    base_path = os.path.join(P.CKPT, "rowce_base.pt")
    torch.save(base.state_dict(), base_path)
    rows = []
    b_all = qstats(base, data["q_te"], data["card_te"], n_rows)
    rows.append(dict(experiment="reference", condition="base", tenant=-1,
                     region="all", k=0, **b_all))
    print(f"  base: {b_all}", flush=True)

    for t in [int(s) for s in args.tenants.split(",")]:
        tenant = data["country"] == t
        forq = np.where(tenant)[0]
        keep = np.setdiff1d(np.arange(n_rows), forq)
        oracle = train_row_model(data["country"][keep], data["a_bin"][keep],
                                 data["d_bin"][keep], seed=0, epochs=60)
        q_in, c_in = data["q_tenant"][t], data["card_tenant"][t]
        o_all = qstats(oracle, data["q_te"], data["card_te"], n_rows)
        o_in = qstats(oracle, q_in, c_in, n_rows)
        b_in = qstats(base, q_in, c_in, n_rows)
        rows += [dict(experiment="reference", condition="base", tenant=t,
                      region="in", k=0, **b_in),
                 dict(experiment="reference", condition="retrain", tenant=t,
                      region="all", k=0, **o_all),
                 dict(experiment="reference", condition="retrain", tenant=t,
                      region="in", k=0, **o_in)]
        print(f"  tenant {t}: |rows|={len(forq)} base_in={b_in['qerr_median']:.3f} "
              f"retrain_in={o_in['qerr_median']:.3f} "
              f"retrain_all={o_all['qerr_median']:.3f}", flush=True)

        for method in ("neggradplus", "finetune"):
            cur = base
            for k in range(1, args.replays + 1):
                cur = unlearn_row_model(cur, torch.from_numpy(keep),
                                        torch.from_numpy(forq), method=method,
                                        epochs=args.epochs, lr=args.lr, seed=k,
                                        data=data)
                a = qstats(cur, data["q_te"], data["card_te"], n_rows)
                i_ = qstats(cur, q_in, c_in, n_rows)
                rows.append(dict(experiment="replay", condition=method, tenant=t,
                                 region="all", k=k, **a))
                rows.append(dict(experiment="replay", condition=method, tenant=t,
                                 region="in", k=k, **i_))
                print(f"    {method:12s} k={k} all={a['qerr_median']:.3f} "
                      f"in={i_['qerr_median']:.3f}", flush=True)

        # ---- service-level comparison ------------------------------------
        rid2idx = {record_identity("dbrows", f"row/{i}", version_digest(str(i).encode())): i
                   for i in range(n_rows)}
        rid_of = {i: r for r, i in rid2idx.items()}
        A = [rid_of[i] for i in forq]
        B = [rid_of[i] for i in np.where(data["country"] == (t + 1) % 10)[0]]
        for mode in ("stateless", "key", "semantic", "su"):
            td = tempfile.mkdtemp()
            be = RowCEBackend(data, rid2idx, base_path,
                              os.path.join(P.ARTIFACTS, "versions", f"e11/{mode}"),
                              epochs=args.epochs, lr=args.lr)
            svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                                method="neggradplus", epochs=args.epochs)
            svc.bootstrap()
            for r in range(3):
                svc.delete(f"t{t}-retry{r}", f"tenant{t}", A)
            svc.delete(f"t{t}-new", f"tenant{t}", A + B)
            st = svc.stats()
            m = be._model(svc.current_artifact())
            a = qstats(m, data["q_te"], data["card_te"], n_rows)
            i_ = qstats(m, q_in, c_in, n_rows)
            rows.append(dict(experiment="service", condition=mode, tenant=t,
                             region="all", k=3, runs=st["physical_runs"],
                             versions=st["versions"] - 1, **a))
            rows.append(dict(experiment="service", condition=mode, tenant=t,
                             region="in", k=3, runs=st["physical_runs"],
                             versions=st["versions"] - 1, **i_))
            print(f"    service {mode:9s} runs={st['physical_runs']} "
                  f"all={a['qerr_median']:.3f} in={i_['qerr_median']:.3f}", flush=True)
            svc.close()
            import shutil
            shutil.rmtree(td, ignore_errors=True)

    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
