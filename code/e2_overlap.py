"""E2 -- Overlap sweep and the cost/quality frontier of recomputation scope.

For two consecutive requests whose overlap rho = |A n B| / |A|, the *semantic*
target after both is a model trained on D \\ (A u B).  Three implementation
choices are compared:

  delta    run the operator on the new records only            B \\ H
  request  re-run the operator on the whole second request     B
  union    re-run the operator on the whole processed set      H u B   (naive replay)

The experiment reports cost, distance to the *correct* retraining oracle, and
utility, as a function of rho in {0, 0.25, 0.5, 0.75, 1.0}.
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
from su.service import DeleteService
from su.world import World, request_indices

torch.set_num_threads(int(C.NT))

POLICIES = ("delta", "request", "union")
RHOS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="neggradplus,finetune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="e2_overlap.csv")
    ap.add_argument("--forget-mode", default="matched")
    args = ap.parse_args()

    with H.Timer("world"):
        w = World.build(args.dataset, args.arch, args.seed)

    n = w.n
    k = int(round(0.10 * n))
    A_idx = np.sort(request_indices(n, 0.10, 21))
    pool = np.setdiff1d(np.arange(n), A_idx)
    rng = np.random.default_rng(22)
    rows = []
    for rho in RHOS:
        n_keep = int(round(rho * k))
        overlap_part = rng.choice(A_idx, size=n_keep, replace=False)
        n_new = k - n_keep
        new_part = rng.choice(pool, size=n_new, replace=False)
        B_idx = np.sort(np.concatenate([overlap_part, new_part]))
        union_idx = np.union1d(A_idx, B_idx)
        keep = np.setdiff1d(np.arange(n), union_idx)
        oracle_path = w.oracle_for(keep, "e2")
        P_oracle = w.probs(oracle_path)
        P_base = w.probs(w.base_path)
        A_rids = w.rids_of(A_idx)
        B_rids = w.rids_of(B_idx)
        print(f"  rho={rho:.2f} |A|={len(A_idx)} |B|={len(B_idx)} "
              f"|A u B|={len(union_idx)} |B\\A|={len(new_part)}", flush=True)

        for method in args.methods.split(","):
            for policy in POLICIES:
                with tempfile.TemporaryDirectory() as td:
                    be = w.backend(method=method, epochs=args.epochs,
                                   artifact_subdir=f"e2/{method}/{policy}",
                                   forget_mode=args.forget_mode)
                    svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode="su",
                                        scope_policy=policy, method=method,
                                        epochs=args.epochs)
                    svc.bootstrap()
                    r1 = svc.delete("a", "tenant", A_rids)
                    r2 = svc.delete("b", "tenant", B_rids)
                    st = svc.stats()
                    p = MET.probs_of(w.model(svc.current_artifact()), w.Fte)
                    ev = {
                        "test_acc": MET.accuracy_of_probs(p, w.yte),
                        "js_to_oracle": MET.js_divergence(p, P_oracle),
                        "js_to_base": MET.js_divergence(p, P_base),
                    }
                    svc.close()
                rows.append(dict(dataset=w.dataset, arch=w.arch, seed=w.seed,
                                 method=method, policy=policy, rho=rho,
                                 forget_mode=args.forget_mode,
                                 n_A=len(A_idx), n_B=len(B_idx),
                                 n_union=len(union_idx), n_delta_B=max(1, len(new_part)),
                                 physical_runs=st["physical_runs"],
                                 versions=st["versions"] - 1,
                                 cost_examples=st["cost_examples"],
                                 delta_second=r2.delta_size, exec_wall_s=st["exec_wall_s"],
                                 **ev))
                print(f"    {method:12s} {policy:8s} runs={rows[-1]['physical_runs']} "
                      f"cost={rows[-1]['cost_examples']:>10d} acc={ev['test_acc']:.4f} "
                      f"js_oracle={ev['js_to_oracle']:.4f}", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
