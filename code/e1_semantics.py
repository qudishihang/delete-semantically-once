"""E1 -- Semantic idempotence: what each request-level strategy actually costs.

Workloads are built so that the *deletion effect* is identical or nested while
the request encoding differs.  A request-key registry can only see the encoding;
a processed-set service sees the effect.

Scenarios
---------
S1 retry_same_key        A, A, A, A, A with the same key
S2 retry_new_key         same effect, five different keys, shuffled payload
S3 partition             five disjoint parts of A, then A itself
S4 overlap_chain         50% overlapping requests A1..A4
S5 superset              A, then A + B (B = 5% new)

Strategies
----------
stateless  no state at all
key        idempotency key registry (DeleteGuard-style)
payload    key-independent hash of the delivered payload
semantic   processed-set delta planning (no concurrency claims)
su         semantic + effect memo + claim/coalesce + atomic publish (proposed)
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np
import torch

from su import config as C
from su import harness as H
from su.service import DeleteService
from su.world import World, request_indices

torch.set_num_threads(int(C.NT))

MODES = ("stateless", "key", "payload", "semantic", "su")


def build_scenarios(w: World):
    rng = np.random.default_rng(7)
    A = np.array(w.indices_of(w.request_rids(0.10, 11)))
    extra = request_indices(w.n, 0.05, 12)
    extra = np.setdiff1d(extra, A)
    B = extra[: max(1, len(A) // 2)]
    S = {}
    S["S1_retry_same_key"] = [("k1", A)] * 5
    shuf = A.copy()
    rng.shuffle(shuf)
    S["S2_retry_new_key"] = [(f"n{i}", shuf) for i in range(5)]
    parts = np.array_split(A, 5)
    S["S3_partition"] = [(f"p{i}", p) for i, p in enumerate(parts)] + [("p-full", A)]
    chain = [A]
    for i in range(3):
        new = np.setdiff1d(extra, chain[-1])[: len(A) // 2]
        chain.append(np.concatenate([chain[-1][: len(A) // 2], new]))
    S["S4_overlap_chain"] = [(f"o{i}", c) for i, c in enumerate(chain)]
    S["S5_superset"] = [("s1", A), ("s2", np.concatenate([A, B]))]
    return {k: [(key, w.rids_of(idx)) for key, idx in v] for k, v in S.items()}


def run_scenario(w: World, mode: str, scenario: str, requests, method: str,
                 epochs: int, tag: str):
    with tempfile.TemporaryDirectory() as td:
        be = w.backend(method=method, epochs=epochs,
                       artifact_subdir=f"e1/{tag}/{mode}/{scenario}")
        svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                            scope_policy="delta", method=method, epochs=epochs)
        svc.bootstrap()
        created = {}
        lat = []
        for key, rids in requests:
            created[key] = rids
            r = svc.delete(key, "tenant-A", rids)
            lat.append(r.latency_ms)
        st = svc.stats()
        art = svc.current_artifact()
        ev = w.evaluate(art)
        lost = svc.lost_effects(created)
        out = dict(dataset=w.dataset, arch=w.arch, seed=w.seed, method=method,
                   epochs=epochs, mode=mode, scenario=scenario,
                   n_requests=len(requests), n_records=len(requests[-1][1]),
                   physical_runs=st["physical_runs"], versions=st["versions"] - 1,
                   processed=st["processed"], cost_examples=st["cost_examples"],
                   exec_wall_s=st["exec_wall_s"],
                   latency_mean_ms=float(np.mean(lat)), latency_max_ms=float(np.max(lat)),
                   lost_effects=sum(lost.values()) if lost else 0,
                   test_acc=ev["test_acc"], js_to_retrainA=ev["js_to_retrainA"],
                   js_to_base=ev["js_to_base"])
        svc.close()
    print(f"    {mode:9s} {scenario:20s} runs={out['physical_runs']:2d} "
          f"vers={out['versions']:2d} acc={out['test_acc']:.4f} "
          f"js_retrain={out['js_to_retrainA']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="neggradplus,finetune")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--out", default="e1_semantics.csv")
    args = ap.parse_args()

    with H.Timer("world"):
        w = World.build(args.dataset, args.arch, args.seed)
    S = build_scenarios(w)
    rows = []
    for method in args.methods.split(","):
        print(f"  method {method}", flush=True)
        for scenario, reqs in S.items():
            for mode in MODES:
                tag = f"{w.dataset}_{w.arch}_s{w.seed}"
                rows.append(run_scenario(w, mode, scenario, reqs, method, args.epochs, tag))
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
