"""E3b -- concurrency audit with a real model: are commits linearisable?

The synthetic-backend experiment (E3) shows how much work each protocol performs.
This one asks the correctness question that a workload counter cannot answer: after
concurrent requests have committed, does the published model actually contain every
committed deletion?

Audit.  Concurrent requests delete disjoint record sets.  The service reports success
for each of them.  We then evaluate the final model on the union of the committed
rids and compare it with a control set of never-deleted training records of the same
size.  If publications were correctly linearised, the union's training accuracy drops
to the level the operator achieved in the sequential reference; if a commit was based
on a stale parent, the deleted records are still fitted and the deltas recorded in
the version chain no longer describe the model.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import threading
import time

import numpy as np
import torch

from su import config as C
from su import harness as H
from su import metrics as MET
from su.service import DeleteService
from su.world import World, request_indices

torch.set_num_threads(int(C.NT))
MODES = ("stateless", "key", "semantic", "su_opt", "su")


def run_one(w: World, mode: str, conc: int, per_req: int, epochs: int):
    td = tempfile.mkdtemp()
    try:
        return _run_one(w, mode, conc, per_req, epochs, td)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def _run_one(w: World, mode: str, conc: int, per_req: int, epochs: int, td: str):
    rng = np.random.default_rng(555)
    perm = rng.permutation(w.n)
    groups = [np.sort(perm[i * per_req:(i + 1) * per_req]) for i in range(conc)]
    control = np.sort(perm[conc * per_req:(conc + 1) * per_req])
    rids = [w.rids_of(g) for g in groups]
    if True:
        be = w.backend(method="neggradplus", epochs=epochs,
                       artifact_subdir=f"e3b/{mode}/c{conc}")
        svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode=mode,
                            method="neggradplus", epochs=epochs)
        svc.bootstrap()
        errs = []
        bar = threading.Barrier(conc)

        def work(i):
            try:
                bar.wait(timeout=120)
                svc.delete(f"k{i}", "tenant", rids[i])
            except Exception as e:      # pragma: no cover
                errs.append(repr(e))

        ts = [threading.Thread(target=work, args=(i,)) for i in range(conc)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        st = svc.stats()
        model = w.model(svc.current_artifact())
        union = np.sort(np.concatenate(groups))
        acc_deleted = MET.accuracy(model, w.Ftr[torch.tensor(union)],
                                   w.ytr[torch.tensor(union)])
        acc_control = MET.accuracy(model, w.Ftr[torch.tensor(control)],
                                   w.ytr[torch.tensor(control)])
        acc_test = MET.accuracy(model, w.Fte, w.yte)
        versions = svc.versions()
        vid_delta = sum(v["n_delta"] for v in versions[1:])
        processed = svc.processed_count()
        svc.close()
    return dict(mode=mode, concurrency=conc, per_request=per_req,
                n_requests=conc, physical_runs=st["physical_runs"],
                versions=st["versions"] - 1,
                processed=processed, version_delta_sum=vid_delta,
                stale_commits=st["stale_commits"],
                errors=len(errs),
                acc_deleted=acc_deleted, acc_control=acc_control,
                forget_gap=acc_deleted - acc_control,
                acc_test=acc_test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=C.PRIMARY_DATASET)
    ap.add_argument("--arch", default="mlp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--per-request", type=int, default=1000)
    ap.add_argument("--concurrency", default="4,16")
    ap.add_argument("--out", default="e3b_concurrency_audit.csv")
    args = ap.parse_args()
    w = World.build(args.dataset, args.arch, args.seed)
    rows = []
    # sequential reference: same work, no concurrency
    for conc in [int(c) for c in args.concurrency.split(",")]:
        for mode in MODES:
            try:
                r = run_one(w, mode, conc, args.per_request, args.epochs)
            except Exception as e:
                print(f"  c={conc:2d} {mode:9s} FAILED: {e!r}", flush=True)
                continue
            rows.append(r)
            print(f"  c={conc:2d} {mode:9s} runs={r['physical_runs']:3d} "
                  f"vers={r['versions']:3d} processed={r['processed']:5d} "
                  f"acc_del={r['acc_deleted']:.4f} acc_ctl={r['acc_control']:.4f} "
                  f"gap={r['forget_gap']:+.4f} stale={r['stale_commits']} "
                  f"acc_test={r['acc_test']:.4f}", flush=True)
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
