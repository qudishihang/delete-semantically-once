"""E4 -- Crash recovery: does a failed request lose a deletion or publish it twice?

For every (strategy, fault point) the harness
  1. issues a request and injects a crash at the chosen protocol point,
  2. simulates a process restart (reopen the database, run recovery),
  3. retries the request the way a real client would (same key),
  4. audits the durable state: lost deletion effects, duplicate committed
     version transitions, and recovery cost.

Fault points
------------
after_execute             artifact written, nothing committed
before_commit             same, immediately before the commit record
after_commit_before_ack   committed, but the client never sees the acknowledgement
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

from su import config as C
from su import harness as H
from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import CrashInjected, DeleteService

MODES = ("stateless", "key", "semantic", "su")
POINTS = ("after_execute", "before_commit", "after_commit_before_response")


def rids(n, offset=0):
    return [record_identity("bench", f"r/{offset+i}", version_digest(b"x")) for i in range(n)]


def run_one(mode: str, point: str, size: int, workdir: str, retries: int = 2):
    be = SyntheticBackend(os.path.join(workdir, "art"), sleep_s=0.01)
    db = os.path.join(workdir, "svc.sqlite")
    svc = DeleteService(db, be, mode=mode,
                        fault_plan={"point": point, "counter": 1})
    svc.bootstrap()
    A = rids(size)
    crashed = 0
    for attempt in range(retries + 1):
        try:
            svc.delete("k1", "tenant", A)
            break
        except CrashInjected:
            crashed += 1
            svc.restart()        # process restart + recovery
    else:
        pass
    st = svc.stats()
    lost = svc.lost_effects({"k1": A})
    row = svc.request_row("k1")
    out = dict(mode=mode, fault_point=point, request_size=size, crashes=crashed,
               physical_runs=st["physical_runs"], versions=st["versions"] - 1,
               processed=st["processed"],
               lost_effects=sum(lost.values()) if lost else 0,
               acked=(row is not None and row["status"] == "committed"),
               committed_version=(row["vid"] if row is not None else None),
               exec_wall_s=st["exec_wall_s"])
    svc.close()
    return out


def crash_under_load(size: int, concurrency: int, workdir: str):
    """One of c concurrent requests crashes just before commit."""
    be = SyntheticBackend(os.path.join(workdir, "art"), sleep_s=0.02)
    db = os.path.join(workdir, "svc.sqlite")
    svc = DeleteService(db, be, mode="su",
                        fault_plan={"point": "before_commit", "counter": 1})
    svc.bootstrap()
    import threading
    errs = []

    def work(i):
        try:
            svc.delete(f"k{i}", "tenant", rids(size, offset=i * size))
        except CrashInjected:
            errs.append("crash")
        except Exception as e:      # pragma: no cover
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(i,)) for i in range(concurrency)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    svc.restart()
    # retry every client request
    for i in range(concurrency):
        svc.delete(f"k{i}", "tenant", rids(size, offset=i * size))
    st = svc.stats()
    expected = {f"k{i}": rids(size, offset=i * size) for i in range(concurrency)}
    lost = svc.lost_effects(expected)
    out = dict(concurrency=concurrency, crashes=len(errs), physical_runs=st["physical_runs"],
               versions=st["versions"] - 1, processed=st["processed"],
               expected_records=size * concurrency,
               keys_in_versions=st["keys_in_versions"],
               lost_effects=sum(lost.values()) if lost else 0)
    svc.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e4_recovery.csv")
    ap.add_argument("--size", type=int, default=100)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    rows = []
    for mode in MODES:
        for point in POINTS:
            for rep in range(args.reps):
                with tempfile.TemporaryDirectory() as td:
                    r = run_one(mode, point, args.size, td)
                r["rep"] = rep
                rows.append(r)
            sub = [r for r in rows if r["mode"] == mode and r["fault_point"] == point]
            print(f"  {mode:9s} {point:26s} runs={np.mean([s['physical_runs'] for s in sub]):4.1f} "
                  f"vers={np.mean([s['versions'] for s in sub]):4.1f} "
                  f"lost={sum(s['lost_effects'] for s in sub)} "
                  f"acked={np.mean([s['acked'] for s in sub]):.2f}", flush=True)
    load_rows = []
    for c in (4, 16):
        for rep in range(args.reps):
            with tempfile.TemporaryDirectory() as td:
                r = crash_under_load(args.size, c, td)
            r["rep"] = rep
            load_rows.append(r)
        print(f"  crash-under-load c={c}: lost={load_rows[-1]['lost_effects']} "
              f"versions={load_rows[-1]['versions']} "
              f"keys={load_rows[-1].get('keys_in_versions')} "
              f"runs={load_rows[-1]['physical_runs']}", flush=True)
    H.write_csv(H.result_path(args.out), rows + load_rows)


if __name__ == "__main__":
    main()
