"""E12 -- Metadata scale: does request cost grow with the committed set?

The obvious objection to keeping one row per committed record identity is that the
processed set grows without bound.  This benchmark sweeps

  * |H| = 10^3 .. 10^6 committed identities, and
  * request size |A| = 10, 100, 1000,

and measures four request paths:

  replay_durable  -- an exactly committed effect, recorded in the request log
                     (one WAL commit)
  replay_cached   -- an exactly committed effect, answered without a write
  subset          -- a strict subset of a committed request (delta is empty)
  partial         -- half committed, half new (delta + one publish commit)

plus storage and a flood of duplicate requests.  The executor is a 0-ms synthetic
backend, so every number is metadata cost, not unlearning cost.

`--seeds` repeats the whole sweep with different record-identity strings, so the claim
that cost is flat in |H| is not an artefact of one hash layout.  Seed 0 uses the
published strings; larger seeds salt them.
"""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time

import numpy as np

from su import config as C
from su import harness as H
from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import DeleteService

CHUNK = 1000
SALT = ""  # set per seed; seed 0 keeps the published identity strings


def rids(n, offset=0, prefix="r"):
    return [record_identity("scale", f"{SALT}{prefix}/{offset + i}", version_digest(b"s"))
            for i in range(n)]


def pct(xs, q):
    return float(np.percentile(xs, q)) * 1e3 if xs else float("nan")


def timed(f, reps):
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        out.append(time.perf_counter() - t0)
    return out


def run_seed(seed, checkpoints, sizes):
    global SALT
    SALT = "" if seed == 0 else f"s{seed}-"
    td = tempfile.mkdtemp()
    rows = []
    try:
        be = SyntheticBackend(os.path.join(td, "art"), sleep_s=0.0)
        db = os.path.join(td, "svc.sqlite")
        svc = DeleteService(db, be, mode="su")
        svc.bootstrap()
        be_c = SyntheticBackend(os.path.join(td, "art_c"), sleep_s=0.0)
        db_c = os.path.join(td, "svc_cached.sqlite")
        svc_c = DeleteService(db_c, be_c, mode="su", record_replays=False)
        svc_c.bootstrap()

        built = 0
        probe = 0
        for c in checkpoints:
            while built < c:
                batch = rids(min(CHUNK, c - built), offset=built)
                svc.delete(f"build-{built}", "t", batch)
                svc_c.delete(f"build-{built}", "t", batch)
                built += len(batch)
            for n in sizes:
                probe += 1
                P = rids(n, offset=10_000_000 + probe * 1000, prefix="probe")
                # commit the probe effect once on both instances
                svc.delete(f"probe-{probe}", "t", P)
                svc_c.delete(f"probe-{probe}", "t", P)
                lat_dur = timed(lambda: svc.delete(f"d-{time.perf_counter_ns()}", "t", P), 15)
                lat_cac = timed(lambda: svc_c.delete(f"c-{time.perf_counter_ns()}", "t", P), 100)
                sub = P[: max(1, n // 2)]
                lat_sub = timed(lambda: svc.delete(f"s-{time.perf_counter_ns()}", "t", sub), 15)
                old = P[: max(1, n // 2)]
                newr = rids(n - len(old), offset=20_000_000 + probe * 1000, prefix="n")
                lat_part = timed(lambda: svc.delete(f"p-{time.perf_counter_ns()}", "t",
                                                    old + newr), 5)
                size = os.path.getsize(db) / 1e6
                rows.append(dict(
                    seed=seed, h_size=c, request_size=n, processed=svc.processed_count(),
                    db_mb=size,
                    replay_durable_p50_ms=pct(lat_dur, 50),
                    replay_durable_p95_ms=pct(lat_dur, 95),
                    replay_cached_p50_us=1e3 * pct(lat_cac, 50),
                    replay_cached_p95_us=1e3 * pct(lat_cac, 95),
                    subset_p50_us=1e3 * pct(lat_sub, 50),
                    partial_p50_ms=pct(lat_part, 50),
                    jobs_so_far=svc.physical_runs()))
                print(f"  seed={seed} |H|={c:8d} |A|={n:5d} db={size:7.2f}MB "
                      f"dur={pct(lat_dur,50):6.2f}ms cac={1e3*pct(lat_cac,50):7.1f}us "
                      f"sub={1e3*pct(lat_sub,50):7.1f}us "
                      f"part={pct(lat_part,50):6.2f}ms", flush=True)

        # ---- duplicate flood --------------------------------------------------
        F = rids(100, offset=30_000_000, prefix="flood")
        svc.delete("flood-probe-d", "t", F)
        svc_c.delete("flood-probe-c", "t", F)
        runs_before = svc.physical_runs()
        t0 = time.perf_counter()
        lat = timed(lambda: svc.delete(f"fd-{time.perf_counter_ns()}", "t", F), 2000)
        wall = time.perf_counter() - t0
        rows.append(dict(
            seed=seed, h_size=built, request_size=100, processed=svc.processed_count(),
            db_mb=os.path.getsize(db) / 1e6,
            replay_durable_p50_ms=pct(lat, 50), replay_durable_p95_ms=pct(lat, 95),
            replay_cached_p50_us=float("nan"), replay_cached_p95_us=float("nan"),
            subset_p50_us=float("nan"), partial_p50_ms=float("nan"),
            jobs_so_far=svc.physical_runs(), replay_mode="durable",
            flood_requests=2000, flood_wall_s=wall, flood_rps=2000 / wall,
            flood_jobs=svc.physical_runs() - runs_before))
        print(f"  seed={seed} flood(durable,100): 2000 retries in {wall:.2f}s "
              f"({2000/wall:.0f}/s), jobs {svc.physical_runs()-runs_before}", flush=True)

        runs_before = svc_c.physical_runs()
        t0 = time.perf_counter()
        lat = timed(lambda: svc_c.delete(f"fc-{time.perf_counter_ns()}", "t", F), 20000)
        wall = time.perf_counter() - t0
        rows.append(dict(
            seed=seed, h_size=built, request_size=100, processed=svc_c.processed_count(),
            db_mb=os.path.getsize(db_c) / 1e6,
            replay_durable_p50_ms=float("nan"), replay_durable_p95_ms=float("nan"),
            replay_cached_p50_us=1e3 * pct(lat, 50),
            replay_cached_p95_us=1e3 * pct(lat, 95),
            subset_p50_us=float("nan"), partial_p50_ms=float("nan"),
            jobs_so_far=svc_c.physical_runs(), replay_mode="cached",
            flood_requests=20000, flood_wall_s=wall, flood_rps=20000 / wall,
            flood_jobs=svc_c.physical_runs() - runs_before))
        print(f"  seed={seed} flood(cached,100): 20000 retries in {wall:.2f}s "
              f"({20000/wall:.0f}/s), jobs {svc_c.physical_runs()-runs_before}", flush=True)
        svc.close()
        svc_c.close()
        return rows
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", default="1000,10000,100000,1000000")
    ap.add_argument("--sizes", default="10,100,1000")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated identity seeds; 0 is the published run")
    ap.add_argument("--out", default="e12_scale.csv")
    args = ap.parse_args()

    checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
    sizes = [int(x) for x in args.sizes.split(",")]
    rows = []
    for seed in [int(s) for s in args.seeds.split(",")]:
        rows.extend(run_seed(seed, checkpoints, sizes))
    H.write_csv(H.result_path(args.out), rows)


if __name__ == "__main__":
    main()
