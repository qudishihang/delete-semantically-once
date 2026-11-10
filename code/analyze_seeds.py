"""Summarise the multi-seed replication of the adversarial and scale benchmarks.

Reads results/e13_seeds.csv and results/e12_seeds.csv (produced with --seeds 0,1,2)
and prints, per workload / per request path, the spread across record universes.  The
point is to show that the two headline claims of Section VI-G are not artefacts of one
random draw:

  * work amplification under adversarial sequences is the same order of magnitude in
    every universe (and exactly 1.00 for the proposed service in all of them);
  * replay cost stays flat in |H| within each universe, and the level of that flat
    line moves by less than a factor of two between universes.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def read(name):
    path = os.path.join(RES, name)
    if not os.path.exists(path):
        print(f"  (missing {name})")
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    # ---------------- adversarial ------------------------------------------
    rows = read("e13_seeds.csv")
    seeds = sorted({int(r["seed"]) for r in rows})
    print(f"E13: {len(rows)} rows over seeds {seeds}")
    amp = defaultdict(dict)
    for r in rows:
        if r["mode"] in ("stateless", "key", "su"):
            amp[(r["sequence"], r["mode"])][int(r["seed"])] = float(r["job_amp"])
    print(f"  {'sequence':18s} {'mode':10s} " + " ".join(f"seed{s:<5d}" for s in seeds)
          + "   spread")
    for (seq, mode), per in sorted(amp.items()):
        vals = [per[s] for s in seeds]
        print(f"  {seq:18s} {mode:10s} " + " ".join(f"{v:9.2f}" for v in vals)
              + f"   {max(vals) / min(vals):5.2f}x")
    worst = max((max(p.values()) / min(p.values()), k) for k, p in amp.items())
    print(f"  largest seed-to-seed ratio: {worst[0]:.2f}x at {worst[1]}")
    su = [p for (s, m), p in amp.items() if m == "su"]
    print(f"  proposed service: max job amplification over all seeds/workloads = "
          f"{max(max(p.values()) for p in su):.2f}")

    # ---------------- scale ------------------------------------------------
    rows = read("e12_seeds.csv")
    paths = [("replay_cached_p50_us", "cached"), ("replay_durable_p50_ms", "durable"),
             ("subset_p50_us", "subset"), ("partial_p50_ms", "partial")]
    print(f"\nE12: {len(rows)} rows over seeds {sorted({int(r['seed']) for r in rows})}")
    for key, label in paths:
        print(f"  {label:8s} |A|=100   " + "  ".join(
            f"seed{s}:" for s in sorted({int(r["seed"]) for r in rows})))
        for s in sorted({int(r["seed"]) for r in rows}):
            vals = [float(r[key]) for r in rows
                    if int(r["seed"]) == s and r.get("request_size") == "100"
                    and r.get(key) not in ("", "nan")]
            if vals:
                print(f"    seed {s}: n={len(vals):2d} min={min(vals):7.3f} "
                      f"max={max(vals):7.3f} max/min={max(vals)/min(vals):5.2f}x")
    flood = [(int(r["seed"]), float(r["flood_rps"])) for r in rows
             if r.get("flood_rps") not in (None, "", "nan")]
    print("  flood request rate (req/s): " +
          ", ".join(f"seed{s}={v:.0f}" for (s, v) in flood))
    jobs = {int(r["seed"]): float(r["flood_jobs"]) for r in rows
            if r.get("flood_jobs") not in (None, "", "nan")}
    print(f"  flood unlearning jobs per seed: {jobs}")


if __name__ == "__main__":
    main()
