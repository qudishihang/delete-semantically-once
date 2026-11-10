"""Print the c=32 cells of results/e3_concurrency.csv, one line per (workload, service)."""
import csv
import io
import statistics as st
from collections import defaultdict

MODES = ["stateless", "key", "key_atomic", "semantic", "su_opt", "su_nogroup", "su"]
rows = [r for r in csv.DictReader(io.open("results/e3_concurrency.csv", newline=""))
        if r["concurrency"] == "32"]
cells = defaultdict(list)
for r in rows:
    cells[(r["scenario"], r["mode"])].append(r)


def mean(rs, key):
    vals = [float(r[key]) for r in rs if r[key] not in ("", None)]
    return st.mean(vals) if vals else float("nan")


for scen in ["identical", "disjoint", "overlap50"]:
    print("==", scen)
    for mode in MODES:
        rs = cells[(scen, mode)]
        print("  {:11s} jobs={:6.1f} vers={:5.1f} kv={:5.1f} stale={:5.1f} "
              "thr={:6.0f} p95={:6.0f} lost={:3.0f}".format(
                  mode, mean(rs, "physical_runs"), mean(rs, "versions"),
                  mean(rs, "keys_per_version"), mean(rs, "stale_commits"),
                  mean(rs, "throughput_rps"), mean(rs, "p95_ms"),
                  mean(rs, "lost_effects")))
