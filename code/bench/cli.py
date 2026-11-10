"""Command-line driver for the benchmark.

Examples
--------
    python -m bench.cli --list
    python -m bench.cli --workload new_key --service su
    python -m bench.cli --workload concurrent_flood --service stateless
    python -m bench.cli --suite --out results/bench_suite.csv
"""
from __future__ import annotations

import argparse
import os
import sys

from .adapters import SUService, synthetic_factory
from .api import Request
from .examples import StatelessService
from .identity import make_universe
from .runner import run_suite, run_workload
from . import workloads as W

MODES = ("stateless", "key", "payload", "semantic", "su_opt", "su")


def _universe(n: int) -> list[str]:
    return make_universe(n)


def build(workload: str, universe_size: int) -> list[Request]:
    U = _universe(universe_size)
    if workload in ("same_key", "new_key", "partition", "overlap_chain", "superset"):
        return W.replay(workload, U)
    if workload.startswith("overlap:"):
        return W.overlap(U, float(workload.split(":", 1)[1]))
    if workload in W.ADVERSARIAL:
        return W.adversarial(workload, U)
    raise SystemExit(f"unknown workload {workload!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list workloads and services")
    ap.add_argument("--workload", default="new_key")
    ap.add_argument("--service", default="su", choices=MODES + ("stateless-example",))
    ap.add_argument("--records", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--suite", action="store_true", help="run every workload x service")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    if args.list:
        print("workloads:")
        for k, v in W.CATALOGUE.items():
            print(f"  {k:18s} {v}")
        print("services:", ", ".join(MODES + ("stateless-example",)))
        return 0

    def factory():
        if args.service == "stateless-example":
            return StatelessService()
        return SUService(mode=args.service, backend=_backend())

    rows = []
    if args.suite:
        for wl in W.CATALOGUE:
            reqs = build(wl, args.records)
            # the flood is the only workload whose point is simultaneity
            w = len(reqs) if wl == "concurrent_flood" else 1
            for mode in MODES:
                rows.append(run_suite(
                    lambda m=mode: SUService(mode=m, backend=_backend()),
                    reqs, label=wl, workers=w) | {"service": mode})
    else:
        reqs = build(args.workload, args.records)
        rows.append(run_suite(factory, reqs, label=args.workload,
                              workers=args.workers) | {"service": args.service})

    keys = ["workload", "service", "requests", "jobs", "min_jobs", "job_amp",
            "versions", "stale_commits", "delta_amp", "lost_effects", "passed",
            "wall_s", "lat_p50_ms", "lat_p95_ms"]
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(round(r[k], 3)) if isinstance(r[k], float) else str(r[k])
                        for k in keys))
    if args.out:
        from .runner import write_csv
        write_csv(args.out, rows)
        print("wrote", args.out)
    return 0


def _backend(sleep_s: float = 0.002):
    from su.backends import SyntheticBackend
    import tempfile
    td = tempfile.mkdtemp()
    return SyntheticBackend(os.path.join(td, "art"), sleep_s=sleep_s)


if __name__ == "__main__":
    sys.exit(main())
