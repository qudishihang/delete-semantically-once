"""Metric definitions reported by the benchmark.

Each metric answers a question a deletion-service operator would ask.  The set is
deliberately small and implementation-agnostic.

  requests                requests issued
  jobs                    unlearning jobs executed (one job = one pass over retain)
  job_amp                 jobs / minimal_jobs, the redundancy factor
  versions                committed model-version transitions
  stale_commits           versions published from a superseded parent (a lost update)
  forget_visits           record-identities handed to the operator, summed
  delta_amp               forget_visits / |union of effects|, how often a record
                          was forgotten on average
  lost_effects            record identities a *successful* request failed to commit
  latency p50 / p95 / max per-request wall-clock
  wall_s                  end-to-end wall-clock
  exec_wall_s             executor wall-clock, i.e. the work the service bought
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .api import DeleteOutcome, ServiceStats
from .workloads import minimal_jobs, union_size


@dataclass
class Observation:
    """Raw per-request observations collected by the runner."""
    latencies_ms: list[float] = field(default_factory=list)
    deltas: list[int] = field(default_factory=list)
    executed: list[bool] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def summarize(obs: Observation, stats: ServiceStats, requests, wall_s: float,
              lost_effects: int = 0, extra: dict | None = None) -> dict:
    lat = np.asarray(obs.latencies_ms, dtype=float) if obs.latencies_ms else np.zeros(1)
    uni = union_size(requests)
    jmin = minimal_jobs(requests)
    out = {
        "requests": len(requests),
        "union": uni,
        "min_jobs": jmin,
        "jobs": int(stats.jobs),
        "job_amp": stats.jobs / jmin if jmin else float("nan"),
        "versions": int(stats.versions),
        "stale_commits": int(stats.stale_commits),
        "processed": int(stats.processed),
        "forget_visits": float(sum(obs.deltas)),
        "delta_amp": (sum(obs.deltas) / uni) if uni else float("nan"),
        "cost_examples": int(stats.cost_examples),
        "lost_effects": int(lost_effects),
        "errors": len(obs.errors),
        "wall_s": wall_s,
        "exec_wall_s": float(stats.exec_wall_s),
        "lat_p50_ms": float(np.percentile(lat, 50)),
        "lat_p95_ms": float(np.percentile(lat, 95)),
        "lat_max_ms": float(lat.max()),
        "lat_mean_ms": float(lat.mean()),
    }
    if extra:
        out.update(extra)
    return out


def correct(metrics: dict) -> bool:
    """The benchmark's pass/fail predicate for a single service on a workload.

    A service passes if it committed every requested effect, published no version from
    a superseded parent, and never exceeded the work bound (one execution per record).
    Services that are not expected to be correct (e.g. a stateless baseline) simply
    fail these predicates; that is the point of reporting them.
    """
    return (metrics.get("auditable", 0) == 1
            and metrics.get("lost_effects", 0) == 0
            and metrics.get("stale_commits", 0) == 0
            and metrics.get("job_amp", float("inf")) <= 1.0 + 1e-9
            and metrics.get("errors", 0) == 0)
