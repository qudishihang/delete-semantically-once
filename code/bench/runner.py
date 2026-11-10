"""Execute a workload against any service that implements ``bench.api.Service``.

The runner is the only place that touches wall-clock and threads, so a service
implementation never has to know how it is being measured.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from .api import Request, Service
from .metrics import Observation, correct, summarize


@dataclass
class WorkloadResult:
    metrics: dict
    passed: bool
    observation: Observation = field(repr=False, default=None)


def run_workload(service_factory: Callable[[], Service],
                 requests: Sequence[Request],
                 workers: int = 1,
                 fault_plan: Optional[dict] = None,
                 crash_hook: Optional[Callable] = None,
                 retry_after_crash: bool = True) -> WorkloadResult:
    """Run ``requests`` against a fresh service and return the benchmark metrics.

    ``workers=1`` runs the requests sequentially; ``workers>1`` issues them
    concurrently through a thread pool (each request keeps its own key).

    ``fault_plan``/``crash_hook`` inject a crash.  The hook is called by the service
    implementation at the named point; the runner is responsible for restarting the
    service and letting the client retry, which is what a real client does.
    """
    svc = service_factory()
    svc.bootstrap()
    obs = Observation()

    def one(req: Request):
        t0 = time.perf_counter()
        try:
            out = svc.delete(req.key, req.client, req.rids)
            dt = (time.perf_counter() - t0) * 1e3
            obs.latencies_ms.append(dt)
            obs.deltas.append(out.delta_size)
            obs.executed.append(out.executed)
            obs.versions.append(out.versions_created)
            return out
        except Exception as e:                      # pragma: no cover
            obs.errors.append(f"{type(e).__name__}: {e}")
            return None

    t0 = time.perf_counter()
    if workers <= 1:
        for req in requests:
            one(req)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(one, requests))
    wall = time.perf_counter() - t0

    stats = svc.stats()
    lost = 0
    auditable = 0
    requested: dict[str, set[str]] = {}
    for req in requests:
        requested.setdefault(req.key, set()).update(req.rids)
    try:
        have = svc.processed_rids()
    except NotImplementedError:
        have = None
    if have is not None:
        auditable = 1
        for key, rids in requested.items():
            lost += len(rids - have)

    m = summarize(obs, stats, requests, wall, lost_effects=lost,
                  extra={"auditable": auditable})
    return WorkloadResult(metrics=m, passed=correct(m), observation=obs)


def run_suite(service_factory: Callable[[], Service],
              workload: Iterable[Request],
              label: str = "",
              workers: int = 1) -> dict:
    """Convenience wrapper that adds the workload label to the metrics."""
    r = run_workload(service_factory, workload, workers=workers)
    out = dict(r.metrics)
    out["workload"] = label
    out["passed"] = r.passed
    return out


def write_csv(path: str, rows: list[dict]) -> None:
    """Write benchmark rows to CSV, preserving first-seen column order."""
    import csv
    import os

    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
