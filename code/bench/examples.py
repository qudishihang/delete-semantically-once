"""A minimal service written *outside* the reference implementation.

It exists to show that ``bench.api.Service`` is implementable in a few dozen lines and
that the benchmark reports something meaningful about an arbitrary implementation:
this one ignores request identity, so it re-executes every request and its ``job_amp``
equals the number of requests.  It is also a template for adding a new system.
"""
from __future__ import annotations

import time

from .api import DeleteOutcome, ServiceStats


class StatelessService:
    name = "stateless-example"

    def __init__(self, job_seconds: float = 0.002):
        self.job_seconds = job_seconds
        self.jobs = 0
        self.versions = 0
        self.processed: set[str] = set()

    def bootstrap(self) -> None:
        self.versions = 1

    def delete(self, key, client, rids) -> DeleteOutcome:
        t0 = time.perf_counter()
        time.sleep(self.job_seconds)          # stand-in for one unlearning job
        self.jobs += 1
        self.versions += 1
        self.processed.update(rids)
        return DeleteOutcome(executed=True, delta_size=len(rids), versions_created=1,
                             latency_ms=(time.perf_counter() - t0) * 1e3)

    def stats(self) -> ServiceStats:
        return ServiceStats(jobs=self.jobs, versions=self.versions,
                            processed=len(self.processed),
                            cost_examples=self.jobs * 100,
                            exec_wall_s=self.jobs * self.job_seconds)

    def processed_rids(self) -> set[str]:
        return set(self.processed)
