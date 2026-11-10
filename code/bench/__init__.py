"""Reproducible benchmark harness for deletion-service correctness.

The package is deliberately independent of any particular service implementation:
it defines

  ``Service``   the interface a deletion service must expose to be benchmarked,
  workloads     request sequences (replay, overlap, adversarial, concurrent, faulty),
  metrics       the quantities the benchmark reports,
  runner        executes a workload against a service and emits one CSV row,
  adapters      the five reference strategies implemented in ``su.service``.

Adding a new service means implementing ``bench.api.Service`` (four methods) and
passing it to ``bench.runner.run_workload``.  See ``bench/CARD.md``.
"""

from .api import DeleteOutcome, Request, Service, ServiceStats  # noqa: F401
from .runner import run_workload, WorkloadResult  # noqa: F401
