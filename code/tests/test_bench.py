"""Tests for the benchmark harness (implementation-agnostic path)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import workloads as W
from bench.adapters import SUService
from bench.api import DeleteOutcome, ServiceStats
from bench.examples import StatelessService
from bench.identity import make_universe
from bench.runner import run_workload


def _backend():
    import tempfile
    from su.backends import SyntheticBackend
    td = tempfile.mkdtemp()
    return SyntheticBackend(os.path.join(td, "art"), sleep_s=0.0)


U = make_universe(5000)


def test_same_key_passes_and_costs_one_job():
    reqs = W.replay("same_key", U, size=1000, n_requests=5)
    r = run_workload(lambda: SUService(mode="su", backend=_backend()), reqs)
    assert r.metrics["jobs"] == 1
    assert r.metrics["min_jobs"] == 1
    assert r.metrics["job_amp"] == pytest.approx(1.0)
    assert r.metrics["lost_effects"] == 0
    assert r.passed


def test_new_key_breaks_key_dedup():
    reqs = W.replay("new_key", U, size=1000, n_requests=5)
    good = run_workload(lambda: SUService(mode="su", backend=_backend()), reqs)
    bad = run_workload(lambda: SUService(mode="key", backend=_backend()), reqs)
    assert good.metrics["job_amp"] == pytest.approx(1.0)
    assert bad.metrics["job_amp"] == pytest.approx(5.0)
    assert good.passed and not bad.passed


def test_non_tracking_service_is_not_audited():
    reqs = W.replay("new_key", U, size=500, n_requests=3)
    r = run_workload(lambda: SUService(mode="key", backend=_backend()), reqs)
    # a service without a processed set cannot be audited for lost effects
    assert r.metrics["auditable"] == 0
    assert not r.passed


def test_out_of_tree_service_runs():
    reqs = W.adversarial("partition_cycle", U, n_requests=20)
    r = run_workload(lambda: StatelessService(job_seconds=0.0), reqs)
    assert r.metrics["requests"] == 20
    assert r.metrics["min_jobs"] == 10
    assert r.metrics["job_amp"] == pytest.approx(2.0)


def test_lost_effects_are_detected():
    """A service that claims success but drops a record must fail the audit."""

    class Leaky:
        name = "leaky"

        def __init__(self):
            self.seen = set()
            self.jobs = 0

        def bootstrap(self):
            pass

        def delete(self, key, client, rids):
            self.jobs += 1
            self.seen.update(rids[:-1])          # drops the last record
            return DeleteOutcome(True, len(rids), 1, 0.1)

        def stats(self):
            return ServiceStats(jobs=self.jobs, versions=self.jobs + 1,
                                processed=len(self.seen))

        def processed_rids(self):
            return set(self.seen)

    reqs = W.replay("same_key", U, size=100, n_requests=2)
    r = run_workload(Leaky, reqs)
    assert r.metrics["lost_effects"] > 0
    assert not r.passed


def test_metrics_are_implementation_agnostic():
    """The runner works with a service that has no relation to the reference code."""
    reqs = W.replay("superset", U, size=500)
    r = run_workload(lambda: StatelessService(job_seconds=0.0), reqs)
    assert r.metrics["union"] > 0
    assert r.metrics["delta_amp"] >= 1.0
    assert set(["jobs", "versions", "wall_s"]).issubset(r.metrics)
