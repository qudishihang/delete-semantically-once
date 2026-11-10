"""The interface a benchmarked deletion service must implement.

The benchmark treats a service as a black box with four operations.  Nothing about
metadata layout, storage engine, concurrency control or unlearning algorithm is
assumed; the benchmark only needs to know what the service was asked to delete, what
it did in response, and what durable state it claims to hold afterwards.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Request:
    """One deletion request: an idempotency key, a client and a record payload."""
    key: str
    client: str
    rids: tuple[str, ...]
    label: str = ""            # workload-internal tag, e.g. "replay" or "overlap"


@dataclass
class DeleteOutcome:
    """What the service reports for a single request."""
    executed: bool             # did it perform model work?
    delta_size: int            # how many record identities were new
    versions_created: int      # committed model-version transitions caused
    latency_ms: float


@dataclass
class ServiceStats:
    """Durable state the service reports after a workload."""
    jobs: int = 0              # unlearning jobs executed
    versions: int = 0          # committed model versions
    processed: int = 0         # record identities the service claims are deleted
    cost_examples: int = 0     # example-visits paid to the executor
    stale_commits: int = 0     # versions published from a superseded parent
    exec_wall_s: float = 0.0   # executor wall-clock
    extra: Mapping[str, float] = field(default_factory=dict)


@runtime_checkable
class Service(Protocol):
    """A deletion service under test.

    ``delete`` must be callable concurrently from several threads if the benchmark
    workload is concurrent; implementations that serialise internally are allowed.
    """

    name: str

    def bootstrap(self) -> None:
        """Create version 0 and any empty metadata the service needs."""

    def delete(self, key: str, client: str, rids: Sequence[str]) -> DeleteOutcome:
        """Process one request and return what happened."""

    def stats(self) -> ServiceStats:
        """Report the durable state (counters) after the workload."""

    # optional, used by the correctness audit when available
    def processed_rids(self) -> set[str]:
        raise NotImplementedError
