"""Adapters that expose the reference strategies of ``su.service`` to the benchmark."""
from __future__ import annotations

import os
import tempfile
from typing import Optional, Sequence

from .api import DeleteOutcome, ServiceStats
from .api import Request  # noqa: F401  (re-exported for convenience)


class SUService:
    """One strategy of the reference implementation, behind the benchmark interface.

    ``mode`` is one of ``stateless``, ``key``, ``payload``, ``semantic``, ``su_opt``,
    ``su``.  All of them share the same metadata representation and executor, so the
    benchmark compares semantics rather than implementations.
    """

    def __init__(self, mode: str = "su", backend=None, db_dir: Optional[str] = None,
                 **service_kwargs):
        if backend is None:
            raise ValueError("a backend (unlearning executor) is required")
        self.name = mode
        self._backend = backend
        self._tmp = None
        if db_dir is None:
            self._tmp = tempfile.mkdtemp()
            db_dir = self._tmp
        from su.service import DeleteService
        self._svc = DeleteService(os.path.join(db_dir, "svc.sqlite"), backend,
                                  mode=mode, **service_kwargs)

    # -- benchmark interface ---------------------------------------------- #
    def bootstrap(self) -> None:
        self._svc.bootstrap()

    def delete(self, key: str, client: str, rids: Sequence[str]) -> DeleteOutcome:
        before = self._svc.version_count()
        r = self._svc.delete(key, client, list(rids))
        after = self._svc.version_count()
        return DeleteOutcome(executed=bool(r.executed), delta_size=int(r.delta_size),
                             versions_created=int(after - before),
                             latency_ms=float(r.latency_ms))

    def stats(self) -> ServiceStats:
        st = self._svc.stats()
        return ServiceStats(
            jobs=int(st.get("physical_runs", 0)),
            versions=int(st.get("versions", 0)),
            processed=int(st.get("processed", 0)),
            cost_examples=int(st.get("cost_examples", 0)),
            stale_commits=int(st.get("stale_commits", 0)),
            exec_wall_s=float(st.get("exec_wall_s", 0.0)),
            extra={"keys_in_versions": st.get("keys_in_versions", 0)},
        )

    def processed_rids(self) -> set[str]:
        # strategies that keep no processed set cannot be audited for lost effects;
        # the benchmark reports that rather than pretending the deletions were lost.
        if not self._svc.tracks_processed:
            raise NotImplementedError
        return self._svc.processed_set()

    def close(self) -> None:
        self._svc.close()
        if self._tmp:
            import shutil
            shutil.rmtree(self._tmp, ignore_errors=True)


def synthetic_factory(mode: str, sleep_s: float = 0.002):
    """Factory for the zero-cost executor used by the protocol benchmarks."""
    from su.backends import SyntheticBackend

    holder = {}

    def make() -> SUService:
        td = tempfile.mkdtemp()
        holder["td"] = td
        be = SyntheticBackend(os.path.join(td, "art"), sleep_s=sleep_s)
        return SUService(mode=mode, backend=be, db_dir=td)

    return make
