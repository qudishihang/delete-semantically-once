"""Unit tests for the deletion service (synthetic backend, no ML needed)."""
import os
import threading
import time

import pytest

from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import CrashInjected, DeleteService


def mk_rids(n, tag="a"):
    return [record_identity("test", f"{tag}/{i}", version_digest(str(i).encode()))
            for i in range(n)]


def new_service(tmp_path, mode="su", scope="delta", sleep=0.01, fault=None, name="db"):
    be = SyntheticBackend(str(tmp_path / f"art_{name}"), sleep_s=sleep)
    svc = DeleteService(str(tmp_path / f"{name}.sqlite"), be, mode=mode,
                        scope_policy=scope, fault_plan=fault)
    svc.bootstrap()
    return svc


def test_same_key_replay_is_noop(tmp_path):
    svc = new_service(tmp_path)
    A = mk_rids(10)
    r1 = svc.delete("k1", "c", A)
    r2 = svc.delete("k1", "c", A)
    assert r1.executed and r1.delta_size == 10
    assert not r2.executed and r2.path in ("noop", "committed_by_batch", "key_cache", "memo")
    assert svc.version_count() == 2          # base + 1
    assert svc.physical_runs() == 1


def test_different_key_same_effect_is_noop(tmp_path):
    svc = new_service(tmp_path)
    A = mk_rids(10)
    r1 = svc.delete("k1", "c", A)
    r2 = svc.delete("k2", "c2", list(reversed(A)))
    assert r1.executed and not r2.executed
    assert svc.version_count() == 2
    assert svc.physical_runs() == 1


def test_partitioned_request_is_noop(tmp_path):
    svc = new_service(tmp_path)
    A = mk_rids(10)
    svc.delete("k1", "c", A[:6])
    svc.delete("k2", "c", A[6:])
    r = svc.delete("k3", "c", A)
    assert not r.executed
    assert svc.physical_runs() == 2


def test_partial_overlap_executes_only_delta(tmp_path):
    svc = new_service(tmp_path)
    A = mk_rids(10, "a")
    B = mk_rids(4, "a") + mk_rids(6, "b")
    svc.delete("k1", "c", A)
    r = svc.delete("k2", "c", B)
    assert r.executed and r.delta_size == 6
    assert svc.processed_count() == 16


def test_signature_baselines_do_more_work(tmp_path):
    A = mk_rids(10)
    counts = {}
    for mode in ("stateless", "key", "payload", "semantic", "su"):
        svc = new_service(tmp_path, mode=mode, name=mode)
        for k in ("k1", "k2", "k1"):
            svc.delete(k, "c", A)
        counts[mode] = (svc.physical_runs(), svc.version_count())
    # stateless: 3 runs, 4 versions; key: 2 runs; payload: 2 runs (k2 has a new key
    # but identical payload bytes -> payload dedup catches it); semantic/su: 1 run
    assert counts["stateless"][0] == 3
    assert counts["key"][0] == 2
    assert counts["semantic"][0] == 1
    assert counts["su"][0] == 1


def test_concurrent_identical_requests_coalesce(tmp_path):
    svc = new_service(tmp_path, sleep=0.20)
    A = mk_rids(10)
    errs, res = [], []

    def work(i):
        try:
            res.append(svc.delete(f"k{i}", "c", A))
        except Exception as e:      # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(16)]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    dt = time.time() - t0
    assert not errs
    assert svc.physical_runs() == 1, svc.physical_runs()
    assert svc.version_count() == 2
    assert dt < 0.20 * 16


def test_crash_before_commit_publishes_once(tmp_path):
    svc = new_service(tmp_path, sleep=0.01,
                      fault={"point": "before_commit", "counter": 1})
    A = mk_rids(10)
    with pytest.raises(CrashInjected):
        svc.delete("k1", "c", A)
    assert svc.version_count() == 1          # nothing published
    svc.restart()
    r = svc.delete("k1", "c", A)
    assert r.executed
    assert svc.version_count() == 2
    assert svc.processed_count() == 10


def test_crash_after_commit_is_effect_once(tmp_path):
    svc = new_service(tmp_path, fault={"point": "after_commit_before_response",
                                       "counter": 1})
    A = mk_rids(10)
    with pytest.raises(CrashInjected):
        svc.delete("k1", "c", A)
    svc.restart()
    r = svc.delete("k1", "c", A)              # client retry
    assert not r.executed
    assert svc.version_count() == 2
    assert svc.processed_count() == 10


def test_crash_after_commit_retry_with_new_key(tmp_path):
    svc = new_service(tmp_path, fault={"point": "after_commit_before_response",
                                       "counter": 1})
    A = mk_rids(10)
    with pytest.raises(CrashInjected):
        svc.delete("k1", "c", A)
    svc.restart()
    r = svc.delete("k2-retry", "c", A)
    assert not r.executed
    assert svc.version_count() == 2


def test_conflicting_key_reuse_rejected(tmp_path):
    svc = new_service(tmp_path, mode="key")
    A, B = mk_rids(10, "a"), mk_rids(10, "b")
    svc.delete("k1", "c", A)
    with pytest.raises(Exception):
        svc.delete("k1", "c", B)


def test_atomic_visibility(tmp_path):
    """After a crash before commit, the processed set must not have advanced."""
    svc = new_service(tmp_path, fault={"point": "before_commit", "counter": 1})
    A = mk_rids(10)
    with pytest.raises(CrashInjected):
        svc.delete("k1", "c", A)
    assert svc.processed_count() == 0
    assert svc.version_count() == 1


def test_lost_effects_none(tmp_path):
    svc = new_service(tmp_path)
    A = mk_rids(10)
    B = mk_rids(5, "b")
    svc.delete("k1", "c", A)
    svc.delete("k2", "c", A + B)
    assert svc.lost_effects({"k1": A, "k2": A + B}) == {}
