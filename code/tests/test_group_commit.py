"""Tests for group commit, linearised publication and crash behaviour under load."""
import threading
import time

import pytest

from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import CrashInjected, DeleteService


def mk_rids(n, tag="a", off=0):
    return [record_identity("test", f"{tag}/{off+i}", version_digest(str(off + i).encode()))
            for i in range(n)]


def new_service(tmp_path, mode="su", sleep=0.01, fault=None, name="db", timeout=30.0):
    be = SyntheticBackend(str(tmp_path / f"art_{name}"), sleep_s=sleep)
    svc = DeleteService(str(tmp_path / f"{name}.sqlite"), be, mode=mode,
                        fault_plan=fault, claim_timeout_s=timeout)
    svc.bootstrap()
    return svc


def run_concurrent(svc, reqs):
    errs, res = [], []
    bar = threading.Barrier(len(reqs))

    def work(k, r):
        try:
            bar.wait(timeout=60)
            res.append(svc.delete(k, "c", r))
        except Exception as e:      # pragma: no cover
            errs.append(repr(e))

    ts = [threading.Thread(target=work, args=(k, r)) for k, r in reqs]
    t0 = time.time()
    [t.start() for t in ts]
    [t.join() for t in ts]
    return res, errs, time.time() - t0


def test_group_commit_batches_disjoint_requests(tmp_path):
    svc = new_service(tmp_path, sleep=0.05)
    reqs = [(f"d{i}", mk_rids(50, "d", off=i * 50)) for i in range(16)]
    res, errs, wall = run_concurrent(svc, reqs)
    assert not errs
    assert svc.processed_count() == 16 * 50
    assert svc.physical_runs() < 16, svc.physical_runs()
    assert len(svc.lost_effects(dict(reqs))) == 0
    # every request must be committed
    st = svc.stats()
    assert st["requests"].get("committed", 0) == 16


def test_version_chain_is_linear(tmp_path):
    svc = new_service(tmp_path, sleep=0.01)
    reqs = [(f"d{i}", mk_rids(20, "d", off=i * 20)) for i in range(8)]
    run_concurrent(svc, reqs)
    vs = svc.versions()
    assert len(vs) >= 2
    for prev, cur in zip(vs, vs[1:]):
        assert cur["parent_vid"] == prev["vid"]
    # the union of all deltas equals the processed set
    assert sum(v["n_delta"] for v in vs[1:]) == svc.processed_count()


def test_replay_after_group_commit_is_noop(tmp_path):
    svc = new_service(tmp_path, sleep=0.01)
    A = mk_rids(10)
    svc.delete("k1", "c", A)
    for i in range(5):
        r = svc.delete(f"k-replay-{i}", "c", list(reversed(A)))
        assert not r.executed
    assert svc.version_count() == 2
    assert svc.physical_runs() == 1


def test_subset_and_superset_do_not_create_versions(tmp_path):
    svc = new_service(tmp_path)
    A = mk_rids(10)
    svc.delete("k1", "c", A)
    svc.delete("k2", "c", A[:3])
    svc.delete("k3", "c", mk_rids(3, "z"))
    svc.delete("k4", "c", A + mk_rids(3, "z"))
    assert svc.version_count() == 3
    assert svc.processed_count() == 13


def test_crash_before_commit_in_group(tmp_path):
    svc = new_service(tmp_path, sleep=0.01,
                      fault={"point": "before_commit", "counter": 1})
    A = mk_rids(10)
    with pytest.raises(CrashInjected):
        svc.delete("k1", "c", A)
    assert svc.processed_count() == 0
    svc.restart()
    svc.delete("k1", "c", A)
    svc.delete("k1b", "c", A)
    assert svc.version_count() == 2
    assert svc.processed_count() == 10


def test_crash_under_concurrency_no_lost_effect(tmp_path):
    svc = new_service(tmp_path, sleep=0.02,
                      fault={"point": "before_commit", "counter": 1})
    reqs = [(f"c{i}", mk_rids(20, "c", off=i * 20)) for i in range(8)]
    res, errs, _ = run_concurrent(svc, reqs)
    assert any("CrashInjected" in e for e in errs)
    svc.restart()
    # clients retry everything
    for k, r in reqs:
        svc.delete(k, "c", r)
    assert svc.processed_count() == 8 * 20
    assert len(svc.lost_effects(dict(reqs))) == 0


def test_conflict_key_reuse(tmp_path):
    from su.service import ConflictError
    svc = new_service(tmp_path, mode="key")
    A, B = mk_rids(5, "a"), mk_rids(5, "b")
    svc.delete("k1", "c", A)
    with pytest.raises(ConflictError):
        svc.delete("k1", "c", B)
