"""A semantically idempotent deletion service built on a versioned model store.

Guarantees (proposed mode, ``mode="su"``)
-----------------------------------------
G1  Safety.        Every request that returns success has all of its record
                   identities in the committed processed set H, and H only grows.
G2  Semantic
    idempotence.   If A is a subset of the committed H when a request commits, the
                   request commits *no new model version*.  Replay under the same
                   key, a different key, a different payload partition, or a strict
                   superset therefore produces exactly zero additional committed
                   model transitions.
G3  Atomicity.     A model version, the processed-set rows that explain it, and the
                   request-log rows of every request it satisfies are published in
                   one transaction.  A reader never observes H advanced without the
                   corresponding model version, nor a version whose deletion effect
                   is missing from H.
G4  Effect-once.   A record identity is assigned to at most one committed version
                   and is never re-executed once committed.  Physical recomputation
                   can still happen after a crash; the service accounts for it
                   explicitly (``exec_log``), i.e. it provides exactly-once *effect*
                   rather than exactly-once *execution*.
G5  Linearised
    publication.   Model versions form a chain (parent_vid) and every version is
                   computed from its parent, so no concurrent commit can lose
                   another commit's deletion effect.
G6  Progress/group
    commit.        Concurrent deletion requests are batched by the writer: one
                   unlearning job and one version publication can satisfy many
                   requests, and a replayed request inside a batch is absorbed.

Deliberate non-guarantee
------------------------
The service does not claim that a committed deletion is an *effective* unlearning
result.  Effectiveness is reported separately by an optional certificate produced
by the backend verifier (``Backend.verify``).  The contract is about request
completion, not about privacy.

Baseline modes
--------------
``stateless``  no state at all: every request runs the operator on its payload
``key``        idempotency-key registry, checked before the commit (the DeleteGuard
               design as deployed: the registry read and the model commit are not one
               atomic step)
``key_atomic`` the same registry, but the key check and the commit are serialised on a
               per-key lock and published in one transaction, i.e. a *correct*
               key-based baseline
``payload``    key-independent dedup on the *delivered* payload bytes
``semantic``   processed-set delta planning only, no concurrency control
``su_opt``     delta planning + optimistic in-flight claims + version validation
``su_nogroup`` the proposed protocol with group commit switched off: writer
               serialisation and atomic publication, one request per batch
``su``         proposed: delta planning + writer serialisation + group commit
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .identity import NormalisedRequest, delta, effect_hash, normalise_request, payload_digest

MODES = ("stateless", "key", "key_atomic", "payload", "semantic", "su_opt",
         "su_nogroup", "su")
TRACKING = ("semantic", "su_opt", "su_nogroup", "su")
NO_GROUP = ("su_nogroup",)


class ConflictError(RuntimeError):
    """Same idempotency key reused with a different deletion effect."""


class CrashInjected(RuntimeError):
    """Raised by the fault injector to simulate a process failure."""


class _StaleEffect(Exception):
    """The delta (or the base version) changed underneath us; retry."""


# --------------------------------------------------------------------------- #
# backend interface
# --------------------------------------------------------------------------- #
@dataclass
class UnlearnOutcome:
    artifact_path: str
    artifact_digest: str
    cost_examples: int
    wall_time_s: float = 0.0
    info: dict = field(default_factory=dict)


@dataclass
class UnlearnTask:
    """Everything an executor needs to know about one deletion step."""
    method: str
    epochs: int
    seed: int
    delta: list[str]          # new deletion work implied by the request(s)
    scope: list[str]          # what the chosen policy hands to the operator
    processed: list[str]      # committed H before this step
    base_vid: int = 0


class Backend:
    """Pluggable unlearning executor.

    ``unlearn`` must be pure with respect to service state: it receives the base
    artifact and a task description and returns a new immutable artifact.  It is
    always called outside a database transaction.
    """

    name = "backend"
    model_tag = "generic"

    def initial_artifact(self) -> UnlearnOutcome:
        raise NotImplementedError

    def unlearn(self, base_artifact: str, task: UnlearnTask) -> UnlearnOutcome:
        raise NotImplementedError

    def verify(self, artifact_path: str, rids: Sequence[str]) -> dict:
        return {}

    def artifact_exists(self, path: str) -> bool:
        return path == "" or os.path.exists(path)


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS config (k TEXT PRIMARY KEY, v TEXT);

CREATE TABLE IF NOT EXISTS versions (
    vid INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_vid INTEGER,
    artifact TEXT NOT NULL,
    digest TEXT NOT NULL,
    method TEXT,
    epochs INTEGER,
    n_delta INTEGER,
    n_scope INTEGER,
    n_keys INTEGER,
    cost_examples INTEGER,
    committed_at REAL,
    certificate TEXT,
    cause_key TEXT
);

CREATE TABLE IF NOT EXISTS processed (
    rid TEXT PRIMARY KEY,
    vid INTEGER NOT NULL,
    committed_at REAL
);

CREATE TABLE IF NOT EXISTS effect_memo (
    effect_hash TEXT PRIMARY KEY,
    vid INTEGER NOT NULL,
    committed_at REAL
);

CREATE TABLE IF NOT EXISTS requests (
    key TEXT PRIMARY KEY,
    client TEXT,
    effect_hash TEXT,
    dedup_hash TEXT,
    n_rids INTEGER,
    vid INTEGER,
    status TEXT,
    batched INTEGER DEFAULT 0,
    physical_runs INTEGER DEFAULT 0,
    latency_ms REAL,
    created_at REAL,
    committed_at REAL
);

CREATE TABLE IF NOT EXISTS inflight (
    effect_hash TEXT PRIMARY KEY,
    base_vid INTEGER,
    owner TEXT,
    started REAL
);

CREATE TABLE IF NOT EXISTS wal (
    wid INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL,
    kind TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS exec_log (
    eid INTEGER PRIMARY KEY AUTOINCREMENT,
    rkey TEXT,
    base_vid INTEGER,
    delta_hash TEXT,
    n_delta INTEGER,
    n_scope INTEGER,
    n_batch INTEGER,
    cost_examples INTEGER,
    wall_time_s REAL,
    ts REAL
);

CREATE INDEX IF NOT EXISTS idx_processed_vid ON processed(vid);
"""

_CHUNK = 500
_BIG_REQUEST = 5000


def _chunks(seq, n=_CHUNK):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


@dataclass
class DeleteResult:
    key: str
    vid: int
    executed: bool
    coalesced: bool
    delta_size: int
    latency_ms: float
    physical_runs: int
    mode: str
    path: str = ""
    batch_size: int = 1
    certificate: dict = field(default_factory=dict)


class DeleteService:
    def __init__(self, db_path: str, backend: Backend, mode: str = "su",
                 scope_policy: str = "delta", method: str = "neggradplus",
                 epochs: int = 10, seed: int = 0, claim_timeout_s: float = 30.0,
                 fault_plan: Optional[dict] = None, use_effect_memo: bool = True,
                 batch_window_s: float = 0.0, record_replays: bool = True):
        assert mode in MODES, mode
        self.db_path = db_path
        self.backend = backend
        self.mode = mode
        self.scope_policy = scope_policy
        self.method = method
        self.epochs = epochs
        self.seed = seed
        self.claim_timeout_s = claim_timeout_s
        self.fault_plan = dict(fault_plan or {})
        self.use_effect_memo = use_effect_memo
        self.batch_window_s = batch_window_s
        # when False, a replay answered from the key/memo cache is not written to
        # the request log: the effect is already durable, so a retry costs one
        # lookup instead of one commit.
        self.record_replays = record_replays
        self.tracks_processed = mode in TRACKING
        self._lk = threading.RLock()
        self._tls = threading.local()
        self._writer = threading.Lock()
        self._pending: dict[str, NormalisedRequest] = {}
        self._pending_lk = threading.Lock()
        self.group_commit = mode not in NO_GROUP
        # per-key mutexes for the atomic key registry baseline
        self._key_lk_guard = threading.Lock()
        self._key_lk: dict[str, threading.Lock] = {}
        self._fault_counts: dict[str, int] = {}
        self.stale_commits = 0        # versions published from a base that was no longer current
        self._open()
        with self._lk:
            self.conn.executescript(SCHEMA)
        if mode in ("su", "su_opt", "su_nogroup"):
            self.recover(force=True)

    # -- plumbing ---------------------------------------------------------- #
    def _open(self):
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False,
                                    isolation_level=None, timeout=120.0)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

    @contextmanager
    def _txn(self):
        with self._lk:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
            except BaseException:
                self.conn.execute("ROLLBACK")
                raise
            else:
                self.conn.execute("COMMIT")

    def _maybe_crash(self, point: str):
        if not self.fault_plan or self.fault_plan.get("point") != point:
            return
        n = self._fault_counts.get(point, 0) + 1
        self._fault_counts[point] = n
        if n == int(self.fault_plan.get("counter", 1)):
            raise CrashInjected(f"injected crash at {point}")

    def close(self):
        with self._lk:
            try:
                self.conn.close()
            except Exception:
                pass

    def restart(self) -> dict:
        self.close()
        self._open()
        return self.recover(force=True) if self.mode in ("su", "su_opt") \
            else {"recovered": 0, "versions": 0}

    # -- recovery ---------------------------------------------------------- #
    def recover(self, force: bool = False) -> dict:
        with self._txn() as c:
            if force:
                abandoned = c.execute("SELECT COUNT(*) n FROM inflight").fetchone()["n"]
                c.execute("DELETE FROM inflight")
            else:
                cut = time.time() - self.claim_timeout_s
                abandoned = max(c.execute("DELETE FROM inflight WHERE started < ?",
                                          (cut,)).rowcount, 0)
            c.execute("INSERT INTO wal(ts, kind, payload) VALUES (?,?,?)",
                      (time.time(), "recover", json.dumps({"abandoned_inflight": abandoned})))
        with self._lk:
            rows = list(self.conn.execute("SELECT vid, artifact FROM versions ORDER BY vid"))
        missing = [r["vid"] for r in rows if not self.backend.artifact_exists(r["artifact"])]
        return {"recovered": abandoned, "versions": len(rows), "missing_artifacts": missing}

    # -- state accessors --------------------------------------------------- #
    def current_vid(self) -> int:
        with self._lk:
            r = self.conn.execute("SELECT v FROM config WHERE k='current_vid'").fetchone()
        return int(r["v"]) if r else 0

    def current_artifact(self) -> Optional[str]:
        with self._lk:
            r = self.conn.execute(
                "SELECT artifact FROM versions ORDER BY vid DESC LIMIT 1").fetchone()
        return r["artifact"] if r else None

    def version_count(self) -> int:
        with self._lk:
            return int(self.conn.execute("SELECT COUNT(*) n FROM versions").fetchone()["n"])

    def processed_set(self) -> set[str]:
        with self._lk:
            return {r["rid"] for r in self.conn.execute("SELECT rid FROM processed")}

    def processed_count(self) -> int:
        with self._lk:
            return int(self.conn.execute("SELECT COUNT(*) n FROM processed").fetchone()["n"])

    def physical_runs(self) -> int:
        with self._lk:
            return int(self.conn.execute("SELECT COUNT(*) n FROM exec_log").fetchone()["n"])

    def request_row(self, key: str):
        with self._lk:
            return self.conn.execute("SELECT * FROM requests WHERE key=?", (key,)).fetchone()

    def versions(self) -> list[dict]:
        with self._lk:
            return [dict(r) for r in self.conn.execute(
                "SELECT * FROM versions ORDER BY vid")]

    def bootstrap(self):
        if self.current_artifact() is not None:
            return self.current_vid()
        out = self.backend.initial_artifact()
        with self._txn() as c:
            c.execute("INSERT INTO versions(parent_vid, artifact, digest, method, epochs,"
                      " n_delta, n_scope, n_keys, cost_examples, committed_at, cause_key)"
                      " VALUES (NULL,?,?,?,?,0,0,0,0,?,?)",
                      (out.artifact_path, out.artifact_digest, "base", 0, time.time(), "@base"))
            vid = int(c.execute("SELECT last_insert_rowid() i").fetchone()["i"])
            c.execute("INSERT OR REPLACE INTO config(k, v) VALUES('current_vid', ?)", (str(vid),))
        return vid

    # -- main entry point -------------------------------------------------- #
    def delete(self, key: str, client: str, rids: Sequence[str],
               max_retries: int = 6) -> DeleteResult:
        t0 = time.perf_counter()
        req = normalise_request(key, client, rids)
        if self.mode == "key":
            hit = self._key_lookup(req)
            if hit is not None:
                return self._res(key, hit, False, 0, t0, "key_cache")
        elif self.mode == "payload":
            hit = self._payload_lookup(req)
            if hit is not None:
                return self._res(key, hit, False, 0, t0, "payload_cache")
        if self.mode == "key_atomic":
            # a *correct* key registry: the registry read, the unlearning job and the
            # commit are serialised per key, so two concurrent retries of the same
            # request cannot both execute
            with self._key_lk_guard:
                lk = self._key_lk.setdefault(req.key, threading.Lock())
            with lk:
                hit = self._key_lookup(req)
                if hit is not None:
                    return self._res(key, hit, False, 0, t0, "key_cache")
                return self._delete_simple(req, t0, max_retries)
        if self.mode == "su" or self.mode == "su_nogroup":
            return self._delete_grouped(req, t0)
        return self._delete_simple(req, t0, max_retries)

    def _res(self, key, vid, executed, delta_size, t0, path, certified=None,
             batch_size=1, coalesced=False):
        return DeleteResult(key, vid, executed, coalesced, delta_size,
                            (time.perf_counter() - t0) * 1e3, 1 if executed else 0,
                            self.mode, path, batch_size, certified or {})

    # ------------------------------------------------------------------ #
    # proposed mode: writer serialisation + group commit
    # ------------------------------------------------------------------ #
    def _delete_grouped(self, req: NormalisedRequest, t0: float) -> DeleteResult:
        # --- O(1) fast paths that never touch the processed set ------------ #
        row = self._key_lookup(req)
        if row is not None:
            if self.record_replays:
                self._commit_noop(req, row)
            return self._res(req.key, row, False, 0, t0, "key_cache")
        if self.use_effect_memo:
            vid = self._memo_lookup(req.effect_hash)
            if vid is not None:
                if self.record_replays:
                    self._commit_noop(req, vid)
                return self._res(req.key, vid, False, 0, t0, "memo")

        with self._pending_lk:
            self._pending[req.key] = req
        try:
            with self._writer:
                row = self._key_lookup(req)
                if row is not None:
                    return self._res(req.key, row, False, 0, t0, "committed_by_batch")
                if self.use_effect_memo:
                    vid = self._memo_lookup(req.effect_hash)
                    if vid is not None:
                        self._commit_noop(req, vid)
                        return self._res(req.key, vid, False, 0, t0, "memo")
                if self.batch_window_s > 0:
                    # give concurrently arriving requests time to join this group
                    time.sleep(self.batch_window_s)
                with self._pending_lk:
                    if self.group_commit:
                        batch = list(self._pending.values())
                    else:
                        batch = [self._pending.get(req.key, req)]
                    for b in batch:
                        self._pending.pop(b.key, None)
                base_vid = self.current_vid()
                # delta against the committed set, by indexed lookup: cost is
                # O(|A| log|H|), independent of the size of H.
                union: list[str] = []
                seen: set[str] = set()
                for b in batch:
                    for r in b.rids:
                        if r not in seen:
                            seen.add(r)
                            union.append(r)
                if len(union) > _BIG_REQUEST:
                    new = self._delta_vs_db(union)
                else:
                    H = self._processed_subset(union)
                    new = [r for r in union if r not in H]
                if not new:
                    if self.record_replays:
                        for b in batch:
                            self._commit_noop(b, base_vid)
                    return self._res(req.key, base_vid, False, 0, t0, "noop")
                if self.scope_policy == "delta":
                    scope = new
                elif self.scope_policy == "request":
                    s = set()
                    for b in batch:
                        s |= set(b.rids)
                    scope = sorted(s)
                else:
                    scope = self._scope_union(batch)
                # only the execute path needs the whole processed set, and only when
                # the backend uses it to build the retained set
                if getattr(self.backend, "needs_processed", True):
                    H_full = self.processed_set()
                else:
                    H_full = set()
                outcome = self._execute(base_vid, new, scope, req.key, H_full,
                                        len(batch))
                self._maybe_crash("after_execute")
                cert = self.backend.verify(outcome.artifact_path, new)
                self._maybe_crash("before_commit")
                vid = self._commit_batch(batch, new, scope, base_vid, outcome, cert)
                self._maybe_crash("after_commit_before_response")
                return self._res(req.key, vid, True, len(new), t0, "execute",
                                 cert, batch_size=len(batch))
        finally:
            with self._pending_lk:
                self._pending.pop(req.key, None)

    def _processed_subset(self, rids) -> set[str]:
        """Membership test restricted to `rids` (indexed, independent of |H|)."""
        out: set[str] = set()
        if not self.tracks_processed:
            return out
        with self._lk:
            for part in _chunks(rids):
                q = ",".join("?" * len(part))
                out.update(r[0] for r in self.conn.execute(
                    f"SELECT rid FROM processed WHERE rid IN ({q})", tuple(part)))
        return out

    def _delta_vs_db(self, rids) -> list[str]:
        """Delta for very large requests, streamed in chunks."""
        new: list[str] = []
        for part in _chunks(rids, _BIG_REQUEST):
            done = self._processed_subset(part)
            new.extend(r for r in part if r not in done)
        return new

    def _scope_union(self, batch):
        s = set()
        for b in batch:
            s |= set(b.rids)
        return sorted(s)

    # ------------------------------------------------------------------ #
    # baselines and the optimistic variant
    # ------------------------------------------------------------------ #
    def _delete_simple(self, req: NormalisedRequest, t0: float, max_retries: int) -> DeleteResult:
        D: list[str] = []
        for _attempt in range(max_retries):
            base_vid = self.current_vid()
            if self.tracks_processed:
                # indexed membership test, so that the baseline differs from the
                # proposed service in its *semantics*, not in how it reads H
                Hsub = self._processed_subset(req.rids)
                D = [r for r in req.rids if r not in Hsub]
                H = Hsub
            else:
                H = set()
                D = list(req.rids)
            if self.mode == "su_opt" and self.use_effect_memo and D:
                if self._memo_lookup(req.effect_hash) is not None:
                    D = []
            if not D:
                vid = self._commit_noop(req, base_vid)
                return self._res(req.key, vid, False, 0, t0, "noop")

            scope = self._scope(req, H, D)
            owner = uuid.uuid4().hex
            if self.mode == "su_opt":
                if not self._claim(D, base_vid, owner):
                    self._wait_for_change(base_vid, D)
                    continue
                self._tls.claim = effect_hash(D)
            try:
                outcome = self._execute(base_vid, D, scope, req.key, H, 1)
                self._maybe_crash("after_execute")
                cert = self.backend.verify(outcome.artifact_path, D)
                self._maybe_crash("before_commit")
                vid = self._commit_batch([req], D, scope, base_vid, outcome, cert)
                self._maybe_crash("after_commit_before_response")
                return self._res(req.key, vid, True, len(D), t0, "execute", cert)
            except CrashInjected:
                raise
            except _StaleEffect:
                continue
            finally:
                if self.mode == "su_opt":
                    self._release()

        vid = self.current_vid()
        self._record_uncommitted(req, vid, "retry_exhausted")
        return self._res(req.key, vid, False, len(D), t0, "retry_exhausted", coalesced=True)

    # -- lookups ----------------------------------------------------------- #
    def _dedup_hash(self, req: NormalisedRequest) -> str:
        if self.mode == "payload":
            return payload_digest({"rids": list(req.rids), "raw_size": req.raw_size})
        return req.effect_hash

    def _key_lookup(self, req: NormalisedRequest) -> Optional[int]:
        with self._lk:
            r = self.conn.execute("SELECT effect_hash, vid, status FROM requests WHERE key=?",
                                  (req.key,)).fetchone()
        if r and r["status"] == "committed":
            if r["effect_hash"] != req.effect_hash:
                raise ConflictError("idempotency key reused with a different deletion effect")
            return int(r["vid"])
        return None

    def _payload_lookup(self, req: NormalisedRequest) -> Optional[int]:
        with self._lk:
            r = self.conn.execute(
                "SELECT vid FROM requests WHERE status='committed' AND dedup_hash=?",
                (self._dedup_hash(req),)).fetchone()
        return int(r["vid"]) if r else None

    def _memo_lookup(self, eh: str) -> Optional[int]:
        with self._lk:
            r = self.conn.execute("SELECT vid FROM effect_memo WHERE effect_hash=?",
                                  (eh,)).fetchone()
        return int(r["vid"]) if r else None

    # -- optimistic claims (su_opt) ---------------------------------------- #
    def _claim(self, D: Sequence[str], base_vid: int, owner: str) -> bool:
        eh = effect_hash(D)
        now = time.time()
        with self._txn() as c:
            row = c.execute("SELECT owner, started FROM inflight WHERE effect_hash=?",
                            (eh,)).fetchone()
            if row is not None and now - row["started"] < self.claim_timeout_s:
                return False
            c.execute("INSERT OR REPLACE INTO inflight(effect_hash, base_vid, owner, started)"
                      " VALUES (?,?,?,?)", (eh, base_vid, owner, now))
        return True

    def _release(self):
        eh = getattr(self._tls, "claim", None)
        self._tls.claim = None
        if not eh:
            return
        try:
            with self._txn() as c:
                c.execute("DELETE FROM inflight WHERE effect_hash=?", (eh,))
        except Exception:
            pass

    def _wait_for_change(self, base_vid: int, D: Sequence[str] = ()):
        deadline = time.time() + self.claim_timeout_s
        while time.time() < deadline:
            if self.current_vid() != base_vid or self._any_processed(D):
                return
            time.sleep(0.001)

    def _any_processed(self, D: Sequence[str]) -> bool:
        if not self.tracks_processed or not len(D):
            return False
        with self._lk:
            for part in _chunks(D):
                q = ",".join("?" * len(part))
                if self.conn.execute(f"SELECT COUNT(*) n FROM processed WHERE rid IN ({q})",
                                     tuple(part)).fetchone()["n"]:
                    return True
        return False

    # -- execution --------------------------------------------------------- #
    def _scope(self, req: NormalisedRequest, H: set, D) -> Sequence[str]:
        if self.scope_policy == "delta":
            return list(D)
        if self.scope_policy == "request":
            return list(req.rids)
        if self.scope_policy == "union":
            return sorted(set(req.rids) | set(H))
        raise ValueError(self.scope_policy)

    def _execute(self, base_vid: int, D, scope, key: str, H: set,
                 batch_size: int = 1) -> UnlearnOutcome:
        art = self.current_artifact()
        if art is None:
            self.bootstrap()
            art = self.current_artifact()
        t = time.time()
        task = UnlearnTask(method=self.method, epochs=self.epochs, seed=self.seed,
                           delta=list(D), scope=list(scope), processed=sorted(H),
                           base_vid=base_vid)
        out = self.backend.unlearn(art, task)
        if not out.wall_time_s:
            out.wall_time_s = time.time() - t
        self._maybe_crash("after_artifact_write")
        with self._txn() as c:
            c.execute("INSERT INTO exec_log(rkey, base_vid, delta_hash, n_delta, n_scope,"
                      " n_batch, cost_examples, wall_time_s, ts)"
                      " VALUES (?,?,?,?,?,?,?,?,?)",
                      (key, base_vid, effect_hash(D), len(D), len(scope), batch_size,
                       out.cost_examples, out.wall_time_s, time.time()))
        return out

    # -- commit ------------------------------------------------------------ #
    def _commit_batch(self, batch, D, scope, base_vid, outcome, cert) -> int:
        now = time.time()
        validate = self.mode in ("su", "su_opt")
        with self._txn() as c:
            row = c.execute("SELECT v FROM config WHERE k='current_vid'").fetchone()
            cur_vid = int(row["v"]) if row else 0
            stale = cur_vid != base_vid
            if validate and stale:
                raise _StaleEffect()
            if self.tracks_processed:
                for part in _chunks(D):
                    q = ",".join("?" * len(part))
                    if c.execute(f"SELECT COUNT(*) n FROM processed WHERE rid IN ({q})",
                                 tuple(part)).fetchone()["n"]:
                        raise _StaleEffect()
            if stale:
                self.stale_commits += 1
            keys = [b.key for b in batch]
            c.execute(
                "INSERT INTO versions(parent_vid, artifact, digest, method, epochs,"
                " n_delta, n_scope, n_keys, cost_examples, committed_at, certificate, cause_key)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (cur_vid, outcome.artifact_path, outcome.artifact_digest, self.method,
                 self.epochs, len(D), len(scope), len(keys), outcome.cost_examples, now,
                 json.dumps(cert), json.dumps(keys)))
            vid = int(c.execute("SELECT last_insert_rowid() i").fetchone()["i"])
            if self.tracks_processed:
                c.executemany(
                    "INSERT OR IGNORE INTO processed(rid, vid, committed_at) VALUES (?,?,?)",
                    [(r, vid, now) for r in D])
                c.executemany(
                    "INSERT OR REPLACE INTO effect_memo(effect_hash, vid, committed_at)"
                    " VALUES (?,?,?)", [(b.effect_hash, vid, now) for b in batch])
            c.executemany(
                "INSERT OR REPLACE INTO requests(key, client, effect_hash, dedup_hash, n_rids,"
                " vid, status, batched, physical_runs, latency_ms, created_at, committed_at)"
                " VALUES (?,?,?,?,?,?,'committed',?,1,0,?,?)",
                [(b.key, b.client, b.effect_hash, self._dedup_hash(b), len(b.rids), vid,
                  1 if len(batch) > 1 else 0, now, now) for b in batch])
            c.execute("INSERT OR REPLACE INTO config(k, v) VALUES('current_vid', ?)", (str(vid),))
            c.execute("INSERT INTO wal(ts, kind, payload) VALUES (?,?,?)",
                      (now, "commit", json.dumps({"vid": vid, "keys": keys,
                                                  "n_delta": len(D)})))
        return vid

    def _commit_noop(self, req, vid: int) -> int:
        now = time.time()
        with self._txn() as c:
            c.execute("INSERT OR REPLACE INTO requests(key, client, effect_hash, dedup_hash,"
                      " n_rids, vid, status, batched, physical_runs, latency_ms, created_at,"
                      " committed_at) VALUES (?,?,?,?,?,?,'committed',0,0,0,?,?)",
                      (req.key, req.client, req.effect_hash, self._dedup_hash(req),
                       len(req.rids), vid, now, now))
            c.execute("INSERT INTO wal(ts, kind, payload) VALUES (?,?,?)",
                      (now, "noop", json.dumps({"key": req.key, "vid": vid})))
        return vid

    def _record_uncommitted(self, req, vid: int, why: str):
        with self._txn() as c:
            c.execute("INSERT OR REPLACE INTO requests(key, client, effect_hash, dedup_hash,"
                      " n_rids, vid, status, batched, physical_runs, latency_ms, created_at,"
                      " committed_at) VALUES (?,?,?,?,?,?,?,0,0,0,?,NULL)",
                      (req.key, req.client, req.effect_hash, self._dedup_hash(req),
                       len(req.rids), vid, why, time.time()))

    # -- reporting --------------------------------------------------------- #
    def stats(self) -> dict:
        with self._lk:
            v = int(self.conn.execute("SELECT COUNT(*) n FROM versions").fetchone()["n"])
            reqs = self.conn.execute(
                "SELECT status, COUNT(*) n FROM requests GROUP BY status").fetchall()
            runs = int(self.conn.execute("SELECT COUNT(*) n FROM exec_log").fetchone()["n"])
            cost = int(self.conn.execute("SELECT COALESCE(SUM(cost_examples),0) s FROM exec_log"
                                        ).fetchone()["s"])
            wall = float(self.conn.execute("SELECT COALESCE(SUM(wall_time_s),0) s FROM exec_log"
                                          ).fetchone()["s"])
            nkeys = int(self.conn.execute("SELECT COALESCE(SUM(n_keys),0) s FROM versions"
                                         ).fetchone()["s"])
        return {"versions": v, "physical_runs": runs, "cost_examples": cost,
                "stale_commits": self.stale_commits,
                "exec_wall_s": wall, "requests": {r["status"]: r["n"] for r in reqs},
                "processed": self.processed_count(), "keys_in_versions": nkeys}

    def lost_effects(self, requests: dict[str, list[str]]) -> dict:
        """G1 check for services that maintain a processed set."""
        if not self.tracks_processed:
            return {}
        H = self.processed_set()
        out = {}
        with self._lk:
            done = {r["key"]: r for r in self.conn.execute("SELECT * FROM requests")}
        for key, rids in requests.items():
            row = done.get(key)
            if row is None or row["status"] != "committed":
                continue
            missing = [r for r in rids if r not in H]
            if missing:
                out[key] = len(missing)
        return out
