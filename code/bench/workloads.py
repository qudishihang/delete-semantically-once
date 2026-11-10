"""Request workloads for deletion services.

Every workload is a pure function of a record universe (a list of record identities)
and a seed, and returns a list of ``bench.api.Request``.  Nothing here knows about
service implementations, models or storage engines, so the same workload can be
replayed against any service that implements ``bench.api.Service``.

Families
--------
replay(kind)      the five replay patterns of the paper (same key, new key, partition,
                  overlap chain, superset)
overlap(rho)      two requests sharing a fraction rho of their payload
adversarial(name) six sequences designed to maximise recomputation of a naive service
crash(plan)       a request sequence with a fault schedule attached
"""
from __future__ import annotations

import random
from typing import Callable, Iterable, Sequence

from .api import Request


# --------------------------------------------------------------------------- #
# the five replay patterns
# --------------------------------------------------------------------------- #
def replay(kind: str, universe: Sequence[str], size: int = 5000,
           n_requests: int = 5, seed: int = 11) -> list[Request]:
    """The replay patterns a deletion service actually receives."""
    rng = random.Random(seed)
    if len(universe) < size:
        raise ValueError("universe smaller than the request size")
    A = list(universe[:size])
    extra = list(universe[size:size + max(1, size // 2)])
    out: list[Request] = []
    if kind == "same_key":
        out = [Request("k", "tenant", tuple(A), "replay") for _ in range(n_requests)]
    elif kind == "new_key":
        shuf = list(A)
        rng.shuffle(shuf)
        out = [Request(f"n{i}", "tenant", tuple(shuf), "replay")
               for i in range(n_requests)]
    elif kind == "partition":
        parts = [A[i::n_requests] for i in range(n_requests)]
        out = [Request(f"p{i}", "tenant", tuple(p), "partition")
               for i, p in enumerate(parts)]
        out.append(Request("p-full", "tenant", tuple(A), "partition"))
    elif kind == "overlap_chain":
        cur = list(A)
        out = [Request("o0", "tenant", tuple(cur), "overlap")]
        for i in range(1, n_requests):
            keep = cur[: max(1, len(cur) // 2)]
            new = list(extra[: max(1, len(cur) - len(keep))])
            cur = keep + new
            out.append(Request(f"o{i}", "tenant", tuple(cur), "overlap"))
    elif kind == "superset":
        out = [Request("s0", "tenant", tuple(A), "superset"),
               Request("s1", "tenant", tuple(A + extra), "superset")]
    else:
        raise ValueError(kind)
    return out


# --------------------------------------------------------------------------- #
# overlap sweep
# --------------------------------------------------------------------------- #
def overlap(universe: Sequence[str], rho: float, size: int = 5000,
            seed: int = 21) -> list[Request]:
    rng = random.Random(seed)
    A = list(universe[:size])
    pool = list(universe[size:])
    keep = int(round(rho * size))
    B = list(rng.sample(A, keep)) + list(rng.sample(pool, size - keep))
    return [Request("a", "tenant", tuple(A), "overlap"),
            Request("b", "tenant", tuple(B), "overlap")]


# --------------------------------------------------------------------------- #
# adversarial sequences
# --------------------------------------------------------------------------- #
ADVERSARIAL = ("sliding_window", "growing_superset", "partition_cycle",
               "dual_cover", "random_retries", "concurrent_flood")


def adversarial(kind: str, universe: Sequence[str], n_requests: int = 100,
                window: int = 1000, seed: int = 0) -> list[Request]:
    U = list(universe)
    if kind == "sliding_window":
        return [Request(f"p{i}", "tenant", tuple(U[i:i + window]), kind)
                for i in range(n_requests)]
    if kind == "growing_superset":
        core = U[:window]
        return [Request(f"s{i}", "tenant", tuple(core + U[window:window + i]), kind)
                for i in range(n_requests)]
    if kind == "partition_cycle":
        slices = 10
        step = len(U) // slices
        return [Request(f"q{i}", "tenant",
                        tuple(U[(i % slices) * step:((i % slices) + 1) * step]), kind)
                for i in range(n_requests)]
    if kind == "dual_cover":
        a = U[: len(U) // 2]
        b = U[len(U) // 4: 3 * len(U) // 4]
        return [Request(f"r{i}", "tenant", tuple(a if i % 2 == 0 else b), kind)
                for i in range(n_requests)]
    if kind == "random_retries":
        rng = random.Random(seed)
        return [Request(f"t{i}", "tenant", tuple(rng.sample(U, 2000)), kind)
                for i in range(n_requests)]
    if kind == "concurrent_flood":
        return [Request(f"f{i}", "tenant", tuple(U[:1000]), kind) for i in range(32)]
    raise ValueError(kind)


def concurrent(requests: Sequence[Request]) -> tuple[list[Request], int]:
    """Mark a workload as concurrent; returns (requests, worker count)."""
    return list(requests), len(requests)


# --------------------------------------------------------------------------- #
# fault schedules
# --------------------------------------------------------------------------- #
FAULT_POINTS = ("after_execute", "before_commit", "after_commit_before_response")


def faulty(universe: Sequence[str], size: int = 100, point: str = "before_commit",
           retries: int = 2) -> list[Request]:
    """One request, retried ``retries`` times; the runner injects the crash."""
    A = tuple(universe[:size])
    return [Request(f"k{i}", "tenant", A, f"fault:{point}") for i in range(retries + 1)]


# --------------------------------------------------------------------------- #
# helpers used by the metrics
# --------------------------------------------------------------------------- #
def union_size(requests: Iterable[Request]) -> int:
    return len({r for req in requests for r in req.rids})


def minimal_jobs(requests: Iterable[Request]) -> int:
    """Number of requests that carry at least one record not requested before.

    This is the number of unlearning jobs that *any* correct service must execute for
    the sequence, because each such job has at least one record to forget.
    """
    seen: set[str] = set()
    n = 0
    for req in requests:
        if any(r not in seen for r in req.rids):
            n += 1
        seen.update(req.rids)
    return n


CATALOGUE: dict[str, str] = {
    "same_key": "the same request delivered five times under one key",
    "new_key": "the same effect delivered under five different keys",
    "partition": "five disjoint partitions of a payload, then the union",
    "overlap_chain": "four requests, each sharing half its records with the previous",
    "superset": "a request, then a strict superset of it",
    "sliding_window": "a 1000-record window sliding by one record per request",
    "growing_superset": "a fixed core plus a tail that grows by one record",
    "partition_cycle": "ten disjoint slices submitted in a cycle",
    "dual_cover": "two 50%-overlapping sets, alternated",
    "random_retries": "i.i.d. random 2000-record subsets",
    "concurrent_flood": "32 identical requests issued at once",
}
