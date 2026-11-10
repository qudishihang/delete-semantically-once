"""Record-identity helpers for the benchmark.

The benchmark never invents record identities; it takes a universe of strings from the
caller.  This module provides a deterministic generator so that a run can be
reproduced without shipping a dataset.
"""
from __future__ import annotations

import hashlib


def make_universe(n: int, namespace: str = "bench") -> list[str]:
    """A deterministic, order-stable list of ``n`` record identities."""
    out = []
    for i in range(n):
        h = hashlib.sha256(f"{namespace}/{i}".encode()).hexdigest()
        out.append(h)
    return out
