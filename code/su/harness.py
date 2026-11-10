"""Small helpers shared by the experiment scripts."""
from __future__ import annotations

import csv
import json
import os
import statistics
import time
from typing import Iterable

import numpy as np

from . import config as C


def write_csv(path: str, rows: list[dict]):
    if not rows:
        print(f"  (no rows) {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def write_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"  wrote {path}")


def summarise_latencies(xs: Iterable[float]) -> dict:
    xs = sorted(xs)
    if not xs:
        return {}
    n = len(xs)
    return {
        "n": n,
        "mean_ms": 1000 * statistics.fmean(xs),
        "p50_ms": 1000 * xs[n // 2],
        "p95_ms": 1000 * xs[min(n - 1, int(round(0.95 * (n - 1))))],
        "max_ms": 1000 * xs[-1],
    }


class Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        print(f"  [{self.label}] {time.time() - self.t0:.1f}s", flush=True)


def result_path(name: str) -> str:
    return os.path.join(C.RESULTS, name)
