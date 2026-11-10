"""Probe: does the *insert* (new-effect) path of the service grow with |H|?

The replay paths are flat in |H| by design (memo hit / indexed delta).  This probe asks
the other question, because the metadata-scale sweep first has to *build* the committed
set: it times each 1000-record commit as |H| grows.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time

from su.backends import SyntheticBackend
from su.identity import record_identity, version_digest
from su.service import DeleteService


def rids(n, offset=0):
    return [record_identity("probe", f"r/{offset + i}", version_digest(b"s"))
            for i in range(n)]


def main():
    td = tempfile.mkdtemp()
    try:
        be = SyntheticBackend(os.path.join(td, "art"), sleep_s=0.0)
        svc = DeleteService(os.path.join(td, "svc.sqlite"), be, mode="su")
        svc.bootstrap()
        built = 0
        for step in range(1, 41):
            t0 = time.perf_counter()
            svc.delete(f"b{step}", "t", rids(1000, offset=built))
            dt = time.perf_counter() - t0
            built += 1000
            if step % 5 == 0 or step <= 3:
                print(f"  |H|={built:7d} commit(1000 new) {dt*1e3:8.1f} ms "
                      f"({dt*1e6/1000:7.0f} us/record)", flush=True)
        svc.close()
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
