# Benchmark card: deletion-service correctness under replay, overlap and failure

This card describes the benchmark used in *Delete Semantically Once*.  It is meant to
be usable by someone who wants to measure a **different** deletion service, not just to
reproduce our numbers.

## What is measured

A deletion service receives requests of the form `(key, client, payload)` where the
payload is a set of record identities.  The benchmark asks four questions:

1. **Does the service do the minimum work?**  For any request sequence, at most one
   unlearning job is needed per request that carries a record not requested before
   (`min_jobs`), and each record should be handed to the operator at most once.
2. **Is the published state coherent?**  Every record a successful request claimed to
   delete must be in the committed processed set (`lost_effects`), and no version may
   be published from a parent that has since been superseded (`stale_commits`).
3. **Does it survive load?**  Memory/metadata cost must not grow with the committed
   set, and cheap replays must stay cheap at 10^6 committed identities.
4. **What does correctness cost?**  Jobs, executor wall-clock, end-to-end wall-clock and
   latency percentiles.

## Workloads

Generated from a record universe of `n` synthetic identities; every workload is a pure
function of `(universe, seed)`.

| workload | what it contains |
|---|---|
| `same_key` | the same request delivered five times under one key |
| `new_key` | the same effect delivered under five different keys |
| `partition` | five disjoint partitions of a payload, then the union |
| `overlap_chain` | four requests, each sharing half its records with the previous |
| `superset` | a request, then a strict superset of it |
| `overlap:ρ` | two requests sharing a fraction ρ of the payload |
| `sliding_window` | a 1000-record window sliding by one record per request |
| `growing_superset` | a fixed core plus a tail that grows by one record |
| `partition_cycle` | ten disjoint slices submitted in a cycle |
| `dual_cover` | two 50%-overlapping sets, alternated |
| `random_retries` | i.i.d. random 2000-record subsets |
| `concurrent_flood` | 32 identical requests issued at once |
| `fault:*` | a request with a crash injected at `after_execute`, `before_commit` or `after_commit_before_response`, then retried |

## Metrics

| metric | definition |
|---|---|
| `jobs` | unlearning jobs executed |
| `min_jobs` | requests carrying at least one not-previously-requested record |
| `job_amp` | `jobs / min_jobs` — the redundancy factor (1.0 is optimal) |
| `versions` | committed model-version transitions |
| `stale_commits` | versions published from a superseded parent (a lost update) |
| `forget_visits` | record identities handed to the operator, summed |
| `delta_amp` | `forget_visits / |union of all effects|` |
| `lost_effects` | identities a successful request failed to commit (needs `processed_rids`) |
| `auditable` | 1 if the service can report its processed set |
| `passed` | `auditable ∧ lost_effects=0 ∧ stale_commits=0 ∧ job_amp≤1 ∧ no errors` |
| `lat_p50_ms`, `lat_p95_ms`, `lat_max_ms` | per-request latency |
| `wall_s`, `exec_wall_s` | end-to-end and executor time |

## Adding a service

Implement four methods (`bench/api.py`):

```python
class MyService:
    name = "my-service"

    def bootstrap(self) -> None: ...            # create version 0
    def delete(self, key, client, rids) -> DeleteOutcome: ...
    def stats(self) -> ServiceStats: ...
    def processed_rids(self) -> set[str]: ...   # or raise NotImplementedError
```

`bench/examples.py` contains a complete 40-line implementation written without the
reference codebase; it ignores request identity, so the benchmark reports
`job_amp = number of requests`.  Then

```python
from bench.runner import run_workload
from bench import workloads as W
from bench.identity import make_universe

U = make_universe(20000)
res = run_workload(lambda: MyService(), W.replay("new_key", U))
print(res.metrics, res.passed)
```

or from the shell:

```bash
python -m bench.cli --list
python -m bench.cli --workload new_key --service su --records 20000
python -m bench.cli --workload concurrent_flood --service stateless-example
python -m bench.cli --suite --out results/bench_suite.csv
```

## Reference services shipped

`bench/adapters.py` exposes the six strategies of the reference implementation
(`stateless`, `key`, `payload`, `semantic`, `su_opt`, `su`) behind the same interface.
All of them share one metadata layout, one indexed delta computation and one executor,
so a comparison between them isolates request-level semantics rather than
implementation quality.

## Results in the paper

The paper's Tables II–VI are produced by the experiment scripts `e1`–`e10`; this
package is the cleaned-up, implementation-independent version of the same workloads and
metrics.  It reproduces the qualitative ordering with a 0-ms executor:
`su` reaches `job_amp = 1.0` on every workload, `key` and `stateless` do not.
