# ICDE 2027 EAB — "Delete Semantically Once"

Target venue: **ICDE 2027, Experiments, Analysis & Benchmarks (EAB)**.  The paper is
framed as a benchmark plus a reference implementation that passes it, not as a
protocol-design paper.

Everything in the paper is produced by the code in `code/`; the tables are generated from
`results/*.csv` (Table I by `code/gen_table1.py`), not transcribed.  `reproduce.ps1` runs one
command per table; `SHA256SUMS.txt` covers every shipped file.

## What the benchmark covers

| Dimension | Instances |
|---|---|
| Replay patterns | same key; new key, same effect; partitioned; 50%-overlapping chain; superset |
| Overlap | $\rho=0,\ 0.25,\ 0.5,\ 0.75,\ 1$ |
| Concurrency | 1, 2, 4, 8, 16, 32 simultaneous requests; identical / 50%-overlap / disjoint |
| Adversarial | sliding window, growing superset, partition cycle, dual cover, random retries, 32-way flood |
| Faults | crash after execute, before commit, after commit before ack; crash under load |
| Scale | $\lvert H\rvert=10^3..10^5$, request size 10..1000, 20 000-retry flood |
| Metrics | jobs, versions, stale commits, latency p50/p95, wall-clock, executor time, forget-visits, `DAR_LOO`, `A_paired`, Q-error, MIA advantage, accuracy |
| Platforms | CIFAR-10 last-layer; converged CIFAR-10 CNN; learned CE on query records; learned CE on table rows |
| Baselines | stateless; key dedup (industry idempotency-token pattern); payload dedup; delta only; optimistic claim; proposed |

## Headline results (all from `results/*.csv`)

* **Calibrated statistics** (`e7_stats.csv`, 10 trajectories × 3 seeds): NegGrad+
  `DAR_LOO = 1.60–1.69` at k=2, `3.35–3.67` at k=3, `5.97–6.48` at k=4; fine-tuning
  `1.01–1.09` at k=2 (p ≤ 0.004) growing to `1.30–1.51` at k=4.  The older
  all-pairs estimator reads `1-1/n` under perfect idempotence.
* **Key dedup covers 1 of 5 replay patterns** (`e1_semantics.csv`), and under
  concurrency it misses even that one.
* **Stale-parent publication** (`e3`, `e10`): delta planning without a publish
  protocol commits versions computed from superseded parents — 31 of 32 at 32-way
  concurrency, with both synthetic and real executors.  The proposed service: 0.
  Two ablations (`e3_concurrency.csv`) separate the two contributions: an *atomic* key
  registry (`key_atomic`) collapses the identical workload from 32 jobs / 31 stale
  versions to 1 / 0, but on disjoint effects still executes 32 jobs with 31 stale
  versions; our protocol with group commit switched off (`su_nogroup`) has 0 stale
  versions but 32 jobs and 32 versions on the same workload.  Correctness comes from
  atomic publication, batching from group commit.
* **Group commit, real backend** (`e10_concurrency_real.csv`): 32 requests → 5 jobs,
  67 s wall vs 416 s, 66 s executor time vs 3325 s, p95 37 s vs 183 s, test accuracy
  87.23% vs 85.62%, JS to retrain 0.023 vs 0.042.
* **Adversarial sequences** (`e13_adversarial.csv`): naive services execute up to 50×
  the necessary jobs and forget up to 95× the necessary records; delta planning alone
  still executes 30× on a 32-way flood; the proposed service is at the bound (1.00) on
  every sequence.  Replicated on three record universes (`e13_seeds.csv`,
  `code/analyze_seeds.py`): every job ratio is identical in all three (50.0×, 10.0×,
  32.0×, 1.0×), only the i.i.d. retry row moves (1.30–1.37×), and the service is
  exactly 1.00 in all 18 cells.
* **Metadata scale** (`e12_scale.csv`, `e12_million.csv`): replay cost flat in |H|
  across **three** orders of magnitude (10^3 → 10^6): 46–117 µs for 10–100 records and
  1.2–2.5 ms for 1000 records, dominated by payload hashing rather than any lookup;
  storage ~190 B/record (182 MB at 10^6); durable replay adds one WAL commit
  (0.3–0.6 ms); a 20 000-retry flood runs at 3 300–4 500 req/s (cached) or
  1 300–2 000 req/s (durable) with **zero** unlearning jobs.  Replicated on three
  record universes (`e12_seeds.csv`): within a universe the 100-record replay is flat
  at 177/186/179/182 µs across |H| = 10^3..10^6 (1.05×) and 1.06× in the second,
  while the whole line moves 3.8× between sweeps taken under different background
  load; the flood gives 6 000 req/s (cached) and 2 450 (durable) with zero jobs in
  every seed.
* **Crash recovery** (`e4_recovery.csv`): effect-once holds; only the stateless service
  duplicates an effect after a post-commit crash.
* **Full recomputation as the exactness reference** (`e14_grow.csv`, `e14_repeat.csv`):
  five requests.  Repeated payload: recompute 5 jobs / 4.80M retained-record passes /
  JS 0.000; stateless 5 / 2.40M / 0.078 (82.01% test acc); the service **1 job / 0.48M /
  0.020 (87.09%)** — 5× less wall-clock than either, and closest to the retrained model.
  Growing payload: recompute 5 / 4.68M / 0.000, stateless 5 / 2.34M / 0.066, service
  5 / 2.34M / 0.080 — neither selective strategy reproduces retraining at this operator
  budget, and the operator overshoots the requested effect ~4–5×.
* **Batching vs unlearning quality** (`e15_batched.csv`, `e15_sequential.csv`): 32
  disjoint 200-record deletions, single run.  Group commit takes the work from 32 jobs to
  2; the median *signed* per-request gap to the exact-retraining oracle moves from
  −0.105 to −0.020 (median absolute gap 0.105 → 0.020, mean absolute 0.110 → 0.030).
  But the batched model's worst request is 0.220 below the oracle, its test accuracy is
  84.37% against 87.22%, and its MIA advantage on deleted records is 0.082 against
  0.011 — a cost/quality trade-off, not an effectiveness guarantee.  The 25/4 split of
  per-request gaps (over-/under-forgetting) is why the paper reports distributions, not
  means.
* **Portable benchmark harness** (`code/bench/`, `results/bench_suite.csv`): eleven  workloads behind a one-method `Service` interface, evaluated for jobs against the
  minimum, lost effects, stale commits and end-to-end cost.  Reference service passes
  11/11, effect-level deduplication without the memo 10/11 (fails the 32-way concurrent
  flood: 18 jobs, two versions), the three syntactic strategies cannot be certified at
  all because they do not record which effects they committed.
* **Database, query records** (`e5_dbtask.csv`): in-region Q-error 1.14 → 1.40 (k=2),
  10.5 (k=3), 53.7 (k=4), 1037 (k=5); service keeps it at 1.13–1.16.
* **Database, table rows** (`e11_dbrows.csv`): 1.073 → target 1.135; one NegGrad+
  invocation overshoots to 1.189, five to 1.243; fine-tuning approaches from below.
* **Operator/structure sensitivity** (`e6_*`): amplification monotone in β
  (0.90 → 3.50, 0.95 → 1.54, 0.99 → 0.96); class-concentrated requests are 2.4× more
  sensitive than random ones of the same size.
* **Converged full-network replication** (`e9_fullnet_conv.json`, `e9_conv_s1.json`,
  `e9_conv_s2.json`): three 4-conv networks trained from scratch for 80 epochs
  (79.85%, 79.82%, 80.43% test accuracy), 5 trajectories each.  Deleted-set accuracy
  falls 99.9% → 92.8/92.4/92.3% → 84.2/84.3/85.3% → 77.9/78.6/78.7%; test accuracy
  79.9% → 78.4/78.2/78.0% → 77.1/77.1/77.8% → 76.4/76.3/77.2%.  At k=3 all three
  reject the noise-floor hypothesis (`A_paired = 1.48, 1.28, 1.09`, p < 0.001); at
  k=2 two of three do (`1.27, 1.17`, p < 0.001) and the third sits at `1.04`
  (p = 0.14).  The same architecture trained for 12 epochs gave `0.97` (n.s.): the
  contrast is between operating points, not platforms.

## Layout

```
code/su/                 library: identity, service, methods, metrics, stats,
                         dbtask (query records), dbtask_rows (rows), backends
code/e0_platform.py      base models + retraining oracles
code/e0b_calibrate.py    unlearning-budget calibration
code/e1_semantics.py     replay-pattern ladder
code/e2_overlap.py       replay scope / overlap
code/e3_concurrency.py   concurrency protocol benchmark
code/e3b_concurrency_audit.py  real-model concurrency audit
code/e4_recovery.py      crash injection
code/e5_dbtask.py        learned CE, query records
code/e6_sensitivity.py   operator / budget / request / structure sweeps
code/e7_stats.py         calibrated replay statistics
code/e8_batching.py      group-commit latency/work frontier
code/e9_fullnet.py       converged full-network replication
code/e10_concurrency_real.py   32-way concurrency, real executor
code/e11_dbrows.py       learned CE, deleted table rows
code/e12_scale.py        metadata scale + duplicate flood
code/e13_adversarial.py  adversarial request sequences
code/e14_recompute.py    full recomputation as the exactness/cost reference
code/analyze_seeds.py    multi-seed spread of the adversarial and scale benchmarks
code/make_table_conc.py  regenerate the concurrency table from results/e3_concurrency.csv
code/probe_insert_scale.py  commit-path cost vs |H| (the path that is not flat)
code/bench/              portable benchmark: api (Service protocol), workloads,
                         metrics, runner, adapters, cli, CARD.md (benchmark card)
code/gen_table1.py       regenerate Table I from the CSV
code/make_figures.py     figures
code/tests/              19 protocol unit tests + 6 harness tests
results/                 raw CSVs used by the paper
paper/main.tex           IEEEtran manuscript (13 pp., body 12, refs 1)
```

## Reproducing

```powershell
$env:PYTHONPATH="code"
python -m pytest code/tests -q
python code/e0_platform.py --datasets cifar10 --archs mlp --seeds 0,1,2
python code/e7_stats.py --runs 10
python code/e1_semantics.py
python code/e3_concurrency.py
python code/e10_concurrency_real.py
python code/e4_recovery.py
python code/e5_dbtask.py --epochs 3 --lr 0.0001 --clip 1.0
python code/e11_dbrows.py --tenants 0,1,2 --replays 5 --epochs 10 --lr 0.002
python code/e6_sensitivity.py --runs 3
python code/e12_scale.py --checkpoints 1000,10000,100000
python code/e13_adversarial.py --universe 20000 --requests 100
python code/e12_scale.py --seeds 0,1,2 --out e12_seeds.csv
python code/e13_adversarial.py --seeds 0,1,2 --out e13_seeds.csv
python code/analyze_seeds.py          # multi-seed spread of both benchmarks
python code/make_table_conc.py        # regenerate the concurrency table
python code/e9_fullnet.py --base-epochs 80 --kmax 3 --with-oracle
python code/e14_recompute.py --pattern repeat --requests 5 --core 2000 --ue-epochs 10 --out e14_repeat.csv
python code/e14_recompute.py --pattern grow   --requests 5 --core 2000 --chunk 400 --ue-epochs 10 --out e14_grow.csv
python -m bench.cli --suite --out results/bench_suite.csv
python code/gen_table1.py; python code/make_figures.py
cd paper; pdflatex main; bibtex main; pdflatex main; pdflatex main
```

## Scoring a different service with `code/bench/`

Implement `bench.api.Service` (`delete`, `stats`, `current_model_id`) and run
`python -m bench.cli --service your.module:factory --suite`.  The harness generates the
workloads, drives them, and writes one row per (workload, service) with: jobs executed
against the theoretical minimum, effects acknowledged but not committed (`lost_effects`,
requires the service to expose them), stale publications, cost in examples, wall-clock
and latency percentiles, and a pass/fail verdict.  `bench/CARD.md` is the benchmark card;
`code/tests/test_bench.py` tests the harness itself.

## Honest limitations (also in the paper)

* Most vision experiments use the last-layer regime; the converged full-network
  replication is reported separately and its magnitudes are smaller.
* Cardinality-estimation workloads are generated locally, not DMV/CEBench.
* Three to ten trajectories per cell; no operator is claimed to be exactly idempotent.
* The protocol is single-node; a distributed model store would need an outbox/2PC
  publish.
* The replay paths are flat in |H|; the path that commits *new* records is not
  (`code/probe_insert_scale.py`: 12 µs/record at |H|=2·10^4 rising to ~240 µs at
  2·10^5).  The contract bounds execution, not our metadata store.
* Every timing in this artifact was measured on one shared laptop, so absolute levels
  move with background load.  The paper therefore reports the trend within a run and
  the spread across three universes (`code/analyze_seeds.py`), not a single number.
