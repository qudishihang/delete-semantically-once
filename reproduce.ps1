# Reproduce every table and claim in the paper.
#
#   pwsh -File reproduce.ps1            # tests + the cheap tables
#   pwsh -File reproduce.ps1 -Heavy     # adds the GPU-free but slow sweeps (hours)
#
# Every step writes a CSV under results/; the paper reads only those files.

param([switch]$Heavy)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "code"

Write-Host "== unit tests (19 protocol + 6 harness) =="
python -m pytest code/tests -q

Write-Host "== benchmark harness: 11 workloads x 6 services =="
python -m bench.cli --suite --out results/bench_suite.csv

Write-Host "== table IV (concurrency) from a fresh 3-repetition run =="
python code/e3_concurrency.py
python code/make_table_conc.py

Write-Host "== table III (full recomputation reference) =="
python code/e14_recompute.py --pattern repeat --requests 5 --core 2000 --ue-epochs 10 --out e14_repeat.csv
python code/e14_recompute.py --pattern grow   --requests 5 --core 2000 --chunk 400 --ue-epochs 10 --out e14_grow.csv

Write-Host "== multi-seed replication of the adversarial and scale benchmarks =="
python code/e13_adversarial.py --seeds 0,1,2 --out e13_seeds.csv
python code/e12_scale.py --seeds 0,1,2 --out e12_seeds.csv
python code/analyze_seeds.py | Tee-Object -FilePath results/seed_summary.txt

Write-Host "== batching vs effectiveness (E15) =="
python code/e15_batching_quality.py --requests 32 --size 200 --out e15_batching_quality.csv

if ($Heavy) {
    Write-Host "== heavy: platform, statistics, real backend, database, scale =="
    python code/e0_platform.py --datasets cifar10 --archs mlp --seeds 0,1,2
    python code/e7_stats.py --runs 10
    python code/e1_semantics.py
    python code/e4_recovery.py
    python code/e10_concurrency_real.py
    python code/e5_dbtask.py --epochs 3 --lr 0.0001 --clip 1.0
    python code/e11_dbrows.py --tenants 0,1,2 --replays 5 --epochs 10 --lr 0.002
    python code/e6_sensitivity.py --runs 3
    python code/e9_fullnet.py --base-epochs 80 --kmax 3 --with-oracle
}

Write-Host "== rebuild the paper =="
Push-Location paper
pdflatex -interaction=nonstopmode main.tex | Out-Null
bibtex main | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null
Pop-Location
Write-Host "done: paper/main.pdf"
