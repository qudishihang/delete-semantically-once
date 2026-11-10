"""su — Semantic Unlearning service toolkit.

Sub-modules
-----------
platform  : data + frozen-encoder feature extraction, base model training, artifact store
methods   : approximate unlearning operators (finetune, NegGrad+, regularized, noisy, SalUn-lite)
metrics   : JS divergence, accuracy, MIA advantage, Q-error
identity  : canonical record identity, request normalisation, effect hashing
stats     : leave-one-run-out DAR, paired drift, clustered bootstrap, permutation test
service   : transactional, semantically idempotent deletion service (SQLite + WAL)
planner   : delta planner / cost-aware action selection
backends  : pluggable unlearning executors used by the service benchmarks
"""

__version__ = "0.1.0"
