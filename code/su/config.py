"""Shared experiment configuration."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
ARTIFACTS = os.path.join(ROOT, "artifacts")
FIGURES = os.path.join(ROOT, "figures")
LOGS = os.path.join(ROOT, "logs")
for _d in (RESULTS, ARTIFACTS, FIGURES, LOGS):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------- #
# platform
# --------------------------------------------------------------------------- #
FEATURE_TAG = "resnet18_128"
PRIMARY_DATASET = "cifar10"
SECOND_DATASET = "cifar100"
ARCHS = ("mlp", "linear")
TRAIN_SEEDS = (0, 1, 2)

BASE_EPOCHS = 20
BASE_LR = 0.05
BASE_WD = 5e-4
BATCH = 256

# default deletion request: 10% of the training pool
REQUEST_RATIO = 0.10
REQUEST_SEED = 1234

# default unlearning invocation
UE_EPOCHS = 10
UE_LR = 0.01
UE_WD = 5e-4
UE_BETA = 0.95

METHODS = ("finetune", "neggradplus", "salun", "finetune_l1", "finetune_noisy")

NT = os.environ.get("SU_THREADS", "10")
