"""Backends: the unlearning executors plugged into the deletion service."""
from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from . import metrics as MET
from . import platform as P
from .identity import record_identity
from .methods import UnlearnConfig, example_cost, run as run_method
from .service import Backend, UnlearnOutcome, UnlearnTask


# --------------------------------------------------------------------------- #
# synthetic backend (system correctness experiments)
# --------------------------------------------------------------------------- #
class SyntheticBackend(Backend):
    """Deterministic stand-in executor for system-level tests.

    The artifact is a small file holding a monotonically increasing counter, so
    that version transitions and effect-once semantics can be audited without a
    model in the loop.  ``sleep_s`` models the cost of a real unlearning job.
    """

    name = "synthetic"
    # The synthetic executor only looks at the delta and the scope, so the service
    # does not have to hand it the whole committed processed set (an O(|H|) read per
    # job that would dominate a metadata-scale sweep).
    needs_processed = False

    def __init__(self, workdir: str, sleep_s: float = 0.05):
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        self.sleep_s = sleep_s
        self._init_path = os.path.join(workdir, "v0.artifact")
        self._write(self._init_path, 0)

    @staticmethod
    def _write(path: str, value: int, retries: int = 20):
        tmp = path + f".{os.getpid()}.{time.time_ns()}.tmp"
        with open(tmp, "w") as f:
            f.write(f"{value}\n")
            f.flush()
            os.fsync(f.fileno())
        for k in range(retries):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                time.sleep(0.002 * (k + 1))
        os.unlink(tmp)
        raise PermissionError(f"could not publish artifact {path}")

    def _read(self, path: str) -> int:
        with open(path) as f:
            return int(f.readline().strip())

    def initial_artifact(self) -> UnlearnOutcome:
        return UnlearnOutcome(self._init_path, "0", 0, 0.0)

    def unlearn(self, base_artifact: str, task: UnlearnTask) -> UnlearnOutcome:
        t0 = time.time()
        time.sleep(self.sleep_s)
        value = self._read(base_artifact) + 1
        tag = hashlib.sha256(("|".join(sorted(task.delta))).encode()).hexdigest()[:10]
        uniq = hashlib.sha256(os.urandom(16)).hexdigest()[:8]
        path = os.path.join(self.workdir, f"v{value}_{tag}_{uniq}.artifact")
        self._write(path, value)
        return UnlearnOutcome(path, f"{value}-{uniq}",
                              cost_examples=len(task.delta) * task.epochs,
                              wall_time_s=time.time() - t0)


# --------------------------------------------------------------------------- #
# torch backend (real unlearning)
# --------------------------------------------------------------------------- #
@dataclass
class TorchBackendConfig:
    dataset: str = "cifar10"
    arch: str = "mlp"
    train_seed: int = 0
    lr: float = 0.01
    wd: float = 5e-4
    beta: float = 0.95
    l1_lambda: float = 0.0
    noise_sigma: float = 0.0
    batch_size: int = 256
    grad_clip: float = 5.0
    verify: bool = False
    n_ref: int = 4000
    forget_mode: str = "matched"


class TorchBackend(Backend):
    """Last-layer unlearning over cached encoder features."""

    name = "torch"

    def __init__(self, cfg: TorchBackendConfig, artifact_dir: str, rid2idx: dict,
                 base_state: str):
        self.cfg = cfg
        self.model_tag = f"{cfg.dataset}_{cfg.arch}_s{cfg.train_seed}"
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.rid2idx = rid2idx
        self.base_state = base_state
        self.dim = 512
        self.n_classes = 10 if cfg.dataset == "cifar10" else 100
        self._base_digest = None

    def attach_data(self, Ftr, ytr, Fte, yte):
        self.Ftr, self.ytr, self.Fte, self.yte = Ftr, ytr, Fte, yte
        if self.cfg.n_ref:
            self.ref_idx = torch.arange(min(self.cfg.n_ref, len(Fte)))
        else:
            self.ref_idx = torch.arange(0)

    # -- model io ---------------------------------------------------------- #
    def _model(self, path: str):
        m = P.make_head(self.cfg.arch, self.dim, self.n_classes)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        return m

    def initial_artifact(self) -> UnlearnOutcome:
        m = self._model(self.base_state)
        dg = P.state_digest(m)
        self._base_digest = dg
        return UnlearnOutcome(self.base_state, dg, 0, 0.0)

    # -- execution --------------------------------------------------------- #
    def unlearn(self, base_artifact: str, task: UnlearnTask) -> UnlearnOutcome:
        t0 = time.time()
        cfg = UnlearnConfig(method=task.method, epochs=task.epochs, lr=self.cfg.lr,
                            weight_decay=self.cfg.wd, beta=self.cfg.beta,
                            l1_lambda=self.cfg.l1_lambda, noise_sigma=self.cfg.noise_sigma,
                            batch_size=self.cfg.batch_size, grad_clip=self.cfg.grad_clip,
                            seed=task.seed, forget_mode=self.cfg.forget_mode)
        base = self._model(base_artifact)
        forget_idx = torch.tensor(
            sorted({self.rid2idx[r] for r in task.scope if r in self.rid2idx}), dtype=torch.long)
        excluded = {self.rid2idx[r] for r in task.processed if r in self.rid2idx}
        excluded.update(forget_idx.tolist())
        keep = torch.ones(len(self.Ftr), dtype=torch.bool)
        if excluded:
            keep[torch.tensor(sorted(excluded), dtype=torch.long)] = False
        retain_idx = torch.where(keep)[0]
        model = run_method(base, self.Ftr, self.ytr, retain_idx, forget_idx, cfg)
        digest = P.state_digest(model)
        path = os.path.join(self.artifact_dir, f"{digest[:20]}.pt")
        if not os.path.exists(path):
            tmp = path + f".{os.getpid()}.{time.time_ns()}.tmp"
            torch.save(model.state_dict(), tmp)
            for k in range(20):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    time.sleep(0.002 * (k + 1))
            else:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        cost = example_cost(len(retain_idx), len(forget_idx), cfg)
        return UnlearnOutcome(path, digest, cost_examples=cost,
                              wall_time_s=time.time() - t0,
                              info={"n_forget": int(len(forget_idx)),
                                    "n_retain": int(len(retain_idx)),
                                    "n_delta": len(task.delta)})

    # -- effectiveness certificate ----------------------------------------- #
    def verify(self, artifact_path: str, rids: Sequence[str]) -> dict:
        if not self.cfg.verify:
            return {}
        idx = sorted({self.rid2idx[r] for r in rids if r in self.rid2idx})
        if not idx:
            return {}
        m = self._model(artifact_path)
        fidx = torch.tensor(idx, dtype=torch.long)
        ref = self.ref_idx
        mia = MET.mia_advantage(m, self.Ftr[fidx], self.ytr[fidx],
                                self.Fte[ref], self.yte[ref])
        acc_f = MET.accuracy(m, self.Ftr[fidx], self.ytr[fidx])
        return {"forget_acc": acc_f, "mia_auc": mia["mia_auc"],
                "mia_advantage": mia["mia_advantage"], "n": len(idx)}


# --------------------------------------------------------------------------- #
# cardinality-estimation backend (the database task)
# --------------------------------------------------------------------------- #
class CEBackend(Backend):
    """Backend for the learned cardinality estimator (regression)."""

    name = "ce"

    def __init__(self, artifact_dir: str, rid2idx: dict, base_state: str,
                 Xtr, ytr, Xte, yte, card_te, hidden: int = 256,
                 lr: float = 1e-3, wd: float = 1e-5, beta: float = 0.95,
                 batch_size: int = 256, grad_clip: float = 5.0):
        self.artifact_dir = artifact_dir
        os.makedirs(artifact_dir, exist_ok=True)
        self.rid2idx = rid2idx
        self.base_state = base_state
        self.Xtr, self.ytr = Xtr, ytr
        self.Xte, self.yte, self.card_te = Xte, yte, card_te
        self.hidden = hidden
        self.lr, self.wd, self.beta = lr, wd, beta
        self.batch_size, self.grad_clip = batch_size, grad_clip
        self.dim = Xtr.shape[1]

    def _model(self, path: str):
        import torch.nn as nn
        m = nn.Sequential(nn.Linear(self.dim, self.hidden), nn.ReLU(),
                          nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                          nn.Linear(self.hidden, 1))
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        return m

    def initial_artifact(self) -> UnlearnOutcome:
        return UnlearnOutcome(self.base_state, "ce-base", 0, 0.0)

    def unlearn(self, base_artifact: str, task: UnlearnTask) -> UnlearnOutcome:
        t0 = time.time()
        cfg = UnlearnConfig(method=task.method, epochs=task.epochs, lr=self.lr,
                            weight_decay=self.wd, beta=self.beta,
                            batch_size=self.batch_size, grad_clip=self.grad_clip,
                            seed=task.seed, loss="mse")
        base = self._model(base_artifact)
        forget_idx = torch.tensor(
            sorted({self.rid2idx[r] for r in task.scope if r in self.rid2idx}), dtype=torch.long)
        excluded = {self.rid2idx[r] for r in task.processed if r in self.rid2idx}
        excluded.update(forget_idx.tolist())
        keep = torch.ones(len(self.Xtr), dtype=torch.bool)
        if excluded:
            keep[torch.tensor(sorted(excluded), dtype=torch.long)] = False
        retain_idx = torch.where(keep)[0]
        model = run_method(base, self.Xtr, self.ytr, retain_idx, forget_idx, cfg)
        digest = P.state_digest(model)
        path = os.path.join(self.artifact_dir, f"{digest[:20]}.pt")
        if not os.path.exists(path):
            tmp = path + f".{os.getpid()}.{time.time_ns()}.tmp"
            torch.save(model.state_dict(), tmp)
            for k in range(20):
                try:
                    os.replace(tmp, path)
                    break
                except PermissionError:
                    time.sleep(0.002 * (k + 1))
        return UnlearnOutcome(path, digest,
                              cost_examples=example_cost(len(retain_idx), len(forget_idx), cfg),
                              wall_time_s=time.time() - t0,
                              info={"n_forget": int(len(forget_idx))})

    def qerror(self, artifact_path: str) -> dict:
        m = self._model(artifact_path)
        with torch.no_grad():
            pred = m(torch.from_numpy(self.Xte)).squeeze(-1).numpy()
        est = np.expm1(pred)
        return MET.q_error_stats(est, self.card_te)


# --------------------------------------------------------------------------- #
# helpers to build record identities
# --------------------------------------------------------------------------- #
def build_rid_map(dataset_name: str, Xtr: torch.Tensor, stride: int = 1) -> tuple[list, dict]:
    idx2rid, rid2idx = [], {}
    for i in range(0, len(Xtr)):
        h = hashlib.sha256(Xtr[i].numpy().tobytes()).hexdigest()
        rid = record_identity(dataset_name, f"train/{i}", h)
        idx2rid.append(rid)
        rid2idx[rid] = i
    return idx2rid, rid2idx


def rid_map_path(tag: str) -> str:
    return os.path.join(P.ARTIFACTS, f"ridmap_{tag}.npz")


def save_rid_map(tag: str, idx2rid: list):
    np.savez_compressed(rid_map_path(tag), idx2rid=np.array(idx2rid, dtype=object))


def load_rid_map(tag: str) -> tuple[list, dict]:
    z = np.load(rid_map_path(tag), allow_pickle=True)
    idx2rid = list(z["idx2rid"])
    return idx2rid, {r: i for i, r in enumerate(idx2rid)}
