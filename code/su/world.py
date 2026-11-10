"""A 'world' bundles a trained base model, its exact retraining oracle, the
cached encoder features and the record-identity map, so that individual
experiment scripts stay short and share one artefact set.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from . import config as C
from . import metrics as MET
from . import platform as P
from .backends import TorchBackend, TorchBackendConfig, load_rid_map, save_rid_map, build_rid_map


def request_indices(n: int, ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(round(ratio * n))
    return np.sort(rng.choice(n, size=k, replace=False))


@dataclass
class World:
    dataset: str = "cifar10"
    arch: str = "mlp"
    seed: int = 0
    request_ratio: float = C.REQUEST_RATIO
    request_seed: int = C.REQUEST_SEED
    Ftr: torch.Tensor = None
    ytr: torch.Tensor = None
    Fte: torch.Tensor = None
    yte: torch.Tensor = None
    idx2rid: list = field(default_factory=list)
    rid2idx: dict = field(default_factory=dict)
    base_path: str = ""
    retrain_path: str = ""
    n_classes: int = 10
    dim: int = 512
    _cache: dict = field(default_factory=dict)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def build(cls, dataset="cifar10", arch="mlp", seed=0, ratio=C.REQUEST_RATIO,
              request_seed=C.REQUEST_SEED) -> "World":
        ds = P.load_image_dataset(dataset)
        Ftr, Fte = P.extract_features(ds, tag=C.FEATURE_TAG)
        tag = f"{dataset}_{arch}_s{seed}"
        base_path = os.path.join(P.CKPT, f"{tag}_base.pt")
        retrain_path = os.path.join(P.CKPT, f"{tag}_retrainA.pt")
        if not os.path.exists(base_path):
            base = P.train_head(Ftr, ds.ytr, arch, ds.n_classes, epochs=C.BASE_EPOCHS,
                                lr=C.BASE_LR, weight_decay=C.BASE_WD, seed=seed, verbose=True)
            P.save_state(base, f"{tag}_base")
        if not os.path.exists(retrain_path):
            A = request_indices(len(Ftr), ratio, request_seed)
            keep = np.ones(len(Ftr), dtype=bool)
            keep[A] = False
            R = np.where(keep)[0]
            retr = P.train_head(Ftr[R], ds.ytr[R], arch, ds.n_classes, epochs=C.BASE_EPOCHS,
                                lr=C.BASE_LR, weight_decay=C.BASE_WD, seed=seed, verbose=True)
            P.save_state(retr, f"{tag}_retrainA")
        # record identities
        if os.path.exists(os.path.join(P.ARTIFACTS, f"ridmap_{dataset}.npz")):
            idx2rid, rid2idx = load_rid_map(dataset)
        else:
            idx2rid, rid2idx = build_rid_map(dataset, ds.Xtr)
            save_rid_map(dataset, idx2rid)
        return cls(dataset=dataset, arch=arch, seed=seed, request_ratio=ratio,
                   request_seed=request_seed, Ftr=Ftr, ytr=ds.ytr, Fte=Fte, yte=ds.yte,
                   idx2rid=idx2rid, rid2idx=rid2idx, base_path=base_path,
                   retrain_path=retrain_path, n_classes=ds.n_classes)

    # -- models ------------------------------------------------------------ #
    def model(self, path: str):
        m = P.make_head(self.arch, self.dim, self.n_classes)
        m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        m.eval()
        return m

    def head(self) -> torch.nn.Module:
        return self.model(self.base_path)

    def oracle(self) -> torch.nn.Module:
        return self.model(self.retrain_path)

    def oracle_for(self, keep_idx: np.ndarray, tag: str) -> str:
        """Exact retraining reference on an arbitrary retained subset (cached)."""
        import hashlib
        key = hashlib.sha256(np.sort(np.asarray(keep_idx)).tobytes()).hexdigest()[:12]
        name = f"{self.dataset}_{self.arch}_s{self.seed}_oracle_{tag}_{key}"
        path = os.path.join(P.CKPT, name + ".pt")
        if not os.path.exists(path):
            m = P.train_head(self.Ftr[keep_idx], self.ytr[keep_idx], self.arch,
                             self.n_classes, epochs=C.BASE_EPOCHS, lr=C.BASE_LR,
                             weight_decay=C.BASE_WD, seed=self.seed, verbose=False)
            P.save_state(m, name)
        return path

    def probs(self, path: str) -> np.ndarray:
        key = ("probs", path)
        if key not in self._cache:
            self._cache[key] = MET.probs_of(self.model(path), self.Fte)
        return self._cache[key]

    def probs_model(self, m) -> np.ndarray:
        return MET.probs_of(m, self.Fte)

    # -- data -------------------------------------------------------------- #
    @property
    def n(self) -> int:
        return len(self.Ftr)

    def request_rids(self, ratio=None, seed=None, kind="random", param=None) -> list[str]:
        ratio = self.request_ratio if ratio is None else ratio
        seed = self.request_seed if seed is None else seed
        if kind == "random":
            idx = request_indices(self.n, ratio, seed)
        elif kind == "class":
            # concentrate the request on `param` classes
            lab = self.ytr.numpy()
            classes = list(param)
            cand = np.where(np.isin(lab, classes))[0]
            k = int(round(ratio * self.n))
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(cand, size=min(k, len(cand)), replace=False))
        elif kind == "explicit":
            idx = np.asarray(param)
        else:
            raise ValueError(kind)
        return [self.idx2rid[i] for i in idx]

    def rids_of(self, idx) -> list[str]:
        return [self.idx2rid[int(i)] for i in idx]

    def indices_of(self, rids) -> np.ndarray:
        return np.array(sorted(self.rid2idx[r] for r in rids), dtype=np.int64)

    # -- backend ----------------------------------------------------------- #
    def backend(self, method="neggradplus", epochs=C.UE_EPOCHS, lr=C.UE_LR,
                beta=C.UE_BETA, wd=C.UE_WD, l1=0.0, sigma=0.0, verify=False,
                artifact_subdir=None, forget_mode="matched") -> TorchBackend:
        cfg = TorchBackendConfig(dataset=self.dataset, arch=self.arch,
                                 train_seed=self.seed, lr=lr, wd=wd, beta=beta,
                                 l1_lambda=l1, noise_sigma=sigma, verify=verify,
                                 forget_mode=forget_mode)
        sub = artifact_subdir or f"{self.dataset}_{self.arch}_s{self.seed}/{method}"
        be = TorchBackend(cfg, os.path.join(P.ARTIFACTS, "versions", sub),
                          self.rid2idx, self.base_path)
        be.attach_data(self.Ftr, self.ytr, self.Fte, self.yte)
        return be

    # -- evaluation -------------------------------------------------------- #
    def evaluate(self, model_or_path) -> dict:
        m = self.model(model_or_path) if isinstance(model_or_path, str) else model_or_path
        p = MET.probs_of(m, self.Fte)
        pb = self.probs(self.base_path)
        pr = self.probs(self.retrain_path)
        from .metrics import js_divergence
        return {
            "test_acc": MET.accuracy_of_probs(p, self.yte),
            "js_to_base": js_divergence(p, pb),
            "js_to_retrainA": js_divergence(p, pr),
        }
