from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


BACKEND_DEFAULTS = {
    "logistic": {"max_iter": 2000, "C": 1.0},
    "xgboost": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8, "n_jobs": 1},
    "tabpfn_v35": {"n_estimators": 8, "version": "v3.5"},
}


@dataclass(frozen=True)
class ProbabilityPrediction:
    probabilities: np.ndarray
    classes: tuple[int, ...]
    metadata: dict[str, Any]


def fit_predict_joint(backend: str, X, labels, X_query, seed: int, device: str) -> ProbabilityPrediction:
    X, labels, X_query = np.asarray(X), np.asarray(labels), np.asarray(X_query)
    if tuple(np.unique(labels)) != (0, 1, 2, 3):
        raise ValueError("joint probability backend requires all four joint classes")
    if backend == "logistic":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**BACKEND_DEFAULTS[backend], random_state=seed).fit(X, labels)
    elif backend == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(**BACKEND_DEFAULTS[backend], random_state=seed, objective="multi:softprob",
                              num_class=4, eval_metric="mlogloss").fit(X, labels)
    elif backend == "tabpfn_v35":
        from ..tabpfn.backend import OfficialTabPFNBackend
        factory = OfficialTabPFNBackend({"tabpfn": {"model_version": "v3.5",
            "checkpoint_revision": "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
            "n_estimators": 8,
            "n_preprocessing_jobs": 1, "device": device, "ignore_pretraining_limits": True}})
        model = factory.create_classifier(random_state=seed, role="partial-id-joint").fit(X, labels)
    else:
        raise ValueError(f"unknown backend: {backend}")
    probability = np.asarray(model.predict_proba(X_query), dtype=float)
    classes = tuple(int(v) for v in model.classes_)
    if classes != (0, 1, 2, 3):
        raise RuntimeError(f"unexpected class ordering: {classes}")
    error = float(np.max(np.abs(probability.sum(1) - 1)))
    if error > 1e-6 or not np.isfinite(probability).all() or (probability < 0).any():
        raise RuntimeError("invalid probability output")
    probability /= probability.sum(1, keepdims=True)
    return ProbabilityPrediction(probability, classes, {"backend": backend, "seed": seed, "device": device,
                                                         "pre_normalization_max_error": error})
