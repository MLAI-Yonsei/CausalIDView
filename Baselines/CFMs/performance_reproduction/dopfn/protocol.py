"""Literal Do-PFN split and CID/CATE evaluation protocol."""

from __future__ import annotations

import importlib
import os
import random
import sys
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

import numpy as np

from ..common.schemas import TaskSpec
from .data import DOPFN_ROOT


OFFICIAL_SEED = 82718


@dataclass(frozen=True)
class DoPFNTaskResult:
    task_id: str
    cid_nmse: float
    cate_nmse: float
    pred_interventional_y: np.ndarray
    true_interventional_y: np.ndarray
    pred_cate: np.ndarray
    true_cate: np.ndarray
    runtime_seconds: float


def official_split(dataset: Any, *, split_number: int = 1, n_splits: int = 5) -> tuple[Any, Any]:
    """Use the release's legacy one-based split numbering."""

    if split_number < 1 or split_number > n_splits:
        raise ValueError("Do-PFN split_number must be one-based and within n_splits")
    train, test = dataset.generate_valid_split(split_number=split_number, n_splits=n_splits)
    if train is None or test is None:
        raise RuntimeError("Do-PFN could not generate the requested valid split")
    return train, test


def normalized_mse(y_true: Any, y_pred: Any) -> float:
    """Match the official normalized MSE: both values use target min/max."""

    target = _as_array(y_true, "target")
    prediction = _as_array(y_pred, "prediction")
    if target.shape != prediction.shape or target.size == 0:
        raise ValueError("Do-PFN metric arrays must be non-empty and shape-aligned")
    low = float(np.min(target))
    scale = float(np.max(target) - low)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Do-PFN target range must be finite and positive")
    value = float(np.mean(((target - low) / scale - (prediction - low) / scale) ** 2))
    if not np.isfinite(value):
        raise ValueError("Do-PFN normalized MSE is non-finite")
    return value


def fit_dopfn(root: str | Path = DOPFN_ROOT, *, device: str = "cpu") -> Any:
    """Construct the bundled DoPFNRegressor under its release root."""

    release = Path(root).resolve()
    if not release.is_dir():
        raise FileNotFoundError(f"missing Do-PFN release root: {release}")
    with _release_context(release):
        # The arena worktree also has a top-level ``scripts`` package.  The
        # vendored release has no scripts/__init__.py, so put its scripts
        # directory ahead of the worktree before importing the official model.
        scripts_root = str(release / "scripts")
        sys.path.insert(0, scripts_root)
        package = types.ModuleType("scripts")
        package.__path__ = [scripts_root]  # type: ignore[attr-defined]
        sys.modules["scripts"] = package
        try:
            module = importlib.import_module("scripts.transformer_prediction_interface.base")
            model = module.DoPFNRegressor()
            model._dopfn_root = release
        finally:
            sys.path.remove(scripts_root)
    model.device = device
    return model


def set_official_seed(seed: int = OFFICIAL_SEED) -> None:
    """Seed the RNGs used by the released Do-PFN inference stack."""

    if seed != OFFICIAL_SEED:
        raise ValueError(f"Do-PFN seed must be the official value {OFFICIAL_SEED}")
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_dopfn_cate_arrays(
    model: Any,
    X_support: np.ndarray,
    treatment_support: np.ndarray,
    outcome_support: np.ndarray,
    X_query: np.ndarray,
    *,
    seed: int = OFFICIAL_SEED,
) -> np.ndarray:
    """Fit Do-PFN on treatment-first support and return query ``p1 - p0``."""

    set_official_seed(seed)
    X_fit = np.asarray(X_support, dtype=np.float64)
    treatment = np.asarray(treatment_support, dtype=np.float64)
    outcome = np.asarray(outcome_support, dtype=np.float64)
    query = np.asarray(X_query, dtype=np.float64)
    if (
        X_fit.ndim != 2
        or query.ndim != 2
        or X_fit.shape[1] != query.shape[1]
    ):
        raise ValueError(
            "Do-PFN support/query covariates must be feature-aligned matrices"
        )
    if treatment.shape != (len(X_fit),) or outcome.shape != (len(X_fit),):
        raise ValueError("Do-PFN treatment/outcome must align with support rows")
    if set(np.unique(treatment)) != {0.0, 1.0}:
        raise ValueError("Do-PFN support treatment must contain both binary arms")
    if not all(
        np.isfinite(value).all()
        for value in (X_fit, treatment, outcome, query)
    ):
        raise ValueError("Do-PFN support/query arrays must be finite")
    support = np.column_stack((treatment, X_fit))
    query_zero = np.column_stack((np.zeros(len(query), dtype=np.float64), query))
    query_one = np.column_stack((np.ones(len(query), dtype=np.float64), query))
    torch = importlib.import_module("torch")
    query_zero_tensor = torch.as_tensor(query_zero.copy(), dtype=torch.float32)
    query_one_tensor = torch.as_tensor(query_one.copy(), dtype=torch.float32)
    model_root = getattr(model, "_dopfn_root", None)
    context = (
        _release_context(Path(model_root))
        if model_root is not None
        else nullcontext()
    )
    with context:
        model.fit(support, outcome)
        p0 = _as_array(
            model.predict(query_zero_tensor), "control prediction"
        ).reshape(-1)
        p1 = _as_array(
            model.predict(query_one_tensor), "treated prediction"
        ).reshape(-1)
    if p0.shape != (len(query),) or p1.shape != (len(query),):
        raise ValueError("Do-PFN potential outcomes must be query-aligned vectors")
    return p1 - p0


def evaluate_dopfn_task(model: Any, dataset: Any, task: TaskSpec) -> DoPFNTaskResult:
    """Fit a fresh model on the official split and score CID plus CATE."""

    if task.model != "dopfn":
        raise ValueError(f"expected Do-PFN task, got {task.model!r}")
    started = perf_counter()
    model_root = getattr(model, "_dopfn_root", None)
    context = _release_context(Path(model_root)) if model_root is not None else nullcontext()
    with context:
        train, test = official_split(dataset)
        model.fit(train.x, train.y)
        test_x = _clone(test.x)
        cid_prediction = _as_array(model.predict_full(test_x)["mean"], "CID prediction").reshape(-1)
        true_interventional = _as_array(
            test.y_int if hasattr(test, "y_int") else test.y, "CID target"
        ).reshape(-1)
        cate_prediction = _as_array(model.predict_cate(_clone(test.x)), "CATE prediction").reshape(-1)
        true_cate = _as_array(test.cate, "CATE target").reshape(-1)
    if cid_prediction.shape != true_interventional.shape or cate_prediction.shape != true_cate.shape:
        raise ValueError("Do-PFN prediction and target shapes are misaligned")
    return DoPFNTaskResult(
        task_id=task.task_id,
        cid_nmse=normalized_mse(true_interventional, cid_prediction),
        cate_nmse=normalized_mse(true_cate, cate_prediction),
        pred_interventional_y=cid_prediction,
        true_interventional_y=true_interventional,
        pred_cate=cate_prediction,
        true_cate=true_cate,
        runtime_seconds=perf_counter() - started,
    )


@contextmanager
def _release_context(root: Path) -> Iterator[None]:
    previous_cwd = Path.cwd()
    root_text = str(root)
    sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass


def _clone(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    raise TypeError("Do-PFN prediction input must support clone/copy")


def _as_array(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"Do-PFN {label} must be finite and non-empty")
    return np.array(array, dtype=np.float64, copy=True)
