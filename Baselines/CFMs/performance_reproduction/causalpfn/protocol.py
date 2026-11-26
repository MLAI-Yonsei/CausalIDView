"""Literal CausalPFN Table 1 task evaluation protocol."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import math
import random
import sys
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from ..common.schemas import TaskSpec
from .data import VENDOR_ROOT


OFFICIAL_SEED = 82718


@dataclass(frozen=True)
class CausalPFNTaskResult:
    task_id: str
    pehe: float
    ate_relative_error: float
    cate_prediction: np.ndarray
    true_cate: np.ndarray
    ate_prediction: float
    true_ate: float
    runtime_seconds: float

    @property
    def cate_pred(self) -> np.ndarray:
        """Compatibility name for the persisted CATE prediction."""

        return self.cate_prediction


@dataclass(frozen=True)
class ParallelEquivalenceReport:
    passed: bool
    sequential_hashes: dict[str, str]
    isolated_hashes: dict[str, str]

    def __getitem__(self, key: str) -> Any:
        return {"seed": OFFICIAL_SEED, "passed": self.passed,
                "sequential_hashes": self.sequential_hashes,
                "isolated_hashes": self.isolated_hashes}[key]


def set_official_seed(seed: int = OFFICIAL_SEED) -> None:
    """Apply the seed used by the official notebook to available RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_path(root: str | Path) -> Path:
    """Locate the single official checkpoint within the vendor cache."""

    vendor_root = Path(root).resolve()
    direct = vendor_root / "causalpfn_v0.pt"
    candidates = [direct] if direct.is_file() else []
    candidates.extend((vendor_root / ".cache").glob("**/causalpfn_v0.pt"))
    candidates = sorted(path.resolve() for path in candidates if path.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError("expected exactly one official causalpfn_v0.pt checkpoint")
    return candidates[0]


def estimate_causalpfn_cate_arrays(
    X_support: np.ndarray,
    treatment_support: np.ndarray,
    outcome_support: np.ndarray,
    X_query: np.ndarray,
    *,
    model_root: str | Path,
    device: str = "cuda:0",
    seed: int = OFFICIAL_SEED,
    estimator_factory: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Run the released CausalPFN CATE API on explicit support/query arrays."""

    if device != "cuda:0":
        raise ValueError("failure-search CausalPFN must use logical cuda:0")
    if seed != OFFICIAL_SEED:
        raise ValueError(f"CausalPFN seed must be the official value {OFFICIAL_SEED}")
    X_fit, treatment, outcome, query = _validate_cate_arrays(
        X_support, treatment_support, outcome_support, X_query
    )
    checkpoint = checkpoint_path(model_root)
    set_official_seed(seed)
    if estimator_factory is not None:
        estimator_context = nullcontext(estimator_factory)
    else:
        estimator_context = _official_cate_estimator(Path(model_root))
    with estimator_context as cate_type:
        estimator = cate_type(device=device, model_path=str(checkpoint))
        estimator.fit(X_fit, treatment, outcome)
        prediction = _finite_array(
            estimator.estimate_cate(X=query), "CATE prediction"
        )
    if prediction.shape != (len(query),):
        raise ValueError("CausalPFN CATE prediction must be a query-aligned vector")
    return prediction


@contextmanager
def _official_cate_estimator(root: Path) -> Iterator[type[Any]]:
    with _official_estimators(root) as (cate_type, _):
        yield cate_type


def _validate_cate_arrays(
    X_support: np.ndarray,
    treatment_support: np.ndarray,
    outcome_support: np.ndarray,
    X_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            "CausalPFN support/query covariates must be feature-aligned matrices"
        )
    if treatment.shape != (len(X_fit),) or outcome.shape != (len(X_fit),):
        raise ValueError("CausalPFN treatment/outcome must align with support rows")
    if set(np.unique(treatment)) != {0.0, 1.0}:
        raise ValueError("CausalPFN support treatment must contain both binary arms")
    if not all(
        np.isfinite(value).all()
        for value in (X_fit, treatment, outcome, query)
    ):
        raise ValueError("CausalPFN support/query arrays must be finite")
    return X_fit, treatment, outcome, query


def relative_ate_error(predicted: float, true: float) -> float:
    """Compute the official finite relative ATE error."""

    if not math.isfinite(predicted) or not math.isfinite(true):
        raise ValueError("ATE prediction and true ATE must be finite")
    if true == 0.0:
        raise ValueError("true ATE is zero")
    return abs(predicted - true) / abs(true)


def evaluate_causalpfn_task(
    task: TaskSpec,
    catalog: Any,
    device: str,
    checkpoint: str | Path,
    *,
    estimator_factory: Callable[..., Any] | None = None,
) -> CausalPFNTaskResult:
    """Evaluate one realization using fresh official CATE and ATE estimators."""

    if task.model != "causalpfn":
        raise ValueError(f"expected a CausalPFN task, got {task.model!r}")
    if estimator_factory is not None:
        if len(inspect.signature(estimator_factory).parameters) == 0:
            cate_type, ate_type = estimator_factory()
            return _evaluate(task, catalog, device, Path(checkpoint), cate_type, ate_type)
        return _evaluate_with_factory(task, catalog, device, Path(checkpoint), estimator_factory)
    with _official_estimators() as (cate_type, ate_type):
        return _evaluate(task, catalog, device, Path(checkpoint), cate_type, ate_type)


def _evaluate(
    task: TaskSpec,
    catalog: Any,
    device: str,
    checkpoint: Path,
    cate_type: type[Any],
    ate_type: type[Any],
) -> CausalPFNTaskResult:
    cate_data, ate_data = catalog[task.index]
    started = perf_counter()
    cate_estimator = cate_type(device=device, model_path=str(checkpoint))
    cate_estimator.fit(cate_data.X_train, cate_data.t_train, cate_data.y_train)
    cate_prediction = _finite_array(cate_estimator.estimate_cate(X=cate_data.X_test), "CATE prediction")
    true_cate = _finite_array(cate_data.true_cate, "true CATE")
    if cate_prediction.shape != true_cate.shape or cate_prediction.size == 0:
        raise ValueError("CATE prediction and true CATE must be finite, non-empty, and shape-aligned")
    ate_estimator = ate_type(device=device, model_path=str(checkpoint))
    ate_estimator.fit(ate_data.X, ate_data.t, ate_data.y)
    ate_prediction = _finite_scalar(ate_estimator.estimate_ate(), "ATE prediction")
    true_ate = _finite_scalar(ate_data.true_ate, "true ATE")
    pehe = math.sqrt(float(np.mean((cate_prediction - true_cate) ** 2)))
    return CausalPFNTaskResult(
        task_id=task.task_id,
        pehe=pehe,
        ate_relative_error=relative_ate_error(ate_prediction, true_ate),
        cate_prediction=cate_prediction,
        true_cate=true_cate,
        ate_prediction=ate_prediction,
        true_ate=true_ate,
        runtime_seconds=perf_counter() - started,
    )


def _evaluate_with_factory(
    task: TaskSpec, catalog: Any, device: str, checkpoint: Path,
    estimator_factory: Callable[[str, str, str], Any],
) -> CausalPFNTaskResult:
    """Adapt the test-only direct factory to the same literal call contract."""

    cate_type, ate_type = type("CATE", (), {}), type("ATE", (), {})
    cate_type.__new__ = staticmethod(lambda cls, **_: estimator_factory("cate", device, str(checkpoint)))
    ate_type.__new__ = staticmethod(lambda cls, **_: estimator_factory("ate", device, str(checkpoint)))
    return _evaluate(task, catalog, device, checkpoint, cate_type, ate_type)


def verify_parallel_equivalence(
    tasks: Sequence[TaskSpec],
    catalog: Any,
    device: str,
    checkpoint: str | Path,
    *,
    estimator_factory: Callable[..., Any] | None = None,
) -> ParallelEquivalenceReport:
    """Prove separately seeded tasks have bit-identical CATE and ATE outputs."""

    set_official_seed()
    if len(tasks) != 2:
        raise ValueError("parallel equivalence requires exactly two tasks")
    sequential = [
        evaluate_causalpfn_task(task, catalog, device, checkpoint, estimator_factory=estimator_factory)
        for task in tasks
    ]
    isolated = []
    for task in tasks:
        set_official_seed()
        isolated.append(evaluate_causalpfn_task(task, catalog, device, checkpoint, estimator_factory=estimator_factory))
    sequential_hashes = _prediction_hashes(sequential)
    isolated_hashes = _prediction_hashes(isolated)
    report = ParallelEquivalenceReport(
        passed=sequential_hashes == isolated_hashes,
        sequential_hashes=sequential_hashes,
        isolated_hashes=isolated_hashes,
    )
    return report


def _finite_array(value: Any, label: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite and non-empty")
    return np.array(array, dtype=np.float64, copy=True)


def _finite_scalar(value: Any, label: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size != 1 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite scalar")
    return float(array.reshape(-1)[0])


def _prediction_hashes(results: list[CausalPFNTaskResult]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for result in results:
        payload = np.concatenate((result.cate_prediction.ravel(), np.array([result.ate_prediction], dtype=np.float64)))
        hashes[result.task_id] = hashlib.sha256(payload.tobytes()).hexdigest()
    return hashes


@contextmanager
def _official_estimators(root: Path = VENDOR_ROOT) -> Iterator[tuple[type[Any], type[Any]]]:
    """Load official estimator classes without leaking vendor modules globally."""

    source_root = root.resolve() / "src"
    if not source_root.is_dir():
        raise FileNotFoundError(f"missing official CausalPFN source: {source_root}")
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module("causalpfn")
        yield module.CATEEstimator, module.ATEEstimator
    finally:
        for name in tuple(sys.modules):
            if name == "causalpfn" or name.startswith("causalpfn."):
                sys.modules.pop(name, None)
        try:
            sys.path.remove(str(source_root))
        except ValueError:
            pass
