from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold

from .exceptions import CATEValidationError
from .validation import validate_fit_data


PredictionMethod = Literal["predict", "predict_proba"]


@dataclass(frozen=True)
class FoldRecord:
    fold_index: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    train_treatment_counts: dict[int, int]
    test_treatment_counts: dict[int, int]


@dataclass(frozen=True)
class FallbackRecord:
    fallback_used: bool
    fallback_reason: str
    fallback_component: str
    fold_index: int | None = None
    treatment_arm: int | None = None


@dataclass(frozen=True)
class CrossFitResult:
    predictions: NDArray[np.float64]
    effective_fold_count: int
    folds: tuple[FoldRecord, ...]
    training_ids_by_row: tuple[frozenset[int], ...]
    fallback_records: tuple[FallbackRecord, ...] = ()


@dataclass(frozen=True)
class ArmOutcomeCrossFitResult:
    mu0: NDArray[np.float64]
    mu1: NDArray[np.float64]
    effective_fold_count: int
    folds: tuple[FoldRecord, ...]
    fallback_records: tuple[FallbackRecord, ...] = ()


@dataclass(frozen=True)
class PropensityDiagnostics:
    min_raw_propensity: float
    max_raw_propensity: float
    clipping_fraction: float
    lower_bound: float
    upper_bound: float


def _treatment_counts(treatment: NDArray[np.int64]) -> dict[int, int]:
    values, counts = np.unique(treatment, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts)}


def _effective_fold_count(
    treatment: NDArray[np.int64],
    requested: int,
) -> int:
    if requested < 2:
        raise CATEValidationError("n_splits must be at least 2")
    smallest_arm = min(_treatment_counts(treatment).values())
    if smallest_arm < 2:
        raise CATEValidationError(
            "cross-fitting requires at least two observations in each treatment arm"
        )
    return min(requested, smallest_arm)


def _predict(
    estimator: Any,
    X: NDArray[np.float64],
    method: PredictionMethod,
) -> NDArray[np.float64]:
    if method == "predict_proba":
        probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        classes = np.asarray(estimator.classes_)
        if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
            raise CATEValidationError("predict_proba columns must match classes_")
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size == 1:
            prediction = probabilities[:, int(positive_columns[0])]
        elif classes.size == 1 and classes[0] == 0:
            prediction = np.zeros(len(X), dtype=np.float64)
        else:
            raise CATEValidationError("binary estimator classes_ must contain 0 or 1")
    else:
        prediction = np.asarray(estimator.predict(X), dtype=np.float64)
    if prediction.ndim == 2 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
        raise CATEValidationError(
            f"OOF prediction must be finite with shape ({len(X)},)"
        )
    return prediction


def fit_prediction_model(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    prediction_method: PredictionMethod,
    clone_estimator: bool = True,
    fallback_component: str = "prediction_model",
    fold_index: int | None = None,
    treatment_arm: int | None = None,
) -> Any:
    """Fit a clone, using the empirical constant for degenerate binary folds."""

    y_array = np.asarray(y)
    if prediction_method == "predict_proba":
        classes = np.unique(y_array)
        if classes.size == 1 and classes[0] in (0, 1):
            fitted = DummyClassifier(
                strategy="constant",
                constant=int(classes[0]),
            ).fit(X, y_array)
            fitted._causalarena_fallback_record = FallbackRecord(
                fallback_used=True,
                fallback_reason="single_class_outcome",
                fallback_component=fallback_component,
                fold_index=fold_index,
                treatment_arm=treatment_arm,
            )
            return fitted
    target = clone(estimator) if clone_estimator else estimator
    return target.fit(X, y_array)


def fitted_fallback_record(estimator: Any) -> FallbackRecord | None:
    record = getattr(estimator, "_causalarena_fallback_record", None)
    return record if isinstance(record, FallbackRecord) else None


def _fold_record(
    fold_index: int,
    train_indices: NDArray[np.int64],
    test_indices: NDArray[np.int64],
    treatment: NDArray[np.int64],
) -> FoldRecord:
    return FoldRecord(
        fold_index=fold_index,
        train_indices=tuple(int(index) for index in train_indices),
        test_indices=tuple(int(index) for index in test_indices),
        train_treatment_counts=_treatment_counts(treatment[train_indices]),
        test_treatment_counts=_treatment_counts(treatment[test_indices]),
    )


def cross_fit_predictions(
    estimator: Any,
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    treatment: NDArray[np.int64],
    n_splits: int = 5,
    random_state: int = 0,
    prediction_method: PredictionMethod = "predict",
    fallback_component: str = "prediction_model",
) -> CrossFitResult:
    """Generate leakage-safe OOF predictions from cloned estimators."""

    X_array, treatment_array, y_array = validate_fit_data(
        X,
        treatment,
        y,
        outcome_type="continuous",
    )
    effective = _effective_fold_count(treatment_array, n_splits)
    splitter = StratifiedKFold(
        n_splits=effective,
        shuffle=True,
        random_state=random_state,
    )
    predictions = np.empty(len(X_array), dtype=np.float64)
    folds: list[FoldRecord] = []
    fallback_records: list[FallbackRecord] = []
    training_ids: list[frozenset[int] | None] = [None] * len(X_array)
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(X_array, treatment_array)
    ):
        fitted = fit_prediction_model(
            estimator,
            X_array[train_indices],
            y_array[train_indices],
            prediction_method=prediction_method,
            fallback_component=fallback_component,
            fold_index=fold_index,
        )
        fallback = fitted_fallback_record(fitted)
        if fallback is not None:
            fallback_records.append(fallback)
        predictions[test_indices] = _predict(
            fitted,
            X_array[test_indices],
            prediction_method,
        )
        train_id_set = frozenset(int(index) for index in train_indices)
        for test_index in test_indices:
            training_ids[int(test_index)] = train_id_set
        folds.append(
            _fold_record(
                fold_index,
                train_indices,
                test_indices,
                treatment_array,
            )
        )
    if any(ids is None for ids in training_ids):
        raise RuntimeError("cross-fitting did not assign every observation")
    return CrossFitResult(
        predictions=predictions,
        effective_fold_count=effective,
        folds=tuple(folds),
        training_ids_by_row=tuple(ids for ids in training_ids if ids is not None),
        fallback_records=tuple(fallback_records),
    )


def cross_fit_arm_outcomes(
    estimator: Any,
    X: NDArray[np.float64],
    treatment: NDArray[np.int64],
    y: NDArray[np.float64],
    *,
    n_splits: int = 5,
    random_state: int = 0,
    prediction_method: PredictionMethod = "predict",
) -> ArmOutcomeCrossFitResult:
    """Generate OOF potential-outcome nuisance predictions for both arms."""

    X_array, treatment_array, y_array = validate_fit_data(
        X,
        treatment,
        y,
        outcome_type="continuous",
    )
    effective = _effective_fold_count(treatment_array, n_splits)
    splitter = StratifiedKFold(
        n_splits=effective,
        shuffle=True,
        random_state=random_state,
    )
    mu0 = np.empty(len(X_array), dtype=np.float64)
    mu1 = np.empty(len(X_array), dtype=np.float64)
    folds: list[FoldRecord] = []
    fallback_records: list[FallbackRecord] = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(X_array, treatment_array)
    ):
        for arm, destination in ((0, mu0), (1, mu1)):
            arm_train = train_indices[treatment_array[train_indices] == arm]
            fitted = fit_prediction_model(
                estimator,
                X_array[arm_train],
                y_array[arm_train],
                prediction_method=prediction_method,
                fallback_component="outcome",
                fold_index=fold_index,
                treatment_arm=arm,
            )
            fallback = fitted_fallback_record(fitted)
            if fallback is not None:
                fallback_records.append(fallback)
            destination[test_indices] = _predict(
                fitted,
                X_array[test_indices],
                prediction_method,
            )
        folds.append(
            _fold_record(
                fold_index,
                train_indices,
                test_indices,
                treatment_array,
            )
        )
    return ArmOutcomeCrossFitResult(
        mu0=mu0,
        mu1=mu1,
        effective_fold_count=effective,
        folds=tuple(folds),
        fallback_records=tuple(fallback_records),
    )


def clip_propensity(
    raw_propensity: NDArray[np.float64],
    *,
    lower: float = 0.01,
    upper: float = 0.99,
) -> tuple[NDArray[np.float64], PropensityDiagnostics]:
    raw = np.asarray(raw_propensity, dtype=np.float64)
    if raw.ndim != 1 or raw.size == 0 or not np.isfinite(raw).all():
        raise CATEValidationError("raw propensity must be a nonempty finite vector")
    if not 0.0 < lower < upper < 1.0:
        raise CATEValidationError("propensity bounds must satisfy 0 < lower < upper < 1")
    clipped = np.clip(raw, lower, upper)
    diagnostics = PropensityDiagnostics(
        min_raw_propensity=float(np.min(raw)),
        max_raw_propensity=float(np.max(raw)),
        clipping_fraction=float(np.mean(clipped != raw)),
        lower_bound=lower,
        upper_bound=upper,
    )
    return clipped, diagnostics


def _pseudo_outcome_inputs(
    treatment: NDArray[np.int64],
    y: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    treatment_array = np.asarray(treatment, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    propensity_array = np.asarray(propensity, dtype=np.float64)
    arrays = (treatment_array, y_array, propensity_array)
    if any(array.ndim != 1 for array in arrays):
        raise CATEValidationError("pseudo-outcome inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise CATEValidationError("pseudo-outcome inputs must have equal length")
    if any(not np.isfinite(array).all() for array in arrays):
        raise CATEValidationError("pseudo-outcome inputs must be finite")
    if not np.isin(treatment_array, [0.0, 1.0]).all():
        raise CATEValidationError("treatment may contain only 0 and 1")
    if not ((propensity_array > 0.0) & (propensity_array < 1.0)).all():
        raise CATEValidationError("propensity must be strictly between 0 and 1")
    return treatment_array, y_array, propensity_array


def pw_pseudo_outcome(
    treatment: NDArray[np.int64],
    y: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> NDArray[np.float64]:
    treatment_array, y_array, propensity_array = _pseudo_outcome_inputs(
        treatment,
        y,
        propensity,
    )
    return (
        treatment_array * y_array / propensity_array
        - (1.0 - treatment_array) * y_array / (1.0 - propensity_array)
    )


def dr_pseudo_outcome(
    treatment: NDArray[np.int64],
    y: NDArray[np.float64],
    propensity: NDArray[np.float64],
    mu0: NDArray[np.float64],
    mu1: NDArray[np.float64],
) -> NDArray[np.float64]:
    treatment_array, y_array, propensity_array = _pseudo_outcome_inputs(
        treatment,
        y,
        propensity,
    )
    mu0_array = np.asarray(mu0, dtype=np.float64)
    mu1_array = np.asarray(mu1, dtype=np.float64)
    for name, value in (("mu0", mu0_array), ("mu1", mu1_array)):
        if value.shape != y_array.shape or not np.isfinite(value).all():
            raise CATEValidationError(
                f"{name} must be finite with shape {y_array.shape}"
            )
    return (
        mu1_array
        - mu0_array
        + treatment_array * (y_array - mu1_array) / propensity_array
        - (1.0 - treatment_array)
        * (y_array - mu0_array)
        / (1.0 - propensity_array)
    )
