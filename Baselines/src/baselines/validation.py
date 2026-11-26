from __future__ import annotations

from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from .exceptions import CATEValidationError


OutcomeType = Literal["continuous", "binary"]


def _numeric_array(value: object, *, name: str) -> NDArray[np.float64]:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CATEValidationError(f"{name} must be numeric") from error


def validate_fit_data(
    X: object,
    treatment: object,
    y: object,
    *,
    outcome_type: OutcomeType,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64]]:
    """Validate and normalize shared estimator fit inputs."""

    if outcome_type not in {"continuous", "binary"}:
        raise CATEValidationError(
            "outcome_type must be either 'continuous' or 'binary'"
        )

    X_array = _numeric_array(X, name="X")
    if X_array.ndim != 2:
        raise CATEValidationError("X must be two-dimensional")
    if not np.isfinite(X_array).all():
        raise CATEValidationError("X must contain only finite values")

    treatment_array = _numeric_array(treatment, name="treatment")
    if treatment_array.ndim != 1:
        raise CATEValidationError("treatment must be one-dimensional")
    if not np.isfinite(treatment_array).all():
        raise CATEValidationError("treatment must contain only finite values")
    if not np.isin(treatment_array, [0.0, 1.0]).all():
        raise CATEValidationError("treatment may contain only 0 and 1")
    if set(np.unique(treatment_array)) != {0.0, 1.0}:
        raise CATEValidationError("treatment must contain both 0 and 1")

    y_array = _numeric_array(y, name="y")
    if y_array.ndim != 1:
        raise CATEValidationError("y must be one-dimensional")
    if not np.isfinite(y_array).all():
        raise CATEValidationError("y must contain only finite values")

    if not (len(X_array) == len(treatment_array) == len(y_array)):
        raise CATEValidationError("X, treatment, and y must have the same number of rows")

    return (
        X_array,
        treatment_array.astype(np.int64, copy=False),
        y_array,
    )


def validate_predict_X(
    X: object,
    *,
    alpha: float = 0.05,
) -> NDArray[np.float64]:
    """Validate and normalize shared estimator prediction inputs."""

    X_array = _numeric_array(X, name="X")
    if X_array.ndim != 2:
        raise CATEValidationError("X must be two-dimensional")
    if not np.isfinite(X_array).all():
        raise CATEValidationError("X must contain only finite values")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise CATEValidationError("alpha must be strictly between 0 and 1")
    return cast(NDArray[np.float64], X_array)

