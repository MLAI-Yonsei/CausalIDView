from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .exceptions import CATEValidationError


@dataclass(frozen=True)
class CATEPrediction:
    cate: NDArray[np.float64]
    mu0: NDArray[np.float64] | None = None
    mu1: NDArray[np.float64] | None = None
    cate_lower: NDArray[np.float64] | None = None
    cate_upper: NDArray[np.float64] | None = None
    metadata: Mapping[str, Any] | None = None


def _normalize_field(
    value: NDArray[np.float64] | None,
    *,
    field_name: str,
    n_rows: int,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CATEValidationError(f"{field_name} must be numeric") from error
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise CATEValidationError(
            f"{field_name} must have shape ({n_rows},), got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise CATEValidationError(f"{field_name} must contain only finite values")
    return array


def normalize_prediction(
    prediction: CATEPrediction,
    n_rows: int,
) -> CATEPrediction:
    """Return a prediction whose numeric fields satisfy the public contract."""

    if n_rows < 0:
        raise CATEValidationError("n_rows must be non-negative")
    if prediction.cate is None:
        raise CATEValidationError("cate is required")
    cate = _normalize_field(prediction.cate, field_name="cate", n_rows=n_rows)
    assert cate is not None
    return CATEPrediction(
        cate=cate,
        mu0=_normalize_field(prediction.mu0, field_name="mu0", n_rows=n_rows),
        mu1=_normalize_field(prediction.mu1, field_name="mu1", n_rows=n_rows),
        cate_lower=_normalize_field(
            prediction.cate_lower,
            field_name="cate_lower",
            n_rows=n_rows,
        ),
        cate_upper=_normalize_field(
            prediction.cate_upper,
            field_name="cate_upper",
            n_rows=n_rows,
        ),
        metadata=prediction.metadata,
    )
