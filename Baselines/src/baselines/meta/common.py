from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CATEValidationError
from ..model_factories import model_factory_metadata
from ..validation import OutcomeType, validate_predict_X


def conditional_mean(
    model: Any,
    X: NDArray[np.float64],
    outcome_type: OutcomeType,
) -> NDArray[np.float64]:
    if outcome_type == "continuous":
        values = np.asarray(model.predict(X), dtype=np.float64)
    else:
        probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
        classes = np.asarray(model.classes_)
        positive_columns = np.flatnonzero(classes == 1)
        if positive_columns.size == 1:
            values = probabilities[:, int(positive_columns[0])]
        elif classes.size == 1 and classes[0] == 0:
            values = np.zeros(len(X), dtype=np.float64)
        elif classes.size == 1 and classes[0] == 1:
            values = np.ones(len(X), dtype=np.float64)
        else:
            raise CATEValidationError("binary model does not expose class 1")
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.shape != (len(X),) or not np.isfinite(values).all():
        raise CATEValidationError("conditional mean prediction has invalid shape or values")
    return values


class ConfiguredMetaEstimator:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = deepcopy(dict(config))
        self.seed = int(self.config.get("seed", 0))

    @property
    def n_splits(self) -> int:
        return int(self.config.get("cross_fitting", {}).get("n_splits", 5))

    @property
    def propensity_bounds(self) -> tuple[float, float]:
        bounds = self.config.get("propensity_clip", [0.01, 0.99])
        return float(bounds[0]), float(bounds[1])

    def base_learner_metadata(self) -> Mapping[str, Any]:
        return model_factory_metadata(self.config)

    def _validated_predict_X(
        self,
        X: NDArray[np.float64],
        *,
        alpha: float,
    ) -> NDArray[np.float64]:
        self._require_fitted()  # type: ignore[attr-defined]
        X_array = validate_predict_X(X, alpha=alpha)
        if X_array.shape[1] != self.n_features_in_:
            raise CATEValidationError(
                f"X must have {self.n_features_in_} features, got {X_array.shape[1]}"
            )
        return X_array
