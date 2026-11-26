from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ..base import BaseCATEEstimator
from ..crossfit import fit_prediction_model
from ..model_factories import make_outcome_model
from ..prediction import CATEPrediction, normalize_prediction
from ..validation import OutcomeType, validate_fit_data
from .common import ConfiguredMetaEstimator, conditional_mean


class SLearnerEstimator(ConfiguredMetaEstimator, BaseCATEEstimator):
    model_id = "s_learner"

    def __init__(self, config: Mapping[str, Any], *, outcome_model: Any = None) -> None:
        super().__init__(config)
        self._outcome_model_template = outcome_model

    def fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> Self:
        X_array, treatment_array, y_array = validate_fit_data(
            X, treatment, y, outcome_type=outcome_type
        )
        template = self._outcome_model_template or make_outcome_model(
            outcome_type, self.config, random_state=self.seed
        )
        method = "predict_proba" if outcome_type == "binary" else "predict"
        self.model_ = fit_prediction_model(
            template,
            np.column_stack([X_array, treatment_array]),
            y_array,
            prediction_method=method,
        )
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self._is_fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float64],
        *,
        return_po: bool = False,
        return_interval: bool = False,
        alpha: float = 0.05,
    ) -> CATEPrediction:
        X_array = self._validated_predict_X(X, alpha=alpha)
        X0 = np.column_stack([X_array, np.zeros(len(X_array))])
        X1 = np.column_stack([X_array, np.ones(len(X_array))])
        stacked = conditional_mean(
            self.model_, np.vstack([X0, X1]), self.outcome_type_
        )
        mu0, mu1 = np.split(stacked, 2)
        return normalize_prediction(
            CATEPrediction(
                cate=mu1 - mu0,
                mu0=mu0 if return_po else None,
                mu1=mu1 if return_po else None,
                metadata=self.get_metadata(),
            ),
            len(X_array),
        )

    def get_metadata(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "seed": self.seed,
            "base_learner": self.base_learner_metadata(),
            "binary_outcome_handling": "native predict_proba risk difference",
        }
