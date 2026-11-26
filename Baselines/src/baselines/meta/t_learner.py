from __future__ import annotations

from typing import Any, Mapping

from numpy.typing import NDArray
from typing_extensions import Self
import numpy as np

from ..base import BaseCATEEstimator
from ..crossfit import fit_prediction_model
from ..model_factories import make_outcome_model
from ..prediction import CATEPrediction, normalize_prediction
from ..validation import OutcomeType, validate_fit_data
from .common import ConfiguredMetaEstimator, conditional_mean


class TLearnerEstimator(ConfiguredMetaEstimator, BaseCATEEstimator):
    model_id = "t_learner"

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
        self.model0_ = fit_prediction_model(
            template,
            X_array[treatment_array == 0],
            y_array[treatment_array == 0],
            prediction_method=method,
            fallback_component="outcome",
            treatment_arm=0,
        )
        self.model1_ = fit_prediction_model(
            template,
            X_array[treatment_array == 1],
            y_array[treatment_array == 1],
            prediction_method=method,
            fallback_component="outcome",
            treatment_arm=1,
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
        mu0 = conditional_mean(self.model0_, X_array, self.outcome_type_)
        mu1 = conditional_mean(self.model1_, X_array, self.outcome_type_)
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
