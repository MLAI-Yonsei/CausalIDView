from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from typing_extensions import Self

from ..base import BaseCATEEstimator
from ..crossfit import clip_propensity, cross_fit_predictions
from ..model_factories import (
    make_effect_model,
    make_outcome_model,
    make_propensity_model,
)
from ..prediction import CATEPrediction, normalize_prediction
from ..validation import OutcomeType, validate_fit_data
from .common import ConfiguredMetaEstimator


class RLearnerEstimator(ConfiguredMetaEstimator, BaseCATEEstimator):
    model_id = "r_learner"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        outcome_model: Any = None,
        propensity_model: Any = None,
        final_model: Any = None,
    ) -> None:
        super().__init__(config)
        self._outcome_model_template = outcome_model
        self._propensity_model_template = propensity_model
        self._final_model_template = final_model

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
        outcome_template = self._outcome_model_template or make_outcome_model(
            outcome_type, self.config, random_state=self.seed
        )
        propensity_template = self._propensity_model_template or make_propensity_model(
            self.config, random_state=self.seed
        )
        outcome_oof = cross_fit_predictions(
            outcome_template,
            X_array,
            y_array,
            treatment=treatment_array,
            n_splits=self.n_splits,
            random_state=self.seed,
            prediction_method="predict_proba" if outcome_type == "binary" else "predict",
        )
        propensity_oof = cross_fit_predictions(
            propensity_template,
            X_array,
            treatment_array.astype(np.float64),
            treatment=treatment_array,
            n_splits=self.n_splits,
            random_state=self.seed,
            prediction_method="predict_proba",
        )
        propensity, self.propensity_diagnostics_ = clip_propensity(
            propensity_oof.predictions,
            lower=self.propensity_bounds[0],
            upper=self.propensity_bounds[1],
        )
        treatment_residual = treatment_array.astype(np.float64) - propensity
        outcome_residual = y_array - outcome_oof.predictions
        pseudo_target = outcome_residual / treatment_residual
        sample_weight = treatment_residual**2
        final_template = self._final_model_template or make_effect_model(
            self.config, random_state=self.seed
        )
        self.final_model_ = clone(final_template).fit(
            X_array,
            pseudo_target,
            sample_weight=sample_weight,
        )
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self.effective_fold_count_ = min(
            outcome_oof.effective_fold_count,
            propensity_oof.effective_fold_count,
        )
        self.fold_counts_ = [
            {
                "train": fold.train_treatment_counts,
                "test": fold.test_treatment_counts,
            }
            for fold in propensity_oof.folds
        ]
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
        cate = np.asarray(self.final_model_.predict(X_array), dtype=np.float64)
        if self.outcome_type_ == "binary":
            cate = np.clip(cate, -1.0, 1.0)
        return normalize_prediction(
            CATEPrediction(cate=cate, metadata=self.get_metadata()),
            len(X_array),
        )

    def get_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "seed": self.seed,
            "base_learner": self.base_learner_metadata(),
            "implementation_variant": "weighted R-loss",
            "binary_outcome_handling": "probability nuisance and conditional-mean difference",
        }
        if getattr(self, "_is_fitted", False):
            metadata.update(
                effective_fold_count=self.effective_fold_count_,
                treatment_counts_per_fold=self.fold_counts_,
                propensity_diagnostics=asdict(self.propensity_diagnostics_),
            )
        return metadata
