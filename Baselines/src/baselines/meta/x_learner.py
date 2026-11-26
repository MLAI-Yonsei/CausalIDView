from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from typing_extensions import Self

from ..base import BaseCATEEstimator
from ..crossfit import (
    clip_propensity,
    cross_fit_arm_outcomes,
    fitted_fallback_record,
    fit_prediction_model,
)
from ..exceptions import CATEValidationError
from ..model_factories import (
    make_effect_model,
    make_outcome_model,
    make_propensity_model,
)
from ..prediction import CATEPrediction, normalize_prediction
from ..validation import OutcomeType, validate_fit_data
from .common import ConfiguredMetaEstimator, conditional_mean


def combine_x_learner_effects(
    tau0: NDArray[np.float64],
    tau1: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> NDArray[np.float64]:
    tau0_array = np.asarray(tau0, dtype=np.float64)
    tau1_array = np.asarray(tau1, dtype=np.float64)
    propensity_array = np.asarray(propensity, dtype=np.float64)
    if not (tau0_array.shape == tau1_array.shape == propensity_array.shape):
        raise CATEValidationError("X-Learner effect and propensity shapes must match")
    return propensity_array * tau0_array + (1.0 - propensity_array) * tau1_array


def _target_summary(values: NDArray[np.float64]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, [0.0, 0.05, 0.5, 0.95, 1.0])
    return {
        "n": int(len(array)),
        "mean": float(np.mean(array)),
        "std_ddof0": float(np.std(array, ddof=0)),
        "min": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "max": float(quantiles[4]),
    }


class XLearnerEstimator(ConfiguredMetaEstimator, BaseCATEEstimator):
    model_id = "x_learner"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        outcome_model: Any = None,
        effect_model: Any = None,
        propensity_model: Any = None,
    ) -> None:
        super().__init__(config)
        self._outcome_model_template = outcome_model
        self._effect_model_template = effect_model
        self._propensity_model_template = propensity_model

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
        method = "predict_proba" if outcome_type == "binary" else "predict"
        oof = cross_fit_arm_outcomes(
            outcome_template,
            X_array,
            treatment_array,
            y_array,
            n_splits=self.n_splits,
            random_state=self.seed,
            prediction_method=method,
        )
        imputed_control = oof.mu1[treatment_array == 0] - y_array[treatment_array == 0]
        imputed_treated = y_array[treatment_array == 1] - oof.mu0[treatment_array == 1]
        self.imputed_control_ = np.asarray(imputed_control, dtype=np.float64)
        self.imputed_treated_ = np.asarray(imputed_treated, dtype=np.float64)
        effect_template = self._effect_model_template or make_effect_model(
            self.config, random_state=self.seed
        )
        self.tau0_model_ = clone(effect_template).fit(
            X_array[treatment_array == 0], imputed_control
        )
        self.tau1_model_ = clone(effect_template).fit(
            X_array[treatment_array == 1], imputed_treated
        )
        propensity_template = self._propensity_model_template or make_propensity_model(
            self.config, random_state=self.seed
        )
        self.propensity_model_ = clone(propensity_template).fit(X_array, treatment_array)
        raw_propensity = conditional_mean(self.propensity_model_, X_array, "binary")
        _, self.propensity_diagnostics_ = clip_propensity(
            raw_propensity,
            lower=self.propensity_bounds[0],
            upper=self.propensity_bounds[1],
        )
        self.model0_ = fit_prediction_model(
            outcome_template,
            X_array[treatment_array == 0],
            y_array[treatment_array == 0],
            prediction_method=method,
            fallback_component="outcome",
            treatment_arm=0,
        )
        self.model1_ = fit_prediction_model(
            outcome_template,
            X_array[treatment_array == 1],
            y_array[treatment_array == 1],
            prediction_method=method,
            fallback_component="outcome",
            treatment_arm=1,
        )
        self.fallback_records_ = list(oof.fallback_records)
        self.fallback_records_.extend(
            record
            for record in (
                fitted_fallback_record(self.model0_),
                fitted_fallback_record(self.model1_),
            )
            if record is not None
        )
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self.effective_fold_count_ = oof.effective_fold_count
        self.fold_counts_ = [
            {
                "train": {
                    str(arm): count
                    for arm, count in fold.train_treatment_counts.items()
                },
                "test": {
                    str(arm): count
                    for arm, count in fold.test_treatment_counts.items()
                },
            }
            for fold in oof.folds
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
        tau0 = np.asarray(self.tau0_model_.predict(X_array), dtype=np.float64)
        tau1 = np.asarray(self.tau1_model_.predict(X_array), dtype=np.float64)
        raw_propensity = conditional_mean(self.propensity_model_, X_array, "binary")
        propensity, _ = clip_propensity(
            raw_propensity,
            lower=self.propensity_bounds[0],
            upper=self.propensity_bounds[1],
        )
        cate = combine_x_learner_effects(tau0, tau1, propensity)
        if self.outcome_type_ == "binary":
            cate = np.clip(cate, -1.0, 1.0)
        mu0 = conditional_mean(self.model0_, X_array, self.outcome_type_) if return_po else None
        mu1 = conditional_mean(self.model1_, X_array, self.outcome_type_) if return_po else None
        return normalize_prediction(
            CATEPrediction(cate=cate, mu0=mu0, mu1=mu1, metadata=self.get_metadata()),
            len(X_array),
        )

    def get_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "algorithm_variant": "cross-fitted X-Learner",
            "seed": self.seed,
            "base_learner": self.base_learner_metadata(),
            "binary_outcome_handling": "probability nuisance models and conditional-mean effect regression",
        }
        if getattr(self, "_is_fitted", False):
            metadata.update(
                effective_fold_count=self.effective_fold_count_,
                treatment_counts_per_fold=self.fold_counts_,
                propensity_diagnostics=asdict(self.propensity_diagnostics_),
                pseudo_outcome_summary={
                    "control_side_mu1_minus_y0": _target_summary(self.imputed_control_),
                    "treated_side_y1_minus_mu0": _target_summary(self.imputed_treated_),
                },
            )
        return metadata
