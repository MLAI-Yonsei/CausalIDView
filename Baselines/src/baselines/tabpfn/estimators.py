from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from typing_extensions import Self

from ..base import BaseCATEEstimator
from ..crossfit import (
    clip_propensity,
    cross_fit_arm_outcomes,
    cross_fit_predictions,
    dr_pseudo_outcome,
    fitted_fallback_record,
    fit_prediction_model,
)
from ..meta.common import ConfiguredMetaEstimator, conditional_mean
from ..meta.s_learner import SLearnerEstimator
from ..meta.t_learner import TLearnerEstimator
from ..meta.x_learner import XLearnerEstimator
from ..model_factories import make_effect_model
from ..prediction import CATEPrediction, normalize_prediction
from ..validation import OutcomeType, validate_fit_data
from .backend import OfficialTabPFNBackend, TabPFNBackend


def _outcome_template(
    backend: TabPFNBackend,
    outcome_type: OutcomeType,
    seed: int,
) -> Any:
    if outcome_type == "binary":
        return backend.create_classifier(random_state=seed, role="outcome")
    return backend.create_regressor(random_state=seed, role="outcome")


def _select_s_learner_features(
    X: NDArray[np.float64],
    *,
    feature_budget: int,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64], int | None]:
    """Select X only, reserving one mandatory feature slot for treatment."""

    X_array = np.asarray(X, dtype=np.float64)
    if feature_budget < 2:
        raise ValueError("TabPFN-S feature budget must reserve at least one X and T")
    if X_array.shape[1] + 1 <= feature_budget:
        indices = np.arange(X_array.shape[1], dtype=np.int64)
        return X_array, indices, None
    digest = hashlib.sha256(
        f"tabpfn_s|{seed}|{X_array.shape[1]}|{feature_budget}".encode("utf-8")
    ).digest()
    subset_seed = int.from_bytes(digest[:8], "little", signed=False)
    rng = np.random.default_rng(subset_seed)
    indices = np.sort(
        rng.choice(X_array.shape[1], size=feature_budget - 1, replace=False)
    ).astype(np.int64)
    return X_array[:, indices], indices, subset_seed


def _backend_provenance(backend: TabPFNBackend) -> dict[str, Any]:
    provider = getattr(backend, "provenance", None)
    if callable(provider):
        return dict(provider())
    return {
        "base_learner": {
            "family": "tabpfn",
            "backend": backend.backend_name,
        },
        "tabpfn_model_identifier": backend.backend_name,
        "tabpfn_revision": "test_or_custom_backend",
        "tabpfn_ensemble_size": None,
        "tabpfn_device": None,
    }


def _fallback_metadata(records: list[Any]) -> dict[str, Any]:
    events = [asdict(record) for record in records if record is not None]
    return {
        "fallback_used": bool(events),
        "fallback_reason": sorted({event["fallback_reason"] for event in events}) or None,
        "fallback_component": sorted({event["fallback_component"] for event in events}) or None,
        "fallback_events": events,
    }


def _internal_feature_metadata(
    backend: TabPFNBackend,
    models: list[Any],
    *,
    n_features: int,
    seed: int,
) -> dict[str, Any]:
    budgets = []
    for model in models:
        if hasattr(model, "get_inference_config"):
            budgets.append(int(backend.effective_feature_budget(model)))
    if not budgets:
        return {
            "feature_subsampling_used": False,
            "feature_subset_seed": None,
            "effective_feature_count": n_features,
        }
    minimum_budget = min(budgets)
    used = n_features > minimum_budget
    return {
        "feature_subsampling_used": used,
        "feature_subset_seed": seed if used else None,
        "effective_feature_count": min(n_features, minimum_budget),
        "tabpfn_stage_feature_budgets": sorted(set(budgets)),
    }


class TabPFNSLearner(SLearnerEstimator):
    model_id = "tabpfn_s"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: TabPFNBackend | None = None,
    ) -> None:
        super().__init__(config)
        self.backend = backend or OfficialTabPFNBackend(config)

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
        template = _outcome_template(self.backend, outcome_type, self.seed)
        feature_budget = int(self.backend.effective_feature_budget(template))
        selected_X, selected_indices, subset_seed = _select_s_learner_features(
            X_array,
            feature_budget=feature_budget,
            seed=self.seed,
        )
        method = "predict_proba" if outcome_type == "binary" else "predict"
        self.model_ = fit_prediction_model(
            template,
            np.column_stack([selected_X, treatment_array]),
            y_array,
            prediction_method=method,
            clone_estimator=False,
            fallback_component="outcome",
        )
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self.feature_budget_ = feature_budget
        self.selected_x_feature_indices_ = selected_indices
        self.feature_subset_seed_ = subset_seed
        self.fallback_records_ = [fitted_fallback_record(self.model_)]
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
        selected_X = X_array[:, self.selected_x_feature_indices_]
        X0 = np.column_stack([selected_X, np.zeros(len(selected_X))])
        X1 = np.column_stack([selected_X, np.ones(len(selected_X))])
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
        metadata = dict(super().get_metadata())
        metadata.update(_backend_provenance(self.backend))
        metadata.update(
            backend=self.backend.backend_name,
            implementation_variant="single-model g-computation CATE",
        )
        if getattr(self, "_is_fitted", False):
            selected = self.selected_x_feature_indices_.tolist()
            metadata.update(
                feature_subsampling_used=len(selected) < self.n_features_in_,
                feature_subset_seed=self.feature_subset_seed_,
                effective_feature_count=len(selected) + 1,
                original_x_feature_count=self.n_features_in_,
                selected_x_feature_indices=selected,
                treatment_feature_pinned=True,
                tabpfn_stage_feature_budgets=[self.feature_budget_],
            )
            metadata.update(_fallback_metadata(self.fallback_records_))
        return metadata


class TabPFNTLearner(TLearnerEstimator):
    model_id = "tabpfn_t"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: TabPFNBackend | None = None,
    ) -> None:
        super().__init__(config)
        self.backend = backend or OfficialTabPFNBackend(config)

    def fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> Self:
        self._outcome_model_template = _outcome_template(
            self.backend, outcome_type, self.seed
        )
        super().fit(X, treatment, y, outcome_type=outcome_type)
        self.fallback_records_ = [
            fitted_fallback_record(self.model0_),
            fitted_fallback_record(self.model1_),
        ]
        return self

    def get_metadata(self) -> Mapping[str, Any]:
        metadata = dict(super().get_metadata())
        metadata.update(_backend_provenance(self.backend))
        metadata.update(
            backend=self.backend.backend_name,
            implementation_variant="two-model g-computation CATE",
        )
        if getattr(self, "_is_fitted", False):
            metadata.update(_internal_feature_metadata(
                self.backend,
                [self.model0_, self.model1_],
                n_features=self.n_features_in_,
                seed=self.seed,
            ))
            metadata.update(_fallback_metadata(self.fallback_records_))
        return metadata


class TabPFNXLearner(XLearnerEstimator):
    model_id = "tabpfn_x"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: TabPFNBackend | None = None,
    ) -> None:
        super().__init__(config)
        self.backend = backend or OfficialTabPFNBackend(config)

    def fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> Self:
        self._outcome_model_template = _outcome_template(
            self.backend, outcome_type, self.seed
        )
        self._effect_model_template = self.backend.create_regressor(
            random_state=self.seed, role="effect"
        )
        propensity_mode = str(
            self.config.get("propensity", {}).get("mode", "learned_tabpfn")
        )
        if propensity_mode == "constant_context_rate":
            self._propensity_model_template = DummyClassifier(strategy="prior")
        elif propensity_mode == "learned_tabpfn":
            self._propensity_model_template = self.backend.create_classifier(
                random_state=self.seed, role="propensity"
            )
        else:
            raise ValueError(
                f"unsupported TabPFN X-Learner propensity mode: {propensity_mode}"
            )
        self.propensity_mode_ = propensity_mode
        super().fit(X, treatment, y, outcome_type=outcome_type)
        return self

    def get_metadata(self) -> Mapping[str, Any]:
        metadata = dict(super().get_metadata())
        metadata.update(_backend_provenance(self.backend))
        metadata.update(
            backend=self.backend.backend_name,
            implementation_variant="OOF TabPFN X-Learner",
            propensity_mode=getattr(self, "propensity_mode_", "learned_tabpfn"),
        )
        if getattr(self, "_is_fitted", False):
            metadata.update(_internal_feature_metadata(
                self.backend,
                [
                    self.model0_, self.model1_, self.tau0_model_, self.tau1_model_,
                    self.propensity_model_,
                ],
                n_features=self.n_features_in_,
                seed=self.seed,
            ))
            metadata.update(_fallback_metadata(self.fallback_records_))
        return metadata


class TabPFNDRLearner(ConfiguredMetaEstimator, BaseCATEEstimator):
    model_id = "tabpfn_dr"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: TabPFNBackend | None = None,
    ) -> None:
        super().__init__(config)
        self.backend = backend or OfficialTabPFNBackend(config)

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
        outcome_template = _outcome_template(self.backend, outcome_type, self.seed)
        method = "predict_proba" if outcome_type == "binary" else "predict"
        outcome_oof = cross_fit_arm_outcomes(
            outcome_template,
            X_array,
            treatment_array,
            y_array,
            n_splits=self.n_splits,
            random_state=self.seed,
            prediction_method=method,
        )
        propensity_template = self.backend.create_classifier(
            random_state=self.seed, role="propensity"
        )
        propensity_oof = cross_fit_predictions(
            propensity_template,
            X_array,
            treatment_array.astype(np.float64),
            treatment=treatment_array,
            n_splits=self.n_splits,
            random_state=self.seed,
            prediction_method="predict_proba",
            fallback_component="propensity",
        )
        propensity, self.propensity_diagnostics_ = clip_propensity(
            propensity_oof.predictions,
            lower=self.propensity_bounds[0],
            upper=self.propensity_bounds[1],
        )
        pseudo_target = dr_pseudo_outcome(
            treatment_array,
            y_array,
            propensity,
            outcome_oof.mu0,
            outcome_oof.mu1,
        )
        self.final_model_ = self.backend.create_regressor(
            random_state=self.seed, role="dr_effect"
        ).fit(X_array, pseudo_target)
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
        self.fallback_records_ = [
            *outcome_oof.fallback_records,
            *propensity_oof.fallback_records,
            fitted_fallback_record(self.model0_),
            fitted_fallback_record(self.model1_),
        ]
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self.effective_fold_count_ = min(
            outcome_oof.effective_fold_count,
            propensity_oof.effective_fold_count,
        )
        self.fold_counts_ = [
            {"train": fold.train_treatment_counts, "test": fold.test_treatment_counts}
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
        mu0 = conditional_mean(self.model0_, X_array, self.outcome_type_) if return_po else None
        mu1 = conditional_mean(self.model1_, X_array, self.outcome_type_) if return_po else None
        return normalize_prediction(
            CATEPrediction(cate=cate, mu0=mu0, mu1=mu1, metadata=self.get_metadata()),
            len(X_array),
        )

    def get_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "seed": self.seed,
            "backend": self.backend.backend_name,
            "implementation_variant": "OOF TabPFN DR pseudo-outcome and TabPFN final regressor",
        }
        metadata.update(_backend_provenance(self.backend))
        if getattr(self, "_is_fitted", False):
            metadata.update(
                effective_fold_count=self.effective_fold_count_,
                treatment_counts_per_fold=self.fold_counts_,
                propensity_diagnostics=asdict(self.propensity_diagnostics_),
            )
            metadata.update(_internal_feature_metadata(
                self.backend,
                [self.model0_, self.model1_, self.final_model_],
                n_features=self.n_features_in_,
                seed=self.seed,
            ))
            metadata.update(_fallback_metadata(self.fallback_records_))
        return metadata


class TabPFNDMLEstimator(ConfiguredMetaEstimator, BaseCATEEstimator):
    model_id = "tabpfn_dml"

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        backend: TabPFNBackend | None = None,
    ) -> None:
        super().__init__(config)
        self.backend = backend or OfficialTabPFNBackend(config)

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
        outcome_template = _outcome_template(self.backend, outcome_type, self.seed)
        outcome_oof = cross_fit_predictions(
            outcome_template,
            X_array,
            y_array,
            treatment=treatment_array,
            n_splits=self.n_splits,
            random_state=self.seed,
            prediction_method="predict_proba" if outcome_type == "binary" else "predict",
        )
        propensity_template = self.backend.create_classifier(
            random_state=self.seed, role="propensity"
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
        pseudo_target = (y_array - outcome_oof.predictions) / treatment_residual
        final_model = make_effect_model(self.config, random_state=self.seed)
        self.final_model_ = final_model.fit(
            X_array,
            pseudo_target,
            sample_weight=treatment_residual**2,
        )
        self.outcome_type_ = outcome_type
        self.n_features_in_ = X_array.shape[1]
        self.effective_fold_count_ = min(
            outcome_oof.effective_fold_count,
            propensity_oof.effective_fold_count,
        )
        self.fold_counts_ = [
            {"train": fold.train_treatment_counts, "test": fold.test_treatment_counts}
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
            CATEPrediction(cate=cate, metadata=self.get_metadata()), len(X_array)
        )

    def get_metadata(self) -> Mapping[str, Any]:
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "seed": self.seed,
            "nuisance_backend": self.backend.backend_name,
            "final_stage": "RandomForestRegressor weighted R-loss",
        }
        if getattr(self, "_is_fitted", False):
            metadata.update(
                effective_fold_count=self.effective_fold_count_,
                treatment_counts_per_fold=self.fold_counts_,
                propensity_diagnostics=asdict(self.propensity_diagnostics_),
            )
        return metadata
