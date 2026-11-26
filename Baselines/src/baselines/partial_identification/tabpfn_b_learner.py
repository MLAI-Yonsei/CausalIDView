from __future__ import annotations

import time
import gc
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin


from .assets import SENSITIVITY_GAMMA, SENSITIVITY_GAMMA_INDEX


TABPFN_B_LEARNER_METHODS = (
    "b_learner_tabpfn_final",
    "b_learner_tabpfn_all",
)


@dataclass(frozen=True)
class TabPFNBLearnerPrediction:
    lower: np.ndarray
    upper: np.ndarray
    unit_id: np.ndarray
    metadata: dict[str, Any]


def build_b_learner(
    method: str,
    *,
    gamma: float,
    seed: int,
    device: str,
    n_estimators: int = 8,
):
    if method not in TABPFN_B_LEARNER_METHODS:
        raise ValueError(f"unknown TabPFN B-Learner variant: {method}")

    from ._blearner.blearner import BLearner
    from sklearn.ensemble import (
        RandomForestClassifier,
        RandomForestRegressor,
    )
    from sklearn_quantile import RandomForestQuantileRegressor

    tau = float(gamma / (1.0 + gamma))
    forest_parameters = {
        "n_estimators": 100,
        "min_samples_leaf": 8,
        "max_features": 0.8,
        "n_jobs": 1,
        "random_state": seed,
    }
    final_model = LazyTabPFNRegressor(
        seed=seed,
        device=device,
        n_estimators=n_estimators,
        role="b-learner-final-bound",
    )

    if method == "b_learner_tabpfn_final":
        propensity_model = RandomForestClassifier(**forest_parameters)
        quantile_plus_model = RandomForestQuantileRegressor(
            q=tau, **forest_parameters
        )
        quantile_minus_model = RandomForestQuantileRegressor(
            q=1.0 - tau, **forest_parameters
        )
        mu_model = RandomForestRegressor(**forest_parameters)
    else:
        propensity_model = LazyTabPFNClassifier(
            seed=seed, device=device, n_estimators=n_estimators
        )
        quantile_plus_model = LazyTabPFNQuantileRegressor(
            quantile=tau, seed=seed, device=device, n_estimators=n_estimators
        )
        quantile_minus_model = LazyTabPFNQuantileRegressor(
            quantile=1.0 - tau,
            seed=seed,
            device=device,
            n_estimators=n_estimators,
        )
        mu_model = LazyTabPFNRegressor(
            seed=seed,
            device=device,
            n_estimators=n_estimators,
            role="b-learner-nuisance",
        )

    learner = BLearner(
        propensity_model=propensity_model,
        quantile_plus_model=quantile_plus_model,
        quantile_minus_model=quantile_minus_model,
        mu_model=mu_model,
        cate_bounds_model=final_model,
        use_rho=True,
        gamma=float(gamma),
        cv=5,
        random_state=seed,
    )
    if method == "b_learner_tabpfn_all":
        learner.nuisance_model = SequentialTabPFNNuisanceModel(
            gamma=float(gamma),
            seed=seed,
            device=device,
            n_estimators=n_estimators,
        )
    return learner


def estimate_tabpfn_b_learner(
    world,
    method: str,
    *,
    seed: int,
    device: str,
    n_estimators: int = 8,
    learner_builder: Callable[..., Any] = build_b_learner,
) -> TabPFNBLearnerPrediction:
    if method not in TABPFN_B_LEARNER_METHODS:
        raise ValueError(f"unknown TabPFN B-Learner variant: {method}")
    context = np.flatnonzero(world.arrays["split"] == 0)
    query = np.flatnonzero(world.arrays["split"] == 1)
    X = np.asarray(world.arrays["X"])
    treatment = np.asarray(world.arrays["T"])
    outcome = np.asarray(world.arrays["Y"])
    # Retain the original seed assigned to Γ=2 in the experiment grid.
    learner = learner_builder(
        method,
        gamma=SENSITIVITY_GAMMA,
        seed=seed + 1000 * SENSITIVITY_GAMMA_INDEX,
        device=device,
        n_estimators=n_estimators,
    )
    started = time.perf_counter()
    learner.fit(X[context], treatment[context], outcome[context])
    fit_time = time.perf_counter() - started
    started = time.perf_counter()
    lower, upper = learner.effect(X[query])
    prediction_time = time.perf_counter() - started

    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError(f"non-finite {method} sensitivity bounds")
    return TabPFNBLearnerPrediction(
        lower=np.atleast_2d(lower),
        upper=np.atleast_2d(upper),
        unit_id=np.asarray(world.arrays["unit_id"])[query],
        metadata={
            "implementation": "B-Learner with pinned TabPFN-v3.5 adapters",
            "variant": method,
            "tabpfn_version": "v3.5",
            "tabpfn_revision": "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
            "tabpfn_n_estimators": n_estimators,
            "nuisance_backend": (
                "authors' random-forest nuisance models"
                if method == "b_learner_tabpfn_final"
                else "TabPFN classifier/regressor/quantile adapters"
            ),
            "final_backend": "TabPFN regressor",
            "cross_fitting_folds": 5,
            "use_rho": True,
            "device": device,
            "per_gamma_fit_time_sec": [fit_time],
            "per_gamma_prediction_time_sec": [prediction_time],
            "fit_time_sec": fit_time,
            "prediction_time_sec": prediction_time,
            "query_used_for_fit_or_tuning": False,
            "actual_fit_rows": int(len(context)),
            "binary_outcome": True,
            "world_id": int(world.world_id),
        },
    )


def _backend(device: str, n_estimators: int):
    from ..tabpfn.backend import OfficialTabPFNBackend

    return OfficialTabPFNBackend(
        {
            "tabpfn": {
                "model_version": "v3.5",
                "checkpoint_revision": "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e",
                "n_estimators": n_estimators,
                "n_preprocessing_jobs": 1,
                "device": device,
                "ignore_pretraining_limits": True,
            }
        }
    )


class LazyTabPFNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, *, seed: int, device: str, n_estimators: int = 8):
        self.seed = seed
        self.device = device
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model_ = _backend(self.device, self.n_estimators).create_classifier(
            random_state=self.seed, role="b-learner-propensity"
        )
        self.model_.fit(X, y)
        self.classes_ = np.asarray(self.model_.classes_)
        return self

    def predict_proba(self, X):
        return np.asarray(self.model_.predict_proba(X), dtype=float)

    def release(self):
        if hasattr(self, "model_"):
            del self.model_
        _release_cuda_cache()


class LazyTabPFNRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self, *, seed: int, device: str, n_estimators: int = 8, role: str = "b-learner-regression"
    ):
        self.seed = seed
        self.device = device
        self.n_estimators = n_estimators
        self.role = role

    def fit(self, X, y):
        self.model_ = _backend(self.device, self.n_estimators).create_regressor(
            random_state=self.seed, role=self.role
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return np.asarray(self.model_.predict(X), dtype=float).reshape(-1)

    def release(self):
        if hasattr(self, "model_"):
            del self.model_
        _release_cuda_cache()


class LazyTabPFNQuantileRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        quantile: float,
        seed: int,
        device: str,
        n_estimators: int = 8,
    ):
        self.quantile = quantile
        self.seed = seed
        self.device = device
        self.n_estimators = n_estimators

    def fit(self, X, y):
        self.model_ = _backend(self.device, self.n_estimators).create_regressor(
            random_state=self.seed, role="b-learner-quantile"
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        prediction = self.model_.predict(
            X, output_type="quantiles", quantiles=[float(self.quantile)]
        )
        return np.asarray(prediction[0], dtype=float).reshape(-1)

    def release(self):
        if hasattr(self, "model_"):
            del self.model_
        _release_cuda_cache()


def _release_cuda_cache():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class SequentialTabPFNNuisanceModel(BaseEstimator):
    """Compute one fold's B-Learner nuisances with one resident TabPFN at a time."""

    def __init__(
        self,
        *,
        gamma: float,
        seed: int,
        device: str,
        n_estimators: int = 8,
        estimator_factory=None,
    ):
        self.gamma = gamma
        self.seed = seed
        self.device = device
        self.n_estimators = n_estimators
        self.estimator_factory = estimator_factory

    def fit(self, X, treatment, outcome, X_val=None, A_val=None, Y_val=None):
        self.X_ = np.asarray(X, dtype=float)
        self.treatment_ = np.asarray(treatment, dtype=int)
        self.outcome_ = np.asarray(outcome, dtype=float)
        return self

    def _make_estimator(self, kind: str, *, quantile=None, offset: int):
        parameters = {
            "seed": int(self.seed + offset),
            "device": self.device,
            "n_estimators": self.n_estimators,
        }
        if self.estimator_factory is not None:
            return self.estimator_factory(kind, quantile=quantile, **parameters)
        if kind == "classifier":
            return LazyTabPFNClassifier(**parameters)
        if kind == "quantile":
            return LazyTabPFNQuantileRegressor(
                quantile=float(quantile), **parameters
            )
        return LazyTabPFNRegressor(
            role=f"b-learner-{kind}", **parameters
        )

    def _fit_predict(
        self, kind: str, X_train, y_train, X_predict, *, offset: int, quantile=None
    ) -> np.ndarray:
        estimator = self._make_estimator(kind, quantile=quantile, offset=offset)
        try:
            estimator.fit(X_train, y_train)
            if kind == "classifier":
                return np.asarray(estimator.predict_proba(X_predict), float)[:, 1]
            return np.asarray(estimator.predict(X_predict), float).reshape(-1)
        finally:
            estimator.release()
            del estimator

    def predict(self, X):
        X_query = np.asarray(X, dtype=float)
        propensity = np.clip(
            self._fit_predict(
                "classifier",
                self.X_,
                self.treatment_,
                X_query,
                offset=0,
            ),
            1.0e-6,
            1.0 - 1.0e-6,
        )
        tau = float(self.gamma / (1.0 + self.gamma))
        arm_predictions: list[tuple[np.ndarray, ...]] = []
        offset = 1
        for arm in (0, 1):
            selected = self.treatment_ == arm
            X_arm = self.X_[selected]
            y_arm = self.outcome_[selected]
            combined = np.vstack([X_arm, X_query])
            q_plus_all = self._fit_predict(
                "quantile",
                X_arm,
                y_arm,
                combined,
                offset=offset,
                quantile=tau,
            )
            offset += 1
            q_minus_all = self._fit_predict(
                "quantile",
                X_arm,
                y_arm,
                combined,
                offset=offset,
                quantile=1.0 - tau,
            )
            offset += 1
            q_plus_train, q_plus_query = (
                q_plus_all[: len(X_arm)],
                q_plus_all[len(X_arm) :],
            )
            q_minus_train, q_minus_query = (
                q_minus_all[: len(X_arm)],
                q_minus_all[len(X_arm) :],
            )
            mu_query = self._fit_predict(
                "mu", X_arm, y_arm, X_query, offset=offset
            )
            offset += 1
            inverse_gamma = 1.0 / float(self.gamma)
            tail_weight = 1.0 / (1.0 - tau)
            rho_plus_target = inverse_gamma * y_arm + (1.0 - inverse_gamma) * (
                q_plus_train
                + tail_weight * np.maximum(y_arm - q_plus_train, 0.0)
            )
            rho_minus_target = inverse_gamma * y_arm + (1.0 - inverse_gamma) * (
                q_minus_train
                + tail_weight * np.minimum(y_arm - q_minus_train, 0.0)
            )
            rho_plus_query = self._fit_predict(
                "rho-plus",
                X_arm,
                rho_plus_target,
                X_query,
                offset=offset,
            )
            offset += 1
            rho_minus_query = self._fit_predict(
                "rho-minus",
                X_arm,
                rho_minus_target,
                X_query,
                offset=offset,
            )
            offset += 1
            arm_predictions.append(
                (
                    q_plus_query,
                    q_minus_query,
                    mu_query,
                    rho_plus_query,
                    rho_minus_query,
                )
            )
        zero, one = arm_predictions
        return np.column_stack(
            [
                propensity,
                zero[0],
                one[0],
                zero[1],
                one[1],
                zero[2],
                one[2],
                zero[3],
                one[3],
                zero[4],
                one[4],
            ]
        )
