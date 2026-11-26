"""Full MLP (neural network) nuisance and final-stage P-Learner configuration.

Reuses the same generic full-backend nuisance/score pipeline as the XGBoost
and TabPFN-v3 full backends (`compute_full_tabpfn_oof_score`); only the
classifier/regressor factories differ. The MLP implementation
(`baselines.neural.MLPNuisance`) is the same sklearn-style
early-stopped MLP already used and validated for the "NN + FD Plug-in"
baseline in the Front-Door panel.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .full_tabpfn import (
    FullPLearnerBackendSpec,
    FullTabPFNFinalResult,
    FullTabPFNScoreResult,
    compute_full_tabpfn_oof_score,
    fit_full_tabpfn_final_stage,
)


FULL_NN_MODEL_ID = "p_learner_nn_full"
FULL_NN_DISPLAY_NAME = "NN + P-Learner"
FULL_NN_NUISANCE_BACKEND_ID = "nn_full"
FULL_NN_BRIDGE_CONFIG = {
    "algorithm": "Sverdrup-Cui-Algorithm-1",
    "ridge": 0.01,
    "outcome_bridge_slope_clip": [-10.0, 10.0],
    "propensity_clip": [0.02, 0.98],
    "treatment_bridge_q_clip": [-25.0, 25.0],
}
FULL_NN_NUISANCE_CONFIG = {
    "backend_id": FULL_NN_NUISANCE_BACKEND_ID,
    "crossfit_folds": 5,
    "stratification_target": "T",
    "shuffle": True,
    "constant_class_policy": "fail_with_informative_error",
    "independent_model_per_nuisance_and_arm": True,
    "counterfactual_proxy_queries_batched": True,
}
FULL_NN_BACKEND_SPEC = FullPLearnerBackendSpec(
    model_id=FULL_NN_MODEL_ID,
    display_name=FULL_NN_DISPLAY_NAME,
    nuisance_backend_id=FULL_NN_NUISANCE_BACKEND_ID,
    nuisance_backend_name="early-stopped MLP (128-128, BCE/MSE, validation-selected epochs)",
    estimator_name="P-Learner with MLP nuisances and final regression",
    score_namespace="p-learner-nn-full-score-v1",
    nuisance_config=FULL_NN_NUISANCE_CONFIG,
    bridge_config=FULL_NN_BRIDGE_CONFIG,
)


def nn_backend_metadata(route_seed: int, training_log: list[dict[str, Any]]) -> dict[str, Any]:
    from ..neural import CONFIG

    selected_epochs = [entry["selected_epochs"] for entry in training_log]
    return {
        "backend": "NN (MLP, same implementation as the Front-Door NN Plug-in baseline)",
        "architecture": dict(CONFIG),
        "random_state": int(route_seed),
        "hpo_performed": False,
        "parameter_updates": True,
        "models_trained": len(training_log),
        "mean_selected_epochs": float(np.mean(selected_epochs)) if selected_epochs else None,
    }


class _ClassesAwareMLPClassifier:
    """Adapter exposing sklearn's ``classes_`` contract for ``MLPNuisance``.

    The shared full-backend pipeline's probability helper (``_probability`` in
    ``full_tabpfn_p_learner.py``) calls the project's generic
    ``conditional_mean(model, X, "binary")``, which reads ``model.classes_``
    to locate the positive-class column. ``MLPNuisance`` (used unmodified,
    identical to the validated Front-Door NN Plug-in baseline) does not
    define that attribute, so this thin wrapper adds it without touching
    ``baselines/neural.py`` at all.
    """

    def __init__(self, seed: int, training_log: list[dict[str, Any]]) -> None:
        from ..neural import MLPNuisance

        self._model = MLPNuisance(int(seed), True, training_log)
        self.classes_ = np.array([0, 1])

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_ClassesAwareMLPClassifier":
        self._model.fit(x, y)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self._model.predict(x)


def nn_factories(training_log: list[dict[str, Any]]) -> tuple[
    Callable[[int, str], Any], Callable[[int, str], Any]
]:
    from ..neural import MLPNuisance

    def classifier(seed: int, role: str) -> Any:
        del role
        return _ClassesAwareMLPClassifier(int(seed), training_log)

    def regressor(seed: int, role: str) -> Any:
        del role
        return MLPNuisance(int(seed), False, training_log)

    return classifier, regressor


def compute_full_nn_oof_score(
    X_context: np.ndarray,
    Z_context: np.ndarray,
    W_context: np.ndarray,
    T_context: np.ndarray,
    Y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    route_seed: int,
    input_hash: str,
    context_query_split_hash: str,
) -> tuple[FullTabPFNScoreResult, list[dict[str, Any]]]:
    training_log: list[dict[str, Any]] = []
    classifier, regressor = nn_factories(training_log)
    score = compute_full_tabpfn_oof_score(
        X_context,
        Z_context,
        W_context,
        T_context,
        Y_context,
        X_query,
        route_seed=route_seed,
        classifier_factory=classifier,
        regressor_factory=regressor,
        tabpfn_metadata=nn_backend_metadata(route_seed, training_log),
        input_hash=input_hash,
        context_query_split_hash=context_query_split_hash,
        backend_spec=FULL_NN_BACKEND_SPEC,
    )
    return score, training_log


def fit_full_nn_final_stage(
    X_context: np.ndarray,
    Gamma_oof: np.ndarray,
    X_query: np.ndarray,
    *,
    seed: int,
    training_log: list[dict[str, Any]] | None = None,
) -> FullTabPFNFinalResult:
    log: list[dict[str, Any]] = training_log if training_log is not None else []
    _, regressor = nn_factories(log)
    return fit_full_tabpfn_final_stage(
        X_context,
        Gamma_oof,
        X_query,
        regressor_factory=lambda: regressor(seed, "full-p-learner-final-x-to-gamma"),
    )
