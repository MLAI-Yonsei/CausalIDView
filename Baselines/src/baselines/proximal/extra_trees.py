"""ExtraTrees P-Learner and shared binary-proxy score functions."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

import numpy as np

from data.integrity import array_hash


P_LEARNER_BRIDGE_CONFIG = {
    "algorithm": "Sverdrup-Cui-Algorithm-1",
    "bridge_backend": "ExtraTrees",
    "crossfit_folds": 5,
    "ridge": 0.01,
    "outcome_bridge_slope_clip": [-10.0, 10.0],
    "propensity_clip": [0.02, 0.98],
    "treatment_bridge_q_clip": [-25.0, 25.0],
    "n_estimators": 300,
    "min_samples_leaf": {"bridge_regressor": 6, "bridge_classifier": 8},
    "max_features": 0.8,
}


def _config_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class PLearnerScoreBundle:
    """Backend-independent output of the fixed P-Learner bridge/score stage."""

    x_context: np.ndarray
    x_query: np.ndarray
    gamma_oof: np.ndarray
    fold_ids: np.ndarray
    gamma_hash: str
    score_hash: str
    fold_hash: str
    bridge_config_hash: str
    context_query_split_hash: str
    diagnostics: dict[str, Any]

    @property
    def pseudo(self) -> np.ndarray:
        """Backward-compatible name for the OOF proximal score Gamma."""
        return self.gamma_oof

    @property
    def outcome_bridge_diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostics["outcome_bridge_diagnostics"])

    @property
    def treatment_bridge_diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostics["treatment_bridge_diagnostics"])

    @property
    def q_clipping_diagnostics(self) -> dict[str, Any]:
        return dict(self.diagnostics["q_clipping_diagnostics"])

    @property
    def pseudo_summary(self) -> dict[str, Any]:
        return dict(self.diagnostics["pseudo_outcome_summary"])


def p_learner_score_hash(
    gamma_oof: np.ndarray, fold_ids: np.ndarray,
    bridge_config_hash: str, context_query_split_hash: str,
) -> str:
    """Hash the complete backend-independent P-Learner score contract."""
    digest = hashlib.sha256()
    digest.update(b"p-learner-shared-score-v1\0")
    for value in (gamma_oof, fold_ids):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    digest.update(bridge_config_hash.encode("ascii"))
    digest.update(context_query_split_hash.encode("ascii"))
    return digest.hexdigest()


def _extra_trees(seed: int, *, min_leaf: int = 5):
    from sklearn.ensemble import ExtraTreesRegressor

    return ExtraTreesRegressor(
        n_estimators=300, min_samples_leaf=min_leaf, max_features=0.8,
        n_jobs=1, random_state=seed,
    )


def p_learner_score(
    treatment: np.ndarray, q_factual: np.ndarray, outcome: np.ndarray,
    h_factual: np.ndarray, h0: np.ndarray, h1: np.ndarray,
) -> np.ndarray:
    """Literal P-Learner OOF score used by every final-stage backend."""
    values = [np.asarray(value, dtype=np.float64).reshape(-1) for value in
              (treatment, q_factual, outcome, h_factual, h0, h1)]
    if len({value.shape for value in values}) != 1:
        raise ValueError("P-Learner score components are not aligned")
    t, q, y, h, h_zero, h_one = values
    gamma = (2.0 * t - 1.0) * q * (y - h) + h_one - h_zero
    if not np.isfinite(gamma).all():
        raise RuntimeError("P-Learner produced non-finite cross-fitted pseudo outcomes")
    return gamma


def solve_regularized_binary_outcome_bridge(
    mu0: np.ndarray,
    mu1: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    ridge: float = 0.01,
    slope_clip: tuple[float, float] = (-10.0, 10.0),
    intercept_clip: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the repository's regularized two-point outcome bridge.

    Returns ``(h0, h1, raw_slope)``.  This is the vectorized slope/intercept
    implementation historically used inline by the ExtraTrees P-Learner; it
    is shared by nuisance backends so the bridge definition does not change.

    ``intercept_clip`` is ``None`` by default, which reproduces the historical
    unclipped behavior byte-for-byte (needed for ExtraTrees/TabPFN backends,
    whose ``mu0``/``mu1`` are already bounded by construction). Backends whose
    raw regressor can extrapolate outside the training-target range (e.g. a
    boosted-tree final stage) may opt into a symmetric bound here, applied
    identically to both ``h0`` and ``h1`` after the slope clip, as a
    numerical-stability safeguard analogous to ``slope_clip``.
    """
    mu_zero, mu_one, probability_zero, probability_one = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (mu0, mu1, p0, p1)
    ]
    if len({value.shape for value in (
        mu_zero, mu_one, probability_zero, probability_one,
    )}) != 1:
        raise ValueError("binary outcome-bridge inputs are not aligned")
    delta = probability_one - probability_zero
    raw_slope = (mu_one - mu_zero) * delta / (np.square(delta) + ridge)
    slope = np.clip(raw_slope, *slope_clip)
    intercept = 0.5 * (
        (mu_zero - slope * probability_zero)
        + (mu_one - slope * probability_one)
    )
    h0, h1 = intercept, intercept + slope
    if intercept_clip is not None:
        h0 = np.clip(h0, *intercept_clip)
        h1 = np.clip(h1, *intercept_clip)
    return h0, h1, raw_slope


def solve_regularized_binary_treatment_bridge(
    target0: np.ndarray,
    target1: np.ndarray,
    s0: np.ndarray,
    s1: np.ndarray,
    *,
    ridge: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the repository's regularized two-point treatment bridge.

    Clipping q to [-25, 25] is intentionally left to factual evaluation,
    matching the historical P-Learner implementation.
    """
    target_zero, target_one, probability_zero, probability_one = [
        np.asarray(value, dtype=np.float64).reshape(-1)
        for value in (target0, target1, s0, s1)
    ]
    if len({value.shape for value in (
        target_zero, target_one, probability_zero, probability_one,
    )}) != 1:
        raise ValueError("binary treatment-bridge inputs are not aligned")
    delta = probability_one - probability_zero
    slope = (target_one - target_zero) * delta / (np.square(delta) + ridge)
    intercept = 0.5 * (
        (target_zero - slope * probability_zero)
        + (target_one - slope * probability_one)
    )
    return intercept, intercept + slope


def _binary_operator_singular_values(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    matrices = np.empty((len(p0), 2, 2), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - p0
    matrices[:, 0, 1] = p0
    matrices[:, 1, 0] = 1.0 - p1
    matrices[:, 1, 1] = p1
    return np.linalg.svd(matrices, compute_uv=False)


def _residual_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_abs": float(np.mean(np.abs(array))),
        "rmse": float(np.sqrt(np.mean(np.square(array)))),
        "max_abs": float(np.max(np.abs(array))),
        "mean": float(np.mean(array)),
    }


def _operator_summary(singular_values: np.ndarray) -> dict[str, float]:
    minimum = np.asarray(singular_values, dtype=np.float64)[:, -1]
    maximum = np.asarray(singular_values, dtype=np.float64)[:, 0]
    condition = np.divide(
        maximum, minimum, out=np.full(len(minimum), np.inf), where=minimum > 0.0,
    )
    return {
        "min_singular_value": float(np.min(minimum)),
        "max_condition_number": float(np.max(condition)),
    }


def compute_p_learner_oof_score(
    arrays: dict[str, np.ndarray], context: np.ndarray, query_rows: np.ndarray, seed: int,
) -> PLearnerScoreBundle:
    """Fit the frozen bridges once and return the shared OOF proximal score."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold

    x = np.asarray(arrays["X"][context], dtype=np.float64)
    t = np.asarray(arrays["T"][context], dtype=np.int64)
    z = np.asarray(arrays["Z_P"][context], dtype=np.float64)
    w = np.asarray(arrays["W_P"][context], dtype=np.float64)
    y = np.asarray(arrays["Y"][context], dtype=np.float64)
    x_query = np.asarray(arrays["X"][query_rows], dtype=np.float64)
    folds = min(5, int(np.min(np.bincount(t))))
    if folds < 2:
        raise RuntimeError("P-Learner needs both treatment arms in at least two folds")
    ridge = 0.01
    pseudo = np.full(len(context), np.nan, dtype=np.float64)
    fold_ids = np.full(len(context), -1, dtype=np.int64)
    bridge_denominators: list[float] = []
    q_values: list[float] = []
    q_before_clip: list[float] = []
    q_after_clip: list[float] = []
    outcome_moment_residuals: list[float] = []
    treatment_moment_residuals: list[float] = []
    outcome_operator_singular_values: list[np.ndarray] = []
    treatment_operator_singular_values: list[np.ndarray] = []
    q_clip_trigger_count = 0
    q_value_count = 0
    propensity_clip_trigger_count = 0
    outcome_slope_clip_trigger_count = 0
    fit_started = time.perf_counter()

    def regressor(local_seed: int, min_leaf: int = 6):
        return _extra_trees(local_seed, min_leaf=min_leaf)

    def classifier(local_seed: int, min_leaf: int = 8):
        return ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=min_leaf, max_features=0.8,
            n_jobs=1, random_state=local_seed,
        )

    def probability(model: Any, features: np.ndarray) -> np.ndarray:
        classes = np.asarray(model.classes_)
        if not np.any(classes == 1):
            return np.zeros(len(features), dtype=np.float64)
        column = int(np.flatnonzero(classes == 1)[0])
        return np.asarray(model.predict_proba(features)[:, column], dtype=np.float64)

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (train_index, valid_index) in enumerate(splitter.split(x, t)):
        fold_ids[valid_index] = fold
        xv, zv, wv, tv, yv = x[valid_index], z[valid_index], w[valid_index], t[valid_index], y[valid_index]
        propensity = classifier(seed + 1000 * fold + 1).fit(
            np.column_stack([x[train_index], w[train_index]]), t[train_index],
        )
        h_by_arm: dict[int, np.ndarray] = {}
        h_observed_by_arm: dict[int, np.ndarray] = {}
        q_by_arm: dict[int, np.ndarray] = {}
        for treatment in (0, 1):
            arm_train = train_index[t[train_index] == treatment]
            if len(arm_train) < 50:
                raise RuntimeError(f"P-Learner treatment arm {treatment} too small: {len(arm_train)}")

            outcome_regression = regressor(seed + 1000 * fold + 10 + treatment).fit(
                np.column_stack([x[arm_train], z[arm_train]]), y[arm_train],
            )
            w_regression = regressor(seed + 1000 * fold + 20 + treatment).fit(
                np.column_stack([x[arm_train], z[arm_train]]), w[arm_train],
            )
            z0 = np.zeros(len(xv), dtype=np.float64)
            z1 = np.ones(len(xv), dtype=np.float64)
            mu_y0 = outcome_regression.predict(np.column_stack([xv, z0]))
            mu_y1 = outcome_regression.predict(np.column_stack([xv, z1]))
            p_w0 = np.clip(w_regression.predict(np.column_stack([xv, z0])), 0.0, 1.0)
            p_w1 = np.clip(w_regression.predict(np.column_stack([xv, z1])), 0.0, 1.0)
            delta_w = p_w1 - p_w0
            h_zero, h_one, raw_h_slope = solve_regularized_binary_outcome_bridge(
                mu_y0, mu_y1, p_w0, p_w1, ridge=ridge,
            )
            outcome_slope_clip_trigger_count += int(np.sum((raw_h_slope < -10.0) | (raw_h_slope > 10.0)))
            h_slope = h_one - h_zero
            h_intercept = h_zero
            h_observed_by_arm[treatment] = np.where(wv == 1.0, h_one, h_zero)
            h_by_arm[treatment] = h_observed_by_arm[treatment]
            bridge_denominators.extend(np.abs(delta_w).tolist())
            outcome_moment_residuals.extend((mu_y0 - (h_intercept + h_slope * p_w0)).tolist())
            outcome_moment_residuals.extend((mu_y1 - (h_intercept + h_slope * p_w1)).tolist())
            outcome_operator_singular_values.append(_binary_operator_singular_values(p_w0, p_w1))

            z_regression = classifier(seed + 1000 * fold + 30 + treatment).fit(
                np.column_stack([x[arm_train], w[arm_train]]), z[arm_train].astype(np.int64),
            )
            w0 = np.zeros(len(xv), dtype=np.float64)
            w1 = np.ones(len(xv), dtype=np.float64)
            p_z0 = probability(z_regression, np.column_stack([xv, w0]))
            p_z1 = probability(z_regression, np.column_stack([xv, w1]))
            raw_e1_w0 = probability(propensity, np.column_stack([xv, w0]))
            raw_e1_w1 = probability(propensity, np.column_stack([xv, w1]))
            propensity_clip_trigger_count += int(np.sum((raw_e1_w0 < 0.02) | (raw_e1_w0 > 0.98)))
            propensity_clip_trigger_count += int(np.sum((raw_e1_w1 < 0.02) | (raw_e1_w1 > 0.98)))
            e1_w0 = np.clip(raw_e1_w0, 0.02, 0.98)
            e1_w1 = np.clip(raw_e1_w1, 0.02, 0.98)
            target0 = 1.0 / (e1_w0 if treatment == 1 else 1.0 - e1_w0)
            target1 = 1.0 / (e1_w1 if treatment == 1 else 1.0 - e1_w1)
            delta_z = p_z1 - p_z0
            q_zero, q_one = solve_regularized_binary_treatment_bridge(
                target0, target1, p_z0, p_z1, ridge=ridge,
            )
            q_slope = q_one - q_zero
            q_intercept = q_zero
            raw_q = np.where(zv == 1.0, q_one, q_zero)
            q_clip_trigger_count += int(np.sum((raw_q < -25.0) | (raw_q > 25.0)))
            q_value_count += len(raw_q)
            clipped_q = np.clip(raw_q, -25.0, 25.0)
            q_before_clip.extend(raw_q.tolist())
            q_after_clip.extend(clipped_q.tolist())
            q_by_arm[treatment] = clipped_q
            treatment_moment_residuals.extend((target0 - (q_intercept + q_slope * p_z0)).tolist())
            treatment_moment_residuals.extend((target1 - (q_intercept + q_slope * p_z1)).tolist())
            treatment_operator_singular_values.append(_binary_operator_singular_values(p_z0, p_z1))

        h_observed = np.where(tv == 1, h_observed_by_arm[1], h_observed_by_arm[0])
        q_observed = np.where(tv == 1, q_by_arm[1], q_by_arm[0])
        pseudo[valid_index] = p_learner_score(
            tv, q_observed, yv, h_observed, h_by_arm[0], h_by_arm[1],
        )
        q_values.extend(q_observed.tolist())

    if np.any(fold_ids < 0) or not np.isfinite(pseudo).all():
        raise RuntimeError("P-Learner OOF score/fold assignment is incomplete")
    outcome_singular = np.concatenate(outcome_operator_singular_values, axis=0)
    treatment_singular = np.concatenate(treatment_operator_singular_values, axis=0)
    all_singular = np.concatenate([outcome_singular, treatment_singular], axis=0)
    outcome_residual_summary = _residual_summary(outcome_moment_residuals)
    treatment_residual_summary = _residual_summary(treatment_moment_residuals)
    outcome_operator_summary = _operator_summary(outcome_singular)
    treatment_operator_summary = _operator_summary(treatment_singular)
    all_operator_summary = _operator_summary(all_singular)
    q_before = np.asarray(q_before_clip, dtype=np.float64)
    q_after = np.asarray(q_after_clip, dtype=np.float64)
    q_clipping_diagnostics = {
        "q_clip_lower": -25.0,
        "q_clip_upper": 25.0,
        "q_clip_trigger_count": q_clip_trigger_count,
        "q_clip_trigger_fraction": float(q_clip_trigger_count / q_value_count),
        "q_before_clip_min": float(np.min(q_before)),
        "q_before_clip_max": float(np.max(q_before)),
        "q_after_clip_min": float(np.min(q_after)),
        "q_after_clip_max": float(np.max(q_after)),
    }
    pseudo_summary = {
        "gamma_mean": float(np.mean(pseudo)),
        "gamma_std": float(np.std(pseudo)),
        "gamma_min": float(np.min(pseudo)),
        "gamma_q01": float(np.quantile(pseudo, 0.01)),
        "gamma_q05": float(np.quantile(pseudo, 0.05)),
        "gamma_median": float(np.quantile(pseudo, 0.50)),
        "gamma_q95": float(np.quantile(pseudo, 0.95)),
        "gamma_q99": float(np.quantile(pseudo, 0.99)),
        "gamma_max": float(np.max(pseudo)),
        "nonfinite_gamma_count": int(np.sum(~np.isfinite(pseudo))),
    }
    fold_counts = {
        str(fold): int(np.sum(fold_ids == fold)) for fold in np.unique(fold_ids)
    }
    diagnostics = {
        "estimator": "P-Learner (Algorithm 1, cross-fitted proximal pseudo-outcome)",
        "reference": "Sverdrup and Cui, ICML 2023, PMLR 202:33285-33298 (arXiv:2301.10913)",
        "author_reference_code_available": False,
        "identification_wrapper_used": True,
        "bridge_backend": "ExtraTrees",
        "nuisance_backend": "regularized binary-proxy bridge equations with sklearn ExtraTrees",
        "cross_fitting_folds": folds, "crossfit_folds": folds, "ridge": ridge,
        "fold_counts": fold_counts,
        "q_clip_bounds": [-25.0, 25.0], "q_clip_trigger_count": q_clip_trigger_count,
        "q_clip_trigger_fraction": q_clipping_diagnostics["q_clip_trigger_fraction"],
        "q_clipping_diagnostics": q_clipping_diagnostics,
        "q_value_count": q_value_count,
        "propensity_clip_bounds": [0.02, 0.98],
        "propensity_clip_trigger_count": propensity_clip_trigger_count,
        "outcome_bridge_slope_clip_bounds": [-10.0, 10.0],
        "outcome_bridge_slope_clip_trigger_count": outcome_slope_clip_trigger_count,
        "pseudo_outcome_quantiles": np.quantile(pseudo, [0.0, 0.05, 0.5, 0.95, 1.0]).tolist(),
        "pseudo_outcome_summary": pseudo_summary,
        "abs_outcome_bridge_denominator_quantiles": np.quantile(bridge_denominators, [0.05, 0.5, 0.95]).tolist(),
        "treatment_bridge_q_quantiles": np.quantile(q_values, [0.05, 0.5, 0.95]).tolist(),
        "outcome_bridge_moment_residual": outcome_residual_summary,
        "treatment_bridge_moment_residual": treatment_residual_summary,
        "outcome_bridge_diagnostics": {
            "moment_residual": outcome_residual_summary,
            **outcome_operator_summary,
        },
        "treatment_bridge_diagnostics": {
            "moment_residual": treatment_residual_summary,
            **treatment_operator_summary,
        },
        "conditioning_diagnostic_representation": (
            "per-row 2x2 binary conditional-probability operator implied by the "
            "learned two-point linear bridge before ridge regularization"
        ),
        "bridge_min_singular_value": all_operator_summary["min_singular_value"],
        "bridge_max_condition_number": all_operator_summary["max_condition_number"],
        "bridge_fit_time_sec": float(time.perf_counter() - fit_started),
        "final_query_schema": ["X"], "proxy_in_final_query": False,
    }
    bridge_config_hash = _config_hash(P_LEARNER_BRIDGE_CONFIG)
    context_query_split_hash = array_hash(
        np.asarray(context, dtype=np.int64), np.asarray(query_rows, dtype=np.int64),
        np.asarray(arrays["unit_id"])[context], np.asarray(arrays["unit_id"])[query_rows],
    )
    gamma_hash = array_hash(pseudo)
    return PLearnerScoreBundle(
        x_context=x, x_query=x_query, gamma_oof=pseudo, fold_ids=fold_ids,
        gamma_hash=gamma_hash,
        score_hash=p_learner_score_hash(
            pseudo, fold_ids, bridge_config_hash, context_query_split_hash,
        ),
        fold_hash=array_hash(fold_ids), bridge_config_hash=bridge_config_hash,
        context_query_split_hash=context_query_split_hash, diagnostics=diagnostics,
    )


def fit_existing_p_learner_final_stage(
    bundle: PLearnerScoreBundle, seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the frozen ExtraTrees final stage to an already computed score."""
    final_fit_started = time.perf_counter()
    final_model = _extra_trees(seed + 900000, min_leaf=10).fit(
        bundle.x_context, bundle.gamma_oof,
    )
    final_fit_time = time.perf_counter() - final_fit_started
    prediction_started = time.perf_counter()
    prediction = np.asarray(final_model.predict(bundle.x_query), dtype=np.float64)
    prediction_time = time.perf_counter() - prediction_started
    if prediction.shape != (len(bundle.x_query),) or not np.isfinite(prediction).all():
        raise ValueError("existing P-Learner final stage returned invalid predictions")
    return prediction, {
        "final_backend": "ExtraTrees",
        "final_backend_parameters": {
            "n_estimators": 300,
            "min_samples_leaf": 10,
            "max_features": 0.8,
            "n_jobs": 1,
            "random_state": int(seed + 900000),
        },
        "final_fit_time_sec": float(final_fit_time),
        "inference_time_sec": float(prediction_time),
    }


def run(arrays: dict[str, np.ndarray], context: np.ndarray, query_rows: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Existing ExtraTrees P-Learner, now consuming the shared score API."""
    bundle = compute_p_learner_oof_score(arrays, context, query_rows, seed)
    prediction, final_metadata = fit_existing_p_learner_final_stage(bundle, seed)
    metadata = dict(bundle.diagnostics)
    metadata.update({
        "parent_score_model_id": "p_learner",
        "score_artifact_hash": bundle.score_hash,
        "score_hash": bundle.score_hash, "fold_hash": bundle.fold_hash,
        "gamma_hash": bundle.gamma_hash,
        "bridge_config_hash": bundle.bridge_config_hash,
        "context_query_split_hash": bundle.context_query_split_hash,
        **final_metadata,
        "fit_time_sec": float(
            metadata.pop("bridge_fit_time_sec") + final_metadata["final_fit_time_sec"]
        ),
    })
    return prediction, metadata
