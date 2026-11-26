"""Full frozen-TabPFN nuisance P-Learner.

TabPFN estimates only observable conditional means/probabilities and the final
conditional mean.  The outcome and treatment bridges are explicit calls to the
same regularized binary-proxy equations used by the ExtraTrees P-Learner.
Calling sklearn-compatible ``fit`` conditions frozen pretrained TabPFN-v3 on
the fold-specific context; there is no parameter update or fine-tuning.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Callable, Mapping

import numpy as np

from baselines.meta.common import conditional_mean

from ..frontdoor import _release
from data.integrity import array_hash
from .extra_trees import (
    PLearnerScoreBundle,
    _binary_operator_singular_values,
    _operator_summary,
    _residual_summary,
    p_learner_score,
    solve_regularized_binary_outcome_bridge,
    solve_regularized_binary_treatment_bridge,
)


FULL_TABPFN_MODEL_ID = "p_learner_tabpfn_v3_full"
FULL_TABPFN_DISPLAY_NAME = "Full TabPFN-nuisance P-Learner"
FULL_TABPFN_NUISANCE_BACKEND_ID = "tabpfn_v3_full"
FULL_TABPFN_BRIDGE_CONFIG = {
    "algorithm": "Sverdrup-Cui-Algorithm-1",
    "ridge": 0.01,
    "outcome_bridge_slope_clip": [-10.0, 10.0],
    "propensity_clip": [0.02, 0.98],
    "treatment_bridge_q_clip": [-25.0, 25.0],
}
FULL_TABPFN_NUISANCE_CONFIG = {
    "backend_id": FULL_TABPFN_NUISANCE_BACKEND_ID,
    "crossfit_folds": 5,
    "stratification_target": "T",
    "shuffle": True,
    "constant_class_policy": "fail_with_informative_error",
    "estimator_contexts_per_fold": 7,
    "independent_context_per_nuisance_and_arm": True,
    "counterfactual_proxy_queries_batched": True,
}


@dataclass(frozen=True)
class FullPLearnerBackendSpec:
    """Identity and locked configuration for a full P-Learner backend."""

    model_id: str
    display_name: str
    nuisance_backend_id: str
    nuisance_backend_name: str
    estimator_name: str
    score_namespace: str
    nuisance_config: Mapping[str, Any]
    bridge_config: Mapping[str, Any]


FULL_TABPFN_BACKEND_SPEC = FullPLearnerBackendSpec(
    model_id=FULL_TABPFN_MODEL_ID,
    display_name=FULL_TABPFN_DISPLAY_NAME,
    nuisance_backend_id=FULL_TABPFN_NUISANCE_BACKEND_ID,
    nuisance_backend_name="frozen pretrained TabPFN-v3 in-context estimation",
    estimator_name="P-Learner with frozen pretrained TabPFN-v3 nuisances",
    score_namespace="p-learner-tabpfn-v3-full-score-v1",
    nuisance_config=FULL_TABPFN_NUISANCE_CONFIG,
    bridge_config=FULL_TABPFN_BRIDGE_CONFIG,
)


def _metadata_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def full_tabpfn_score_hash(
    gamma_oof: np.ndarray,
    fold_ids: np.ndarray,
    *,
    input_hash: str,
    context_query_split_hash: str,
    nuisance_config_hash: str,
    tabpfn_source_revision_hash: str,
    bridge_config_hash: str,
    route_seed: int,
) -> str:
    """Hash the complete full-nuisance score contract."""
    return full_backend_score_hash(
        gamma_oof,
        fold_ids,
        input_hash=input_hash,
        context_query_split_hash=context_query_split_hash,
        nuisance_config_hash=nuisance_config_hash,
        backend_source_revision_hash=tabpfn_source_revision_hash,
        bridge_config_hash=bridge_config_hash,
        route_seed=route_seed,
        score_namespace=FULL_TABPFN_BACKEND_SPEC.score_namespace,
    )


def full_backend_score_hash(
    gamma_oof: np.ndarray,
    fold_ids: np.ndarray,
    *,
    input_hash: str,
    context_query_split_hash: str,
    nuisance_config_hash: str,
    backend_source_revision_hash: str,
    bridge_config_hash: str,
    route_seed: int,
    score_namespace: str,
) -> str:
    """Hash a full-backend P-Learner score without mixing backend families."""
    digest = hashlib.sha256()
    digest.update(score_namespace.encode("utf-8"))
    digest.update(b"\0")
    for value in (gamma_oof, fold_ids):
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(array.tobytes(order="C"))
    for value in (
        input_hash,
        context_query_split_hash,
        nuisance_config_hash,
        backend_source_revision_hash,
        bridge_config_hash,
        str(int(route_seed)),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FullTabPFNScoreResult:
    bundle: PLearnerScoreBundle
    input_hash: str
    nuisance_config_hash: str
    tabpfn_source_revision_hash: str
    route_seed: int


@dataclass(frozen=True)
class FullTabPFNFinalResult:
    tau_hat: np.ndarray
    model: Any
    timing: dict[str, float]


def _two_level_query(x: np.ndarray) -> np.ndarray:
    zeros = np.zeros((len(x), 1), dtype=np.float64)
    ones = np.ones((len(x), 1), dtype=np.float64)
    return np.concatenate(
        [np.column_stack([x, zeros]), np.column_stack([x, ones])], axis=0,
    )


def _split_two_level_prediction(values: np.ndarray, rows: int) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(values, dtype=np.float64).reshape(-1)
    if prediction.shape != (2 * rows,):
        raise ValueError("batched binary-proxy prediction has invalid shape")
    return prediction[:rows], prediction[rows:]


def _binary_counts(target: np.ndarray) -> dict[str, int]:
    values = np.asarray(target, dtype=np.int64).reshape(-1)
    return {"0": int(np.sum(values == 0)), "1": int(np.sum(values == 1))}


def _require_two_classes(target: np.ndarray, *, role: str, fold: int, arm: int | None) -> None:
    counts = _binary_counts(target)
    if not counts["0"] or not counts["1"]:
        suffix = "" if arm is None else f", treatment_arm={arm}"
        raise RuntimeError(
            f"full TabPFN nuisance requires both binary classes: role={role}, "
            f"fold={fold}{suffix}, counts={counts}; configured policy is "
            "fail_with_informative_error (no ExtraTrees fallback)"
        )


def _probability(model: Any, features: np.ndarray) -> np.ndarray:
    """Use the project's class-1-safe common conditional-mean helper."""
    return np.clip(
        np.asarray(conditional_mean(model, features, "binary"), dtype=np.float64),
        0.0,
        1.0,
    )


def _probability_summary(values: list[np.ndarray]) -> dict[str, float | int]:
    array = np.concatenate([np.asarray(value, dtype=np.float64) for value in values])
    return {
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "nonfinite_count": int(np.sum(~np.isfinite(array))),
    }


def compute_full_tabpfn_oof_score(
    X_context: np.ndarray,
    Z_context: np.ndarray,
    W_context: np.ndarray,
    T_context: np.ndarray,
    Y_context: np.ndarray,
    X_query: np.ndarray,
    *,
    route_seed: int,
    classifier_factory: Callable[[int, str], Any],
    regressor_factory: Callable[[int, str], Any],
    tabpfn_metadata: Mapping[str, Any],
    input_hash: str,
    context_query_split_hash: str,
    backend_spec: FullPLearnerBackendSpec = FULL_TABPFN_BACKEND_SPEC,
    outcome_bridge_intercept_clip: tuple[float, float] | None = None,
) -> FullTabPFNScoreResult:
    """Create leakage-safe OOF Gamma using frozen TabPFN nuisance contexts only.

    Oracle tau, potential outcomes, and latent variables are absent from this
    estimator-facing signature by construction.
    """
    from sklearn.model_selection import StratifiedKFold

    x = np.asarray(X_context, dtype=np.float64)
    z = np.asarray(Z_context, dtype=np.int64).reshape(-1)
    w = np.asarray(W_context, dtype=np.int64).reshape(-1)
    t = np.asarray(T_context, dtype=np.int64).reshape(-1)
    y = np.asarray(Y_context, dtype=np.float64).reshape(-1)
    x_query = np.asarray(X_query, dtype=np.float64)
    n = len(x)
    if x.ndim != 2 or x_query.ndim != 2 or x.shape[1] != x_query.shape[1]:
        raise ValueError("full TabPFN P-Learner requires aligned 2D context/query X")
    if any(value.shape != (n,) for value in (z, w, t, y)):
        raise ValueError("full TabPFN P-Learner context arrays are not row aligned")
    if set(np.unique(t)) != {0, 1} or not set(np.unique(z)).issubset({0, 1}) or not set(np.unique(w)).issubset({0, 1}):
        raise ValueError("T, Z_P, and W_P must be binary")
    if not all(np.isfinite(value).all() for value in (x, z, w, t, y, x_query)):
        raise ValueError("full TabPFN P-Learner inputs must be finite")
    if int(np.min(np.bincount(t))) < 5:
        raise RuntimeError("full TabPFN P-Learner requires five stratified folds")

    nuisance_config_hash = _metadata_hash(backend_spec.nuisance_config)
    bridge_config_hash = _metadata_hash(backend_spec.bridge_config)
    source_fields = {
        key: value for key, value in dict(tabpfn_metadata).items()
        if key in {
            "base_learner", "package_version", "tabpfn_model_identifier",
            "tabpfn_revision", "tabpfn_ensemble_size", "tabpfn_device",
            "tabpfn_checkpoint_filenames", "hpo_performed", "parameter_updates",
            "backend", "locked_parameters", "random_state",
        }
    }
    tabpfn_source_revision_hash = _metadata_hash(source_fields)
    pseudo = np.full(n, np.nan, dtype=np.float64)
    fold_ids = np.full(n, -1, dtype=np.int64)
    assignment_counts = np.zeros(n, dtype=np.int64)
    fold_diagnostics: list[dict[str, Any]] = []
    outcome_residuals: list[float] = []
    treatment_residuals: list[float] = []
    outcome_singular: list[np.ndarray] = []
    treatment_singular: list[np.ndarray] = []
    outcome_probabilities: list[np.ndarray] = []
    treatment_probabilities: list[np.ndarray] = []
    raw_propensities: list[np.ndarray] = []
    clipped_propensities: list[np.ndarray] = []
    q_raw_values: list[np.ndarray] = []
    q_clipped_values: list[np.ndarray] = []
    h_slope_clip_count = 0
    h_intercept_clip_count = 0
    q_clip_count = 0
    propensity_clip_count = 0
    bridge_nonfinite_count = 0
    timing = {
        "propensity_sec": 0.0,
        "outcome_reduced_form_sec": 0.0,
        "w_operator_sec": 0.0,
        "z_operator_sec": 0.0,
        "bridge_solve_sec": 0.0,
        "score_construction_sec": 0.0,
    }
    started_total = time.perf_counter()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(route_seed))
    for fold, (train_index, valid_index) in enumerate(splitter.split(x, t)):
        overlap = np.intersect1d(train_index, valid_index)
        if len(overlap):
            raise RuntimeError(f"cross-fit leakage in fold {fold}: {len(overlap)} rows")
        if np.any(assignment_counts[valid_index]):
            raise RuntimeError(f"duplicate validation assignment in fold {fold}")
        assignment_counts[valid_index] += 1
        fold_ids[valid_index] = fold
        xv = x[valid_index]
        zv = z[valid_index]
        wv = w[valid_index]
        tv = t[valid_index]
        yv = y[valid_index]
        query_w = _two_level_query(xv)
        query_z = query_w

        fold_record: dict[str, Any] = {
            "fold": fold,
            "train_size": int(len(train_index)),
            "validation_size": int(len(valid_index)),
            "train_validation_overlap_count": int(len(overlap)),
            "treatment_counts_train": _binary_counts(t[train_index]),
            "treatment_counts_validation": _binary_counts(tv),
            "proxy_class_counts_by_arm": {},
        }
        _require_two_classes(t[train_index], role="propensity_T", fold=fold, arm=None)
        nuisance_started = time.perf_counter()
        propensity = classifier_factory(
            route_seed + 1000 * fold + 1, f"full-p-learner-fold-{fold}-propensity",
        )
        propensity.fit(np.column_stack([x[train_index], w[train_index]]), t[train_index])
        raw_e0, raw_e1 = _split_two_level_prediction(
            _probability(propensity, query_w), len(xv),
        )
        timing["propensity_sec"] += time.perf_counter() - nuisance_started
        _release(propensity)
        raw_propensities.extend([raw_e0, raw_e1])
        propensity_clip_count += int(np.sum((raw_e0 < 0.02) | (raw_e0 > 0.98)))
        propensity_clip_count += int(np.sum((raw_e1 < 0.02) | (raw_e1 > 0.98)))
        e0 = np.clip(raw_e0, 0.02, 0.98)
        e1 = np.clip(raw_e1, 0.02, 0.98)
        clipped_propensities.extend([e0, e1])

        h_by_arm: dict[int, np.ndarray] = {}
        q_by_arm: dict[int, np.ndarray] = {}
        for arm in (0, 1):
            arm_train = train_index[t[train_index] == arm]
            fold_record["proxy_class_counts_by_arm"][str(arm)] = {
                "rows": int(len(arm_train)),
                "W_P": _binary_counts(w[arm_train]),
                "Z_P": _binary_counts(z[arm_train]),
            }
            _require_two_classes(w[arm_train], role="W_operator", fold=fold, arm=arm)
            _require_two_classes(z[arm_train], role="Z_operator", fold=fold, arm=arm)

            nuisance_started = time.perf_counter()
            outcome_model = regressor_factory(
                route_seed + 1000 * fold + 10 + arm,
                f"full-p-learner-fold-{fold}-arm-{arm}-outcome",
            )
            outcome_model.fit(np.column_stack([x[arm_train], z[arm_train]]), y[arm_train])
            mu0, mu1 = _split_two_level_prediction(
                outcome_model.predict(query_z), len(xv),
            )
            timing["outcome_reduced_form_sec"] += time.perf_counter() - nuisance_started
            _release(outcome_model)

            nuisance_started = time.perf_counter()
            w_model = classifier_factory(
                route_seed + 1000 * fold + 20 + arm,
                f"full-p-learner-fold-{fold}-arm-{arm}-w-operator",
            )
            w_model.fit(np.column_stack([x[arm_train], z[arm_train]]), w[arm_train])
            p_w0, p_w1 = _split_two_level_prediction(
                _probability(w_model, query_z), len(xv),
            )
            timing["w_operator_sec"] += time.perf_counter() - nuisance_started
            _release(w_model)
            outcome_probabilities.extend([p_w0, p_w1])

            nuisance_started = time.perf_counter()
            z_model = classifier_factory(
                route_seed + 1000 * fold + 30 + arm,
                f"full-p-learner-fold-{fold}-arm-{arm}-z-operator",
            )
            z_model.fit(np.column_stack([x[arm_train], w[arm_train]]), z[arm_train])
            p_z0, p_z1 = _split_two_level_prediction(
                _probability(z_model, query_w), len(xv),
            )
            timing["z_operator_sec"] += time.perf_counter() - nuisance_started
            _release(z_model)
            treatment_probabilities.extend([p_z0, p_z1])

            bridge_started = time.perf_counter()
            h0, h1, raw_h_slope = solve_regularized_binary_outcome_bridge(
                mu0, mu1, p_w0, p_w1,
                ridge=float(backend_spec.bridge_config["ridge"]),
                intercept_clip=outcome_bridge_intercept_clip,
            )
            h_slope_clip_count += int(np.sum((raw_h_slope < -10.0) | (raw_h_slope > 10.0)))
            if outcome_bridge_intercept_clip is not None:
                lower, upper = outcome_bridge_intercept_clip
                h_intercept_clip_count += int(
                    np.sum((h0 <= lower) | (h0 >= upper))
                    + np.sum((h1 <= lower) | (h1 >= upper))
                )
            h_by_arm[arm] = np.where(wv == 1, h1, h0)
            fitted_mu0 = (1.0 - p_w0) * h0 + p_w0 * h1
            fitted_mu1 = (1.0 - p_w1) * h0 + p_w1 * h1
            outcome_residuals.extend((mu0 - fitted_mu0).tolist())
            outcome_residuals.extend((mu1 - fitted_mu1).tolist())
            outcome_singular.append(_binary_operator_singular_values(p_w0, p_w1))

            target0 = 1.0 / (e0 if arm == 1 else 1.0 - e0)
            target1 = 1.0 / (e1 if arm == 1 else 1.0 - e1)
            q0, q1 = solve_regularized_binary_treatment_bridge(
                target0, target1, p_z0, p_z1,
                ridge=float(backend_spec.bridge_config["ridge"]),
            )
            raw_q = np.where(zv == 1, q1, q0)
            q_raw_values.append(raw_q)
            q_clip_count += int(np.sum((raw_q < -25.0) | (raw_q > 25.0)))
            clipped_q = np.clip(raw_q, -25.0, 25.0)
            q_clipped_values.append(clipped_q)
            q_by_arm[arm] = clipped_q
            fitted_target0 = (1.0 - p_z0) * q0 + p_z0 * q1
            fitted_target1 = (1.0 - p_z1) * q0 + p_z1 * q1
            treatment_residuals.extend((target0 - fitted_target0).tolist())
            treatment_residuals.extend((target1 - fitted_target1).tolist())
            treatment_singular.append(_binary_operator_singular_values(p_z0, p_z1))
            bridge_nonfinite_count += int(sum(
                np.sum(~np.isfinite(value)) for value in (h0, h1, q0, q1)
            ))
            timing["bridge_solve_sec"] += time.perf_counter() - bridge_started

        score_started = time.perf_counter()
        h_factual = np.where(tv == 1, h_by_arm[1], h_by_arm[0])
        q_factual = np.where(tv == 1, q_by_arm[1], q_by_arm[0])
        pseudo[valid_index] = p_learner_score(
            tv, q_factual, yv, h_factual, h_by_arm[0], h_by_arm[1],
        )
        timing["score_construction_sec"] += time.perf_counter() - score_started
        fold_diagnostics.append(fold_record)

    zero_assignments = int(np.sum(assignment_counts == 0))
    duplicate_assignments = int(np.sum(assignment_counts > 1))
    if zero_assignments or duplicate_assignments or np.any(fold_ids < 0):
        raise RuntimeError(
            f"incomplete OOF assignment: zero={zero_assignments}, duplicate={duplicate_assignments}"
        )
    if not np.isfinite(pseudo).all():
        raise RuntimeError("full TabPFN P-Learner produced non-finite Gamma")

    outcome_singular_array = np.concatenate(outcome_singular, axis=0)
    treatment_singular_array = np.concatenate(treatment_singular, axis=0)
    raw_propensity_array = np.concatenate(raw_propensities)
    clipped_propensity_array = np.concatenate(clipped_propensities)
    raw_q_array = np.concatenate(q_raw_values)
    clipped_q_array = np.concatenate(q_clipped_values)
    gamma_hash = array_hash(pseudo)
    fold_hash = array_hash(fold_ids)
    score_hash = full_backend_score_hash(
        pseudo,
        fold_ids,
        input_hash=input_hash,
        context_query_split_hash=context_query_split_hash,
        nuisance_config_hash=nuisance_config_hash,
        backend_source_revision_hash=tabpfn_source_revision_hash,
        bridge_config_hash=bridge_config_hash,
        route_seed=route_seed,
        score_namespace=backend_spec.score_namespace,
    )
    propensity_count = len(raw_propensity_array)
    nuisance_total = sum(timing[key] for key in (
        "propensity_sec", "outcome_reduced_form_sec", "w_operator_sec", "z_operator_sec",
    ))
    timing["nuisance_total_sec"] = float(nuisance_total)
    timing["score_stage_wall_sec"] = float(time.perf_counter() - started_total)
    diagnostics: dict[str, Any] = {
        "model_id": backend_spec.model_id,
        "display_name": backend_spec.display_name,
        "estimator": backend_spec.estimator_name,
        "reference": "Sverdrup and Cui, ICML 2023, PMLR 202:33285-33298 (arXiv:2301.10913)",
        "nuisance_backend_id": backend_spec.nuisance_backend_id,
        "nuisance_backend": backend_spec.nuisance_backend_name,
        "bridge_backend": "explicit regularized binary-proxy equations",
        "parameter_updates": False,
        "fine_tuning": False,
        "crossfit_folds": 5,
        "fold_counts": {str(i): int(np.sum(fold_ids == i)) for i in range(5)},
        "fold_diagnostics": fold_diagnostics,
        "train_validation_overlap_count": int(sum(
            item["train_validation_overlap_count"] for item in fold_diagnostics
        )),
        "rows_receiving_zero_gamma": zero_assignments,
        "rows_receiving_multiple_gamma": duplicate_assignments,
        "constant_class_policy": backend_spec.nuisance_config["constant_class_policy"],
        "constant_class_trigger_count": 0,
        "propensity_diagnostics": {
            "raw_min": float(np.min(raw_propensity_array)),
            "raw_max": float(np.max(raw_propensity_array)),
            "clipped_min": float(np.min(clipped_propensity_array)),
            "clipped_max": float(np.max(clipped_propensity_array)),
            "clipping_count": propensity_clip_count,
            "clipping_fraction": float(propensity_clip_count / propensity_count),
        },
        "outcome_operator_diagnostics": {
            **_operator_summary(outcome_singular_array),
            **_probability_summary(outcome_probabilities),
        },
        "treatment_operator_diagnostics": {
            **_operator_summary(treatment_singular_array),
            **_probability_summary(treatment_probabilities),
        },
        "outcome_bridge_diagnostics": {
            "moment_residual": _residual_summary(outcome_residuals),
            "slope_clipping_count": h_slope_clip_count,
            "intercept_clip_bounds": (
                list(outcome_bridge_intercept_clip)
                if outcome_bridge_intercept_clip is not None else None
            ),
            "intercept_clipping_count": h_intercept_clip_count,
            "nonfinite_bridge_count": bridge_nonfinite_count,
        },
        "treatment_bridge_diagnostics": {
            "moment_residual": _residual_summary(treatment_residuals),
            "q_clipping_count": q_clip_count,
            "q_before_clip_min": float(np.min(raw_q_array)),
            "q_before_clip_max": float(np.max(raw_q_array)),
            "q_after_clip_min": float(np.min(clipped_q_array)),
            "q_after_clip_max": float(np.max(clipped_q_array)),
            "nonfinite_bridge_count": bridge_nonfinite_count,
        },
        "pseudo_outcome_summary": {
            "gamma_mean": float(np.mean(pseudo)),
            "gamma_std": float(np.std(pseudo)),
            "gamma_min": float(np.min(pseudo)),
            "gamma_max": float(np.max(pseudo)),
            "nonfinite_gamma_count": int(np.sum(~np.isfinite(pseudo))),
            "score_hash": score_hash,
        },
        "runtime": timing,
        "runtime_boundary": (
            "nuisance timers include each context-conditioning fit and batched validation "
            "prediction; bridge/score are separate; final regression is timed by the runner"
        ),
        "final_query_schema": ["X"],
        "proxy_in_final_query": False,
        "oracle_used_in_score_construction": False,
        "backend_metadata": dict(tabpfn_metadata),
        "nuisance_config_hash": nuisance_config_hash,
        "tabpfn_source_revision_hash": tabpfn_source_revision_hash,
        "backend_source_revision_hash": tabpfn_source_revision_hash,
        "bridge_config_hash": bridge_config_hash,
        "input_hash": input_hash,
        "context_query_split_hash": context_query_split_hash,
        "score_hash": score_hash,
        **(
            {"tabpfn_metadata": dict(tabpfn_metadata)}
            if backend_spec.nuisance_backend_id == FULL_TABPFN_NUISANCE_BACKEND_ID
            else {}
        ),
    }
    bundle = PLearnerScoreBundle(
        x_context=x,
        x_query=x_query,
        gamma_oof=pseudo,
        fold_ids=fold_ids,
        gamma_hash=gamma_hash,
        score_hash=score_hash,
        fold_hash=fold_hash,
        bridge_config_hash=bridge_config_hash,
        context_query_split_hash=context_query_split_hash,
        diagnostics=diagnostics,
    )
    return FullTabPFNScoreResult(
        bundle=bundle,
        input_hash=input_hash,
        nuisance_config_hash=nuisance_config_hash,
        tabpfn_source_revision_hash=tabpfn_source_revision_hash,
        route_seed=int(route_seed),
    )


def fit_full_tabpfn_final_stage(
    X_context: np.ndarray,
    Gamma_oof: np.ndarray,
    X_query: np.ndarray,
    *,
    regressor_factory: Callable[[], Any],
) -> FullTabPFNFinalResult:
    """Condition frozen TabPFN on ``X_context -> Gamma`` and query X only."""
    x = np.asarray(X_context, dtype=np.float64)
    gamma = np.asarray(Gamma_oof, dtype=np.float64).reshape(-1)
    x_query = np.asarray(X_query, dtype=np.float64)
    if x.ndim != 2 or x_query.ndim != 2 or x.shape[1] != x_query.shape[1]:
        raise ValueError("final full TabPFN context/query X is invalid")
    if gamma.shape != (len(x),):
        raise ValueError("final full TabPFN Gamma is not context aligned")
    if not all(np.isfinite(value).all() for value in (x, gamma, x_query)):
        raise ValueError("final full TabPFN inputs must be finite")
    model = regressor_factory()
    fit_started = time.perf_counter()
    model.fit(x, gamma)
    fit_time = time.perf_counter() - fit_started
    predict_started = time.perf_counter()
    tau_hat = np.asarray(model.predict(x_query), dtype=np.float64).reshape(-1)
    prediction_time = time.perf_counter() - predict_started
    if tau_hat.shape != (len(x_query),) or not np.isfinite(tau_hat).all():
        raise ValueError("final full TabPFN prediction must be finite and query aligned")
    return FullTabPFNFinalResult(
        tau_hat=tau_hat,
        model=model,
        timing={
            "fit_sec": float(fit_time),
            "prediction_sec": float(prediction_time),
            "total_sec": float(fit_time + prediction_time),
        },
    )
