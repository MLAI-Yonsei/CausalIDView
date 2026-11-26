"""Figure 2 TabPFN-v3.5 worker routines from the Rebuttal benchmark."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

def _model_seed(regime: str, world_id: int) -> int:
    if regime == "BD":
        return 20300901 + 101 * int(world_id)
    if regime == "BD_S":
        return 20290901 + 101 * int(world_id)
    if regime == "BD_DR":
        return 20360901 + 101 * int(world_id)
    if regime in {"FD", "IV"}:
        return 20360901 + 1009 * int(world_id)
    if regime == "PX":
        from ..fig2 import specialist_seed

        return specialist_seed("p_learner", world_id)
    raise ValueError(regime)


def _run_bd(view: dict[str, Any], seed: int, version: str) -> tuple[np.ndarray, dict[str, Any]]:
    from baselines.registry import create_estimator
    from .fig2_config import estimator_config

    arrays = view["payload"]
    route = view["route"]
    model = {"BD": "tabpfn_x", "BD_S": "tabpfn_s", "BD_DR": "tabpfn_dr"}[route]
    estimator = create_estimator(
        model, estimator_config(seed, version),
    )
    fit_started = time.perf_counter()
    estimator.fit(
        arrays["X_context"], arrays["treatment"].astype(np.int64),
        arrays["outcome"], outcome_type="continuous",
    )
    fit_sec = time.perf_counter() - fit_started
    prediction_started = time.perf_counter()
    estimate = np.asarray(
        estimator.predict_cate(arrays["X_query"]), dtype=np.float64,
    ).reshape(-1)
    prediction_sec = time.perf_counter() - prediction_started
    nuisances = {
        "BD": [
            "arm-specific outcome regressions", "imputed-effect regressions",
            "treatment propensity predict_proba",
        ],
        "BD_S": ["joint outcome regression E[Y|X,U,T]"],
        "BD_DR": [
            "cross-fitted arm-specific outcome regressions",
            "cross-fitted treatment propensity predict_proba",
            "final DR pseudo-outcome regression",
        ],
    }[route]
    return estimate, {
        "fit_time_sec": fit_sec,
        "inference_time_sec": prediction_sec,
        "estimator_metadata": estimator.get_metadata(),
        "nuisance_components": nuisances,
    }


def _run_fd(view: dict[str, Any], seed: int, version: str) -> tuple[np.ndarray, dict[str, Any]]:
    from ..frontdoor import run

    with tempfile.TemporaryDirectory(prefix="tabpfn_v35_fd_") as temporary:
        directory = Path(temporary)
        input_path = directory / "input.npz"
        prediction_path = directory / "prediction.npz"
        metadata_path = directory / "metadata.json"
        np.savez_compressed(input_path, **view["payload"])
        run(
            input_path, prediction_path, metadata_path, "tabpfn", seed,
            fold_count=5, integration_method="exact", integration_points=None,
            integration_batch_size=2048, tabpfn_version=version,
        )
        with np.load(prediction_path, allow_pickle=False) as archive:
            estimate = np.asarray(archive["tau_hat"], dtype=np.float64).reshape(-1)
        metadata = json.loads(metadata_path.read_text())
    return estimate, metadata


def _run_iv(view: dict[str, Any], seed: int, version: str) -> tuple[np.ndarray, dict[str, Any]]:
    from ..iv.wald import wald

    predictions, metadata = wald(
        view["payload"], "tabpfn", seed, tabpfn_version=version,
    )
    estimate = np.asarray(predictions["tau_hat"], dtype=np.float64).reshape(-1)
    metadata["delta_y"] = _summary(np.asarray(predictions["delta_Y"], dtype=np.float64))
    metadata["delta_t"] = _summary(np.asarray(predictions["delta_T"], dtype=np.float64))
    return estimate, metadata


def _run_px(view: dict[str, Any], seed: int, version: str) -> tuple[np.ndarray, dict[str, Any]]:
    from ..proximal.full_tabpfn import (
        FULL_TABPFN_BACKEND_SPEC,
        compute_full_tabpfn_oof_score,
        fit_full_tabpfn_final_stage,
    )
    from .fig2_config import fixed_tabpfn_backend

    backend, tabpfn_metadata = fixed_tabpfn_backend(seed, version=version, device="cuda:0")
    version_label = version.removeprefix("v")
    nuisance_id = f"tabpfn_v{version_label.replace('.', '_')}_full"
    base_spec = FULL_TABPFN_BACKEND_SPEC
    nuisance_config = {**dict(base_spec.nuisance_config), "backend_id": nuisance_id}
    backend_spec = replace(
        base_spec,
        model_id=f"p_learner_tabpfn_v{version_label.replace('.', '_')}_full",
        display_name=f"Full TabPFN-{version_label}-nuisance P-Learner",
        nuisance_backend_id=nuisance_id,
        nuisance_backend_name=f"frozen pretrained TabPFN-{version_label} in-context estimation",
        estimator_name=f"P-Learner with frozen pretrained TabPFN-{version_label} nuisances",
        score_namespace=f"p-learner-tabpfn-v{version_label}-full-score-v1",
        nuisance_config=nuisance_config,
    )
    arrays = view["payload"]

    def classifier_factory(local_seed: int, role: str) -> Any:
        return backend.create_classifier(random_state=int(local_seed), role=role)

    def regressor_factory(local_seed: int, role: str) -> Any:
        return backend.create_regressor(random_state=int(local_seed), role=role)

    score = compute_full_tabpfn_oof_score(
        arrays["X_context"], arrays["Z_context"], arrays["W_context"],
        arrays["treatment"], arrays["outcome"], arrays["X_query"],
        route_seed=seed,
        classifier_factory=classifier_factory,
        regressor_factory=regressor_factory,
        tabpfn_metadata=tabpfn_metadata,
        input_hash=view["input_hash"],
        context_query_split_hash=view["context_query_split_hash"],
        backend_spec=backend_spec,
    )
    final_seed = seed + 1_200_000
    final = fit_full_tabpfn_final_stage(
        score.bundle.x_context, score.bundle.gamma_oof, score.bundle.x_query,
        regressor_factory=lambda: backend.create_regressor(
            random_state=final_seed, role="full-p-learner-final-x-to-gamma",
        ),
    )
    metadata = {
        **score.bundle.diagnostics,
        "tabpfn_metadata": tabpfn_metadata,
        "gamma_hash": score.bundle.gamma_hash,
        "fold_hash": score.bundle.fold_hash,
        "score_hash": score.bundle.score_hash,
        "fold_ids": score.bundle.fold_ids.tolist(),
        "final_seed": final_seed,
        "final_backend": f"TabPFN-{version_label}",
        "final_tabpfn_timing": final.timing,
    }
    return final.tau_hat, metadata


def _summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "shape": list(array.shape), "finite": bool(np.isfinite(array).all()),
        "min": float(np.min(array)), "max": float(np.max(array)),
        "mean": float(np.mean(array)), "std": float(np.std(array)),
    }
