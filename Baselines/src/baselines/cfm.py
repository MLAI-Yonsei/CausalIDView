"""Dependency-light isolated entrypoint for the three official CFM environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np


WORKSPACE = Path(
    os.environ.get(
        "CAUSALARENA_WORKSPACE",
        str(Path(__file__).resolve().parents[3]),
    )
).resolve()
CAUSALFM_PREPROCESSING_CONTRACT = "support_zscore_x_y_v1"
CAUSALFM_SAVE_PEAK_MEM_THRESHOLD = 20_000
CAUSALFM_SAVE_PEAK_MEM_FACTOR = 8


def _stable_array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cfm_input_audit(
    model: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    d_covariate = int(arrays["X_context"].shape[1])
    preprocessed = int(metadata.get("model_input_feature_dim", d_covariate))
    if model == "dopfn":
        stage_widths = {"support_with_treatment": d_covariate + 1, "query_with_treatment": d_covariate + 1}
        treatment_handling = "treatment prepended; identical factual covariates used for T=0 and T=1 queries"
        final_context = [len(arrays["X_context"]), d_covariate + 1]
        final_query = [2 * len(arrays["X_query"]), d_covariate + 1]
    elif model == "causalpfn":
        stage_widths = {"checkpoint_covariates": preprocessed, "treatment_argument": 1}
        treatment_handling = "separate treatment argument; internal T=0/T=1 queries duplicate one covariate matrix"
        final_context = [len(arrays["X_context"]), preprocessed]
        final_query = [2 * len(arrays["X_query"]), preprocessed]
    else:
        stage_widths = {"per_feature_encoder_covariates": d_covariate, "treatment_argument": 1}
        treatment_handling = "separate support treatment; native direct-CATE query API"
        final_context = [len(arrays["X_context"]), d_covariate]
        final_query = [len(arrays["X_query"]), d_covariate]
    return {
        "adapter_input_shape": {
            "context": list(arrays["X_context"].shape),
            "query": list(arrays["X_query"].shape),
        },
        "preprocessed_input_shape": {
            "context_covariates": [len(arrays["X_context"]), preprocessed],
            "query_covariates": [len(arrays["X_query"]), preprocessed],
        },
        "final_model_input_shape": {"context": final_context, "query": final_query},
        "final_stage_widths": stage_widths,
        "covariate_width_received": d_covariate,
        "treatment_handling": treatment_handling,
        "potential_outcome_covariates_fixed": True,
        "post_columns_explicitly_removed": False,
        "adapter_context_hash": _stable_array_hash(arrays["X_context"]),
        "adapter_query_hash": _stable_array_hash(arrays["X_query"]),
    }


def _causalfm_save_peak_mem_factor(n_context: int, n_query: int) -> int | None:
    """Use the released model's exact, non-truncating low-memory inference path."""

    if n_context < 0 or n_query < 0:
        raise ValueError("CausalFM row counts must be nonnegative")
    return (
        CAUSALFM_SAVE_PEAK_MEM_FACTOR
        if n_context + n_query > CAUSALFM_SAVE_PEAK_MEM_THRESHOLD
        else None
    )


def _load_input(
    path: Path,
    *,
    model: str,
    graph_regime: str,
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"X_context", "treatment", "outcome", "X_query"}
        if model == "causalfm" and graph_regime in {"frontdoor", "conditional_iv"}:
            required.add("graph_aux_context")
        if set(archive.files) != required:
            raise ValueError(f"CFM input arrays mismatch: {sorted(archive.files)}")
        return {name: np.asarray(archive[name]) for name in archive.files}


def _standardize_causalfm_inputs(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], float, dict[str, Any]]:
    """Apply leakage-free support statistics for the standard-CATE checkpoint."""

    X_context = np.asarray(arrays["X_context"], dtype=np.float64)
    X_query = np.asarray(arrays["X_query"], dtype=np.float64)
    treatment = np.asarray(arrays["treatment"], dtype=np.float64)
    outcome = np.asarray(arrays["outcome"], dtype=np.float64)
    if X_context.ndim != 2 or X_query.ndim != 2:
        raise ValueError("CausalFM covariates must be two-dimensional")
    if X_context.shape[1] != X_query.shape[1]:
        raise ValueError("CausalFM support/query covariates must be feature aligned")
    if treatment.shape != (len(X_context),) or outcome.shape != (len(X_context),):
        raise ValueError("CausalFM treatment/outcome must align with support rows")
    if not all(
        np.isfinite(value).all()
        for value in (X_context, X_query, treatment, outcome)
    ):
        raise ValueError("CausalFM standardization inputs must be finite")

    x_mean = np.mean(X_context, axis=0, dtype=np.float64)
    x_observed_scale = np.std(X_context, axis=0, dtype=np.float64)
    constant_features = x_observed_scale <= np.finfo(np.float64).eps
    x_scale = np.where(constant_features, 1.0, x_observed_scale)

    outcome_mean = float(np.mean(outcome, dtype=np.float64))
    outcome_observed_scale = float(np.std(outcome, dtype=np.float64))
    outcome_scale = (
        1.0
        if outcome_observed_scale <= np.finfo(np.float64).eps
        else outcome_observed_scale
    )

    standardized = {
        "X_context": (X_context - x_mean) / x_scale,
        "treatment": treatment,
        "outcome": (outcome - outcome_mean) / outcome_scale,
        "X_query": (X_query - x_mean) / x_scale,
    }
    metadata = {
        "preprocessing_contract": CAUSALFM_PREPROCESSING_CONTRACT,
        "statistics_source": "support_only",
        "treatment_standardized": False,
        "cate_inverse_transform": "multiply_by_support_outcome_std",
        "constant_feature_count": int(np.sum(constant_features)),
        "outcome_mean": outcome_mean,
        "outcome_observed_std": outcome_observed_scale,
        "outcome_scale": outcome_scale,
    }
    return standardized, outcome_scale, metadata


def _run_causalpfn(arrays: dict[str, np.ndarray], device: str) -> tuple[np.ndarray, dict[str, Any]]:
    from Baselines.CFMs.performance_reproduction.causalpfn.protocol import (
        checkpoint_path,
        estimate_causalpfn_cate_arrays,
    )

    root = WORKSPACE / "Baselines" / "CFMs" / "CausalPFN"
    checkpoint = checkpoint_path(root)
    import torch

    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = raw["model_config"]
    feature_limit = (
        int(config["model"]["max_num_features"]) - 1
        if config.get("model_type") == "tabdpt"
        else None
    )
    prediction = estimate_causalpfn_cate_arrays(
        arrays["X_context"], arrays["treatment"], arrays["outcome"], arrays["X_query"],
        model_root=root, device=device, seed=82718,
    )
    offered = len(arrays["X_context"])
    return prediction, {
        "context_mechanism": "official_cepo_weak_learner_retrieval",
        "n_context_offered": offered,
        "n_context_effective": min(offered, 4096),
        "feature_mechanism": "official_truncated_svd" if feature_limit and arrays["X_context"].shape[1] > feature_limit else "identity",
        "effective_feature_dim": min(arrays["X_context"].shape[1], feature_limit) if feature_limit else arrays["X_context"].shape[1],
        "arbitrary_context_truncation": False,
    }


def _run_dopfn(
    arrays: dict[str, np.ndarray], device: str, *, model_root: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    from Baselines.CFMs.performance_reproduction.dopfn.protocol import (
        evaluate_dopfn_cate_arrays,
        fit_dopfn,
    )

    root = model_root if model_root is not None else WORKSPACE / "Baselines" / "CFMs" / "Do-PFN"
    model = fit_dopfn(root, device=device)
    prediction = evaluate_dopfn_cate_arrays(
        model, arrays["X_context"], arrays["treatment"], arrays["outcome"], arrays["X_query"], seed=82718,
    )
    return prediction, {
        "context_mechanism": "official_full_context_api",
        "n_context_offered": len(arrays["X_context"]),
        "n_context_effective": len(arrays["X_context"]),
        "feature_mechanism": "official_release",
        "effective_feature_dim": arrays["X_context"].shape[1],
        "arbitrary_context_truncation": False,
    }


def _run_causalfm(arrays: dict[str, np.ndarray], device: str) -> tuple[np.ndarray, dict[str, Any]]:
    from Baselines.CFMs.performance_reproduction.causalfm.protocol import (
        estimate_causalfm_cate_arrays,
        load_causalfm_model,
        set_causalfm_seed,
    )

    root = WORKSPACE / "Baselines" / "CFMs" / "CausalFM"
    standardized, outcome_scale, preprocessing = _standardize_causalfm_inputs(arrays)
    set_causalfm_seed(0)
    model = load_causalfm_model(root, device)
    save_peak_mem_factor = _causalfm_save_peak_mem_factor(
        len(standardized["X_context"]), len(standardized["X_query"])
    )
    if save_peak_mem_factor is not None:
        model.reset_save_peak_mem_factor(save_peak_mem_factor)
    standardized_prediction = estimate_causalfm_cate_arrays(
        model,
        standardized["X_context"],
        standardized["treatment"],
        standardized["outcome"],
        standardized["X_query"],
        device=device,
    )
    prediction = np.asarray(standardized_prediction, dtype=np.float64) * outcome_scale
    return prediction, {
        "context_mechanism": "official_full_context_api",
        "n_context_offered": len(arrays["X_context"]),
        "n_context_effective": len(arrays["X_context"]),
        "feature_mechanism": "official_per_feature_encoder",
        "effective_feature_dim": arrays["X_context"].shape[1],
        "arbitrary_context_truncation": False,
        "save_peak_mem_factor": save_peak_mem_factor,
        **preprocessing,
    }


def _v2_sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _v2_reset_peak_memory(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def _v2_peak_memory_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    import torch

    return float(torch.cuda.max_memory_allocated() / (1024.0 * 1024.0))


def _v2_batched(function: Any, query: np.ndarray, microbatch_size: int) -> np.ndarray:
    chunks = [
        np.asarray(function(query[start:start + microbatch_size]), dtype=np.float64).reshape(-1)
        for start in range(0, len(query), microbatch_size)
    ]
    prediction = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float64)
    if prediction.shape != (len(query),):
        raise ValueError("CFM micro-batch output is not query aligned")
    return prediction


def _v2_timed(function: Any, query: np.ndarray, microbatch_size: int, device: str) -> tuple[np.ndarray, float, list[float]]:
    _v2_batched(function, query, microbatch_size)
    timings: list[float] = []
    prediction: np.ndarray | None = None
    for _ in range(3):
        _v2_sync(device)
        started = time.perf_counter()
        prediction = _v2_batched(function, query, microbatch_size)
        _v2_sync(device)
        timings.append(float(time.perf_counter() - started))
    assert prediction is not None
    return prediction, float(np.median(np.asarray(timings))), timings


def _v2_gmm_quantile(pi: Any, mu: Any, sigma: Any, probability: float) -> Any:
    """Return exact numerical quantiles of the direct-CATE Gaussian mixture."""

    import torch

    if not 0.0 < probability < 1.0:
        raise ValueError("GMM quantile probability must be in (0,1)")
    sigma = torch.clamp(sigma, min=torch.finfo(sigma.dtype).eps)
    low = torch.min(mu - 12.0 * sigma, dim=-1).values
    high = torch.max(mu + 12.0 * sigma, dim=-1).values
    target = torch.full_like(low, probability)
    root_two = math.sqrt(2.0)
    for _ in range(80):
        midpoint = (low + high) / 2.0
        cdf = torch.sum(
            pi * 0.5 * (1.0 + torch.erf((midpoint[..., None] - mu) / (sigma * root_two))),
            dim=-1,
        )
        low = torch.where(cdf < target, midpoint, low)
        high = torch.where(cdf >= target, midpoint, high)
    return (low + high) / 2.0


def _run_causalpfn_v2(
    arrays: dict[str, np.ndarray], device: str, microbatch_size: int,
    *, compute_uq: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from Baselines.CFMs.performance_reproduction.causalpfn.protocol import (
        _official_cate_estimator,
        checkpoint_path,
        set_official_seed,
    )

    root = WORKSPACE / "Baselines" / "CFMs" / "CausalPFN"
    checkpoint = checkpoint_path(root)
    set_official_seed(82718)
    load_started = time.perf_counter()
    with _official_cate_estimator(root) as cate_type:
        estimator = cate_type(device=device, model_path=str(checkpoint))
        estimator.load_model()
        model_load_time = float(time.perf_counter() - load_started)
        raw_feature_dim = int(arrays["X_context"].shape[1])
        context_rank_safe_svd = False
        model_input_feature_dim = raw_feature_dim
        if estimator.max_feature_size is not None and raw_feature_dim > estimator.max_feature_size:
            # The released estimator always asks ARPACK for the checkpoint's
            # full feature limit.  For short/wide inputs that violates
            # scipy's strict k < min(X.shape) contract.  Keep the same
            # context-fitted TruncatedSVD mechanism, reducing only k to the
            # largest mathematically valid checkpoint-facing dimension.
            model_input_feature_dim = min(
                int(estimator.max_feature_size),
                int(arrays["X_context"].shape[0]) - 1,
                raw_feature_dim - 1,
            )
            if model_input_feature_dim < 1:
                raise ValueError("CausalPFN context-rank-safe SVD has no valid component")
            if model_input_feature_dim < int(estimator.max_feature_size):
                from sklearn.decomposition import TruncatedSVD

                estimator.x_dim_transformer = TruncatedSVD(
                    n_components=model_input_feature_dim, algorithm="arpack",
                )
                context_rank_safe_svd = True
        uses_official_svd = model_input_feature_dim < raw_feature_dim
        # Preserve the official fit implementation while avoiding a second
        # checkpoint load; loading/import time is excluded from the headline.
        estimator.load_model = lambda: None
        conditioning_started = time.perf_counter()
        estimator.fit(arrays["X_context"], arrays["treatment"], arrays["outcome"])
        conditioning_time = float(time.perf_counter() - conditioning_started)
        prediction, point_only_time, repeats = _v2_timed(
            estimator.estimate_cate, arrays["X_query"], microbatch_size, device,
        )
        lower_chunks: list[np.ndarray] = []
        upper_chunks: list[np.ndarray] = []
        uq_started = time.perf_counter()
        if compute_uq:
            for start in range(0, len(arrays["X_query"]), microbatch_size):
                interval = estimator.estimate_cate_CI(
                    arrays["X_query"][start:start + microbatch_size], alpha=0.10, n_samples=10_000,
                )
                lower_chunks.append(np.asarray(interval["lower_bound"], dtype=np.float64).reshape(-1))
                upper_chunks.append(np.asarray(interval["upper_bound"], dtype=np.float64).reshape(-1))
            _v2_sync(device)
        uq_time = float(time.perf_counter() - uq_started) if compute_uq else 0.0
    lower = np.concatenate(lower_chunks) if compute_uq else None
    upper = np.concatenate(upper_chunks) if compute_uq else None
    return prediction, lower, upper, {
        "fit_time_sec": 0.0,
        "inference_time_sec": conditioning_time + point_only_time,
        "conditioning_time_sec": conditioning_time,
        "point_only_repeat_median_sec": point_only_time,
        "inference_repeats_sec": repeats,
        "model_load_time_excluded_sec": model_load_time,
        "uq_time_sec": uq_time,
        "uq_supported": bool(compute_uq),
        "uq_status": "success" if compute_uq else "not_requested_by_synthetic_protocol",
        "uq_method": (
            "CausalPFN.estimate_cate_CI(alpha=0.10,n_samples=10000); direct CATE posterior samples"
            if compute_uq else None
        ),
        # effective_feature_dim is benchmark-facing and verifies that all raw
        # columns reached the official estimator API. The checkpoint-facing
        # dimension is recorded separately below.
        "effective_feature_dim": raw_feature_dim,
        "raw_feature_dim": raw_feature_dim,
        "model_input_feature_dim": model_input_feature_dim,
        "n_context_effective": int(len(arrays["X_context"])),
        "feature_mechanism": (
            "context_rank_safe_truncated_svd_context_fit"
            if context_rank_safe_svd
            else "official_truncated_svd_context_fit"
            if uses_official_svd
            else "official_identity"
        ),
        "context_rank_safe_svd": context_rank_safe_svd,
        "feature_transformer": (
            "sklearn.decomposition.TruncatedSVD(algorithm=arpack)"
            if uses_official_svd
            else None
        ),
        "context_mechanism": "official_cepo_context_conditioning",
        "actual_fit_rows": int(len(arrays["X_context"])),
        "validation_rows": 0,
        "statistics_source": "context_only",
        "query_used_for_fit_or_tuning": False,
        "point_output_scale": "original_outcome",
        "interval_output_scale": "original_outcome",
    }


def _run_dopfn_v2(
    arrays: dict[str, np.ndarray], device: str, microbatch_size: int, *, model_root: Path | None = None,
) -> tuple[np.ndarray, None, None, dict[str, Any]]:
    from Baselines.CFMs.performance_reproduction.dopfn.protocol import (
        evaluate_dopfn_cate_arrays,
        fit_dopfn,
    )

    root = model_root if model_root is not None else WORKSPACE / "Baselines" / "CFMs" / "Do-PFN"
    construct_started = time.perf_counter()
    model = fit_dopfn(root, device=device)
    construct_time = float(time.perf_counter() - construct_started)

    def predict(query: np.ndarray) -> np.ndarray:
        return evaluate_dopfn_cate_arrays(
            model,
            arrays["X_context"], arrays["treatment"], arrays["outcome"], query,
            seed=82718,
        )

    prediction, inference_time, repeats = _v2_timed(
        predict, arrays["X_query"], microbatch_size, device,
    )
    return prediction, None, None, {
        "fit_time_sec": 0.0,
        "inference_time_sec": inference_time,
        "inference_repeats_sec": repeats,
        "model_construct_time_excluded_sec": construct_time,
        "uq_time_sec": 0.0,
        "uq_supported": False,
        "uq_status": "unsupported_native_cate_interval",
        "uq_method": None,
        "uq_reason": "released distributions are potential-outcome marginals, not a joint/direct CATE distribution",
        "effective_feature_dim": int(arrays["X_context"].shape[1]),
        "n_context_effective": int(len(arrays["X_context"])),
        "feature_mechanism": "canonical_input_retained_with_official_dopfn_preprocessing",
        "maximum_feature_limit": int(getattr(model, "max_num_features_", -1)),
        "context_mechanism": "official_full_context_api",
        "actual_fit_rows": int(len(arrays["X_context"])),
        "validation_rows": 0,
        "statistics_source": "context_only",
        "query_used_for_fit_or_tuning": False,
        "point_output_scale": "original_outcome",
        "interval_output_scale": None,
    }


def _run_causalfm_v2(
    arrays: dict[str, np.ndarray], device: str, microbatch_size: int,
    *, compute_uq: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    from Baselines.CFMs.performance_reproduction.causalfm.protocol import (
        estimate_causalfm_cate_arrays,
        load_causalfm_model,
        set_causalfm_seed,
    )
    import torch

    root = WORKSPACE / "Baselines" / "CFMs" / "CausalFM"
    preprocessing_started = time.perf_counter()
    standardized, outcome_scale, preprocessing = _standardize_causalfm_inputs(arrays)
    x_mean = np.mean(np.asarray(arrays["X_context"], dtype=np.float64), axis=0, dtype=np.float64)
    x_observed_scale = np.std(np.asarray(arrays["X_context"], dtype=np.float64), axis=0, dtype=np.float64)
    x_scale = np.where(x_observed_scale <= np.finfo(np.float64).eps, 1.0, x_observed_scale)
    preprocessing_fit_time = float(time.perf_counter() - preprocessing_started)
    set_causalfm_seed(0)
    load_started = time.perf_counter()
    model = load_causalfm_model(root, device)
    model_load_time = float(time.perf_counter() - load_started)
    save_peak_mem_factor = _causalfm_save_peak_mem_factor(
        len(standardized["X_context"]), microbatch_size,
    )
    if save_peak_mem_factor is not None:
        model.reset_save_peak_mem_factor(save_peak_mem_factor)

    def standardized_query(query: np.ndarray) -> np.ndarray:
        return (np.asarray(query, dtype=np.float64) - x_mean) / x_scale

    def predict(query: np.ndarray) -> np.ndarray:
        value = estimate_causalfm_cate_arrays(
            model,
            standardized["X_context"], standardized["treatment"], standardized["outcome"],
            standardized_query(query), device=device,
        )
        return np.asarray(value, dtype=np.float64) * outcome_scale

    prediction, inference_time, repeats = _v2_timed(
        predict, arrays["X_query"], microbatch_size, device,
    )
    lower_chunks: list[np.ndarray] = []
    upper_chunks: list[np.ndarray] = []
    uq_started = time.perf_counter()
    if compute_uq:
        for start in range(0, len(arrays["X_query"]), microbatch_size):
            query = standardized_query(arrays["X_query"][start:start + microbatch_size])
            tensors = (
                torch.as_tensor(standardized["X_context"], dtype=torch.float32, device=device),
                torch.as_tensor(standardized["treatment"][:, None], dtype=torch.float32, device=device),
                torch.as_tensor(standardized["outcome"][:, None], dtype=torch.float32, device=device),
                torch.as_tensor(query, dtype=torch.float32, device=device),
            )
            with torch.inference_mode():
                output = model.estimate_cate(*tensors, return_gmm=True)
                required = {"gmm_pi", "gmm_mu", "gmm_sigma"}
                if not required.issubset(output):
                    raise ValueError("CausalFM checkpoint does not expose its direct-CATE GMM posterior")
                lower = _v2_gmm_quantile(output["gmm_pi"], output["gmm_mu"], output["gmm_sigma"], 0.05)
                upper = _v2_gmm_quantile(output["gmm_pi"], output["gmm_mu"], output["gmm_sigma"], 0.95)
            lower_chunks.append(np.asarray(lower.detach().cpu(), dtype=np.float64).reshape(-1) * outcome_scale)
            upper_chunks.append(np.asarray(upper.detach().cpu(), dtype=np.float64).reshape(-1) * outcome_scale)
        _v2_sync(device)
    uq_time = float(time.perf_counter() - uq_started) if compute_uq else 0.0
    lower = np.concatenate(lower_chunks) if compute_uq else None
    upper = np.concatenate(upper_chunks) if compute_uq else None
    return prediction, lower, upper, {
        "fit_time_sec": preprocessing_fit_time,
        "inference_time_sec": inference_time,
        "inference_repeats_sec": repeats,
        "model_load_time_excluded_sec": model_load_time,
        "uq_time_sec": uq_time,
        "uq_supported": bool(compute_uq),
        "uq_status": "success" if compute_uq else "not_requested_by_synthetic_protocol",
        "uq_method": (
            "CausalFM direct-CATE GMM posterior mixture CDF quantiles q=0.05,0.95"
            if compute_uq else None
        ),
        "effective_feature_dim": int(arrays["X_context"].shape[1]),
        "n_context_effective": int(len(arrays["X_context"])),
        "feature_mechanism": "official_per_feature_encoder_no_projection",
        "context_mechanism": "official_full_context_api",
        "actual_fit_rows": int(len(arrays["X_context"])),
        "validation_rows": 0,
        "save_peak_mem_factor": save_peak_mem_factor,
        "query_preprocessing_in_timed_call": True,
        "query_used_for_fit_or_tuning": False,
        "point_output_scale": "original_outcome",
        "interval_output_scale": "original_outcome",
        **preprocessing,
    }


def _run_causalfm_graph_v2(
    arrays: dict[str, np.ndarray],
    device: str,
    microbatch_size: int,
    *,
    graph_regime: str,
    compute_uq: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Run the released graph-specific CausalFM checkpoint and API."""

    from Baselines.CFMs.performance_reproduction.causalfm.protocol import (
        _isolated_official_imports,
        set_causalfm_seed,
    )
    import torch

    contracts = {
        "frontdoor": {
            "module": "src.tabpfn.model.causalFM4FD",
            "checkpoint": "checkpoints_FD/best_model.pth",
            "method": "estimate_cate_fd",
            "auxiliary": "factual_mediator",
        },
        "conditional_iv": {
            "module": "src.tabpfn.model.causalFM4IV",
            "checkpoint": "checkpoints_IV_binary/best_model.pth",
            "method": "estimate_cate_iv",
            "auxiliary": "binary_instrument",
        },
    }
    if graph_regime not in contracts:
        raise ValueError(f"unsupported graph-specific CausalFM regime: {graph_regime}")
    contract = contracts[graph_regime]
    auxiliary = np.asarray(arrays["graph_aux_context"], dtype=np.float64).reshape(-1)
    if auxiliary.shape != (len(arrays["X_context"]),) or not np.isfinite(auxiliary).all():
        raise ValueError("CausalFM graph auxiliary must be finite and context aligned")

    preprocessing_started = time.perf_counter()
    standardized, outcome_scale, preprocessing = _standardize_causalfm_inputs(arrays)
    X_context_raw = np.asarray(arrays["X_context"], dtype=np.float64)
    x_mean = np.mean(X_context_raw, axis=0, dtype=np.float64)
    x_observed_scale = np.std(X_context_raw, axis=0, dtype=np.float64)
    x_scale = np.where(x_observed_scale <= np.finfo(np.float64).eps, 1.0, x_observed_scale)
    auxiliary_metadata: dict[str, Any]
    if graph_regime == "frontdoor":
        auxiliary_mean = float(np.mean(auxiliary, dtype=np.float64))
        auxiliary_observed_scale = float(np.std(auxiliary, dtype=np.float64))
        auxiliary_scale = (
            1.0
            if auxiliary_observed_scale <= np.finfo(np.float64).eps
            else auxiliary_observed_scale
        )
        auxiliary_model = (auxiliary - auxiliary_mean) / auxiliary_scale
        auxiliary_metadata = {
            "mediator_standardized": True,
            "mediator_statistics_source": "context_factual_mediator_only",
            "mediator_mean": auxiliary_mean,
            "mediator_observed_std": auxiliary_observed_scale,
            "mediator_scale": auxiliary_scale,
        }
    else:
        unique_instrument = set(np.unique(auxiliary).tolist())
        if unique_instrument != {0.0, 1.0}:
            raise ValueError(
                "CausalFM binary-IV checkpoint requires both binary instrument arms"
            )
        auxiliary_model = auxiliary
        auxiliary_metadata = {
            "instrument_standardized": False,
            "instrument_type": "binary",
            "instrument_statistics_source": "none",
        }
    preprocessing_fit_time = float(time.perf_counter() - preprocessing_started)

    root = WORKSPACE / "Baselines" / "CFMs" / "CausalFM"
    checkpoint_path = root / str(contract["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"missing official CausalFM checkpoint: {checkpoint_path}")
    set_causalfm_seed(0)
    load_started = time.perf_counter()
    with _isolated_official_imports(root) as official:
        model_module = official.import_module(str(contract["module"]))
        model = model_module.PerFeatureTransformerCATE(seed=0).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        model.eval()
    model_load_time = float(time.perf_counter() - load_started)
    save_peak_mem_factor = _causalfm_save_peak_mem_factor(
        len(standardized["X_context"]), microbatch_size,
    )
    if save_peak_mem_factor is not None:
        model.reset_save_peak_mem_factor(save_peak_mem_factor)

    def standardized_query(query: np.ndarray) -> np.ndarray:
        return (np.asarray(query, dtype=np.float64) - x_mean) / x_scale

    def estimate(query: np.ndarray, *, return_gmm: bool = False) -> Any:
        tensors = (
            torch.as_tensor(standardized["X_context"], dtype=torch.float32, device=device),
            torch.as_tensor(standardized["treatment"][:, None], dtype=torch.float32, device=device),
            torch.as_tensor(standardized["outcome"][:, None], dtype=torch.float32, device=device),
            torch.as_tensor(auxiliary_model[:, None], dtype=torch.float32, device=device),
            torch.as_tensor(standardized_query(query), dtype=torch.float32, device=device),
        )
        method = getattr(model, str(contract["method"]))
        with torch.inference_mode():
            return method(*tensors, return_gmm=return_gmm)

    def predict(query: np.ndarray) -> np.ndarray:
        output = estimate(query)
        if not isinstance(output, dict) or "cate" not in output:
            raise ValueError(f"CausalFM {contract['method']} did not return direct CATE")
        return np.asarray(output["cate"].detach().cpu(), dtype=np.float64).reshape(-1) * outcome_scale

    prediction, inference_time, repeats = _v2_timed(
        predict, arrays["X_query"], microbatch_size, device,
    )
    lower_chunks: list[np.ndarray] = []
    upper_chunks: list[np.ndarray] = []
    uq_started = time.perf_counter()
    if compute_uq:
        for start in range(0, len(arrays["X_query"]), microbatch_size):
            output = estimate(
                arrays["X_query"][start:start + microbatch_size], return_gmm=True,
            )
            required = {"gmm_pi", "gmm_mu", "gmm_sigma"}
            if not isinstance(output, dict) or not required.issubset(output):
                raise ValueError(
                    f"CausalFM {graph_regime} checkpoint does not expose direct-CATE GMM parameters"
                )
            lower = _v2_gmm_quantile(
                output["gmm_pi"], output["gmm_mu"], output["gmm_sigma"], 0.05,
            )
            upper = _v2_gmm_quantile(
                output["gmm_pi"], output["gmm_mu"], output["gmm_sigma"], 0.95,
            )
            lower_chunks.append(
                np.asarray(lower.detach().cpu(), dtype=np.float64).reshape(-1) * outcome_scale
            )
            upper_chunks.append(
                np.asarray(upper.detach().cpu(), dtype=np.float64).reshape(-1) * outcome_scale
            )
        _v2_sync(device)
    uq_time = float(time.perf_counter() - uq_started) if compute_uq else 0.0
    lower = np.concatenate(lower_chunks) if compute_uq else None
    upper = np.concatenate(upper_chunks) if compute_uq else None
    return prediction, lower, upper, {
        "graph_regime": graph_regime,
        "graph_checkpoint": str(contract["checkpoint"]),
        "graph_api": str(contract["method"]),
        "graph_auxiliary": str(contract["auxiliary"]),
        "fit_time_sec": preprocessing_fit_time,
        "inference_time_sec": inference_time,
        "inference_repeats_sec": repeats,
        "model_load_time_excluded_sec": model_load_time,
        "uq_time_sec": uq_time,
        "uq_supported": bool(compute_uq),
        "uq_status": "success" if compute_uq else "not_requested_by_synthetic_protocol",
        "uq_method": (
            f"CausalFM {graph_regime} direct-CATE GMM posterior mixture CDF quantiles q=0.05,0.95"
            if compute_uq else None
        ),
        "effective_feature_dim": int(arrays["X_context"].shape[1]),
        "n_context_effective": int(len(arrays["X_context"])),
        "feature_mechanism": "official_per_feature_encoder_no_projection",
        "context_mechanism": "official_full_context_graph_api",
        "actual_fit_rows": int(len(arrays["X_context"])),
        "validation_rows": 0,
        "save_peak_mem_factor": save_peak_mem_factor,
        "query_preprocessing_in_timed_call": True,
        "query_used_for_fit_or_tuning": False,
        "point_output_scale": "original_outcome",
        "interval_output_scale": "original_outcome" if compute_uq else None,
        **preprocessing,
        **auxiliary_metadata,
    }


def _run_frontier_v2(
    model: str,
    arrays: dict[str, np.ndarray],
    device: str,
    microbatch_size: int,
    *,
    compute_uq: bool = True,
    graph_regime: str = "backdoor",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    started = time.perf_counter()
    if model == "causalpfn":
        prediction, lower, upper, metadata = _run_causalpfn_v2(
            arrays, device, microbatch_size, compute_uq=compute_uq,
        )
    elif model == "causalfm":
        if graph_regime in {"frontdoor", "conditional_iv"}:
            prediction, lower, upper, metadata = _run_causalfm_graph_v2(
                arrays, device, microbatch_size,
                graph_regime=graph_regime, compute_uq=compute_uq,
            )
        else:
            prediction, lower, upper, metadata = _run_causalfm_v2(
                arrays, device, microbatch_size, compute_uq=compute_uq,
            )
            metadata["graph_regime"] = graph_regime
            metadata["graph_checkpoint"] = "checkpoints_standard/best_model.pth"
            metadata["graph_api"] = "estimate_cate"
    elif model == "dopfn":
        prediction, lower, upper, metadata = _run_dopfn_v2(arrays, device, microbatch_size)
    else:
        raise KeyError(model)
    metadata["internal_end_to_end_time_sec"] = float(time.perf_counter() - started)
    metadata["query_microbatch_size"] = int(microbatch_size)
    metadata["input_audit"] = _cfm_input_audit(model, arrays, metadata)
    return prediction, lower, upper, metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("causalpfn", "causalfm", "dopfn"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frontier-v2", action="store_true")
    parser.add_argument("--query-microbatch-size", type=int)
    parser.add_argument("--skip-uq", action="store_true")
    parser.add_argument(
        "--graph-regime",
        choices=("backdoor", "frontdoor", "conditional_iv", "hidden_confounding"),
        default="backdoor",
    )
    args = parser.parse_args(argv)
    if args.device != "cuda:0":
        raise ValueError("isolated CFM jobs must use logical cuda:0")
    arrays = _load_input(args.input, model=args.model, graph_regime=args.graph_regime)
    if args.frontier_v2:
        _v2_reset_peak_memory(args.device)
        microbatch_size = args.query_microbatch_size or len(arrays["X_query"])
        if microbatch_size < 1 or microbatch_size > len(arrays["X_query"]):
            raise ValueError("query micro-batch size must be within the evaluation query")
        prediction, lower, upper, metadata = _run_frontier_v2(
            args.model, arrays, args.device, microbatch_size,
            compute_uq=not args.skip_uq,
            graph_regime=args.graph_regime,
        )
        metadata["peak_gpu_memory_mb"] = _v2_peak_memory_mb(args.device)
    else:
        runner = {
            "causalpfn": _run_causalpfn,
            "causalfm": _run_causalfm,
            "dopfn": _run_dopfn,
        }[args.model]
        prediction, metadata = runner(arrays, args.device)
        lower = upper = None
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if prediction.shape != (len(arrays["X_query"]),) or not np.isfinite(prediction).all():
        raise ValueError("CFM prediction is not a finite query-aligned vector")
    args.prediction.parent.mkdir(parents=True, exist_ok=True)
    if args.frontier_v2:
        payload: dict[str, np.ndarray] = {"tau_hat": prediction}
        if lower is not None and upper is not None:
            lower = np.asarray(lower, dtype=np.float64).reshape(-1)
            upper = np.asarray(upper, dtype=np.float64).reshape(-1)
            if lower.shape != prediction.shape or upper.shape != prediction.shape or np.any(lower > upper):
                raise ValueError("CFM native interval is invalid or query misaligned")
            payload.update(lower=lower, upper=upper)
        np.savez_compressed(args.prediction, **payload)
    else:
        np.save(args.prediction, prediction, allow_pickle=False)
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
