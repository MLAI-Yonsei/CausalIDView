"""Train the traditional Table 2 estimators on one original RCT split."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import load_profile, merge_config


TRADITIONAL_MODELS = ("causal_tree", "tarnet", "grf", "cfrnet")
R_MODELS = ("causal_tree", "grf")


def optimization_seeds(model: str, realization: int) -> tuple[int, ...]:
    seed = 20260924 + 1009 * realization + 1000003 * TRADITIONAL_MODELS.index(model)
    return (seed,) if model in R_MODELS else (seed, seed + 1009, seed + 2018)


def fit_predict(
    model: str,
    arrays: dict[str, np.ndarray],
    realization: int,
    *,
    device: str,
    work: Path,
    rscript: str | None = None,
    r_libs: Path | None = None,
) -> np.ndarray:
    seeds = optimization_seeds(model, realization)
    profile = load_profile("default")
    if model in R_MODELS:
        from .r_backend.estimators import CausalTreeEstimator, GRFEstimator
        from .r_backend.runner import RBackendRunner

        estimator_type = CausalTreeEstimator if model == "causal_tree" else GRFEstimator
        r_options = {"r_libs_user": r_libs} if r_libs is not None else {}
        runner = RBackendRunner(
            temp_root=work / "jobs", artifact_root=work / "models",
            rscript=rscript, **r_options,
        )
        estimator_options = {"runner": runner}
    else:
        from .table2_neural.estimators import CFRNetEstimator, TARNetEstimator

        estimator_type = TARNetEstimator if model == "tarnet" else CFRNetEstimator
        estimator_options = {}
        profile = merge_config(profile, {"gpu_device": device, "neural": {"device": device}})

    estimates = []
    for seed in seeds:
        model_config = merge_config(profile, {"seed": seed})
        estimator = estimator_type(config=model_config, **estimator_options)
        estimator.fit(arrays["X_context"], arrays["treatment"], arrays["outcome"], outcome_type="continuous")
        estimates.append(estimator.predict_cate(arrays["X_query"]))
    return np.mean(estimates, axis=0)
