"""Production continuous-mediator front-door plug-in (estimator-v2)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .config import BASELINES_ROOT

ESTIMATOR_VERSION = "fd_plugin_v2_oof_exact"


def _plain(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): return value.item()
    if value is None or isinstance(value, (str, bool, int, float)): return value
    return str(value)


def _release(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass


def _predict_chunks(model: Any, values: np.ndarray, *, probability: bool = False,
                    chunk: int = 256) -> np.ndarray:
    if chunk <= 0: raise ValueError("prediction chunk must be positive")
    rows = []
    for start in range(0, len(values), chunk):
        batch = values[start:start + chunk]
        pred = model.predict_proba(batch)[:, 1] if probability else model.predict(batch)
        rows.append(np.asarray(pred, dtype=np.float64).reshape(-1))
    return np.concatenate(rows) if rows else np.empty(0, dtype=np.float64)


def _factories(
    backend: str, seed: int, tabpfn_version: str = "v3",
) -> tuple[Callable[[int], Any], Callable[[int], Any], dict[str, Any]]:
    if backend == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor, __version__
        from .xgboost_settings import LOCKED_XGBOOST
        params = dict(LOCKED_XGBOOST)
        classifier = lambda offset: XGBClassifier(objective="binary:logistic", random_state=seed + offset, **params)
        regressor = lambda offset: XGBRegressor(objective="reg:squarederror", random_state=seed + offset, **params)
        return classifier, regressor, {"backend": "xgboost", "package_version": __version__,
                                      "locked_parameters": params}
    if backend == "nn":
        from .neural import CONFIG, MLPNuisance

        training_log = []
        return (
            lambda offset: MLPNuisance(seed + offset, True, training_log),
            lambda offset: MLPNuisance(seed + offset, False, training_log),
            {"architecture": CONFIG, "nuisance_training_log": training_log},
        )
    if backend == "tabpfn":
        from baselines.tabpfn.backend import (
            OfficialTabPFNBackend,
            normalize_tabpfn_version,
            tabpfn_version_spec,
        )
        import tabpfn
        version = normalize_tabpfn_version(tabpfn_version)
        spec = tabpfn_version_spec(version)
        factory = OfficialTabPFNBackend({"tabpfn": {
            "model_version": version,
            "checkpoint_revision": spec["checkpoint_revision"],
            "checkpoint_root": str(
                BASELINES_ROOT / "TabPFN" / str(spec["checkpoint_directory"])
            ),
            "n_estimators": 8, "n_preprocessing_jobs": 1, "device": "cuda:0",
            "ignore_pretraining_limits": True,
        }})
        classifier = lambda offset: factory.create_classifier(random_state=seed + offset, role=f"fd-{offset}")
        regressor = lambda offset: factory.create_regressor(random_state=seed + offset, role=f"fd-{offset}")
        return classifier, regressor, {**factory.provenance(), "package_version": tabpfn.__version__}
    raise ValueError(backend)


def fold_seed_from_world_seed(world_seed: int) -> int:
    digest = hashlib.sha256(f"fd_plugin_v2_folds|{int(world_seed)}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def deterministic_fold_assignment(n: int, fold_count: int, fold_seed: int) -> np.ndarray:
    if fold_count < 2 or n < fold_count:
        raise ValueError(f"cannot construct {fold_count} folds from {n} observations")
    permutation = np.random.default_rng(fold_seed).permutation(n)
    assignment = np.empty(n, dtype=np.int64)
    assignment[permutation] = np.arange(n, dtype=np.int64) % fold_count
    if np.any(np.bincount(assignment, minlength=fold_count) == 0):
        raise AssertionError("invalid deterministic fold assignment")
    return assignment


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_oof_mediator_residuals(x_arm: np.ndarray, m_arm: np.ndarray,
                               make_regressor: Callable[[int], Any], *, arm: int,
                               fold_count: int, fold_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit only within one arm; each row receives one held-out prediction."""
    assignment = deterministic_fold_assignment(len(x_arm), fold_count, fold_seed)
    oof = np.full(len(x_arm), np.nan, dtype=np.float64)
    prediction_count = np.zeros(len(x_arm), dtype=np.int64)
    for fold in range(fold_count):
        held_out = assignment == fold
        training = ~held_out
        if np.any(training & held_out) or not np.any(held_out):
            raise AssertionError("invalid OOF split")
        model = make_regressor(100 + 10 * arm + fold)
        model.fit(x_arm[training], m_arm[training])
        oof[held_out] = _predict_chunks(model, x_arm[held_out])
        prediction_count[held_out] += 1
        _release(model)
    if not np.all(prediction_count == 1) or not np.isfinite(oof).all():
        raise AssertionError("each arm observation must receive exactly one finite OOF prediction")
    return np.asarray(m_arm, dtype=np.float64) - oof, assignment


def validate_propensity(probability: np.ndarray) -> np.ndarray:
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    if not np.isfinite(probability).all():
        raise ValueError("propensity predict_proba returned NaN/Inf")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("propensity predict_proba returned a value outside [0,1]")
    return probability


def deterministic_quantile_residuals(residuals: np.ndarray, points: int) -> np.ndarray:
    if points <= 0: raise ValueError("quantile integration points must be positive")
    probabilities = (np.arange(points, dtype=np.float64) + 0.5) / points
    return np.quantile(np.asarray(residuals, dtype=np.float64), probabilities, method="linear")


def empirical_frontdoor_integral(outcome_model: Any, x_query: np.ndarray,
                                 mediator_mean: np.ndarray, residuals: np.ndarray,
                                 propensity: np.ndarray, *, integration_batch_size: int) -> np.ndarray:
    """Exact empirical average, with query blocks containing complete pools."""
    x_query = np.asarray(x_query, dtype=np.float64)
    mediator_mean = np.asarray(mediator_mean, dtype=np.float64).reshape(-1)
    residuals = np.asarray(residuals, dtype=np.float64).reshape(-1)
    propensity = validate_propensity(propensity)
    if len(x_query) != len(mediator_mean) or len(x_query) != len(propensity) or not len(residuals):
        raise ValueError("front-door integration inputs are not aligned")
    if integration_batch_size <= 0: raise ValueError("integration_batch_size must be positive")
    queries_per_block = max(1, integration_batch_size // len(residuals))
    result = np.empty(len(x_query), dtype=np.float64)
    for start in range(0, len(x_query), queries_per_block):
        stop = min(len(x_query), start + queries_per_block)
        mediator = mediator_mean[start:stop, None] + residuals[None, :]
        repeated_x = np.repeat(x_query[start:stop], len(residuals), axis=0)
        flat_m = mediator.reshape(-1)
        q0 = np.column_stack([repeated_x, flat_m, np.zeros(len(flat_m))])
        q1 = np.column_stack([repeated_x, flat_m, np.ones(len(flat_m))])
        mu0 = _predict_chunks(outcome_model, q0, chunk=integration_batch_size).reshape(stop-start, -1)
        mu1 = _predict_chunks(outcome_model, q1, chunk=integration_batch_size).reshape(stop-start, -1)
        weights = propensity[start:stop, None]
        result[start:stop] = np.mean((1.0-weights)*mu0 + weights*mu1, axis=1)
    return result


def _residual_metadata(raw: np.ndarray, centered: np.ndarray) -> dict[str, Any]:
    return {"raw_mean": float(np.mean(raw)), "raw_sd": float(np.std(raw)),
            "centered_mean": float(np.mean(centered)), "centered_sd": float(np.std(centered)),
            "count": int(len(raw))}


def run(input_path: Path, prediction_path: Path, metadata_path: Path, backend: str,
        seed: int, *, fold_count: int = 5,
        integration_method: str = "exact", integration_points: int | None = None,
        integration_batch_size: int = 8192,
        tabpfn_version: str = "v3") -> None:
    """Fit the production OOF-residual, deterministic-integration estimator."""
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    expected = {"X_context", "treatment", "mediator", "outcome", "X_query"}
    if set(arrays) != expected: raise ValueError(f"FD assisted input mismatch: {sorted(arrays)}")
    x = np.asarray(arrays["X_context"], dtype=np.float64)
    t = np.asarray(arrays["treatment"], dtype=np.int64)
    m = np.asarray(arrays["mediator"], dtype=np.float64)
    y = np.asarray(arrays["outcome"], dtype=np.float64)
    xq = np.asarray(arrays["X_query"], dtype=np.float64)
    if not (x.ndim == xq.ndim == 2 and x.shape[1] == xq.shape[1]
            and len(x) == len(t) == len(m) == len(y)):
        raise ValueError("FD context/query feature contract mismatch")
    if integration_method not in {"exact", "quantile"}: raise ValueError("invalid integration method")
    if integration_method == "quantile" and integration_points not in {256, 512}:
        raise ValueError("production quantile fallback permits only 256 or 512 points")

    make_classifier, make_regressor, provenance = (
        _factories(backend, seed, tabpfn_version)
        if backend == "tabpfn" and tabpfn_version != "v3"
        else _factories(backend, seed)
    )
    peak_gpu_memory_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass
    fit_time = prediction_time = 0.0
    started = time.perf_counter(); treatment_model = make_classifier(1); treatment_model.fit(x, t)
    fit_time += time.perf_counter() - started
    started = time.perf_counter()
    propensity = validate_propensity(_predict_chunks(treatment_model, xq, probability=True))
    prediction_time += time.perf_counter() - started
    _release(treatment_model)

    fold_seed = fold_seed_from_world_seed(seed)
    means, residuals, assignments, stats, arm_counts = {}, {}, {}, {}, {}
    for arm in (0, 1):
        selected = t == arm
        arm_counts[str(arm)] = int(np.sum(selected))
        if np.sum(selected) < max(30, fold_count): raise RuntimeError(f"FD treatment arm {arm} too small")
        started = time.perf_counter()
        raw, assignment = fit_oof_mediator_residuals(x[selected], m[selected], make_regressor,
            arm=arm, fold_count=fold_count, fold_seed=fold_seed)
        centered = raw - np.mean(raw)
        residuals[arm], assignments[arm], stats[str(arm)] = centered, assignment, _residual_metadata(raw, centered)
        final_model = make_regressor(10 + arm); final_model.fit(x[selected], m[selected])
        fit_time += time.perf_counter() - started
        started = time.perf_counter(); means[arm] = _predict_chunks(final_model, xq)
        prediction_time += time.perf_counter() - started; _release(final_model)

    started = time.perf_counter(); outcome_model = make_regressor(30)
    outcome_model.fit(np.column_stack([x, m, t]), y); fit_time += time.perf_counter() - started
    theta, point_counts = {}, {}
    started = time.perf_counter()
    for arm in (0, 1):
        pool = residuals[arm]
        if integration_method == "quantile":
            pool = deterministic_quantile_residuals(pool, int(integration_points))
        point_counts[str(arm)] = int(len(pool))
        theta[arm] = empirical_frontdoor_integral(outcome_model, xq, means[arm], pool, propensity,
                                                  integration_batch_size=integration_batch_size)
    tau_hat = theta[1] - theta[0]
    prediction_time += time.perf_counter() - started; _release(outcome_model)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_memory_mb = float(torch.cuda.max_memory_allocated() / 1024**2)
    except (ImportError, RuntimeError):
        pass
    if tau_hat.shape != (len(xq),) or not np.isfinite(tau_hat).all():
        raise ValueError("FD plug-in returned invalid predictions")

    version = ESTIMATOR_VERSION if integration_method == "exact" else f"fd_plugin_v2_oof_quantile{integration_points}"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(prediction_path, tau_hat=tau_hat, theta_0=theta[0], theta_1=theta[1])
    metadata = {
        "estimator_version": version, "backend": backend, "seed": seed,
        "tabpfn_version": (
            provenance.get("tabpfn_version") if backend == "tabpfn" else None
        ),
        "estimator_contract_version": "fd_v2",
        "formula": "continuous-mediator front-door empirical plug-in",
        "mediator_type": "continuous",
        "nuisance_components": ["p(T|X)", "p(M|T,X)", "E(Y|M,T,X)"],
        "fold_count": fold_count, "crossfit_folds": fold_count, "fold_seed": fold_seed,
        "fold_assignment_hash": _array_hash(assignments[0], assignments[1]),
        "fold_assignment_hash_by_arm": {str(a): _array_hash(assignments[a]) for a in (0, 1)},
        "residual_method": "treatment-arm-specific deterministic KFold OOF",
        "mediator_residuals": "arm_specific_oof",
        "residual_centered": True, "residual_centering": "arm_specific",
        "mediator_location_refit": "full_context_arm_specific",
        "residual_stats_by_arm": stats,
        "integration_method": "exact_empirical" if integration_method == "exact" else "deterministic_empirical_quantiles",
        "integration": "exact_empirical_support" if integration_method == "exact" else "deterministic_empirical_quantiles",
        "stochastic_mediator_draws": 0,
        "number_of_integration_points": point_counts, "integration_batch_size": integration_batch_size,
        "exact_integration": integration_method == "exact", "propensity_clipping": None,
        "propensity_min": float(np.min(propensity)), "propensity_max": float(np.max(propensity)),
        "arm_counts": arm_counts,
        "fit_time_sec": fit_time, "inference_time_sec": prediction_time,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "query_used_for_fit_or_tuning": False, "identification_wrapper_used": True,
        "support_status": "ID-A", "final_query_columns": "X only",
        "final_query_schema": ["X"], "factual_mediator_in_query": False,
        "provenance": provenance,
    }
    metadata_path.write_text(json.dumps(_plain(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("xgboost", "tabpfn"), required=True)
    parser.add_argument("--input", type=Path, required=True); parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--integration-method", choices=("exact", "quantile"), default="exact")
    parser.add_argument("--integration-points", type=int); parser.add_argument("--integration-batch-size", type=int, default=8192)
    parser.add_argument("--tabpfn-version", choices=("v3", "v3.5"), default="v3")
    args = parser.parse_args()
    run(args.input, args.prediction, args.metadata, args.backend, args.seed,
        fold_count=args.fold_count, integration_method=args.integration_method,
        integration_points=args.integration_points, integration_batch_size=args.integration_batch_size,
        tabpfn_version=args.tabpfn_version)
    return 0


if __name__ == "__main__": raise SystemExit(main())
