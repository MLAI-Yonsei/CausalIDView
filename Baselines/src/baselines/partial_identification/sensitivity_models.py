from __future__ import annotations

import sys
import time
from typing import Any

import numpy as np

from ..checkpoints import validate_checkpoint
from ..config import BASELINES_ROOT
from .assets import B_LEARNER_COMMIT, CSA_PFN_SPEC, GAMMAS, NEURALCSA_COMMIT, NEURALCSA_ROOT


def _csa_pfn(arrays: dict[str, np.ndarray], context: np.ndarray, query_rows: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn as nn
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    class FlatPFN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Linear(15, 128)
            layer = nn.TransformerEncoderLayer(
                128, 4, 512, 0.0, "relu", batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, 10)
            self.head = nn.Linear(128, 1)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.head(self.encoder(self.embed(value))).squeeze(-1)

    checkpoint = validate_checkpoint(CSA_PFN_SPEC, BASELINES_ROOT.parent)
    started_load = time.perf_counter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("tabular_embedder_0."):
            mapped["embed." + key.split(".")[-1]] = value
        elif key.startswith("head."):
            mapped[key] = value
        elif key.startswith("transformer_encoder_1."):
            mapped["encoder." + key.split(".", 1)[1]] = value
    model = FlatPFN()
    model.load_state_dict(mapped, strict=True)
    model.to(device).eval()
    load_time = time.perf_counter() - started_load

    # The public checkpoint has a strict 10-feature contract while MasterSCM
    # has 50 X features.  PCA is fit on context X only, then its ten outputs are
    # standardized with context statistics.  This is an explicit OOS adapter,
    # not an oracle or label-informed projection.
    fit_started = time.perf_counter()
    x_scaler = StandardScaler().fit(arrays["X"][context])
    x_context_50 = x_scaler.transform(arrays["X"][context])
    pca = PCA(n_components=10, svd_solver="full").fit(x_context_50)
    x_context = pca.transform(x_context_50)
    x_query = pca.transform(x_scaler.transform(arrays["X"][query_rows]))
    projected_scaler = StandardScaler().fit(x_context)
    x_context = projected_scaler.transform(x_context).astype(np.float32)
    x_query = projected_scaler.transform(x_query).astype(np.float32)
    treatment = np.asarray(arrays["T"][context], dtype=np.float32)
    outcome = np.asarray(arrays["Y"][context], dtype=np.float64)
    y_mean = float(np.mean(outcome))
    y_std = float(np.std(outcome))
    if y_std <= 1.0e-12:
        raise RuntimeError("CSA-PFN outcome normalization has zero scale")
    outcome_normalized = ((outcome - y_mean) / y_std).astype(np.float32)
    n_context = len(context)
    query_tokens = 2 * 2 * len(GAMMAS)
    context_tokens = np.zeros((n_context, 15), dtype=np.float32)
    context_tokens[:, :10] = x_context
    context_tokens[:, 10] = treatment
    context_tokens[:, 11] = outcome_normalized
    context_tokens[:, 14] = 1.0
    fit_time = time.perf_counter() - fit_started

    output = np.empty((len(query_rows), 2, 2, len(GAMMAS)), dtype=np.float64)
    prediction_started = time.perf_counter()
    with torch.inference_mode():
        for batch_start in range(0, len(query_rows), batch_size):
            batch_stop = min(batch_start + batch_size, len(query_rows))
            sequences = np.zeros((batch_stop - batch_start, n_context + query_tokens, 15), dtype=np.float32)
            sequences[:, :n_context] = context_tokens
            for local, query_x in enumerate(x_query[batch_start:batch_stop]):
                cursor = n_context
                for arm in (0, 1):
                    for bound in (0, 1):  # public API: 0 upper, 1 lower
                        for gamma in GAMMAS:
                            sequences[local, cursor, :10] = query_x
                            sequences[local, cursor, 10] = arm
                            sequences[local, cursor, 12] = gamma
                            sequences[local, cursor, 13] = bound
                            cursor += 1
            tensor = torch.from_numpy(sequences).to(device)
            predicted = model(tensor)[:, n_context:].float().cpu().numpy()
            output[batch_start:batch_stop] = predicted.reshape(batch_stop - batch_start, 2, 2, len(GAMMAS))
    prediction_time = time.perf_counter() - prediction_started

    upper_arm = output[:, :, 0, :].transpose(2, 0, 1)  # gamma, query, arm
    lower_arm = output[:, :, 1, :].transpose(2, 0, 1)
    raw_lower = (lower_arm[:, :, 1] - upper_arm[:, :, 0]) * y_std
    raw_upper = (upper_arm[:, :, 1] - lower_arm[:, :, 0]) * y_std
    # Checkpoint's documented post-hoc monotonicization is applied per arm.
    upper_arm = np.maximum.accumulate(upper_arm, axis=0)
    lower_arm = np.minimum.accumulate(lower_arm, axis=0)
    lower = (lower_arm[:, :, 1] - upper_arm[:, :, 0]) * y_std
    upper = (upper_arm[:, :, 1] - lower_arm[:, :, 0]) * y_std
    metadata = {
        "implementation": "public CSA-PFN MSM replication checkpoint",
        "source_repository": "https://huggingface.co/arunkatika/csa-pfn-msm",
        "paper": "Javurek et al. (2026), arXiv:2605.10590",
        "checkpoint": str(checkpoint), "checkpoint_sha256": CSA_PFN_SPEC["sha256"],
        "checkpoint_load_time_sec": load_time, "fit_time_sec": fit_time,
        "prediction_time_sec": prediction_time, "device": device,
        "native_feature_count": 10, "master_scm_feature_count": int(arrays["X"].shape[1]),
        "feature_adapter": "context-only StandardScaler + PCA(50->10) + StandardScaler",
        "feature_adapter_in_native_training_support": False,
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "raw_cate_crossing_rate": float(np.mean(raw_lower > raw_upper)),
        "posthoc_monotonicization": "per-arm cumulative min(lower)/max(upper), as documented by checkpoint authors",
    }
    return lower, upper, metadata


def _b_learner(arrays: dict[str, np.ndarray], context: np.ndarray, query_rows: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from ._blearner.blearner import BLearner
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn_quantile import RandomForestQuantileRegressor

    x = np.asarray(arrays["X"][context], dtype=np.float64)
    treatment = np.asarray(arrays["T"][context], dtype=np.int64)
    outcome = np.asarray(arrays["Y"][context], dtype=np.float64)
    x_query = np.asarray(arrays["X"][query_rows], dtype=np.float64)
    lower = np.empty((len(GAMMAS), len(query_rows)), dtype=np.float64)
    upper = np.empty_like(lower)
    fit_started = time.perf_counter()
    per_gamma_fit: list[float] = []
    per_gamma_prediction: list[float] = []
    for gamma_index, gamma in enumerate(GAMMAS):
        gamma_started = time.perf_counter()
        tau = float(gamma / (1.0 + gamma))
        forest_parameters = dict(
            n_estimators=100, min_samples_leaf=8, max_features=0.8,
            n_jobs=1, random_state=seed + 1000 * gamma_index,
        )
        estimator = BLearner(
            propensity_model=RandomForestClassifier(**forest_parameters),
            quantile_plus_model=RandomForestQuantileRegressor(q=tau, **forest_parameters),
            quantile_minus_model=RandomForestQuantileRegressor(q=1.0 - tau, **forest_parameters),
            mu_model=RandomForestRegressor(**forest_parameters),
            cate_bounds_model=RandomForestRegressor(**forest_parameters),
            use_rho=True, gamma=float(gamma), cv=5,
            random_state=seed + 1000 * gamma_index,
        )
        estimator.fit(x, treatment, outcome)
        per_gamma_fit.append(time.perf_counter() - gamma_started)
        prediction_started = time.perf_counter()
        lower[gamma_index], upper[gamma_index] = estimator.effect(x_query)
        per_gamma_prediction.append(time.perf_counter() - prediction_started)
    fit_time = time.perf_counter() - fit_started - float(np.sum(per_gamma_prediction))
    prediction_time = float(np.sum(per_gamma_prediction))
    return lower, upper, {
        "implementation": "authors' B-Learner repository",
        "source_repository": "https://github.com/CausalML/BLearner",
        "source_commit": B_LEARNER_COMMIT,
        "paper": "Oprescu et al. (ICML 2023), arXiv:2304.10577",
        "nuisance_backend": "authors' CATE_Nuisance_Model; sklearn-quantile and 100-tree random forests",
        "cross_fitting_folds": 5, "use_rho": True,
        "per_gamma_fit_time_sec": per_gamma_fit, "per_gamma_prediction_time_sec": per_gamma_prediction,
        "fit_time_sec": fit_time, "prediction_time_sec": prediction_time,
        "checkpoint_load_time_sec": 0.0, "device": "cpu",
    }


def _neuralcsa(arrays: dict[str, np.ndarray], context: np.ndarray, query_rows: np.ndarray, seed: int, smoke: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit the authors' NeuralCSA model on one frozen observational world."""
    if not (NEURALCSA_ROOT / "models" / "auxillary.py").is_file():
        raise FileNotFoundError("NeuralCSA source missing. Run scripts/fig6/prepare_fig6_bounds.py --only neuralcsa")
    sys.path.insert(0, str(NEURALCSA_ROOT))
    import random
    import torch
    from sklearn.preprocessing import StandardScaler
    from data.data_structures import CausalDataset
    from models.auxillary import JointModel

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    x_scaler = StandardScaler().fit(arrays["X"][context])
    x = x_scaler.transform(arrays["X"][context]).astype(np.float32)
    x_query = x_scaler.transform(arrays["X"][query_rows]).astype(np.float32)
    treatment = np.asarray(arrays["T"][context], dtype=np.float32).reshape(-1, 1)
    outcome_raw = np.asarray(arrays["Y"][context], dtype=np.float64)
    y_mean, y_std = float(np.mean(outcome_raw)), float(np.std(outcome_raw))
    if y_std <= 1.0e-12:
        raise RuntimeError("NeuralCSA outcome scale is zero")
    outcome = ((outcome_raw - y_mean) / y_std).astype(np.float32).reshape(-1, 1)
    d_train = CausalDataset(x=x, a=treatment, y=outcome, a_type="discrete")

    # Smoke validates the complete propensity -> Stage 1 -> Stage 2 path with
    # short training. Full runs use the authors' IHDP MSM hyperparameters.
    propensity_epochs, stage1_epochs, stage2_epochs = (2, 2, 3) if smoke else (30, 50, 60)
    base_stage2 = {
        "neptune": False, "dim_y": 1, "bounds": "both", "a_type": "discrete",
        "batch_size": 64, "count_bins": 4, "spline_bound": 3.5,
        "dim_hidden": 20, "n_samples_train": 100, "epochs": stage2_epochs,
        "nested_epochs": 10, "lr": 0.0005, "alpha": 1.8,
        "lambda_init": 0.05, "mu_init": 0.05, "cnf": True,
        "dim_context": x.shape[1] + 1, "causal_query": {"name": "expectation"},
    }
    stage2_upper = []
    stage2_lower = []
    # Gamma=1 is evaluated by the authors' analytical GMSM routine because
    # the constrained Stage-2 optimization is singular at a zero-width set.
    for gamma in GAMMAS[1:]:
        entry = base_stage2 | {"sensitivity_model": {"name": "msm", "gamma": float(gamma)}}
        stage2_upper.append(dict(entry))
        stage2_lower.append(dict(entry))
    config = {
        "propensity": {"dim_input": x.shape[1], "dim_output": 1, "out_type": "discrete", "neptune": False, "batch_size": 128, "dim_hidden": 20, "dropout": 0.0, "epochs": propensity_epochs, "lr": 0.0001},
        "stage1": {"dim_context": x.shape[1] + 1, "dim_y": 1, "neptune": False, "batch_size": 64, "count_bins": 8, "dim_hidden": 20, "epochs": stage1_epochs, "lr": 0.001},
        "stage2_upper": stage2_upper, "stage2_lower": stage2_lower,
    }
    model = JointModel(config)
    fit_started = time.perf_counter()
    model.fit_propensity(d_train, None)
    model.fit_stage1(d_train, None)
    fits = model.fit_stage2(d_train, seed=seed, savepath=None)
    expected_fits = len(GAMMAS) - 1
    if not isinstance(fits, list) or len(fits) != 2 * expected_fits:
        raise ValueError("NeuralCSA did not complete all ten Stage 2 fits")
    keys = [tuple(result) for result in fits if isinstance(result, dict)]
    if keys.count(("lower",)) != expected_fits or keys.count(("upper",)) != expected_fits:
        raise ValueError("NeuralCSA did not complete all five lower and five upper Stage 2 fits")
    fit_time = time.perf_counter() - fit_started

    data_a1 = CausalDataset(x=x_query, a=np.ones((len(x_query), 1), dtype=np.float32), y=None, a_type="discrete")
    data_a0 = CausalDataset(x=x_query, a=np.zeros((len(x_query), 1), dtype=np.float32), y=None, a_type="discrete")
    prediction_device = next(model.stage1.parameters()).device
    # Lightning teardown may return modules to CPU but cannot migrate torch
    # Distribution objects. Rebuild those objects once more on the final
    # prediction device.
    model.stage1.rebuild_distributions(prediction_device)
    for stage2_model in [*model.stage2_lower, *model.stage2_upper]:
        stage2_model.on_fit_start()
    for dataset in (data_a1, data_a0):
        for key, value in dataset.data.items():
            if isinstance(value, torch.Tensor):
                dataset.data[key] = value.to(prediction_device)
    lower = np.empty((len(GAMMAS), len(x_query)), dtype=np.float64)
    upper = np.empty_like(lower)
    prediction_started = time.perf_counter()
    # Public analytical GMSM path at Gamma=1 (both endpoints coincide).
    g1_arm1 = model.GMSM_bounds(data_a1, n_samples=4096, gammas=torch.tensor(1.0), sensitivity_model="msm")
    g1_arm0 = model.GMSM_bounds(data_a0, n_samples=4096, gammas=torch.tensor(1.0), sensitivity_model="msm")
    lower[0] = (g1_arm1["Q_minus"] - g1_arm0["Q_plus"]).detach().cpu().numpy().reshape(-1) * y_std
    upper[0] = (g1_arm1["Q_plus"] - g1_arm0["Q_minus"]).detach().cpu().numpy().reshape(-1) * y_std
    for gamma_index, (model_lower, model_upper) in enumerate(zip(model.stage2_lower, model.stage2_upper), start=1):
        lower1 = model_lower.test_query(data_a1, n_samples=4096).detach().cpu().numpy().reshape(-1)
        upper1 = model_upper.test_query(data_a1, n_samples=4096).detach().cpu().numpy().reshape(-1)
        lower0 = model_lower.test_query(data_a0, n_samples=4096).detach().cpu().numpy().reshape(-1)
        upper0 = model_upper.test_query(data_a0, n_samples=4096).detach().cpu().numpy().reshape(-1)
        lower[gamma_index] = (lower1 - upper0) * y_std
        upper[gamma_index] = (upper1 - lower0) * y_std
    prediction_time = time.perf_counter() - prediction_started
    return lower, upper, {
        "implementation": "authors' NeuralCSA JointModel (conditional Stage-2 normalizing flows)",
        "source_repository": "https://github.com/DennisFrauen/NeuralCSA",
        "source_commit": NEURALCSA_COMMIT,
        "compatibility_patch": "register nested models and rebuild non-buffer distributions on Lightning device moves",
        "paper": "Frauen et al., ICLR 2024, A Neural Framework for Generalized Causal Sensitivity Analysis",
        "configuration_source": "authors' IHDP MSM model configs",
        "propensity_epochs": propensity_epochs, "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs, "stage2_models_trained": 2 * (len(GAMMAS) - 1),
        "gamma_one_method": "authors' analytical GMSM routine using fitted Stage-1 model",
        "test_mc_samples": 4096, "x_normalization": "context StandardScaler",
        "y_normalization": {"context_mean": y_mean, "context_std": y_std},
        "fit_time_sec": fit_time, "prediction_time_sec": prediction_time,
        "checkpoint_load_time_sec": 0.0, "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "smoke_short_training": smoke,
    }
