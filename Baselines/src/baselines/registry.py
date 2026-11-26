from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping

from .base import BaseCATEEstimator
from .config import load_profile, merge_config


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    category: str
    backend: str
    supports_continuous: bool
    supports_binary: bool
    native_binary_probability: bool
    supports_intervals: bool
    uses_cross_fitting: bool
    implementation_path: str


def _spec(
    model_id: str,
    category: str,
    backend: str,
    implementation_path: str,
    *,
    supports_binary: bool = True,
    native_binary_probability: bool = False,
    supports_intervals: bool = False,
    uses_cross_fitting: bool = False,
) -> ModelSpec:
    return ModelSpec(
        model_id=model_id,
        category=category,
        backend=backend,
        supports_continuous=True,
        supports_binary=supports_binary,
        native_binary_probability=native_binary_probability,
        supports_intervals=supports_intervals,
        uses_cross_fitting=uses_cross_fitting,
        implementation_path=implementation_path,
    )


_MODEL_SPECS = (
    _spec("tabpfn_s", "tabpfn", "python", "baselines.tabpfn.estimators:TabPFNSLearner", native_binary_probability=True),
    _spec("tabpfn_t", "tabpfn", "python", "baselines.tabpfn.estimators:TabPFNTLearner", native_binary_probability=True),
    _spec("tabpfn_x", "tabpfn", "python", "baselines.tabpfn.estimators:TabPFNXLearner", native_binary_probability=True, uses_cross_fitting=True),
    _spec("tabpfn_dr", "tabpfn", "python", "baselines.tabpfn.estimators:TabPFNDRLearner", native_binary_probability=True, uses_cross_fitting=True),
    _spec("tabpfn_dml", "tabpfn", "python", "baselines.tabpfn.estimators:TabPFNDMLEstimator", native_binary_probability=True, uses_cross_fitting=True),
    _spec("s_learner", "meta", "python", "baselines.meta.s_learner:SLearnerEstimator", native_binary_probability=True),
    _spec("t_learner", "meta", "python", "baselines.meta.t_learner:TLearnerEstimator", native_binary_probability=True),
    _spec("x_learner", "meta", "python", "baselines.meta.x_learner:XLearnerEstimator", native_binary_probability=True, uses_cross_fitting=True),
    _spec("r_learner", "meta", "python", "baselines.meta.r_learner:RLearnerEstimator", uses_cross_fitting=True),
    _spec("pw_learner", "orthogonal", "python", "baselines.orthogonal.pw:PWLearnerEstimator", uses_cross_fitting=True),
    _spec("dr_learner", "orthogonal", "python", "baselines.orthogonal.econml_estimators:DRLearnerEstimator", uses_cross_fitting=True),
    _spec("dr_ortho_forest", "orthogonal", "python", "baselines.orthogonal.econml_estimators:DROrthoForestEstimator", supports_intervals=True, uses_cross_fitting=True),
    _spec("causal_tree", "tree", "r", "baselines.r_backend.estimators:CausalTreeEstimator"),
    _spec("grf", "tree", "r", "baselines.r_backend.estimators:GRFEstimator", supports_intervals=True),
    _spec("causal_forest_dml", "orthogonal", "python", "baselines.orthogonal.econml_estimators:CausalForestDMLEstimator", supports_intervals=True, uses_cross_fitting=True),
    _spec("bcf", "bayesian", "r", "baselines.r_backend.estimators:BCFEstimator", supports_binary=False, supports_intervals=True, uses_cross_fitting=True),
    _spec("tarnet", "neural", "pytorch", "baselines.neural.estimators:TARNetEstimator", native_binary_probability=True),
    _spec("cfrnet", "neural", "pytorch", "baselines.neural.estimators:CFRNetEstimator", native_binary_probability=True),
    _spec("dragonnet", "neural", "pytorch", "baselines.neural.estimators:DragonNetEstimator", native_binary_probability=True),
    _spec("ra_net", "neural", "pytorch", "baselines.neural.estimators:RANetEstimator", native_binary_probability=True, uses_cross_fitting=True),
)
_SPECS_BY_ID = {spec.model_id: spec for spec in _MODEL_SPECS}


def list_model_specs() -> list[ModelSpec]:
    return list(_MODEL_SPECS)


def create_estimator(
    model_id: str,
    config: Mapping[str, Any],
) -> BaseCATEEstimator:
    try:
        spec = _SPECS_BY_ID[model_id]
    except KeyError as error:
        raise KeyError(f"Unknown model_id: {model_id}") from error

    profile_name = str(config.get("profile", "default"))
    overrides = {key: value for key, value in config.items() if key != "profile"}
    resolved_config = merge_config(load_profile(profile_name), overrides)
    module_name, class_name = spec.implementation_path.split(":", maxsplit=1)
    estimator_class = getattr(import_module(module_name), class_name)
    estimator = estimator_class(config=resolved_config)
    if not isinstance(estimator, BaseCATEEstimator):
        raise TypeError(f"{spec.implementation_path} is not a BaseCATEEstimator")
    return estimator
