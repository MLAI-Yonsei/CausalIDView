from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as distribution_version
from typing import Any, Mapping

from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .exceptions import DependencyUnavailableError
from .validation import OutcomeType


_LOCKED_XGBOOST_VERSION = "3.1.3"


def _forest_kwargs(config: Mapping[str, Any], random_state: int) -> dict[str, Any]:
    forest = config.get("sklearn", {})
    return {
        "n_estimators": int(forest.get("n_estimators", 500)),
        "min_samples_leaf": int(forest.get("min_samples_leaf", 5)),
        "max_features": forest.get("max_features", "sqrt"),
        "n_jobs": int(forest.get("n_jobs", -1)),
        "random_state": random_state,
    }


def _base_learner_family(config: Mapping[str, Any]) -> str:
    base_learner = config.get("base_learner", {})
    if not isinstance(base_learner, Mapping):
        raise ValueError("base_learner must be a mapping")
    family = str(base_learner.get("family", "random_forest"))
    if family not in {"random_forest", "xgboost"}:
        raise ValueError(f"Unsupported base learner family: {family}")
    return family


def _xgboost_runtime_version() -> str:
    """Return the executing XGBoost identity, rejecting any lock drift."""

    try:
        module = import_module("xgboost")
        installed = distribution_version("xgboost")
    except (ImportError, PackageNotFoundError) as error:
        raise DependencyUnavailableError(
            f"xgboost=={_LOCKED_XGBOOST_VERSION} is required"
        ) from error
    imported = getattr(module, "__version__", None)
    if imported != installed or installed != _LOCKED_XGBOOST_VERSION:
        raise DependencyUnavailableError(
            f"xgboost=={_LOCKED_XGBOOST_VERSION} is required; "
            f"imported={imported!r}, installed={installed!r}"
        )
    return installed


def _xgboost_model(kind: str, config: Mapping[str, Any], random_state: int) -> Any:
    _xgboost_runtime_version()
    try:
        from xgboost import XGBClassifier, XGBRegressor
    except ImportError as error:
        raise DependencyUnavailableError("xgboost==3.1.3 is required") from error
    xgb = config["xgboost"]
    params = {
        "n_estimators": int(xgb["n_estimators"]),
        "max_depth": int(xgb["max_depth"]),
        "learning_rate": float(xgb["learning_rate"]),
        "subsample": float(xgb["subsample"]),
        "colsample_bytree": float(xgb["colsample_bytree"]),
        "n_jobs": int(xgb["n_jobs"]),
        "tree_method": str(xgb["tree_method"]),
        "random_state": random_state,
    }
    if kind == "classifier":
        return XGBClassifier(objective="binary:logistic", **params)
    return XGBRegressor(objective="reg:squarederror", **params)


def model_factory_metadata(config: Mapping[str, Any]) -> Mapping[str, Any]:
    family = _base_learner_family(config)
    if family == "xgboost":
        runtime_version = _xgboost_runtime_version()
        return {
            "family": "xgboost",
            "package": "xgboost",
            "package_version": runtime_version,
        }
    return {
        "family": "random_forest",
        "package": "scikit-learn",
        "package_version": sklearn_version,
    }


def make_outcome_model(
    outcome_type: OutcomeType,
    config: Mapping[str, Any],
    *,
    random_state: int,
) -> Any:
    family = _base_learner_family(config)
    if family == "xgboost":
        return _xgboost_model(
            "classifier" if outcome_type == "binary" else "regressor", config, random_state
        )
    kwargs = _forest_kwargs(config, random_state)
    if outcome_type == "binary":
        return RandomForestClassifier(**kwargs)
    return RandomForestRegressor(**kwargs)


def make_effect_model(
    config: Mapping[str, Any],
    *,
    random_state: int,
) -> Any:
    if _base_learner_family(config) == "xgboost":
        return _xgboost_model("regressor", config, random_state)
    return RandomForestRegressor(**_forest_kwargs(config, random_state))


def make_propensity_model(
    config: Mapping[str, Any],
    *,
    random_state: int,
) -> Any:
    if _base_learner_family(config) == "xgboost":
        return _xgboost_model("classifier", config, random_state)
    return RandomForestClassifier(**_forest_kwargs(config, random_state))
