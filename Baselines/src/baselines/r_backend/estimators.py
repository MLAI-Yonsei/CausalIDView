from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ..base import BaseCATEEstimator
from ..exceptions import CATEValidationError
from ..prediction import CATEPrediction, normalize_prediction
from .sources import get_source_record
from ..validation import OutcomeType, validate_fit_data, validate_predict_X
from .runner import RBackendRunner


SCRIPT_ROOT = Path(__file__).resolve().parent / "scripts"


class _BaseREstimator(BaseCATEEstimator):
    script_name: str

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        runner: RBackendRunner | None = None,
    ) -> None:
        self.config = deepcopy(dict(config))
        self.seed = int(self.config.get("seed", 0))
        self.runner = runner or RBackendRunner()

    def _before_fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        outcome_type: OutcomeType,
    ) -> None:
        return None

    def _transform_X(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return X

    def _algorithm_config(
        self,
        *,
        is_fit: bool,
        X: NDArray[np.float64],
    ) -> dict[str, Any]:
        return {}

    def _runtime_config(
        self,
        *,
        is_fit: bool,
        X: NDArray[np.float64],
    ) -> dict[str, Any]:
        source = get_source_record(self.model_id)
        return {
            "seed": self.seed,
            "thread_count": int(self.config.get("thread_count", 1)),
            "r_timeout_seconds": float(self.config.get("r_timeout_seconds", 3600)),
            "outcome_type": self.outcome_type_,
            "source": source,
            **self._algorithm_config(is_fit=is_fit, X=X),
        }

    def fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> Self:
        X_array, treatment_array, y_array = validate_fit_data(
            X, treatment, y, outcome_type=outcome_type
        )
        self.outcome_type_ = outcome_type
        self._before_fit(X_array, treatment_array, y_array, outcome_type)
        backend_X = self._transform_X(X_array)
        self.train_X_ = backend_X
        self.train_treatment_ = treatment_array
        self.train_y_ = y_array
        self.n_features_in_ = X_array.shape[1]
        result = self.runner.run(
            model_id=self.model_id,
            script=SCRIPT_ROOT / self.script_name,
            mode="fit",
            config=self._runtime_config(is_fit=True, X=X_array),
            train_X=backend_X,
            train_treatment=treatment_array,
            train_y=y_array,
            predict_X=backend_X,
        )
        if result.paths is None:
            raise RuntimeError("R backend result did not report artifact paths")
        self.artifact_dir_ = result.paths.artifact_dir
        self.fit_result_ = result
        self._is_fitted = True
        return self

    def predict(
        self,
        X: NDArray[np.float64],
        *,
        return_po: bool = False,
        return_interval: bool = False,
        alpha: float = 0.05,
    ) -> CATEPrediction:
        self._require_fitted()
        X_array = validate_predict_X(X, alpha=alpha)
        if X_array.shape[1] != self.n_features_in_:
            raise CATEValidationError(
                f"X must have {self.n_features_in_} features, got {X_array.shape[1]}"
            )
        backend_X = self._transform_X(X_array)
        result = self.runner.run(
            model_id=self.model_id,
            script=SCRIPT_ROOT / self.script_name,
            mode="predict",
            config={
                **self._runtime_config(is_fit=False, X=X_array),
                "interval_alpha": float(alpha),
            },
            train_X=self.train_X_,
            train_treatment=self.train_treatment_,
            train_y=self.train_y_,
            predict_X=backend_X,
            artifact_dir=self.artifact_dir_,
        )
        self.last_result_ = result
        if return_interval and (
            result.cate_lower is None or result.cate_upper is None
        ):
            raise CATEValidationError(
                f"{self.model_id} R backend did not return requested intervals"
            )
        return normalize_prediction(
            CATEPrediction(
                cate=result.cate,
                mu0=result.mu0 if return_po else None,
                mu1=result.mu1 if return_po else None,
                cate_lower=result.cate_lower if return_interval else None,
                cate_upper=result.cate_upper if return_interval else None,
                metadata=self.get_metadata(),
            ),
            len(X_array),
        )

    def get_metadata(self) -> Mapping[str, Any]:
        source = get_source_record(self.model_id)
        metadata: dict[str, Any] = {
            "model_id": self.model_id,
            "seed": self.seed,
            "backend": source["backend"],
            "implementation_variant": source["implementation_variant"],
            "source": source,
            "binary_outcome_handling": "numerical conditional-mean difference",
            "config_summary": deepcopy(self.config),
        }
        if getattr(self, "_is_fitted", False):
            metadata["artifact_dir"] = str(self.artifact_dir_)
        return metadata

class CausalTreeEstimator(_BaseREstimator):
    model_id = "causal_tree"
    script_name = "causal_tree.R"

    def _algorithm_config(self, *, is_fit: bool, X: NDArray[np.float64]) -> dict[str, Any]:
        tree = dict(self.config.get("causal_tree", {}))
        return {
            "split_rule": str(tree.get("split_rule", "CT")),
            "cv_option": str(tree.get("cv_option", "CT")),
            "cross_validation_folds": int(tree.get("cross_validation_folds", 5)),
            "minsize": int(tree.get("minsize", 20)),
            "cp": float(tree.get("cp", 0.0)),
        }


class GRFEstimator(_BaseREstimator):
    model_id = "grf"
    script_name = "grf.R"

    def _algorithm_config(self, *, is_fit: bool, X: NDArray[np.float64]) -> dict[str, Any]:
        grf = dict(self.config.get("grf", {}))
        feature_count = int(X.shape[1])
        return {
            "n_estimators": int(grf.get("n_estimators", 2000)),
            "sample_fraction": float(grf.get("sample_fraction", 0.5)),
            "mtry": int(
                grf.get(
                    "mtry",
                    min(int(np.ceil(np.sqrt(feature_count)) + 20), feature_count),
                )
            ),
            "honesty": bool(grf.get("honesty", True)),
            "honesty_fraction": float(grf.get("honesty_fraction", 0.5)),
            "min_node_size": int(grf.get("min_node_size", 5)),
            "honesty_prune_leaves": bool(grf.get("honesty_prune_leaves", True)),
            "alpha": float(grf.get("alpha", 0.05)),
            "imbalance_penalty": float(grf.get("imbalance_penalty", 0.0)),
            "stabilize_splits": bool(grf.get("stabilize_splits", True)),
            "ci_group_size": int(grf.get("ci_group_size", 2)),
            "tune_parameters": str(grf.get("tune_parameters", "none")),
            "compute_oob_predictions": bool(
                grf.get("compute_oob_predictions", True)
            ),
            "estimate_variance": bool(grf.get("estimate_variance", True)),
            "native_nuisance_mode": True,
        }
