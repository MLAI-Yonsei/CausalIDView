from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from .exceptions import NotFittedError
from .prediction import CATEPrediction
from .validation import OutcomeType


class BaseCATEEstimator(ABC):
    model_id: str

    @abstractmethod
    def fit(
        self,
        X: NDArray[np.float64],
        treatment: NDArray[np.int64],
        y: NDArray[np.float64],
        *,
        outcome_type: OutcomeType,
    ) -> Self:
        """Fit the estimator and return itself."""

    @abstractmethod
    def predict(
        self,
        X: NDArray[np.float64],
        *,
        return_po: bool = False,
        return_interval: bool = False,
        alpha: float = 0.05,
    ) -> CATEPrediction:
        """Predict individual conditional treatment effects."""

    def predict_cate(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.predict(X).cate

    def predict_ate(self, X: NDArray[np.float64]) -> float:
        return float(np.mean(self.predict_cate(X)))

    @abstractmethod
    def get_metadata(self) -> Mapping[str, Any]:
        """Return serializable estimator metadata."""

    def _require_fitted(self) -> None:
        if not getattr(self, "_is_fitted", False):
            raise NotFittedError(f"Estimator '{self.model_id}' has not been fitted")

