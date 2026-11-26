from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .assets import SENSITIVITY_GAMMA_INDEX
from .world import PartialWorld


SENSITIVITY_METHODS = (
    "csa_pfn_msm",
    "b_learner",
    "neuralcsa",
    "b_learner_tabpfn_final",
    "b_learner_tabpfn_all",
)


@dataclass(frozen=True)
class SensitivityPrediction:
    lower: np.ndarray
    upper: np.ndarray
    unit_id: np.ndarray
    metadata: dict[str, Any]


def estimate_sensitivity(world: PartialWorld, method: str, seed: int, device: str, smoke: bool = False) -> SensitivityPrediction:
    if method not in SENSITIVITY_METHODS:
        raise ValueError(method)
    context = np.flatnonzero(world.arrays["split"] == 0)
    query = np.flatnonzero(world.arrays["split"] == 1)
    if method.startswith("b_learner_tabpfn_"):
        from .tabpfn_b_learner import estimate_tabpfn_b_learner

        prediction = estimate_tabpfn_b_learner(
            world, method, seed=seed, device=device, n_estimators=8
        )
        lower, upper, metadata = (
            prediction.lower,
            prediction.upper,
            prediction.metadata,
        )
    else:
        from .sensitivity_models import _b_learner, _csa_pfn, _neuralcsa

        if method == "csa_pfn_msm":
            lower, upper, metadata = _csa_pfn(world.arrays, context, query, device, batch_size=4 if smoke else 16)
        elif method == "b_learner":
            lower, upper, metadata = _b_learner(world.arrays, context, query, seed)
        else:
            lower, upper, metadata = _neuralcsa(world.arrays, context, query, seed, smoke)
        # Preserve the original CSA-PFN token grid and monotonicity correction
        # and the NeuralCSA training and Monte Carlo evaluation order.
        target = slice(SENSITIVITY_GAMMA_INDEX, SENSITIVITY_GAMMA_INDEX + 1)
        lower, upper = lower[target], upper[target]
    expected = (1, len(query))
    if lower.shape != expected or upper.shape != expected or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise RuntimeError(f"invalid sensitivity output for {method}")
    metadata.update({"query_used_for_fit_or_tuning": False, "actual_fit_rows": len(context),
                     "binary_outcome": True, "world_id": world.world_id})
    return SensitivityPrediction(lower, upper, world.arrays["unit_id"][query], metadata)
