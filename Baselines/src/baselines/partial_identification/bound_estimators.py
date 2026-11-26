from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .iv_lp import solve_iv_bounds
from .manski import manski_bounds
from .predictive_backends import fit_predict_joint
from .world import PartialWorld


@dataclass(frozen=True)
class BoundPrediction:
    lower: np.ndarray
    upper: np.ndarray
    unit_id: np.ndarray
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]


def _rows(world: PartialWorld):
    a = world.arrays
    return np.flatnonzero(a["split"] == 0), np.flatnonzero(a["split"] == 1)


def estimate_manski(world: PartialWorld, backend: str, seed: int, device: str) -> BoundPrediction:
    context, query = _rows(world)
    a = world.arrays
    labels = 2 * a["T"][context] + a["Y"][context]
    fitted = fit_predict_joint(backend, a["X"][context], labels, a["X"][query], seed, device)
    cells = fitted.probabilities.reshape(-1, 2, 2)
    lower, upper = manski_bounds(cells)
    return BoundPrediction(lower, upper, a["unit_id"][query], {"failure_rate": 0.0}, fitted.metadata)


def estimate_iv(world: PartialWorld, backend: str, seed: int, device: str) -> BoundPrediction:
    context, query = _rows(world)
    a = world.arrays
    labels = 2 * a["T"][context] + a["Y"][context]
    train = np.c_[a["X"][context], a["I"][context]]
    q0, q1 = np.c_[a["X"][query], np.zeros(len(query))], np.c_[a["X"][query], np.ones(len(query))]
    fitted = fit_predict_joint(backend, train, labels, np.r_[q0, q1], seed, device)
    probabilities = fitted.probabilities.reshape(2, len(query), 2, 2).transpose(1, 0, 2, 3)
    lower, upper, raw_failed, repaired, distances = [], [], 0, 0, []
    for cells in probabilities:
        result = solve_iv_bounds(cells, repair=False)
        if not result.feasible:
            raw_failed += 1
            result = solve_iv_bounds(cells, repair=True)
            repaired += int(result.repaired)
        lower.append(result.lower)
        upper.append(result.upper)
        distances.append(result.projection_distance)
    n = len(query)
    diagnostics = {"raw_infeasibility_rate": raw_failed / n, "projection_rate": repaired / n,
                   "mean_projection_distance": float(np.mean(distances)), "failure_rate": float(np.mean(~np.isfinite(lower)))}
    return BoundPrediction(np.asarray(lower), np.asarray(upper), a["unit_id"][query], diagnostics, fitted.metadata)
