from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import linprog, minimize


@dataclass(frozen=True)
class ResponseTypeSystem:
    treatment_types: tuple[tuple[int, int], ...]
    outcome_types: tuple[tuple[int, int], ...]
    types: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    cell_matrix: np.ndarray
    ate_objective: np.ndarray


def _make_system() -> ResponseTypeSystem:
    treatment = ((0, 0), (0, 1), (1, 1))
    outcome = ((0, 0), (0, 1), (1, 0), (1, 1))
    types = tuple(product(treatment, outcome))
    matrix = np.zeros((8, 12))
    objective = np.empty(12)
    for k, (tr, yr) in enumerate(types):
        objective[k] = yr[1] - yr[0]
        for instrument in (0, 1):
            t = tr[instrument]
            y = yr[t]
            matrix[instrument * 4 + t * 2 + y, k] = 1
    return ResponseTypeSystem(treatment, outcome, types, matrix, objective)


TYPE_SYSTEM = _make_system()


@dataclass(frozen=True)
class ProjectionResult:
    cells: np.ndarray
    witness: np.ndarray
    distance: float
    success: bool


@dataclass(frozen=True)
class IVResult:
    lower: float
    upper: float
    feasible: bool
    repaired: bool
    cells: np.ndarray
    lower_witness: np.ndarray | None
    upper_witness: np.ndarray | None
    projection_distance: float


def _endpoint(p: np.ndarray, maximize: bool):
    return linprog(-TYPE_SYSTEM.ate_objective if maximize else TYPE_SYSTEM.ate_objective,
                   A_eq=TYPE_SYSTEM.cell_matrix, b_eq=p, bounds=[(0, None)] * 12, method="highs")


def project_iv_cells(cells: np.ndarray) -> ProjectionResult:
    original = np.asarray(cells, float)
    p = original.reshape(8)
    objective = lambda q: 0.5 * np.sum((TYPE_SYSTEM.cell_matrix @ q - p) ** 2)
    solved = minimize(objective, np.full(12, 1 / 12), method="SLSQP", bounds=[(0, 1)] * 12,
                      constraints={"type": "eq", "fun": lambda q: q.sum() - 1},
                      options={"ftol": 1e-13, "maxiter": 2000})
    projected = (TYPE_SYSTEM.cell_matrix @ solved.x).reshape(original.shape)
    return ProjectionResult(projected, solved.x, float(np.linalg.norm(projected - original)), bool(solved.success))


def solve_iv_bounds(cells: np.ndarray, repair: bool = False) -> IVResult:
    value = np.asarray(cells, float)
    p = value.reshape(8)
    lo, hi = _endpoint(p, False), _endpoint(p, True)
    repaired, distance = False, 0.0
    if not (lo.success and hi.success) and repair:
        projection = project_iv_cells(value)
        value, p, repaired, distance = projection.cells, projection.cells.reshape(8), True, projection.distance
        lo, hi = _endpoint(p, False), _endpoint(p, True)
    feasible = bool(lo.success and hi.success)
    return IVResult(float(lo.fun) if feasible else np.nan,
                    float(-hi.fun) if feasible else np.nan, feasible, repaired, value,
                    lo.x if feasible else None, hi.x if feasible else None, distance)

