from __future__ import annotations

import numpy as np


def manski_bounds(cells: np.ndarray):
    cells = np.asarray(cells, dtype=float)
    q00, q01 = cells[..., 0, 0], cells[..., 0, 1]
    q10, q11 = cells[..., 1, 0], cells[..., 1, 1]
    return -(q10 + q01), q11 + q00
