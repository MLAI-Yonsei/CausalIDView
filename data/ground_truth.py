"""Evaluator-only targets."""

from __future__ import annotations

import numpy as np

from .scm import World


def query_ground_truth(world: World) -> tuple[np.ndarray, np.ndarray]:
    query = world.arrays["split"] == 1
    tau = np.asarray(world.arrays["tau"][query], dtype=np.float64)
    contrast = np.asarray(world.arrays["Y1"][query] - world.arrays["Y0"][query])
    if not np.allclose(tau, contrast, rtol=0.0, atol=1e-12):
        raise ValueError("SCM target invariant tau = Y1 - Y0 failed")
    return np.asarray(world.arrays["unit_id"][query]), tau
