"""Figure 2 metrics and world-level bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(estimate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if estimate.shape != target.shape:
        raise ValueError(f"unaligned predictions/targets: {estimate.shape} != {target.shape}")
    error = estimate - target
    mean_error = float(error.mean())
    centered = error - mean_error
    return {
        "sPEHE_sq": float(np.mean(error**2)),
        "ate_error_sq": mean_error**2,
        "centered_sPEHE_sq": float(np.mean(centered**2)),
    }


def aggregate_worlds(world_metrics: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)
    cell_sizes = world_metrics.groupby(["regime", "model"]).size().unique()
    if len(cell_sizes) != 1:
        raise ValueError("all model/regime cells must contain the same number of worlds")
    bootstrap_indices = rng.integers(0, int(cell_sizes[0]), size=(reps, int(cell_sizes[0])))
    for (regime, model, label, order), part in world_metrics.groupby(
        ["regime", "model", "model_label", "panel_order"], sort=False
    ):
        part = part.sort_values("world_id")
        values = part["sPEHE_sq"].to_numpy(float)
        low, high = np.quantile(values[bootstrap_indices].mean(axis=1), (0.025, 0.975))
        rows.append({
            "regime": regime, "model": model, "model_label": label,
            "panel_order": int(order), "mean": float(values.mean()),
            "ci_low": float(low), "ci_high": float(high),
            "ate_error_sq": float(part["ate_error_sq"].mean()),
            "centered_sPEHE_sq": float(part["centered_sPEHE_sq"].mean()),
            "n_worlds": len(part),
        })
    result = pd.DataFrame(rows)
    result["is_best"] = result["mean"].eq(result.groupby("regime")["mean"].transform("min"))
    return result
