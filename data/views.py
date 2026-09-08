"""Identification-specific observational views and leakage guards."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scm import World


REGIMES = ("BD", "FD", "IV", "PROX")
EXTRA_COLUMNS = {"BD": ("U",), "FD": ("M",), "IV": ("I",), "PROX": ("Z_P", "W_P")}


@dataclass(frozen=True)
class ObservationalView:
    regime: str
    context: dict[str, np.ndarray]
    query: dict[str, np.ndarray]


def observational_view(world: World, regime: str) -> ObservationalView:
    if regime not in REGIMES:
        raise ValueError(f"unsupported regime: {regime}")
    a = world.arrays
    context_rows, query_rows = a["split"] == 0, a["split"] == 1
    keys = ("X", *EXTRA_COLUMNS[regime])
    context = {key: np.asarray(a[key][context_rows]) for key in keys}
    context.update({key: np.asarray(a[key][context_rows]) for key in ("T", "Y", "unit_id")})
    # Factual treatment/outcome and all potential outcomes are deliberately absent.
    query = {key: np.asarray(a[key][query_rows]) for key in keys}
    query["unit_id"] = np.asarray(a["unit_id"][query_rows])
    return ObservationalView(regime, context, query)
