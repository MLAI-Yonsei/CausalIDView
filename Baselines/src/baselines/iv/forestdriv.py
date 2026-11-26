"""Figure 2 EconML ForestDRIV implementation."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def run(arrays: dict[str, np.ndarray], context: np.ndarray, query_rows: np.ndarray, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    from econml.iv.dr import ForestDRIV

    estimator = ForestDRIV(
        n_estimators=100, min_samples_leaf=10, max_depth=12, max_samples=0.45,
        discrete_treatment=True, discrete_instrument=True, cv=3, n_jobs=1,
        random_state=seed,
    )
    fit_started = time.perf_counter()
    estimator.fit(
        arrays["Y"][context], arrays["T"][context], Z=arrays["I"][context],
        X=arrays["X"][context], inference=None,
    )
    fit_time = time.perf_counter() - fit_started
    prediction_started = time.perf_counter()
    prediction = np.asarray(estimator.effect(arrays["X"][query_rows]), dtype=np.float64).reshape(-1)
    prediction_time = time.perf_counter() - prediction_started
    return prediction, {
        "estimator": "econml.iv.dr.ForestDRIV", "econml_version": __import__("econml").__version__,
        "cross_fitting_folds": 3, "n_estimators": 100, "fit_time_sec": fit_time,
        "inference_time_sec": prediction_time,
    }
