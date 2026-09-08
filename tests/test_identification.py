from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from data.ground_truth import query_ground_truth
from data.metrics import point_metrics
from data.scm import sample_world
from data.views import EXTRA_COLUMNS, REGIMES, observational_view


ROOT = Path(__file__).resolve().parents[1]


def small_config():
    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    config["scm"].update(n_context=64, n_query=20, calibration_population=1000)
    return config


def test_potential_outcome_contrast_is_tau():
    world = sample_world(0, small_config())
    unit_id, tau = query_ground_truth(world)
    assert unit_id.shape == tau.shape == (20,)


def test_views_expose_only_identification_contract():
    world = sample_world(1, small_config())
    forbidden = {"tau", "Y0", "Y1", "M0", "M1", "T0", "T1", "gamma_u_x"}
    for regime in REGIMES:
        view = observational_view(world, regime)
        assert set(view.context) == {"X", *EXTRA_COLUMNS[regime], "T", "Y", "unit_id"}
        assert set(view.query) == {"X", *EXTRA_COLUMNS[regime], "unit_id"}
        assert forbidden.isdisjoint(view.context)
        assert forbidden.isdisjoint(view.query)


def test_metric_decomposition():
    target = np.array([-1.0, 0.0, 2.0, 4.0])
    estimate = np.array([-0.5, -0.1, 2.3, 4.8])
    metrics = point_metrics(estimate, target)
    assert np.isclose(
        metrics["sPEHE_sq"], metrics["ate_error_sq"] + metrics["centered_sPEHE_sq"]
    )


def test_frozen_plot_table_contract():
    frame = pd.read_csv(ROOT / "results" / "fig2.csv")
    assert len(frame) == 24
    assert set(frame.regime) == set(REGIMES)
    assert frame.groupby("regime").size().eq(6).all()
    assert np.allclose(frame["mean"], frame["ate_error_sq"] + frame["centered_sPEHE_sq"])


def test_public_files_have_no_local_path_or_identity_leak():
    local_absolute_path = re.compile(r"/(?:home|data\d+)/[A-Za-z0-9._-]+/")
    public = [ROOT / "README.md", ROOT / "configs" / "fig2.yaml"]
    public += list((ROOT / "data").rglob("*.py"))
    public += list((ROOT / "scripts").glob("*.py"))
    for path in public:
        text = path.read_text().lower()
        assert local_absolute_path.search(text) is None, path
