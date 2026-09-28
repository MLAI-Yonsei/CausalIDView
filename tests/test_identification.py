from pathlib import Path
import re

import numpy as np
import pandas as pd
import yaml

from data.estimators import backdoor, frontdoor, iv, proximal
from data.ground_truth import query_ground_truth
from data.metrics import point_metrics
from data.scm import sample_world
from data.views import EXTRA_COLUMNS, REGIMES, observational_view
from scripts.plot_fig2 import render


ROOT = Path(__file__).resolve().parents[1]

CURRENT_PANELS = {
    "BD": (
        ("causalpfn", "CausalPFN"),
        ("dopfn", "Do-PFN"),
        ("causalfm", "CausalFM"),
        ("tabpfn_x", "TabPFN-v3.5 + X-Learner"),
        ("tabpfn_s", "TabPFN-v3.5 + S-Learner"),
        ("tabpfn_dr", "TabPFN-v3.5 + DR-Learner"),
    ),
    "FD": (
        ("causalpfn_fd_xonly", "CausalPFN"),
        ("dopfn_fd_xonly", "Do-PFN"),
        ("causalfm", "CausalFM"),
        ("fd_nn", "NN + FD Plug-in"),
        ("fd_xgboost", "XGBoost + FD Plug-in"),
        ("tabpfn_fd", "TabPFN-v3.5 + FD Plug-in"),
    ),
    "IV": (
        ("causalpfn", "CausalPFN"),
        ("dopfn", "Do-PFN"),
        ("causalfm", "CausalFM"),
        ("forestdriv", "ForestDRIV"),
        ("kiv", "KIV"),
        ("wald_tabpfn", "TabPFN-v3.5 + Wald"),
    ),
    "PROX": (
        ("causalpfn", "CausalPFN"),
        ("dopfn", "Do-PFN"),
        ("causalfm", "CausalFM"),
        ("p_learner", "ExtraTrees + P-Learner"),
        ("p_learner_nn_full", "NN + P-Learner"),
        ("p_learner_tabpfn_v3_full", "TabPFN-v3.5 + P-Learner"),
    ),
}


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


def test_current_panel_contract_is_shared_by_config_adapters_and_results():
    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    configured = {
        regime: tuple((model, label) for model, label in models)
        for regime, models in config["panels"].items()
    }
    assert configured == CURRENT_PANELS

    adapter_models = {
        "BD": backdoor.MODELS,
        "FD": frontdoor.MODELS,
        "IV": iv.MODELS,
        "PROX": proximal.MODELS,
    }
    assert adapter_models == {
        regime: tuple(model for model, _ in models)
        for regime, models in CURRENT_PANELS.items()
    }

    frame = pd.read_csv(ROOT / "results" / "fig2.csv").sort_values(
        ["regime", "panel_order"]
    )
    observed = {
        regime: tuple(zip(part["model"], part["model_label"]))
        for regime, part in frame.groupby("regime", sort=False)
    }
    assert observed == CURRENT_PANELS


def test_frozen_table_contains_current_main_figure_values():
    frame = pd.read_csv(ROOT / "results" / "fig2.csv").set_index(["regime", "model"])
    expected_means = {
        ("BD", "tabpfn_x"): 0.15278665446773287,
        ("BD", "tabpfn_s"): 0.19689784498708918,
        ("BD", "tabpfn_dr"): 0.14341101580691448,
        ("FD", "tabpfn_fd"): 0.15339920346215882,
        ("IV", "wald_tabpfn"): 0.33566673113296863,
        ("PROX", "p_learner_nn_full"): 0.6623746889197375,
        ("PROX", "p_learner_tabpfn_v3_full"): 0.18113783131220423,
    }
    for key, expected in expected_means.items():
        assert np.isclose(frame.loc[key, "mean"], expected, rtol=0.0, atol=1e-15)

    best = frame[frame["is_best"]].reset_index()
    assert dict(zip(best["regime"], best["model"])) == {
        "BD": "tabpfn_dr",
        "FD": "tabpfn_fd",
        "IV": "causalfm",
        "PROX": "p_learner_tabpfn_v3_full",
    }


def test_plotter_renders_current_panel_models(tmp_path):
    rows = []
    for regime, models in CURRENT_PANELS.items():
        for order, (model, label) in enumerate(models):
            rows.append({
                "regime": regime,
                "model": model,
                "model_label": label,
                "panel_order": order,
                "mean": 0.2 + order * 0.01,
                "ci_low": 0.19 + order * 0.01,
                "ci_high": 0.21 + order * 0.01,
                "ate_error_sq": 0.05,
                "centered_sPEHE_sq": 0.15 + order * 0.01,
                "n_worlds": 40,
                "is_best": order == 0,
            })
    source = tmp_path / "fig2.csv"
    output = tmp_path / "fig2.pdf"
    pd.DataFrame(rows).to_csv(source, index=False)

    render(source, output)

    assert output.stat().st_size > 0


def test_public_files_have_no_local_path_or_identity_leak():
    local_absolute_path = re.compile(r"/(?:home|data\d+)/[A-Za-z0-9._-]+/")
    public = [ROOT / "README.md", ROOT / "configs" / "fig2.yaml"]
    public += list((ROOT / "data").rglob("*.py"))
    public += list((ROOT / "scripts").glob("*.py"))
    for path in public:
        text = path.read_text().lower()
        assert local_absolute_path.search(text) is None, path
