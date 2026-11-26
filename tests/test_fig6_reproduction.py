from pathlib import Path
from shutil import copytree, ignore_patterns
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path("scripts") / "fig6"
SCRIPT = ROOT / SCRIPTS / "verify_fig6.py"
MODELS = ("causalpfn", "dopfn", "causalfm", "tabpfn_x", "tabpfn_s", "tabpfn_dr")


def run_fig6(output, *options):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--output-dir", str(output), *map(str, options)],
        capture_output=True, text=True,
    )


def test_bound_runner_uses_bundled_estimators_and_world(tmp_path):
    pytest.importorskip("sklearn")
    release = tmp_path / "release"
    script = release / SCRIPTS / "predict_fig6_bounds.py"
    script.parent.mkdir(parents=True)
    script.write_bytes((ROOT / SCRIPTS / script.name).read_bytes())
    copytree(ROOT / "Baselines" / "src", release / "Baselines" / "src",
             ignore=ignore_patterns("__pycache__"))
    world = release / "data" / "fig6" / "bound_worlds" / "world_000.npz"
    world.parent.mkdir(parents=True)
    world.write_bytes((ROOT / world.relative_to(release)).read_bytes())

    output = tmp_path / "logistic.csv"
    run = subprocess.run(
        [sys.executable, "-I", str(script), "--model", "logistic", "--world", "0",
         "--output", str(output)], cwd=tmp_path, capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    predictions = pd.read_csv(output)
    with np.load(world) as arrays:
        query_ids = arrays["unit_id"][arrays["split"] == 1]
    assert predictions.groupby("leaf").size().to_dict() == {"iv": 100, "manski": 100}
    for _, bounds in predictions.groupby("leaf"):
        np.testing.assert_array_equal(bounds["unit_id"], query_ids)
        assert np.isfinite(bounds[["lower", "upper"]]).all().all()
        assert bounds["lower"].le(bounds["upper"]).all()


def test_fig6_recomputes_gamma2_reference_tables(tmp_path):
    run = run_fig6(tmp_path, "--archived", "--paper-reference")
    assert run.returncode == 0, run.stderr
    bounds = pd.read_csv(tmp_path / "bounds_per_world.csv")
    panel_c = pd.read_csv(tmp_path / "panel_c.csv")
    points = pd.read_csv(tmp_path / "point_per_replicate.csv")
    assert len(bounds) == 360
    assert set(bounds["gamma"].dropna()) == {2.0}
    assert not (tmp_path / "sensitivity_curves.csv").exists()
    assert panel_c.loc[panel_c["partial_information"] == "sensitivity", "n"].eq(40).all()
    assert points.groupby("panel").size().to_dict() == {"a": 480, "b": 1200}
    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert comparison["reference_source"].eq("paper_pdf").sum() == 18
    assert (tmp_path / "fig6.pdf").read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize("panel,filename,count", [
    ("a", "point_per_replicate.csv", 480),
    ("b", "point_per_replicate.csv", 1200),
    ("c", "bounds_per_world.csv", 360),
])
def test_fig6_panel_verifier_writes_only_its_results(tmp_path, panel, filename, count):
    script = ROOT / SCRIPTS / f"verify_fig6{panel}.py"
    run = subprocess.run(
        [sys.executable, str(script), "--archived", "--output-dir", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    scores = pd.read_csv(tmp_path / filename)
    assert len(scores) == count
    if panel in ("a", "b"):
        assert set(scores["panel"]) == {panel}
        assert not (tmp_path / "panel_c.csv").exists()
    else:
        assert set(scores["gamma"].dropna()) == {2.0}
        assert not (tmp_path / "point_summary.csv").exists()
    comparison = pd.read_csv(tmp_path / "comparison.csv")
    assert set(comparison["reference_source"]) == {"v2_reference"}
    np.testing.assert_allclose(comparison["delta"], 0, atol=1e-10, rtol=0)
    assert not (tmp_path / "fig6.pdf").exists()


def test_fig6b_override_changes_only_selected_panel_b_scores(tmp_path):
    reference = ROOT / "results" / "fig6" / "point_per_replicate.csv"
    archived = pd.read_csv(reference)
    replaced = archived["panel"].eq("b") & archived["model"].eq("tabpfn_x")
    replacement = archived.loc[replaced, ["replicate", "model", "condition", "spehe"]].copy()
    replacement["condition"] = replacement["condition"].replace({"observed": "full"})
    replacement["spehe"] = 123.0
    replacement_path = tmp_path / "rerun.csv"
    replacement.to_csv(replacement_path, index=False)

    run = run_fig6(tmp_path, "--archived", "--fig6b-metrics", replacement_path)
    assert run.returncode == 0, run.stderr
    actual = pd.read_csv(tmp_path / "point_per_replicate.csv")
    assert actual.loc[replaced, "spehe"].eq(123).all()
    pd.testing.assert_frame_equal(actual.loc[~replaced], archived.loc[~replaced])


def test_fig6a_override_scores_fresh_query_predictions(tmp_path):
    targets = pd.read_csv(ROOT / "data" / "fig6" / "synthetic_targets.csv.gz")
    overlay = tmp_path / "predict"
    for condition in ("BD", "HC"):
        for model in MODELS:
            directory = overlay / condition / model
            directory.mkdir(parents=True)
            for world, group in targets.groupby("world_id"):
                # A known shift gives exactly 2.0 RMSE; reverse IDs to exercise alignment.
                np.savez(directory / f"world_{world:03d}.npz", unit_id=group["unit_id"].to_numpy()[::-1],
                         tau_hat=group["target"].to_numpy()[::-1] + 2.0)

    run = run_fig6(tmp_path, "--archived", "--fig6a-predictions", overlay)
    assert run.returncode == 0, run.stderr
    points = pd.read_csv(tmp_path / "point_per_replicate.csv")
    np.testing.assert_allclose(points.loc[points["panel"].eq("a"), "spehe"], 2.0)
    archived = pd.read_csv(ROOT / "results" / "fig6" / "point_per_replicate.csv")
    pd.testing.assert_frame_equal(points[points["panel"].eq("b")], archived[archived["panel"].eq("b")])


def test_bound_override_scores_complete_leaf_model_subset(tmp_path):
    oracle = pd.read_csv(ROOT / "data" / "fig6" / "bound_oracles.csv.gz", float_precision="round_trip")
    replacement = oracle[oracle["leaf"].eq("manski")].assign(method="logistic")
    paths = [tmp_path / f"fresh_bounds_{part}.csv" for part in (0, 1)]
    replacement[replacement["world_id"].lt(20)].to_csv(paths[0], index=False)
    replacement[replacement["world_id"].ge(20)].to_csv(paths[1], index=False)

    run = run_fig6(tmp_path, "--archived", "--bound-predictions", paths[0], "--bound-predictions", paths[1])
    assert run.returncode == 0, run.stderr
    bounds = pd.read_csv(tmp_path / "bounds_per_world.csv")
    changed = bounds["partial_information"].eq("manski") & bounds["method"].eq("logistic")
    assert len(bounds) == 360
    assert changed.sum() == 40
    assert bounds.loc[changed, "endpoint_rmse"].eq(0).all()


def test_default_fig6_evaluates_fresh_outputs_without_archived_predictions(tmp_path):
    release = tmp_path / "release"
    copytree(ROOT / "scripts", release / "scripts", ignore=ignore_patterns("__pycache__"))
    copytree(ROOT / "data", release / "data", ignore=ignore_patterns(
        "*predictions.csv.gz", "bound_worlds", "model_input_*.npz", "table2", "__pycache__",
    ))
    copytree(ROOT / "results" / "fig6", release / "results" / "fig6")

    script = [sys.executable, str(release / SCRIPTS / "verify_fig6.py")]
    missing = subprocess.run(script, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "fig6a_predictions" in missing.stderr

    artifacts = release / "artifacts"
    targets = pd.read_csv(release / "data" / "fig6" / "synthetic_targets.csv.gz")
    for view in ("BD", "HC"):
        for model in MODELS:
            directory = artifacts / "fig6a_predictions" / view / model
            directory.mkdir(parents=True)
            for world, group in targets.groupby("world_id"):
                np.savez(directory / f"world_{world:03d}.npz", unit_id=group["unit_id"].to_numpy(),
                         tau_hat=group["target"].to_numpy() + 2.0)

    run = subprocess.run(
        [sys.executable, str(release / SCRIPTS / "verify_fig6a.py")],
        capture_output=True, text=True,
    )
    assert run.returncode == 0, run.stderr
    a_scores = pd.read_csv(artifacts / "fig6a_fresh" / "point_per_replicate.csv")
    np.testing.assert_allclose(a_scores["spehe"], 2.0)

    for condition in ("full", "hidden"):
        for model in MODELS:
            directory = artifacts / "fig6b_predictions" / condition / model
            directory.mkdir(parents=True)
            for trial in range(100):
                evaluator = release / "data" / "ihdp_hc" / f"trial_{trial:03d}" / "evaluator_only.npz"
                with np.load(evaluator, allow_pickle=False) as ground_truth:
                    np.savez(directory / f"trial_{trial:03d}.npz", sample_id=ground_truth["test_sample_id"],
                             tau_hat=ground_truth["test_tau_true"].astype(float) + 3.0)

    bounds = artifacts / "fig6_bounds"
    bounds.mkdir()
    oracle = pd.read_csv(release / "data" / "fig6" / "bound_oracles.csv.gz", float_precision="round_trip")
    for models, leaves in (
        (("logistic", "xgboost", "tabpfn_v35"), ("manski", "iv")),
        (("b_learner_tabpfn_all", "csa_pfn_msm", "neuralcsa"), ("sensitivity",)),
    ):
        for model in models:
            rows = oracle["leaf"].isin(leaves) & (oracle["gamma"].isna() | oracle["gamma"].eq(2))
            oracle.loc[rows].assign(method=model).to_csv(bounds / f"{model}.csv", index=False)

    run = subprocess.run(script, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    output = artifacts / "fig6_fresh"
    points = pd.read_csv(output / "point_per_replicate.csv")
    np.testing.assert_allclose(points.loc[points["panel"].eq("a"), "spehe"], 2.0)
    np.testing.assert_allclose(points.loc[points["panel"].eq("b"), "spehe"], 3.0)
    settings = pd.read_csv(output / "panel_c.csv")
    assert settings.loc[settings["partial_information"].eq("sensitivity"), "n"].eq(40).all()
    assert not (output / "sensitivity_curves.csv").exists()
    report = pd.read_csv(output / "comparison.csv")
    assert set(report["reference_source"]) == {"v2_reference"}
    assert len(report) == 90
    assert report.loc[report["table"].eq("point_summary"), "delta"].abs().gt(0).all()
    assert (output / "fig6.pdf").is_file()
