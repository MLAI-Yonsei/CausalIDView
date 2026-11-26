from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]


def fig3_command(script, *args):
    return subprocess.run(
        [sys.executable, f"scripts/fig3/{script}_fig3.py", *map(str, args)],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_outcome_variants_preserve_pairing_and_change_the_intended_effect():
    from data.fig3 import derive_world
    from data.ground_truth import query_ground_truth
    from data.scm import World

    original = World({
        "split": np.array([0, 1]), "unit_id": np.array([10, 11]),
        "U": np.array([1.0, 2.0]), "gamma_u_x": np.array([2.0, 1.0]),
        "beta": np.array([2.0, 3.0]), "tau": np.array([2.0, 3.0]),
        "M0": np.array([0.0, 1.0]), "M1": np.array([1.0, 2.0]),
        "M": np.array([0.0, 2.0]), "Y0": np.array([1.0, 2.0]),
        "Y1": np.array([3.0, 5.0]), "Y": np.array([1.0, 5.0]),
    }, {})
    stay = derive_world(original, "stay")
    move = derive_world(original, "move")
    np.testing.assert_array_equal(stay.arrays["Y"], [-3.0, 1.0])
    np.testing.assert_array_equal(move.arrays["Y"], [1.0, 11.0])
    np.testing.assert_array_equal(original.arrays["Y"], [1.0, 5.0])
    np.testing.assert_array_equal(query_ground_truth(stay)[1], [3.0])
    np.testing.assert_array_equal(query_ground_truth(move)[1], [6.0])
    assert all(v.arrays["unit_id"] is original.arrays["unit_id"] for v in (stay, move))


def test_scores_pair_query_ids_and_reject_incomplete_or_misaligned_inputs(tmp_path):
    from data.fig3 import prediction_identity, prediction_path, score_world

    for condition, estimate in {"original": [1, 2], "stay": [1, 4], "move": [3, 3]}.items():
        path = prediction_path(tmp_path, "IV", "kiv", condition, 0)
        path.parent.mkdir(parents=True)
        np.savez(path, unit_id=[10, 11], tau_hat=estimate, source_world_sha256="a" * 64,
                 **prediction_identity("IV", "kiv", condition, 0))
    path = prediction_path(tmp_path, "IV", "kiv", "move", 0)
    truth = (np.array([10, 11]), np.array([2.0, 2.0]))
    assert score_world(tmp_path, "IV", "kiv", 0, truth) == pytest.approx(
        {"e_stay": 1 / np.sqrt(2), "e_move": 1 / np.sqrt(8)}
    )
    np.savez(path, unit_id=[11, 10], tau_hat=[3, 3], source_world_sha256="a" * 64,
             **prediction_identity("IV", "kiv", "move", 0))
    with pytest.raises(ValueError, match="query unit alignment"):
        score_world(tmp_path, "IV", "kiv", 0, truth)
    np.savez(path, unit_id=[10, 11], tau_hat=[3, 3], source_world_sha256="b" * 64,
             **prediction_identity("IV", "kiv", "move", 0))
    with pytest.raises(ValueError, match="source world fingerprint"):
        score_world(tmp_path, "IV", "kiv", 0, truth)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="move"):
        score_world(tmp_path, "IV", "kiv", 0, truth)


def test_scoring_rejects_a_prediction_from_another_condition(tmp_path):
    from data.fig3 import prediction_path, score_world

    for condition in ("original", "stay", "move"):
        path = prediction_path(tmp_path, "IV", "kiv", condition, 0)
        path.parent.mkdir(parents=True)
        np.savez(path, unit_id=[10, 11], tau_hat=[1, 2], regime="IV", model="kiv", world_id=0,
                 condition="original", source_world_sha256="a" * 64)
    with pytest.raises(ValueError, match="job identity"):
        score_world(tmp_path, "IV", "kiv", 0, (np.array([10, 11]), np.array([2.0, 2.0])))


def test_complete_inference_dry_run_covers_all_cells_conditions_and_worlds():
    result = fig3_command("predict", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "2880 jobs" in result.stdout


def test_fig3_fd_dopfn_uses_25_query_chunks_and_keeps_fig2_default(monkeypatch):
    from baselines import cfm, fig2_worker

    monkeypatch.setattr(cfm, "_run_dopfn", lambda *args, **kwargs: ("single batch", {}))
    monkeypatch.setattr(cfm, "_run_dopfn_v2", lambda values, device, size, **kw: (size, None, None, {}))
    values = {"X_query": np.zeros((100, 1))}
    assert fig2_worker.run_cfm("dopfn_fd_xonly", values, "cpu", "backdoor") == "single batch"
    assert fig2_worker.run_cfm("dopfn_fd_xonly", values, "cpu", "backdoor", dopfn_fd_microbatch=True) == 25


def test_full_scoring_with_fresh_predictions_and_reference_verdict(tmp_path):
    from data.fig3 import CONDITIONS, load_settings, load_targets, prediction_identity, prediction_path

    panels, worlds = load_settings(ROOT)
    targets = load_targets(ROOT / "results/fig2_world_manifest.json", worlds)
    reference = pd.read_csv(ROOT / "results/fig3/fig3_world_metrics.csv").set_index(["regime", "model", "world_id"])
    predictions = tmp_path / "predictions"
    for regime, models in panels.items():
        for model, _ in models:
            for world in worlds:
                unit_id, target = targets[world].query
                row = reference.loc[(regime, model, world)]
                scales = {"original": 0.0, "stay": row.e_stay, "move": 1.0 + row.e_move}
                for condition in CONDITIONS:
                    path = prediction_path(predictions, regime, model, condition, world)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, unit_id=unit_id, tau_hat=target * scales[condition],
                             source_world_sha256=f"{world:064x}",
                             **prediction_identity(regime, model, condition, world))
    output = tmp_path / "scores"
    scored = fig3_command("run", "--predictions", predictions, "--output", output)
    assert scored.returncode == 0, scored.stderr
    metrics_file = output / "fig3_world_metrics.csv"
    world_metrics = pd.read_csv(metrics_file)
    assert len(world_metrics) == 960
    match = fig3_command("verify", "--output", output)
    assert match.returncode == 0, match.stderr
    assert pd.read_csv(output / "fig3_comparison.csv")["within_tolerance"].all()

    world_metrics.loc[0, "e_stay"] += 0.05
    world_metrics.to_csv(metrics_file, index=False)
    mismatch = fig3_command("verify", "--output", output)
    assert mismatch.returncode != 0
    assert "exceed tolerance" in mismatch.stderr
    assert len(pd.read_csv(output / "fig3_comparison.csv")) == 960 * 2 + 24 * 8


def test_bundled_world_metrics_rebuild_the_agreed_summary(tmp_path):
    result = fig3_command("verify", "--archived", "--output", tmp_path)
    assert result.returncode == 0, result.stderr
    assert "960 world metrics" in result.stdout
    assert pd.read_csv(tmp_path / "fig3_comparison.csv")["within_tolerance"].all()
    plotted = fig3_command("plot", "--input", tmp_path / "fig3.csv", "--output", tmp_path / "fig3.pdf")
    assert plotted.returncode == 0, plotted.stderr
    assert (tmp_path / "fig3.pdf").read_bytes().startswith(b"%PDF")
