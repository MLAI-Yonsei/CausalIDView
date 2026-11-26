from pathlib import Path
from shutil import copyfile
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_reference_metrics_rebuild_the_public_table():
    result = subprocess.run(
        [sys.executable, "scripts/fig2/verify_fig2.py", "--archived"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "960 world metrics" in result.stdout


def test_verification_requires_fresh_world_metrics_by_default(tmp_path, monkeypatch, capsys):
    from scripts.fig2 import verify_fig2

    for path in ("configs/fig2.yaml", "results/fig2.csv", "results/fig2_world_metrics.csv"):
        destination = tmp_path / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        copyfile(ROOT / path, destination)
    monkeypatch.setattr(verify_fig2, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["verify_fig2.py"])

    with pytest.raises(SystemExit):
        verify_fig2.main()

    error = capsys.readouterr().err
    assert "artifacts/fig2_fresh/fig2_world_metrics.csv" in error
    assert "run_fig2.py" in error


@pytest.fixture
def fig2_scoring_inputs(tmp_path):
    from data.generate_world import save_world
    from data.scm import World

    config = {
        "panels": {"IV": [["kiv", "KIV"]]},
        "scm": {"world_seeds": [0]},
        "evaluation": {"bootstrap_replicates": 10, "bootstrap_seed": 0},
    }
    path = tmp_path / "configs/fig2.yaml"
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(config))
    world = World({
        "world_id": np.array([0, 0]), "split": np.array([1, 1]), "unit_id": np.array([10, 11]),
        "tau": np.array([0.0, 2.0]), "Y0": np.array([1.0, 1.0]), "Y1": np.array([1.0, 3.0]),
    }, {})
    save_world(world, tmp_path / "artifacts/worlds/world_000.npz")
    path = tmp_path / "artifacts/predictions/IV/kiv/world_000.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, unit_id=[10, 11], tau_hat=[1.0, 5.0])
    return tmp_path


def test_full_scoring_writes_default_world_metrics_from_ground_truth(fig2_scoring_inputs, monkeypatch):
    from scripts.fig2 import run_fig2

    monkeypatch.setattr(run_fig2, "ROOT", fig2_scoring_inputs)
    monkeypatch.setattr(sys, "argv", ["run_fig2.py"])

    run_fig2.main()

    rows = pd.read_csv(fig2_scoring_inputs / "artifacts/fig2_fresh/fig2_world_metrics.csv")
    assert rows[["sPEHE_sq", "ate_error_sq", "centered_sPEHE_sq"]].values.tolist() == [[5, 4, 1]]
    assert pd.read_csv(fig2_scoring_inputs / "artifacts/fig2_fresh/fig2.csv")["mean"].tolist() == [5]


def test_partial_scoring_preserves_default_world_metrics(fig2_scoring_inputs, monkeypatch):
    from scripts.fig2 import run_fig2

    metrics = fig2_scoring_inputs / "artifacts/fig2_fresh/fig2_world_metrics.csv"
    metrics.parent.mkdir()
    metrics.write_text("existing full metrics\n")
    output = fig2_scoring_inputs / "subset.csv"
    monkeypatch.setattr(run_fig2, "ROOT", fig2_scoring_inputs)
    monkeypatch.setattr(sys, "argv", ["run_fig2.py", "--world", "0", "--output", str(output)])

    run_fig2.main()

    assert pd.read_csv(output)["mean"].tolist() == [5]
    assert metrics.read_text() == "existing full metrics\n"


def test_comparison_records_fresh_and_reference_values_on_mismatch(fig2_scoring_inputs, monkeypatch, capsys):
    from scripts.fig2 import run_fig2, verify_fig2

    root = fig2_scoring_inputs
    reference = root / "results/fig2.csv"
    reference.parent.mkdir()
    reference.write_text(
        "regime,model,model_label,panel_order,mean,ci_low,ci_high,ate_error_sq,centered_sPEHE_sq,n_worlds,is_best\n"
        "IV,kiv,KIV,0,4,4,4,3,1,1,True\n"
    )
    monkeypatch.setattr(run_fig2, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["run_fig2.py"])
    run_fig2.main()
    comparison = root / "artifacts/fig2_fresh/fig2_comparison.csv"
    monkeypatch.setattr(verify_fig2, "ROOT", root)
    monkeypatch.setattr(sys, "argv", ["verify_fig2.py"])

    with pytest.raises(SystemExit, match="exceeds absolute tolerance"):
        verify_fig2.main()

    assert "max absolute delta 1" in capsys.readouterr().out
    row = pd.read_csv(comparison).set_index(["regime", "model", "metric"]).loc["IV", "kiv", "mean"]
    assert row[["actual", "reference", "delta", "abs_delta"]].tolist() == [5, 4, 1, 1]


def test_default_plot_requires_fresh_aggregate(tmp_path, monkeypatch, capsys):
    from scripts.fig2 import plot_fig2

    archive = tmp_path / "results/fig2.csv"
    archive.parent.mkdir()
    copyfile(ROOT / "results/fig2.csv", archive)
    monkeypatch.setattr(plot_fig2, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["plot_fig2.py"])

    with pytest.raises(SystemExit):
        plot_fig2.main()

    error = capsys.readouterr().err
    assert "artifacts/fig2_fresh/fig2.csv" in error
    assert "run_fig2.py" in error


def test_dry_run_covers_every_figure_cell_and_world():
    result = subprocess.run(
        [sys.executable, "scripts/fig2/predict_fig2.py", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "960 jobs" in result.stdout


def test_fd_cfm_paths_and_proxy_learner_inputs():
    import numpy as np
    import yaml

    from baselines.fig2 import model_input
    from data.scm import sample_world

    config = yaml.safe_load((ROOT / "configs/fig2.yaml").read_text())
    config["scm"].update(n_context=32, n_query=10, calibration_population=500)
    world = sample_world(0, config)
    for model in ("causalpfn_fd_xonly", "dopfn_fd_xonly"):
        inputs, graph, _, query_ids = model_input(world, "FD", model)
        assert graph == "backdoor"
        assert set(inputs) == {"X_context", "X_query", "treatment", "outcome"}
        assert inputs["X_context"].shape == (32, 50)
        np.testing.assert_array_equal(query_ids, world.arrays["unit_id"][world.arrays["split"] == 1])
    inputs, graph, _, _ = model_input(world, "FD", "causalfm")
    assert graph == "frontdoor"
    np.testing.assert_array_equal(inputs["graph_aux_context"], world.arrays["M"][world.arrays["split"] == 0])
    inputs, graph, _, _ = model_input(world, "IV", "causalfm")
    assert graph == "conditional_iv"
    np.testing.assert_array_equal(inputs["graph_aux_context"], world.arrays["I"][world.arrays["split"] == 0])
    inputs, _, _, _ = model_input(world, "PROX", "p_learner")
    assert inputs["X_query"].shape == (10, 50)
    assert set(inputs) == {"X_context", "X_query", "treatment", "outcome", "Z_context", "W_context"}


def test_relocated_kiv_worker_writes_aligned_predictions(tmp_path):
    import numpy as np
    import yaml

    from data.scm import sample_world
    from scripts.fig2.predict_fig2 import predict

    config = yaml.safe_load((ROOT / "configs/fig2.yaml").read_text())
    config["scm"].update(n_context=64, n_query=10, calibration_population=500)
    world = sample_world(0, config)
    destination = tmp_path / "world_000.npz"
    predict(world, "IV", "kiv", 0, sys.executable, "cpu", destination)
    with np.load(destination, allow_pickle=False) as result:
        np.testing.assert_array_equal(result["unit_id"], world.arrays["unit_id"][world.arrays["split"] == 1])
        assert result["tau_hat"].shape == (10,)
        assert np.isfinite(result["tau_hat"]).all()


def test_dopfn_preflight_rejects_modified_source_with_valid_weights(tmp_path):
    import hashlib
    import json

    import pytest

    from baselines.checkpoints import CheckpointUnavailable, validate_checkpoint

    source = tmp_path / "Do-PFN-fig2" / "base.py"
    source.parent.mkdir()
    original = b"pass\n"
    source.write_bytes(original)
    manifest = {"base.py": hashlib.sha1(b"blob 5\0" + original).hexdigest()}
    (tmp_path / "source.json").write_text(json.dumps(manifest))
    weights = source.parent / "artifacts" / "weights"
    weights.parent.mkdir()
    weights.write_bytes(b"weights")
    spec = {
        "path": str(weights.relative_to(tmp_path)),
        "source": "https://example.org/weights",
        "sha256": hashlib.sha256(b"weights").hexdigest(),
        "source_revision": "90d6743",
        "source_manifest": "source.json",
    }
    assert validate_checkpoint(spec, tmp_path) == weights
    source.write_text("changed")
    with pytest.raises(CheckpointUnavailable, match="base.py.*90d6743"):
        validate_checkpoint(spec, tmp_path)


def test_fig2_worker_uses_pinned_dopfn_root_for_both_routes(monkeypatch):
    from baselines import cfm, fig2_worker

    release = ROOT / "Baselines/CFMs/Do-PFN-fig2"
    monkeypatch.setattr(cfm, "_run_dopfn", lambda *args, **kwargs: (kwargs.get("model_root"), {}))
    monkeypatch.setattr(cfm, "_run_dopfn_v2", lambda *args, **kwargs: (kwargs.get("model_root"), None, None, {}))
    monkeypatch.setattr(cfm, "_run_frontier_v2", lambda *args, **kwargs: (None, None, None, {}))
    values = {"X_query": [0]}
    for model in ("dopfn", "dopfn_fd_xonly"):
        assert fig2_worker.run_cfm(model, values, "cpu", "backdoor") == release


def test_dopfn_setup_fetches_source_revision_instead_of_weights_revision(tmp_path, monkeypatch):
    import json

    from scripts.fig2 import prepare_fig2

    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()

    def commit(message):
        git("add", "base.py")
        git("-c", "user.name=Test", "-c", "user.email=test@example.org",
            "-c", "commit.gpgsign=false", "commit", "-m", message)
        return git("rev-parse", "HEAD")

    git("init", "--quiet")
    source = upstream / "base.py"
    source.write_text("version = 1\n")
    pinned = commit("source used by Figure 2")
    git("tag", "pinned-source", pinned)
    manifest = {"base.py": git("hash-object", "base.py")}
    source.write_text("version = 2\n")
    newer = commit("checkpoint revision")

    monkeypatch.setattr(prepare_fig2, "ROOT", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    checkpoint = tmp_path / "installed" / "artifacts" / "weights"
    spec = {
        "path": str(checkpoint.relative_to(tmp_path)),
        "source_repo": str(upstream),
        "source_revision": pinned,
        "source_manifest": "manifest.json",
        "revision": newer,
    }
    prepare_fig2.prepare_dopfn_source(spec)
    prepare_fig2.prepare_dopfn_source(spec)
    assert (checkpoint.parents[1] / "base.py").read_text() == "version = 1\n"
    assert source.read_text() == "version = 2\n"


def test_causalfm_download_paths_include_upstream_checkpoints_directory():
    import yaml

    checkpoints = yaml.safe_load((ROOT / "configs/fig2.yaml").read_text())["checkpoints"]
    for name in ("causalfm_standard", "causalfm_fd", "causalfm_iv_binary"):
        spec = checkpoints[name]
        checkpoint_subdir = Path(spec["path"]).parent.name
        assert spec["source"].endswith(f"/{spec['revision']}/checkpoints/{checkpoint_subdir}/best_model.pth")


def test_quadratic_cate_preserves_archived_multiplication_order():
    import numpy as np

    from data.scm import _raw_cate

    x = np.array([[0.53, 0.16]])
    a, b = x[0]
    expected = 0.7 * a + 0.45 * (a * b) + 0.25 * (b * b - 1.0)
    actual = _raw_cate(x, "quadratic_interaction", {"a": np.array([1, 0]), "b": np.array([0, 1])})
    np.testing.assert_array_equal(actual, [expected])
