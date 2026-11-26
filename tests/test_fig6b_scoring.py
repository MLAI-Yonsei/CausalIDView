import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_scoring_emits_only_fresh_scores_and_checks_query_order(tmp_path):
    predictions = tmp_path / "predictions"
    cfm_predictions = tmp_path / "cfm_predictions"
    output = tmp_path / "metrics.csv"
    for trial in range(100):
        evaluator = ROOT / f"data/ihdp_hc/trial_{trial:03d}/evaluator_only.npz"
        with np.load(evaluator, allow_pickle=False) as data:
            ids, truth = data["test_sample_id"], data["test_tau_true"]
        for condition in ("full", "hidden"):
            for model in ("tabpfn_x", "tabpfn_s", "tabpfn_dr"):
                directory = predictions / condition / model
                directory.mkdir(parents=True, exist_ok=True)
                np.savez(directory / f"trial_{trial:03d}.npz", sample_id=ids, tau_hat=truth + 2)
            for model in ("causalpfn", "dopfn", "causalfm"):
                directory = cfm_predictions / condition / model
                directory.mkdir(parents=True, exist_ok=True)
                np.savez(directory / f"trial_{trial:03d}.npz", sample_id=ids, tau_hat=truth + 3)

    command = [sys.executable, str(ROOT / "scripts/fig6/run_fig6b.py"),
               "--predictions", str(predictions), "--output", str(output)]
    missing = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "causalpfn/trial_000.npz" in missing.stderr
    assert not output.exists()
    tabpfn_only = [*command, "--tabpfn-only"]
    result = subprocess.run(tabpfn_only, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    original_output = output.read_bytes()
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 600
    assert {row["model"] for row in rows} == {"tabpfn_x", "tabpfn_s", "tabpfn_dr"}
    assert [float(row["spehe"]) for row in rows] == pytest.approx([2] * 600)

    combined_path = tmp_path / "all_metrics.csv"
    combined = subprocess.run(
        [*command, "--cfm-predictions", str(cfm_predictions), "--output", str(combined_path)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert combined.returncode == 0, combined.stderr
    with combined_path.open() as stream:
        scores = list(csv.DictReader(stream))
    cfm_scores = [float(row["spehe"]) for row in scores if not row["model"].startswith("tabpfn_")]
    assert len(scores) == 1200
    assert cfm_scores == pytest.approx([3] * 600)

    np.savez(predictions / "hidden/tabpfn_dr/trial_099.npz", sample_id=ids[::-1], tau_hat=truth + 2)
    result = subprocess.run(tabpfn_only, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert "trial_099.npz" in result.stderr
    assert output.read_bytes() == original_output
