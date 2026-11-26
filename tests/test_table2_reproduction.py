from pathlib import Path
from shutil import copy, copytree, ignore_patterns
import subprocess
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "table2" / "verify_table2.py"
PREDICT = ROOT / "scripts" / "table2" / "predict_table2.py"
DATA = ROOT / "data" / "table2"
REFERENCE = ROOT / "results" / "table2" / "table2.csv"
SCORE_FILES = ("rct_null_results.csv", "requested_baseline_results.csv")


@pytest.fixture
def isolated_table(tmp_path):
    script_dir = tmp_path / SCRIPT.parent.relative_to(ROOT)
    script_dir.mkdir(parents=True)
    for script in (SCRIPT, PREDICT):
        copy(script, script_dir)
    copytree(DATA, tmp_path / "data" / "table2", ignore=ignore_patterns(*SCORE_FILES))
    reference = tmp_path / REFERENCE.relative_to(ROOT)
    reference.parent.mkdir(parents=True)
    copy(REFERENCE, reference)
    return tmp_path


@pytest.fixture
def fresh_inputs(isolated_table):
    fresh = isolated_table / "artifacts" / "table2" / "fresh"
    fresh.mkdir(parents=True)
    # The known metric rows are fixtures for the CPU verification path.
    for filename in SCORE_FILES:
        copy(DATA / filename, fresh)
    return fresh


@pytest.fixture
def archived_table(isolated_table):
    for filename in SCORE_FILES:
        copy(DATA / filename, isolated_table / "data" / "table2")
    return isolated_table


def run_verifier(root, *args):
    return subprocess.run(
        [sys.executable, str(root / SCRIPT.relative_to(ROOT)), *map(str, args)],
        cwd=root, capture_output=True, text=True,
    )


def test_table2_cli_reproduces_all_reference_cells_from_the_archive(archived_table):
    completed = run_verifier(archived_table, "--archived", "--plot")
    assert completed.returncode == 0, completed.stderr
    assert "90 cells" in completed.stdout
    output = archived_table / "artifacts" / "table2" / "archived" / "table2.csv"
    expected = pd.read_csv(REFERENCE)
    pd.testing.assert_frame_equal(pd.read_csv(output), expected, rtol=0, atol=1e-12)
    assert output.with_suffix(".pdf").read_bytes().startswith(b"%PDF-")


@pytest.mark.parametrize("has_rct_scores", [False, True])
def test_table2_requires_fresh_metrics_by_default(archived_table, has_rct_scores):
    fresh = archived_table / "artifacts" / "table2" / "fresh"
    if has_rct_scores:
        fresh.mkdir(parents=True)
        copy(DATA / SCORE_FILES[0], fresh)

    completed = run_verifier(archived_table)

    assert completed.returncode != 0
    assert SCORE_FILES[1] in completed.stderr
    assert "--archived" in completed.stderr
    assert not (fresh / "table2.csv").exists()


def test_table2_compares_default_fresh_metrics_without_archives(isolated_table, fresh_inputs):
    completed = run_verifier(isolated_table)

    assert completed.returncode == 0, completed.stderr
    pd.testing.assert_frame_equal(
        pd.read_csv(fresh_inputs / "table2.csv"), pd.read_csv(REFERENCE), rtol=0, atol=1e-12,
    )
    comparison = pd.read_csv(fresh_inputs / "table2_comparison.csv")
    assert len(comparison) == 90
    assert comparison.matches.all()
    assert comparison[["mean_delta", "sd_delta"]].abs().max().max() <= 1e-12


def test_table2_writes_fresh_comparison_when_strict_numerics_fail(isolated_table, fresh_inputs):
    rct_scores = fresh_inputs / SCORE_FILES[0]
    results = pd.read_csv(rct_scores)
    first_primary = results.index[results.protocol.eq("primary")][0]
    dataset, model = results.loc[first_primary, ["dataset", "model"]]
    score_columns = ["null_spehe", "null_ate", "null_cpehe"]
    results.loc[first_primary, score_columns] *= 2
    results.to_csv(rct_scores, index=False)

    completed = run_verifier(isolated_table)

    assert completed.returncode != 0
    assert (fresh_inputs / "table2.csv").is_file()
    comparison_path = fresh_inputs / "table2_comparison.csv"
    assert str(comparison_path) in completed.stdout
    comparison = pd.read_csv(comparison_path)
    mismatches = comparison.loc[~comparison.matches]
    assert len(comparison) == 90
    assert set(mismatches[["dataset", "model"]].itertuples(index=False, name=None)) == {(dataset, model)}
    assert (mismatches.mean_delta > 0).all()


def test_table2_cli_rejects_a_mismatched_primary_split(tmp_path):
    results = pd.read_csv(ROOT / "data" / "table2" / "requested_baseline_results.csv")
    first_primary = results.index[results.protocol.eq("primary")][0]
    results.loc[first_primary, "query_ids_hash"] = "a" * 64
    modified = tmp_path / "requested_baseline_results.csv"
    results.to_csv(modified, index=False)
    output = tmp_path / "table2.csv"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--archived", "--requested-results", str(modified), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode == 1
    assert "paired split" in completed.stderr
    assert not output.exists()


def test_table2_input_audit_checks_raw_bytes_without_metric_archives(isolated_table):
    (isolated_table / "ACTG175.txt").write_text("not the official public trial data\n")
    completed = subprocess.run(
        [sys.executable, str(isolated_table / PREDICT.relative_to(ROOT)), "--raw-dir", str(isolated_table),
         "--dataset", "actg175", "--prepare-only"],
        cwd=isolated_table, capture_output=True, text=True,
    )
    assert completed.returncode == 1
    assert "SHA-256 mismatch" in completed.stderr


def test_table2_can_render_changed_rerun_scores(tmp_path):
    results = pd.read_csv(ROOT / "data" / "table2" / "rct_null_results.csv")
    first_primary = results.index[results.protocol.eq("primary")][0]
    results.loc[first_primary, ["null_spehe", "null_ate", "null_cpehe"]] *= 2
    replacement = tmp_path / "new_scores.csv"
    results.to_csv(replacement, index=False)
    output = tmp_path / "table2.csv"
    command = [sys.executable, str(SCRIPT), "--archived", "--rct-results", str(replacement), "--output", str(output)]
    strict = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert strict.returncode != 0
    assert not output.exists()
    aggregate = subprocess.run([*command, "--aggregate-only"], cwd=tmp_path, capture_output=True, text=True)
    assert aggregate.returncode == 0, aggregate.stderr
    reference = pd.read_csv(ROOT / "results" / "table2" / "table2.csv")
    fresh = pd.read_csv(output)
    assert len(fresh) == len(reference) == 90
    assert (fresh["mean"] - reference["mean"]).abs().max() > 1e-12


def test_table2_rejects_agreeing_hashes_that_differ_from_the_frozen_split(tmp_path):
    inputs = []
    for filename in ("rct_null_results.csv", "requested_baseline_results.csv"):
        rows = pd.read_csv(ROOT / "data" / "table2" / filename)
        selected = rows.protocol.eq("primary") & rows.dataset.eq("actg175") & rows.realization.eq(0)
        rows.loc[selected, "query_ids_hash"] = "a" * 64
        path = tmp_path / filename
        rows.to_csv(path, index=False)
        inputs.append(path)

    output = tmp_path / "table2.csv"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--rct-results", str(inputs[0]),
         "--requested-results", str(inputs[1]), "--output", str(output)],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "frozen split" in completed.stderr
    assert not output.exists()
