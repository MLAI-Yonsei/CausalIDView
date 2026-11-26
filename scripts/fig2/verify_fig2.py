#!/usr/bin/env python3
"""Aggregate per-world scores and compare them with the published Figure 2 values."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.metrics import aggregate_worlds

KEY = ["regime", "model"]
VALUE = ["mean", "ci_low", "ci_high", "ate_error_sq", "centered_sPEHE_sq"]
METRICS = ["sPEHE_sq", "ate_error_sq", "centered_sPEHE_sq"]


def rebuild(world_metrics: Path, config: dict) -> pd.DataFrame:
    panels = pd.DataFrame([
        (regime, model, label, order)
        for regime, models in config["panels"].items()
        for order, (model, label) in enumerate(models)
    ], columns=[*KEY, "model_label", "panel_order"])
    expected = {
        (row.regime, row.model, seed)
        for row in panels.itertuples(index=False)
        for seed in config["scm"]["world_seeds"]
    }
    worlds = pd.read_csv(world_metrics, float_precision="round_trip")
    keys = list(worlds[[*KEY, "world_id"]].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("world metrics must cover each configured model and world once")
    values = worlds[METRICS].to_numpy()
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("world metrics must be finite and nonnegative")
    np.testing.assert_allclose(values[:, 0], values[:, 1] + values[:, 2], rtol=0, atol=1e-12)
    worlds = worlds.drop(columns=["model_label", "panel_order"], errors="ignore")
    worlds = panels.merge(worlds, on=KEY, validate="one_to_many")
    settings = config["evaluation"]
    return aggregate_worlds(worlds, settings["bootstrap_replicates"], settings["bootstrap_seed"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--world-metrics", type=Path,
                         help="Per-world CSV (default: artifacts/fig2_fresh/fig2_world_metrics.csv)")
    sources.add_argument("--archived", action="store_true", help="Use the published per-world metric archive")
    parser.add_argument("--reference", type=Path, default=ROOT / "results" / "fig2.csv")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--comparison", type=Path,
                        help="Write comparison values (default for the fresh score file: fig2_comparison.csv)")
    parser.add_argument("--atol", type=float, default=1e-12)
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")
    if args.archived:
        world_metrics = ROOT / "results" / "fig2_world_metrics.csv"
    else:
        world_metrics = args.world_metrics or ROOT / "artifacts" / "fig2_fresh" / "fig2_world_metrics.csv"
    if not world_metrics.is_file():
        message = f"missing world metrics: {world_metrics}"
        if not args.archived and args.world_metrics is None:
            message += ". Run scripts/fig2/run_fig2.py after inference, or pass --archived to check the archive."
        parser.error(message)
    comparison = args.comparison
    if comparison is None and not args.archived and args.world_metrics is None:
        comparison = world_metrics.with_name("fig2_comparison.csv")
    outputs = [path.resolve() for path in (args.output, comparison) if path is not None]
    if len(set(outputs)) != len(outputs) or set(outputs).intersection((world_metrics.resolve(), args.reference.resolve())):
        parser.error("output paths must differ from each other and from the input and reference paths")

    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    result = rebuild(world_metrics, config)
    actual = result.set_index(KEY).sort_index()
    reference = pd.read_csv(args.reference, float_precision="round_trip").set_index(KEY).sort_index()
    metadata = ["model_label", "panel_order", "n_worlds"]
    pd.testing.assert_frame_equal(
        actual[metadata], reference[metadata], check_exact=True,
    )
    differences = actual[VALUE] - reference[VALUE]
    deltas = differences.abs()
    if comparison is not None:
        report = pd.concat([
            pd.DataFrame({
                "metric": metric,
                "actual": actual[metric],
                "reference": reference[metric],
                "delta": differences[metric],
                "abs_delta": deltas[metric],
            })
            for metric in VALUE
        ]).reset_index()
        comparison.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(comparison, index=False, float_format="%.17g")
        print(comparison)
    print(f"{len(actual)} cells, {result.n_worlds.sum()} world metrics; max absolute delta {deltas.max().max():.3g}")
    invalid = (~np.isfinite(deltas) | deltas.gt(args.atol)).any(axis=1)
    if invalid.any():
        print(deltas[invalid].to_string(), file=sys.stderr)
        raise SystemExit(f"Figure 2 exceeds absolute tolerance {args.atol:g}")
    pd.testing.assert_series_equal(actual["is_best"], reference["is_best"], check_exact=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
