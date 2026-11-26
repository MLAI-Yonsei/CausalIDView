#!/usr/bin/env python3
"""Score aligned Figure 2 predictions against SCM ground truth and aggregate by cell."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "Baselines" / "src")]

from baselines.checkpoints import CheckpointUnavailable, validate_checkpoint
from data.generate_world import load_world
from data.ground_truth import query_ground_truth
from data.metrics import aggregate_worlds, point_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fresh = ROOT / "artifacts" / "fig2_fresh"
    default_output = fresh / "fig2.csv"
    default_world_metrics = fresh / "fig2_world_metrics.csv"
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fig2.yaml")
    parser.add_argument("--worlds", type=Path, default=ROOT / "artifacts" / "worlds")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output", type=Path, default=default_output,
                        help="Aggregate CSV (default: artifacts/fig2_fresh/fig2.csv)")
    parser.add_argument("--validate-checkpoints", action="store_true")
    parser.add_argument("--world-metrics", type=Path,
                        help="Per-world CSV (full-run default: artifacts/fig2_fresh/fig2_world_metrics.csv)")
    parser.add_argument("--regime", choices=("BD", "FD", "IV", "PROX"))
    parser.add_argument("--model")
    parser.add_argument("--world", type=int, action="append")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.validate_checkpoints:
        unavailable = []
        for name, spec in config["checkpoints"].items():
            try:
                validate_checkpoint(spec, ROOT)
            except CheckpointUnavailable as error:
                unavailable.append(name)
                print(f"{name}: {error}", file=sys.stderr)
            else:
                print(f"verified checkpoint: {name}", flush=True)
        if unavailable:
            raise SystemExit(f"{len(unavailable)} checkpoint groups are unavailable")
        return

    seeds = config["scm"]["world_seeds"] if args.world is None else args.world
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(config["scm"]["world_seeds"]):
        parser.error("world IDs must be distinct and configured")
    partial = args.world is not None or args.regime is not None or args.model is not None
    world_metrics_path = args.world_metrics
    if partial:
        if args.output.resolve() == default_output.resolve():
            parser.error("pass a separate --output for a partial evaluation")
        if world_metrics_path and world_metrics_path.resolve() == default_world_metrics.resolve():
            parser.error("pass a separate --world-metrics for a partial evaluation")
    else:
        world_metrics_path = world_metrics_path or default_world_metrics
    if world_metrics_path and world_metrics_path.resolve() == args.output.resolve():
        parser.error("--output and --world-metrics must use different paths")

    rows = []
    for regime, models in config["panels"].items():
        if args.regime not in (None, regime):
            continue
        for order, (model, label) in enumerate(models):
            if args.model not in (None, model):
                continue
            for seed in seeds:
                world = load_world(args.worlds / f"world_{int(seed):03d}.npz")
                unit_id, target = query_ground_truth(world)
                path = args.predictions / regime / model / f"world_{int(seed):03d}.npz"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"missing estimator output: {path}\n"
                        "Run scripts/fig2/predict_fig2.py for this cell and world."
                    )
                with np.load(path, allow_pickle=False) as prediction:
                    predicted_ids = np.asarray(prediction["unit_id"])
                    estimate = np.asarray(prediction["tau_hat"], dtype=float)
                if not np.array_equal(predicted_ids, unit_id):
                    raise ValueError(f"query unit alignment failed: {path}")
                if estimate.shape != target.shape or not np.isfinite(estimate).all():
                    raise ValueError(f"invalid prediction values or shape: {path}")
                rows.append({
                    "regime": regime, "model": model, "model_label": label,
                    "panel_order": order, "world_id": int(seed),
                    **point_metrics(estimate, target),
                })
    if not rows:
        parser.error("selection contains no Figure 2 cells")
    world_metrics = pd.DataFrame(rows)
    settings = config["evaluation"]
    result = aggregate_worlds(
        world_metrics, int(settings["bootstrap_replicates"]), int(settings["bootstrap_seed"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.17g")
    if world_metrics_path:
        world_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        world_metrics.to_csv(world_metrics_path, index=False, float_format="%.17g")
        print(world_metrics_path)
    print(args.output)


if __name__ == "__main__":
    main()
