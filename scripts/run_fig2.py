#!/usr/bin/env python3
"""Validate aligned estimator predictions and rebuild results/fig2.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.estimators import validate_checkpoint
from data.generate_world import load_world
from data.ground_truth import query_ground_truth
from data.metrics import aggregate_worlds, point_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fig2.yaml")
    parser.add_argument("--worlds", type=Path, default=ROOT / "artifacts" / "worlds")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "fig2.csv")
    parser.add_argument("--validate-checkpoints", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.validate_checkpoints:
        for name, spec in config["checkpoints"].items():
            validate_checkpoint(spec, ROOT)
            print(f"verified checkpoint: {name}")

    rows = []
    for regime, models in config["panels"].items():
        for order, (model, label) in enumerate(models):
            for seed in config["scm"]["world_seeds"]:
                world = load_world(args.worlds / f"world_{int(seed):03d}.npz")
                unit_id, target = query_ground_truth(world)
                path = args.predictions / regime / model / f"world_{int(seed):03d}.npz"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"missing estimator output: {path}\n"
                        "Create it with the official model implementation using the view contracts "
                        "in data/estimators; each NPZ must contain unit_id and tau_hat."
                    )
                with np.load(path, allow_pickle=False) as prediction:
                    predicted_ids = np.asarray(prediction["unit_id"])
                    estimate = np.asarray(prediction["tau_hat"], dtype=float)
                if not np.array_equal(predicted_ids, unit_id):
                    raise ValueError(f"query unit alignment failed: {path}")
                rows.append({
                    "regime": regime, "model": model, "model_label": label,
                    "panel_order": order, "world_id": int(seed),
                    **point_metrics(estimate, target),
                })
    world_metrics = pd.DataFrame(rows)
    settings = config["evaluation"]
    result = aggregate_worlds(
        world_metrics, int(settings["bootstrap_replicates"]), int(settings["bootstrap_seed"])
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, float_format="%.17g")
    print(args.output)


if __name__ == "__main__":
    main()
