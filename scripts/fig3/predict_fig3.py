#!/usr/bin/env python3
"""Fit the v2 baselines on the three paired Figure 3 outcome conditions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.fig2.predict_fig2 import add_runtime_options, predict, validate_runtimes
from baselines.checkpoints import CheckpointUnavailable
from baselines.fig2 import MODEL_IDS, runtime
from data.fig3 import CONDITIONS, check_world, derive_world, load_targets, prediction_identity, prediction_path
from data.generate_world import load_world
from data.integrity import array_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds-dir", type=Path, default=ROOT / "artifacts" / "worlds")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "fig3_fresh" / "predictions")
    parser.add_argument("--regime", choices=("BD", "FD", "IV", "PROX"))
    parser.add_argument("--model", choices=sorted(MODEL_IDS))
    parser.add_argument("--world", type=int, action="append")
    parser.add_argument("--dry-run", action="store_true")
    add_runtime_options(parser)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    worlds = config["scm"]["world_seeds"]
    seeds = args.world or worlds
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(worlds):
        parser.error("world IDs must be distinct and configured")
    cells = [(regime, model) for regime, models in config["panels"].items()
             if args.regime in (None, regime) for model, _ in models if args.model in (None, model)]
    if not cells:
        parser.error("selection contains no Figure 3 cells")
    print(f"{len(cells)} cells, {len(seeds)} worlds, {len(CONDITIONS)} conditions, "
          f"{len(cells) * len(seeds) * len(CONDITIONS)} jobs", flush=True)
    if args.dry_run:
        for regime, model in cells:
            print(regime, model, *CONDITIONS)
        return

    try:
        validate_runtimes(parser, args, cells, config, config)
        targets = load_targets(ROOT / "results" / "fig2_world_manifest.json", worlds)
        for seed in seeds:
            world = load_world(args.worlds_dir / f"world_{seed:03d}.npz")
            fingerprint = array_hash(*(world.arrays[key] for key in sorted(world.arrays)))
            for condition in CONDITIONS:
                variant = derive_world(world, condition)
                check_world(variant, targets[seed], seed, condition)
                for regime, model in cells:
                    profile, _ = runtime(model, regime)
                    python = getattr(args, f"python_{profile}").absolute()
                    output = prediction_path(args.output, regime, model, condition, seed)
                    metadata = prediction_identity(regime, model, condition, seed)
                    worker_options = ("--dopfn-fd-microbatch",) if model == "dopfn_fd_xonly" else ()
                    predict(variant, regime, model, seed, python, args.device, output,
                            metadata={**metadata, "source_world_sha256": fingerprint},
                            worker_options=worker_options)
                    print(output, flush=True)
    except (OSError, ValueError, CheckpointUnavailable) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
