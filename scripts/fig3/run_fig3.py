#!/usr/bin/env python3
"""Score new Figure 3 predictions against the paired SCM query targets."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.fig3 import SUMMARY_FILE, WORLD_FILE, aggregate, load_settings, load_targets, score_world


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fresh = ROOT / "artifacts" / "fig3_fresh"
    parser.add_argument("--predictions", type=Path, default=fresh / "predictions")
    parser.add_argument("--output", type=Path, default=fresh)
    args = parser.parse_args()
    reference = ROOT / "results" / "fig3"
    if any((args.output / name).resolve() == (reference / name).resolve() for name in (WORLD_FILE, SUMMARY_FILE)):
        parser.error("--output must preserve the reference tables in results/fig3")

    try:
        panels, worlds = load_settings(ROOT)
        targets = load_targets(ROOT / "results" / "fig2_world_manifest.json", worlds)
        fingerprints = {}
        rows = [dict(regime=regime, model=model, world_id=world,
                     **score_world(args.predictions, regime, model, world, targets[world].query, fingerprints))
                for regime, models in panels.items() for model, _ in models for world in worlds]
        per_world = pd.DataFrame(rows)
        summary = aggregate(per_world, panels, worlds)
        args.output.mkdir(parents=True, exist_ok=True)
        per_world.to_csv(args.output / WORLD_FILE, index=False, float_format="%.17g")
        summary.to_csv(args.output / SUMMARY_FILE, index=False, float_format="%.17g")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"{len(summary)} cells, {len(per_world)} world metrics in {args.output}")


if __name__ == "__main__":
    main()
