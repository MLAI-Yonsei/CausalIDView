#!/usr/bin/env python3
"""Verify Figure 3 world metrics and summaries against the agreed response benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.fig3 import (
    SUMMARY_FILE, SUMMARY_METRICS, WORLD_FILE, WORLD_KEYS, WORLD_METRICS, aggregate, load_settings,
)


def comparison(actual, reference, keys, metrics, level, atol, rtol):
    actual = actual.set_index(keys, verify_integrity=True).sort_index()
    reference = reference.set_index(keys, verify_integrity=True).sort_index()
    if not actual.index.equals(reference.index):
        raise ValueError(f"Figure 3 {level} comparison coverage differs from the reference")
    return pd.concat([
        pd.DataFrame({
            "level": level, "metric": metric, "actual": actual[metric], "reference": reference[metric],
            "delta": actual[metric] - reference[metric], "abs_delta": (actual[metric] - reference[metric]).abs(),
            "within_tolerance": np.isclose(actual[metric], reference[metric], atol=atol, rtol=rtol),
        }).reset_index() for metric in metrics
    ], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archived", action="store_true", help="Rebuild the summary from the bundled world metrics")
    source.add_argument("--world-metrics", type=Path, help="Override the fresh per-world CSV")
    parser.add_argument("--output", type=Path, help="Directory for scores and comparison CSV")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    args = parser.parse_args()
    reference = ROOT / "results" / "fig3"
    output = args.output or ROOT / "artifacts" / ("fig3_archived" if args.archived else "fig3_fresh")
    scores = args.world_metrics or output / WORLD_FILE
    if any((output / name).resolve() == (reference / name).resolve() for name in (SUMMARY_FILE, "fig3_comparison.csv")):
        parser.error("--output must preserve the reference tables in results/fig3")
    if not args.archived and not scores.is_file():
        parser.error(f"missing fresh world metrics: {scores}. Run scripts/fig3/run_fig3.py first.")
    if not args.archived and (scores.resolve() == (reference / WORLD_FILE).resolve()
                              or scores.resolve() in {(output / name).resolve()
                                                     for name in (SUMMARY_FILE, "fig3_comparison.csv")}):
        parser.error("--world-metrics must be separate from the reference and output CSVs")

    try:
        manifest = json.loads((reference / "reference_manifest.json").read_text())
        atol = manifest["atol"] if args.atol is None else args.atol
        rtol = manifest["rtol"] if args.rtol is None else args.rtol
        if not np.isfinite([atol, rtol]).all() or min(atol, rtol) < 0:
            raise ValueError("--atol and --rtol must be finite and nonnegative")
        for name in (WORLD_FILE, SUMMARY_FILE):
            if hashlib.sha256((reference / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"Figure 3 reference checksum mismatch: {name}")
        world_reference = pd.read_csv(reference / WORLD_FILE, float_precision="round_trip")
        summary_reference = pd.read_csv(reference / SUMMARY_FILE, float_precision="round_trip")
        worlds = world_reference if args.archived else pd.read_csv(scores, float_precision="round_trip")
        panels, seeds = load_settings(ROOT)
        summary = aggregate(worlds, panels, seeds)
        summaries = comparison(summary, summary_reference, WORLD_KEYS[:2], SUMMARY_METRICS,
                               "summary", atol, rtol)
        report = summaries if args.archived else pd.concat([
            comparison(worlds, world_reference, WORLD_KEYS, WORLD_METRICS, "world", atol, rtol),
            summaries.assign(world_id=pd.NA),
        ], ignore_index=True)
        report["world_id"] = report["world_id"].astype("Int64") if "world_id" in report else pd.NA
        report = report.assign(atol=atol, rtol=rtol)
        output.mkdir(parents=True, exist_ok=True)
        report.to_csv(output / "fig3_comparison.csv", index=False, float_format="%.17g")
        summary.to_csv(output / SUMMARY_FILE, index=False, float_format="%.17g")
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"{len(summary)} cells, {len(worlds)} world metrics; max absolute delta {report.abs_delta.max():.3g}")
    print(output / "fig3_comparison.csv")
    failures = int((~report.within_tolerance).sum())
    if failures:
        raise SystemExit(f"Figure 3: {failures}/{len(report)} values exceed tolerance (atol={atol:g}, rtol={rtol:g})")


if __name__ == "__main__":
    main()
