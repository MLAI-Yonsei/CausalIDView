"""Use the common Figure 6 scoring code for one panel."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
import sys

from .verify_fig6 import (
    ARCHIVE, ARTIFACTS, IHDP, PARTIAL_MODELS, POINT_MODELS, SENSITIVITY_GAMMA,
    SENSITIVITY_MODELS, TRIALS, compare_summaries, score_bounds, score_fig6a, score_fig6b,
    sha256, summarize_panel_c, summarize_points, verify_sources, verify_tables, write_outputs,
)


def main(panel):
    parser = argparse.ArgumentParser(description=f"Score Figure 6({panel}) and compare with the bundled v2 reference.")
    parser.add_argument("--archived", action="store_true", help="Use the bundled archived predictions")
    parser.add_argument("--output-dir", "--output", type=Path)
    parser.add_argument("--atol", type=float, help="Require every summary value to meet this absolute tolerance")
    if panel == "c":
        parser.add_argument("--predictions", "--bound-predictions", type=Path, action="append",
                            help="Bound prediction CSV or Parquet file. Repeat for multiple models or shards.")
    else:
        parser.add_argument("--predictions", f"--fig6{panel}-predictions", type=Path,
                            help="Directory containing the complete query predictions")
    if panel == "b":
        parser.add_argument("--cfm-predictions", "--fig6b-cfm-predictions", type=Path,
                            help="Separate directory for the CFM predictions")
    args = parser.parse_args()
    if args.atol is not None and (not isfinite(args.atol) or args.atol < 0):
        parser.error("--atol must be finite and nonnegative")
    if panel == "b" and args.archived and args.cfm_predictions is not None and args.predictions is None:
        parser.error("--cfm-predictions requires --predictions in archived mode")

    provenance = json.loads((ARCHIVE / "archive_manifest.json").read_text())
    evaluators = []
    replaced_a = False
    replaced_b = set()
    replaced_bounds = set()
    if panel == "c":
        files = args.predictions or []
        if not (files or args.archived):
            files = [ARTIFACTS / "fig6_bounds" / f"{model}.csv"
                     for model in (*PARTIAL_MODELS, *SENSITIVITY_MODELS)]
        inputs = ["bound_targets.csv.gz", "bound_oracles.csv.gz"]
        if args.archived:
            inputs.append("bound_predictions.csv.gz")
        sources = verify_sources(provenance, inputs, evaluators)
        bounds, replaced_bounds = score_bounds(files, args.archived)
        tables = {"bounds_per_world": bounds, "panel_c": summarize_panel_c(bounds)}
    else:
        predictions = args.predictions
        if predictions is None and not args.archived:
            predictions = ARTIFACTS / f"fig6{panel}_predictions"
        if panel == "a":
            inputs = ["synthetic_targets.csv.gz"]
            archived_predictions = "synthetic_predictions.csv.gz"
        else:
            inputs = []
            archived_predictions = "ihdp_predictions.csv.gz"
            evaluators = [IHDP / f"trial_{trial:03d}" / "evaluator_only.npz" for trial in TRIALS]
        if predictions is None:
            inputs.append(archived_predictions)
        sources = verify_sources(provenance, inputs, evaluators)
        if panel == "a":
            points, files = score_fig6a(predictions)
            replaced_a = bool(files)
        else:
            points, files = score_fig6b(evaluators, predictions, args.cfm_predictions)
            replaced_b = set(POINT_MODELS) if files else set()
        tables = {"point_per_replicate": points, "point_summary": summarize_points(points)}

    comparison, checked, hashes = compare_summaries(tables, provenance["reference_files"])
    sources.update(hashes)
    archived_checks = {}
    if args.archived:
        archived_checks, hashes = verify_tables(
            tables, replaced_a, replaced_b, replaced_bounds, provenance["reference_files"],
        )
        sources.update(hashes)
    sources.update((str(path.resolve()), sha256(path)) for path in files)

    mode = "archived" if args.archived else "fresh"
    if args.archived and files:
        mode = "mixed" if panel == "c" else "fresh"
    output = (args.output_dir or ARTIFACTS / f"fig6{panel}_{mode}").resolve()
    manifest = {
        "panel": panel, "input_mode": mode, "sources": sources,
        "reference_comparisons": checked, "archived_verification": archived_checks,
        "absolute_tolerance": args.atol,
    }
    if panel == "c":
        manifest["sensitivity_gamma"] = SENSITIVITY_GAMMA
    if args.atol is not None:
        comparison["within_tolerance"] = comparison["delta"].abs().le(args.atol)
    write_outputs(output, tables, comparison, manifest)
    print(f"Figure 6({panel}): compared {len(comparison)} values. {output / 'comparison.csv'}")
    if args.atol is not None and not comparison["within_tolerance"].all():
        raise ValueError(f"Figure 6({panel}) exceeds absolute tolerance {args.atol:g}")


def cli(panel):
    try:
        main(panel)
    except (AssertionError, ImportError, KeyError, OSError, ValueError) as error:
        sys.exit(f"ERROR: {error}")
