#!/usr/bin/env python3
"""Score fresh IHDP-HC predictions for the requested model families."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fig6.predict_fig6b import (
    CONDITIONS, MODELS, PREDICTIONS, SPLIT_SIZES, TRIAL_COUNT, TRIAL_DATA,
    prediction_path,
)
from data.metrics import point_metrics

CFMS = ("causalpfn", "dopfn", "causalfm")
SCORE_FIELDS = ("replicate", "model", "condition", "spehe")


def score_predictions(input_directory: Path, model_paths: dict[str, Path]) -> tuple[list[tuple], list[Path]]:
    rows = []
    prediction_files = []
    query_shape = (SPLIT_SIZES["test"],)
    for trial in range(TRIAL_COUNT):
        path = input_directory / f"trial_{trial:03d}" / "evaluator_only.npz"
        with np.load(path, allow_pickle=False) as evaluator:
            truth, sample_id = evaluator["test_tau_true"], evaluator["test_sample_id"]
        if (truth.shape != query_shape or sample_id.shape != query_shape
                or np.unique(sample_id).size != len(sample_id) or not np.isfinite(truth).all()):
            raise ValueError(f"Invalid evaluation arrays in {path}")
        for condition in CONDITIONS:
            for model, directory in model_paths.items():
                path = prediction_path(directory, trial, condition, model)
                with np.load(path, allow_pickle=False) as prediction:
                    np.testing.assert_array_equal(prediction["sample_id"], sample_id, err_msg=str(path))
                    estimate = prediction["tau_hat"]
                if estimate.shape != query_shape or not np.isfinite(estimate).all():
                    raise ValueError(f"Invalid predictions in {path}")
                spehe = math.sqrt(point_metrics(estimate, truth)["sPEHE_sq"])
                if not math.isfinite(spehe):
                    raise ValueError(f"Non-finite sPEHE for {path}")
                rows.append((trial, model, condition, spehe))
                prediction_files.append(path)
    return rows, prediction_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRIAL_DATA)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    models = parser.add_mutually_exclusive_group()
    models.add_argument("--cfm-predictions", type=Path, help="Optional separate directory of fresh CFM predictions")
    models.add_argument("--tabpfn-only", action="store_true", help="Score only the three TabPFN models")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "fig6b_trial_metrics.csv")
    args = parser.parse_args()

    model_paths = dict.fromkeys(MODELS, args.predictions)
    if not args.tabpfn_only:
        model_paths.update(dict.fromkeys(CFMS, args.cfm_predictions or args.predictions))
    rows, _ = score_predictions(args.input, model_paths)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temporary:
        pending = Path(temporary) / args.output.name
        with pending.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(SCORE_FIELDS)
            writer.writerows(rows)
        pending.replace(args.output)
    print(f"Wrote {len(rows)} trial metrics to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, KeyError, OSError, ValueError) as error:
        sys.exit(f"Figure 6(b) scoring failed: {error}")
