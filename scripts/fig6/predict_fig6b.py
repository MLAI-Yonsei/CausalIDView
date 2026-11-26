#!/usr/bin/env python3
"""Generate Fig. 6(b) TabPFN predictions on the archived IHDP-HC splits."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TRIAL_DATA = ROOT / "data" / "ihdp_hc"
PREDICTIONS = ROOT / "artifacts" / "fig6b_predictions"
TRIAL_COUNT = 100
SPLIT_SIZES = {"train": 470, "valid": 202, "test": 75}
CONDITIONS = {"full": 25, "hidden": 24}
MODELS = ("tabpfn_x", "tabpfn_s", "tabpfn_dr")


def prediction_path(directory: Path, trial: int, condition: str, model: str) -> Path:
    return directory / condition / model / f"trial_{trial:03d}.npz"


def model_input(source: Path, width: int):
    with np.load(source, allow_pickle=False) as data:
        for split, size in SPLIT_SIZES.items():
            if (data[f"{split}_sample_id"].shape != (size,)
                    or data[f"{split}_X"].shape != (size, width)):
                raise ValueError(f"Unexpected {split} split dimensions in {source}")
        x = np.vstack((data["train_X"], data["valid_X"])).astype(float)
        treatment = np.concatenate((data["train_treatment"], data["valid_treatment"])).astype(int)
        outcome = np.concatenate((data["train_factual_y"], data["valid_factual_y"])).astype(float)
        return x, treatment, outcome, data["test_X"].astype(float), data["test_sample_id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRIAL_DATA)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=TRIAL_COUNT, help="Use 1 for a smoke test")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if not (1 <= args.trials <= TRIAL_COUNT and 0 <= args.shard < args.shards <= args.trials):
        parser.error(f"Require 1 <= trials <= {TRIAL_COUNT} and 0 <= shard < shards <= trials")
    checkpoint_override = args.checkpoint.resolve() if args.checkpoint is not None else None
    if checkpoint_override is not None and checkpoint_override.suffix != ".safetensors":
        parser.error("--checkpoint must point to a .safetensors file")
    trials = range(args.shard, args.trials, args.shards)
    for trial in trials:
        for condition in CONDITIONS:
            for model in MODELS:
                path = prediction_path(args.predictions, trial, condition, model)
                if path.exists():
                    parser.error(f"{path} already exists. Use a fresh prediction directory.")

    sys.path.insert(0, str(ROOT / "Baselines" / "src"))
    from baselines.checkpoints import CheckpointUnavailable, validate_checkpoint
    from baselines.tabpfn.backend import OfficialTabPFNBackend
    from baselines.tabpfn.estimators import (
        TabPFNDRLearner, TabPFNSLearner, TabPFNXLearner,
    )

    spec = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())["checkpoints"]["tabpfn_v35"]
    if checkpoint_override is not None:
        spec["path"] = str(checkpoint_override)
    try:
        checkpoint = validate_checkpoint(spec, ROOT)
    except CheckpointUnavailable as error:
        parser.error(str(error))

    settings = yaml.safe_load((ROOT / "configs" / "fig6b.yaml").read_text())
    config = settings["estimator"]
    config["tabpfn"].update(device=args.device, checkpoint_revision=spec["revision"])
    backend = OfficialTabPFNBackend(
        config, package_version=settings["runtime"]["tabpfn_version"], checkpoint_path=checkpoint,
    )
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("SKB_DATA_DIRECTORY", str(ROOT / "artifacts" / "cache" / "skrub"))
    estimators = dict(zip(MODELS, (TabPFNXLearner, TabPFNSLearner, TabPFNDRLearner), strict=True))
    for trial in trials:
        trial_config = {**config, "seed": trial}
        for condition, width in CONDITIONS.items():
            source = args.input / f"trial_{trial:03d}" / f"model_input_{condition}.npz"
            x, treatment, outcome, query, sample_id = model_input(source, width)
            for model, estimator_class in estimators.items():
                estimator = estimator_class(trial_config, backend=backend)
                estimator.fit(x, treatment, outcome, outcome_type="continuous")
                prediction = np.asarray(estimator.predict_cate(query)).reshape(-1)
                if prediction.shape != sample_id.shape or not np.isfinite(prediction).all():
                    raise ValueError(f"Invalid predictions for {source}, {model}")
                path = prediction_path(args.predictions, trial, condition, model)
                path.parent.mkdir(parents=True, exist_ok=True)
                with TemporaryDirectory(dir=path.parent) as temporary:
                    pending = Path(temporary) / path.name
                    np.savez_compressed(pending, sample_id=sample_id, tau_hat=prediction)
                    # Publish the complete file without overwriting another worker's output.
                    os.link(pending, path)
                print(f"{trial:03d} {condition} {model}", flush=True)


if __name__ == "__main__":
    main()
