#!/usr/bin/env python3
"""Generate Figure 6(b) CausalPFN, Do-PFN, and CausalFM predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "Baselines" / "src")]

from baselines.checkpoints import CheckpointUnavailable, validate_checkpoint
from baselines.fig2 import runtime
from scripts.fig2.predict_fig2 import PATHS, check_runtime
from scripts.fig6.predict_fig6b import CONDITIONS, PREDICTIONS, TRIAL_COUNT, TRIAL_DATA, model_input, prediction_path

MODELS = ("causalpfn", "dopfn", "causalfm")


def estimate(model, device, values):
    from baselines import cfm

    if model == "causalfm":
        prediction, _ = cfm._run_causalfm(values, device)
        return prediction
    if model == "dopfn":
        release = ROOT / "Baselines" / "CFMs" / "Do-PFN-fig2"
        prediction, _ = cfm._run_dopfn(values, device, model_root=release)
        return prediction

    from Baselines.CFMs.performance_reproduction.causalpfn.protocol import (
        _official_cate_estimator, checkpoint_path, set_official_seed,
    )

    release = ROOT / "Baselines" / "CFMs" / "CausalPFN"
    treatment = values["treatment"]
    # The IHDP protocol caps retrieval to the smaller treatment arm.
    neighbours = min(1024, int(np.bincount(treatment).min()))
    set_official_seed(82718)
    with _official_cate_estimator(release) as cate_type:
        estimator = cate_type(device=device, model_path=str(checkpoint_path(release)),
                              num_neighbours=neighbours)
        estimator.fit(values["X_context"], treatment, values["outcome"])
        return estimator.estimate_cate(values["X_query"])


def run_worker(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("device", choices=("cuda:0", "cpu"))
    parser.add_argument("condition", choices=CONDITIONS)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(arguments)
    x, treatment, y, query, sample_id = model_input(args.input, CONDITIONS[args.condition])
    if (not np.array_equal(np.unique(treatment), [0, 1])
            or np.unique(sample_id).size != len(sample_id)
            or not all(np.isfinite(value).all() for value in (x, y, query))):
        raise ValueError(f"Invalid IHDP-HC model input in {args.input}")
    prediction = estimate(args.model, args.device, {
        "X_context": x, "treatment": treatment, "outcome": y, "X_query": query,
    })
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    if prediction.shape != sample_id.shape or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predictions for {args.input}, {args.model}")
    np.savez_compressed(args.output, sample_id=sample_id, tau_hat=prediction)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TRIAL_DATA)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--trials", type=int, default=TRIAL_COUNT, help="Use 1 for a smoke test")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    for model in MODELS:
        parser.add_argument(f"--python-{model}", type=Path,
                            default=ROOT / f".venv-{model}" / "bin" / "python")
    args = parser.parse_args()
    if not (1 <= args.trials <= TRIAL_COUNT and 0 <= args.shard < args.shards <= args.trials):
        parser.error(f"Require 1 <= trials <= {TRIAL_COUNT} and 0 <= shard < shards <= trials")
    trials = range(args.shard, args.trials, args.shards)
    conditions = CONDITIONS if args.condition is None else (args.condition,)
    models = MODELS if args.model is None else (args.model,)
    print(f"{len(trials) * len(conditions) * len(models)} jobs", flush=True)
    if args.dry_run:
        return
    if args.device == "cpu" and "causalfm" in models:
        parser.error("The packaged CausalFM worker requires CUDA; select CausalPFN or Do-PFN for CPU")

    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    interpreters = {}
    for model in models:
        profile, checkpoint = runtime(model, "BD")
        path = getattr(args, f"python_{profile}").absolute()
        if not path.is_file():
            parser.error(f"missing {profile} runtime: {path}; see README")
        try:
            check_runtime(path, profile, args.device)
            validate_checkpoint(config["checkpoints"][checkpoint], ROOT)
        except (CheckpointUnavailable, ValueError) as error:
            parser.error(str(error))
        interpreters[model] = path

    jobs = []
    for trial in trials:
        for condition in conditions:
            source = args.input / f"trial_{trial:03d}" / f"model_input_{condition}.npz"
            if not source.is_file():
                parser.error(f"missing {source}")
            for model in models:
                output = prediction_path(args.predictions, trial, condition, model)
                if output.exists():
                    parser.error(f"{output} already exists; use a fresh prediction directory")
                jobs.append((model, condition, source, output))

    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(map(str, PATHS)),
        "CAUSALARENA_WORKSPACE": str(ROOT),
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }
    if args.device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    for model, condition, source, output in jobs:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
            pending = Path(temporary) / output.name
            subprocess.run(
                [str(interpreters[model]), str(Path(__file__).resolve()), "_worker",
                 model, args.device, condition, str(source.resolve()), str(pending.resolve())],
                check=True, cwd=ROOT, env=environment,
            )
            os.link(pending, output)
        print(output, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_worker"]:
        run_worker(sys.argv[2:])
    else:
        main()
