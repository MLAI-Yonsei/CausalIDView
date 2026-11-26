#!/usr/bin/env python3
"""Generate paired synthetic BD and HC predictions for Figure 6(a)."""

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
from baselines.fig2 import model_input, runtime
from data.generate_world import load_world
from scripts.fig2.predict_fig2 import PATHS, check_runtime

CONDITIONS = ("BD", "HC")
MODELS = ("causalpfn", "dopfn", "causalfm", "tabpfn_x", "tabpfn_s", "tabpfn_dr")
RUNTIMES = ("causalpfn", "dopfn", "causalfm", "tabpfn")


def paired_input(world, condition, model):
    payload, graph, _, unit_id = model_input(world, "BD", model)
    if condition == "HC":
        # The BD view appends U after the original 50 covariates.
        payload["X_context"] = payload["X_context"][:, :-1]
        payload["X_query"] = payload["X_query"][:, :-1]
        graph = "hidden_confounding"
    width = 51 if condition == "BD" else 50
    if (payload["X_context"].shape != (1024, width)
            or payload["X_query"].shape != (100, width)
            or unit_id.shape != (100,) or len(np.unique(unit_id)) != 100
            or not all(np.isfinite(value).all() for value in payload.values())):
        raise ValueError(f"Invalid synthetic {condition} input")
    return payload, graph, unit_id


def worker_prediction(model, world, graph, device, values):
    from baselines.fig2 import TABPFN
    from baselines.fig2_worker import run_cfm

    if model not in TABPFN:
        return run_cfm(model, values, device, graph)
    from baselines.tabpfn.fig2 import _model_seed, _run_bd

    route = TABPFN[model]
    seed = _model_seed(route, world)
    if graph == "hidden_confounding":
        seed += 4
    prediction, _ = _run_bd({"payload": values, "route": route}, seed, "v3.5")
    return prediction


def run_worker(arguments):
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("world", type=int)
    parser.add_argument("graph", choices=("backdoor", "hidden_confounding"))
    parser.add_argument("device", choices=("cpu", "cuda:0"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(arguments)
    with np.load(args.input, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    prediction = worker_prediction(args.model, args.world, args.graph, args.device, values)
    np.save(args.output, np.asarray(prediction, dtype=float).reshape(-1), allow_pickle=False)


def predict(world, world_id, condition, model, python, device, output):
    payload, graph, unit_id = paired_input(world, condition, model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
        directory = Path(temporary)
        observed, raw_prediction = directory / "input.npz", directory / "tau.npy"
        np.savez_compressed(observed, **payload)
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(map(str, PATHS)),
            "CAUSALARENA_WORKSPACE": str(ROOT),
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "TABPFN_DEVICE": device, "TABPFN_DISABLE_TELEMETRY": "1",
            "SKB_DATA_DIRECTORY": os.environ.get(
                "SKB_DATA_DIRECTORY", str(ROOT / "artifacts" / "cache" / "skrub"),
            ),
        }
        if device == "cpu":
            environment["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run(
            [str(python), str(Path(__file__).resolve()), "_worker", model, str(world_id),
             graph, device, str(observed.resolve()), str(raw_prediction.resolve())],
            check=True, cwd=ROOT, env=environment,
        )
        estimate = np.load(raw_prediction, allow_pickle=False)
        if estimate.shape != unit_id.shape or not np.isfinite(estimate).all():
            raise ValueError(f"Invalid synthetic prediction for {condition}/{model}/world_{world_id:03d}")
        pending = directory / output.name
        np.savez_compressed(pending, unit_id=unit_id, tau_hat=estimate)
        os.link(pending, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds-dir", type=Path, default=ROOT / "artifacts" / "worlds")
    parser.add_argument("--predictions", type=Path, default=ROOT / "artifacts" / "fig6a_predictions")
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--world", type=int, action="append")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    for name in RUNTIMES:
        parser.add_argument(f"--python-{name}", type=Path,
                            default=ROOT / f".venv-{name}" / "bin" / "python")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    seeds = config["scm"]["world_seeds"] if args.world is None else args.world
    if len(seeds) != len(set(seeds)) or not set(seeds) <= set(config["scm"]["world_seeds"]):
        parser.error("world IDs must be distinct and configured")
    conditions = CONDITIONS if args.condition is None else (args.condition,)
    models = MODELS if args.model is None else (args.model,)
    print(f"{len(conditions) * len(models) * len(seeds)} jobs", flush=True)
    if args.dry_run:
        for condition in conditions:
            for model in models:
                print(condition, model, *runtime(model, condition))
        return
    if args.device == "cpu" and set(models) - {"causalpfn", "dopfn"}:
        parser.error("CPU execution supports CausalPFN and Do-PFN; select a model")

    interpreters = {}
    for model in models:
        profile, checkpoint = runtime(model, "BD")
        if profile in interpreters:
            continue
        path = getattr(args, f"python_{profile}").absolute()
        if not path.is_file():
            parser.error(f"missing {profile} runtime: {path}; see README")
        try:
            check_runtime(path, profile, args.device)
            validate_checkpoint(config["checkpoints"][checkpoint], ROOT)
        except (CheckpointUnavailable, ValueError) as error:
            parser.error(str(error))
        interpreters[profile] = path

    for world_id in seeds:
        source = args.worlds_dir / f"world_{world_id:03d}.npz"
        if not source.is_file():
            parser.error(f"missing {source}; run scripts/fig2/generate_data.py")
        for condition in conditions:
            for model in models:
                output = args.predictions / condition / model / source.name
                if output.exists():
                    parser.error(f"{output} already exists; use a fresh prediction directory")

    for world_id in seeds:
        world = load_world(args.worlds_dir / f"world_{world_id:03d}.npz")
        if world.metadata["world_seed"] != world_id:
            raise ValueError(f"Input world ID differs from filename for world {world_id}")
        for condition in conditions:
            for model in models:
                profile, _ = runtime(model, condition)
                output = args.predictions / condition / model / f"world_{world_id:03d}.npz"
                predict(world, world_id, condition, model, interpreters[profile], args.device, output)
                print(output, flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["_worker"]:
        run_worker(sys.argv[2:])
    else:
        main()
