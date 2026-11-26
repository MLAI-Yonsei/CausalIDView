#!/usr/bin/env python3
"""Generate Figure 2 predictions with the Rebuttal estimator implementations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
PATHS = (ROOT, ROOT / "Baselines" / "src")
sys.path[:0] = list(map(str, PATHS))
from baselines.checkpoints import validate_checkpoint
from baselines.fig2 import MODEL_IDS, model_input, runtime
from data.generate_world import load_world

RUNTIME_PACKAGES = {
    "classical": ("numpy", "scipy", "scikit-learn", "xgboost", "econml"),
    "neural": ("numpy", "scipy", "scikit-learn", "torch"),
    "causalpfn": ("numpy", "scipy", "scikit-learn", "torch"),
    "dopfn": ("numpy", "scipy", "scikit-learn", "torch"),
    "causalfm": ("numpy", "scipy", "scikit-learn", "torch", "tabpfn"),
    "tabpfn": ("numpy", "scipy", "scikit-learn", "torch", "tabpfn"),
}


def check_runtime(python, profile, device):
    references = json.loads((ROOT / "requirements" / "reference-runtime.json").read_text())
    packages = RUNTIME_PACKAGES[profile]
    check_cuda = profile != "classical" and device == "cuda:0"
    probe = """
import importlib.metadata
import json
import sys

packages = {}
for name in sys.argv[2:]:
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
cuda = None
if sys.argv[1] == 'True' and packages.get('torch'):
    import torch
    cuda = torch.cuda.is_available()
print(json.dumps({'python': list(sys.version_info[:2]), 'packages': packages, 'cuda': cuda}))
"""
    result = subprocess.run(
        [str(python), "-I", "-c", probe, str(check_cuda), *packages],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"could not check {profile} runtime at {python}: {result.stderr.strip()}")
    observed = json.loads(result.stdout)
    expected = references[profile]["packages"]
    mismatches = []
    if observed["python"] != [3, 10]:
        mismatches.append(f"Python 3.10 required, found {'.'.join(map(str, observed['python']))}")
    for name, version in observed["packages"].items():
        required = expected[name]
        actual = version if "+" in required or version is None else version.split("+", 1)[0]
        if actual != required:
            mismatches.append(f"{name}=={required} required, found {version or 'missing'}")
    if check_cuda and not observed["cuda"]:
        mismatches.append("CUDA is unavailable with the selected device visibility")
    if mismatches:
        raise ValueError(f"{profile} runtime at {python}: " + ", ".join(mismatches))


def predict(world, regime, model, seed, python, device, output, *, metadata=None, worker_options=()):
    output = output.absolute()
    payload, graph, train_ids, unit_ids = model_input(world, regime, model)
    train_rows = np.flatnonzero(world.arrays["split"] == 0)
    query_rows = np.flatnonzero(world.arrays["split"] == 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}-", dir=output.parent) as temporary:
        directory = Path(temporary)
        input_path, prediction = directory / "input.npz", directory / "prediction.npz"
        np.savez_compressed(input_path, **payload, train_ids=train_ids, query_ids=unit_ids,
                            train_rows=train_rows, query_rows=query_rows)
        environment = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(map(str, PATHS)),
            "CAUSALARENA_WORKSPACE": str(ROOT),
            "PYTHONNOUSERSITE": "1",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "TABPFN_DISABLE_TELEMETRY": "1",
            "SKB_DATA_DIRECTORY": os.environ.get(
                "SKB_DATA_DIRECTORY", str(ROOT / "artifacts" / "cache" / "skrub"),
            ),
        }
        subprocess.run([
            str(python), "-m", "baselines.fig2_worker",
            model, str(seed), graph, device, str(input_path), str(prediction),
            *worker_options,
        ], check=True, cwd=ROOT, env=environment)
        with np.load(prediction, allow_pickle=False) as archive:
            estimate = archive["tau_hat"]
        if estimate.shape != unit_ids.shape or not np.isfinite(estimate).all():
            raise ValueError(f"invalid prediction for {regime}/{model}/world_{seed:03d}")
        completed = directory / output.name
        np.savez_compressed(completed, unit_id=unit_ids, tau_hat=estimate, **(metadata or {}))
        os.replace(completed, output)


def add_runtime_options(parser):
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    for name in RUNTIME_PACKAGES:
        parser.add_argument(f"--python-{name}", type=Path,
                            default=ROOT / f".venv-{name}" / "bin" / "python")


def validate_runtimes(parser, args, selected, config, canonical):
    if args.device == "cpu" and any(runtime(model, regime)[0] == "tabpfn" for regime, model in selected):
        parser.error("the archived TabPFN-v3.5 routes require a visible CUDA GPU")
    verified = set()
    checked_profiles = set()
    for regime, model in selected:
        profile, checkpoint = runtime(model, regime)
        python = getattr(args, f"python_{profile}")
        if not python.is_file():
            parser.error(f"missing {profile} runtime: {python}; see README or supply --python-{profile}")
        if profile not in checked_profiles:
            try:
                check_runtime(python.absolute(), profile, args.device)
            except ValueError as error:
                parser.error(str(error))
            checked_profiles.add(profile)
        if checkpoint and checkpoint not in verified:
            if config["checkpoints"].get(checkpoint) != canonical["checkpoints"][checkpoint]:
                parser.error(f"checkpoint {checkpoint} must use the published specification")
            validate_checkpoint(canonical["checkpoints"][checkpoint], ROOT)
            verified.add(checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fig2.yaml")
    parser.add_argument("--worlds-dir", type=Path, default=ROOT / "artifacts" / "worlds")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "predictions")
    parser.add_argument("--regime", choices=("BD", "FD", "IV", "PROX"))
    parser.add_argument("--model", choices=sorted(MODEL_IDS))
    parser.add_argument("--world", type=int, action="append")
    add_runtime_options(parser)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    world_ids = config["scm"]["world_seeds"] if args.world is None else args.world
    if not set(world_ids) <= set(config["scm"]["world_seeds"]) or len(set(world_ids)) != len(world_ids):
        parser.error("world IDs must be distinct and configured")
    selected = [
        (regime, model)
        for regime, models in config["panels"].items() if args.regime in (None, regime)
        for model, _ in models if args.model in (None, model)
    ]
    if not selected:
        parser.error("selection contains no Figure 2 cells")
    print(f"{len(selected)} cells, {len(selected) * len(world_ids)} jobs", flush=True)
    if args.dry_run:
        for regime, model in selected:
            print(regime, model, *[value for value in runtime(model, regime) if value])
        return
    canonical = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    validate_runtimes(parser, args, selected, config, canonical)
    for seed in world_ids:
        world = load_world(args.worlds_dir / f"world_{seed:03d}.npz")
        for regime, model in selected:
            profile, _ = runtime(model, regime)
            python = getattr(args, f"python_{profile}").absolute()
            output = args.output / regime / model / f"world_{seed:03d}.npz"
            predict(world, regime, model, seed, python, args.device, output)
            print(output, flush=True)


if __name__ == "__main__":
    main()
