#!/usr/bin/env python3
"""Generate Figure 6 bounds from the bundled observed worlds."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
POINT_BACKENDS = ("logistic", "xgboost", "tabpfn_v35")
SENSITIVITY_MODELS = (
    "b_learner", "b_learner_tabpfn_final", "b_learner_tabpfn_all", "csa_pfn_msm", "neuralcsa",
)
WORLD_COUNT = 40
FIELDS = ("X", "T", "Y", "I", "split", "unit_id")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=POINT_BACKENDS + SENSITIVITY_MODELS, required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--world", type=int, choices=range(WORLD_COUNT), action="append",
                        help="Process all 40 worlds by default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    worlds = range(WORLD_COUNT) if args.world is None else args.world
    if len(set(worlds)) != len(worlds):
        parser.error("World IDs must be distinct")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "Baselines" / "src"))
    if args.source_root is not None:
        # Keep the already-queued GPU03 commands using their local wheel snapshot.
        packages = ROOT / "Baselines" / "partial_identification" / "python_packages"
        if packages.is_dir():
            sys.path.insert(1, str(packages))

    from baselines.partial_identification.assets import SENSITIVITY_GAMMA
    from baselines.partial_identification.bound_estimators import estimate_iv, estimate_manski
    from baselines.partial_identification.sensitivity_adapters import estimate_sensitivity
    from baselines.partial_identification.world import PartialWorld

    if args.model in POINT_BACKENDS:
        routes = (("manski", estimate_manski, (None,)), ("iv", estimate_iv, (None,)))
    else:
        routes = (("sensitivity", estimate_sensitivity, (SENSITIVITY_GAMMA,)),)

    output = args.output or ROOT / "artifacts" / "fig6_bounds" / f"{args.model}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        pending = Path(temporary) / output.name
        with pending.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("world_id", "unit_id", "leaf", "gamma", "method", "lower", "upper"))
            for world_id in worlds:
                path = ROOT / "data" / "fig6" / "bound_worlds" / f"world_{world_id:03d}.npz"
                with np.load(path, allow_pickle=False) as archive:
                    arrays = {field: archive[field] for field in FIELDS}
                world = PartialWorld(arrays, world_id)
                ids = arrays["unit_id"][arrays["split"] == 1]
                if ids.shape != (100,) or np.unique(ids).size != 100:
                    raise ValueError(f"Invalid query IDs in {path}")
                for leaf, estimate, gammas in routes:
                    result = estimate(world, args.model, 20260923 + world_id, args.device)
                    np.testing.assert_array_equal(result.unit_id, ids)
                    lower, upper = np.atleast_2d(result.lower), np.atleast_2d(result.upper)
                    if (lower.shape != (len(gammas), len(ids)) or upper.shape != lower.shape
                            or not np.isfinite(lower).all() or not np.isfinite(upper).all()):
                        raise ValueError(f"Invalid {leaf} endpoints in {path}")
                    for index, gamma in enumerate(gammas):
                        writer.writerows(
                            (world_id, int(unit), leaf, gamma, args.model, float(lo), float(hi))
                            for unit, lo, hi in zip(ids, lower[index], upper[index])
                        )
        pending.replace(output)
    print(output)


if __name__ == "__main__":
    main()
