"""Paired SCM interventions and scoring for the Figure 3 response experiment."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .ground_truth import query_ground_truth
from .integrity import array_hash
from .scm import World


CONDITIONS = ("original", "stay", "move")
WORLD_KEYS = ["regime", "model", "world_id"]
WORLD_METRICS = ["e_stay", "e_move"]
SUMMARY_METRICS = [f"{condition}_{stat}" for condition in ("stay", "move")
                   for stat in ("mean", "se", "ci_low", "ci_high")]
SUMMARY_FILE = "fig3.csv"
WORLD_FILE = "fig3_world_metrics.csv"


@dataclass(frozen=True)
class WorldTargets:
    query: tuple[np.ndarray, np.ndarray]
    context_ids_hash: str
    query_x_hash: str


def load_settings(root: Path) -> tuple[dict, list[int]]:
    config = yaml.safe_load((root / "configs" / "fig2.yaml").read_text())
    return config["panels"], config["scm"]["world_seeds"]


def derive_world(original: World, condition: str) -> World:
    if condition == "original":
        return original
    a = original.arrays
    if condition == "stay":
        shift = 2.0 * a["gamma_u_x"] * a["U"]
        changed = {key: a[key] - shift for key in ("Y", "Y0", "Y1")}
        changed["gamma_u_x"] = -a["gamma_u_x"]
    elif condition == "move":
        changed = {y: a[y] + a["beta"] * a[m]
                   for y, m in (("Y", "M"), ("Y0", "M0"), ("Y1", "M1"))}
        changed.update(beta=2.0 * a["beta"], tau=2.0 * a["tau"])
    else:
        raise ValueError(f"unknown Figure 3 condition: {condition}")
    return World({**a, **changed}, original.metadata)


def load_targets(path: Path, worlds: list[int]) -> dict[int, WorldTargets]:
    records = json.loads(path.read_text())
    if len(records) != len(worlds) or {row["world_id"] for row in records} != set(worlds):
        raise ValueError("SCM query targets must cover each configured world once")
    result = {}
    for row in records:
        unit_id = np.asarray(row["query_unit_id"])
        tau = np.asarray(row["tau_true"], dtype=float)
        if (unit_id.shape != (100,) or unit_id.dtype.kind not in "iu" or len(np.unique(unit_id)) != 100
                or tau.shape != unit_id.shape or not np.isfinite(tau).all()):
            raise ValueError(f"invalid query targets for world {row['world_id']}")
        result[row["world_id"]] = WorldTargets((unit_id, tau), row["train_row_ids_hash"], row["X_query_hash"])
    return result


def check_world(world: World, expected: WorldTargets, seed: int, condition: str = "original") -> None:
    a = world.arrays
    query = a["split"] == 1
    context = a["split"] == 0
    if (context.sum() != 1_024 or len(context) != 1_124
            or array_hash(a["unit_id"][context]) != expected.context_ids_hash
            or array_hash(a["X"][query]) != expected.query_x_hash):
        raise ValueError(f"world_{seed:03d} context IDs or query covariates differ from the Figure 3 manifest")
    unit_id, target = query_ground_truth(world)
    expected_ids, expected_tau = expected.query
    multiplier = 2.0 if condition == "move" else 1.0
    if not np.array_equal(unit_id, expected_ids) or not np.allclose(target, multiplier * expected_tau, rtol=0, atol=1e-12):
        raise ValueError(f"world_{seed:03d} query IDs or CATE differ from the Figure 3 targets")


def prediction_path(root: Path, regime: str, model: str, condition: str, world: int) -> Path:
    return root / regime / model / condition / f"world_{world:03d}.npz"


def prediction_identity(regime: str, model: str, condition: str, world: int) -> dict[str, str | int]:
    return dict(regime=regime, model=model, condition=condition, world_id=world)


def score_world(root: Path, regime: str, model: str, world: int,
                truth: tuple[np.ndarray, np.ndarray],
                world_fingerprints: dict[int, str] | None = None) -> dict[str, float]:
    unit_id, target = truth
    if world_fingerprints is None:
        world_fingerprints = {}
    estimates = []
    for condition in CONDITIONS:
        path = prediction_path(root, regime, model, condition, world)
        if not path.is_file():
            raise FileNotFoundError(f"missing estimator output: {path}. Run scripts/fig3/predict_fig3.py first.")
        with np.load(path, allow_pickle=False) as prediction:
            identity = prediction_identity(regime, model, condition, world)
            if any(key not in prediction or not np.array_equal(prediction[key], value)
                   for key, value in identity.items()):
                raise ValueError(f"Figure 3 job identity differs from its path: {path}")
            digest = str(prediction["source_world_sha256"].item())
            if len(digest) != 64 or world_fingerprints.setdefault(world, digest) != digest:
                raise ValueError(f"Figure 3 source world fingerprint differs across predictions: {path}")
            if not np.array_equal(prediction["unit_id"], unit_id):
                raise ValueError(f"query unit alignment failed: {path}")
            estimate = np.asarray(prediction["tau_hat"], dtype=float)
        if estimate.shape != target.shape or not np.isfinite(estimate).all():
            raise ValueError(f"invalid prediction values or shape: {path}")
        estimates.append(estimate)
    original, stay, move = estimates
    scale = np.sqrt(np.mean(target**2))
    if scale <= 1e-12:
        raise ValueError(f"zero RMS target scale for world {world}")
    return {"e_stay": np.sqrt(np.mean((stay - original)**2)) / scale,
            "e_move": np.sqrt(np.mean((move - original - target)**2)) / scale}


def aggregate(scores: pd.DataFrame, panels: dict, worlds: list[int]) -> pd.DataFrame:
    expected = {(regime, model, world) for regime, models in panels.items()
                for model, _ in models for world in worlds}
    actual = list(scores[WORLD_KEYS].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Figure 3 world metrics must cover every model, view, and world exactly once")
    values = scores[WORLD_METRICS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Figure 3 world metrics must be finite and nonnegative")

    groups = scores.sort_values("world_id").groupby(["regime", "model"])
    rows = []
    for regime, models in panels.items():
        for model, label in models:
            group = groups.get_group((regime, model))
            row = dict(regime=regime, model=model, model_label=label, n_worlds=len(group))
            for condition, metric in zip(("stay", "move"), WORLD_METRICS):
                values = group[metric].to_numpy(dtype=float)
                key = f"fig3/{regime}/{model}/{metric}"
                seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
                rng = np.random.default_rng(seed)
                boot = values[rng.integers(0, len(values), (10_000, len(values)))].mean(axis=1)
                low, high = np.quantile(boot, [0.025, 0.975])
                row.update({f"{condition}_mean": values.mean(),
                            f"{condition}_se": values.std(ddof=1) / np.sqrt(len(values)),
                            f"{condition}_ci_low": low, f"{condition}_ci_high": high})
            rows.append(row)
    return pd.DataFrame(rows)
