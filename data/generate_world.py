"""World serialization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .scm import World, sample_world


def save_world(world: World, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **world.arrays)
    return destination


def load_world(path: Path) -> World:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    return World(arrays, {"world_seed": int(arrays["world_id"][0])})


def generate_world(seed: int, config: Mapping[str, Any], output_dir: Path) -> Path:
    return save_world(sample_world(seed, config), output_dir / f"world_{seed:03d}.npz")
