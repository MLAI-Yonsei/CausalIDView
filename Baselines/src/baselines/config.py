from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml


BASELINES_ROOT = Path(__file__).resolve().parents[2]
PROFILE_ROOT = BASELINES_ROOT.parent / "configs" / "baselines"


def merge_config(*configs: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-merge mappings from lowest to highest precedence."""

    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in config.items():
            current = merged.get(key)
            if isinstance(current, Mapping) and isinstance(value, Mapping):
                merged[key] = merge_config(current, value)
            else:
                merged[key] = deepcopy(value)
    return merged


def load_profile(name: str) -> dict[str, Any]:
    """Load one of the version-controlled benchmark profiles."""

    path = PROFILE_ROOT / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"Unknown configuration profile: {name}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Profile {name!r} must contain a mapping")
    return loaded

