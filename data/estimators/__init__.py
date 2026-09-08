"""Strict adapter contract for external and conventional estimators."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Any


class CheckpointUnavailable(RuntimeError):
    pass


def validate_checkpoint(spec: Mapping[str, Any], repo_root: Path) -> Path:
    path = repo_root / str(spec["path"])
    source = str(spec["source"])
    if not path.is_file():
        gated = " License acceptance/authentication is required." if spec.get("gated") else ""
        raise CheckpointUnavailable(f"missing {path}; obtain it from {source}.{gated}")
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    if digest != spec["sha256"]:
        raise CheckpointUnavailable(f"SHA-256 mismatch for {path}: {digest}")
    return path
