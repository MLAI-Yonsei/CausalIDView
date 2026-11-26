"""Verify the published estimator checkpoints and their runtime files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping, Any


class CheckpointUnavailable(RuntimeError):
    pass


def validate_dopfn_source(root: Path, manifest: Path, revision: str) -> None:
    for name, expected in json.loads(manifest.read_text()).items():
        path = root / name
        if path.is_file():
            content = path.read_bytes()
            header = f"blob {len(content)}\0".encode("ascii")
            if hashlib.sha1(header + content).hexdigest() == expected:
                continue
        raise CheckpointUnavailable(
            f"{path} does not match Do-PFN {revision}. "
            "Run scripts/fig2/prepare_fig2.py --only dopfn to install the pinned source."
        )


def validate_checkpoint(spec: Mapping[str, Any], repo_root: Path) -> Path:
    path = repo_root / str(spec["path"])
    source = str(spec["source"])
    if "source_manifest" in spec:
        validate_dopfn_source(path.parents[1], repo_root / spec["source_manifest"], spec["source_revision"])
    if not path.is_file():
        gated = " License acceptance/authentication is required." if spec.get("gated") else ""
        raise CheckpointUnavailable(f"missing {path}; obtain it from {source}.{gated}")
    if "bytes" in spec and path.stat().st_size != spec["bytes"]:
        raise CheckpointUnavailable(f"unexpected file size for {path}; obtain it from {source}")
    for auxiliary in spec.get("runtime_files", []):
        basename = Path(auxiliary["path"]).name
        validate_checkpoint({**auxiliary, "source": source.rsplit("/", 1)[0] + "/" + basename}, repo_root)
    if "sha256" not in spec:
        return path
    checksum = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            checksum.update(block)
    digest = checksum.hexdigest()
    if digest != spec["sha256"]:
        raise CheckpointUnavailable(f"SHA-256 mismatch for {path}: {digest}")
    return path
