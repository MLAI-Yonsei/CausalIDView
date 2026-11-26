#!/usr/bin/env python3
"""Install and verify the sources and checkpoints required for Figure 2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
from urllib.request import urlopen

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "Baselines" / "src")]

from baselines.checkpoints import (
    CheckpointUnavailable, validate_checkpoint, validate_dopfn_source,
)


def prepare_dopfn_source(spec, local=None):
    destination = (ROOT / spec["path"]).parents[1]
    manifest = ROOT / spec["source_manifest"]
    revision = spec["source_revision"]
    if destination.exists():
        try:
            validate_dopfn_source(destination, manifest, revision)
        except CheckpointUnavailable as error:
            raise SystemExit(
                f"{error} Move the existing directory {destination} aside before reinstalling."
            ) from error
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        source = local
        if source is None:
            source = Path(temporary) / "upstream"
            environment = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
            subprocess.run(["git", "init", str(source)], check=True, env=environment)
            subprocess.run(
                ["git", "-C", str(source), "fetch", "--depth=1", spec["source_repo"], revision],
                check=True, env=environment,
            )
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"],
                           check=True, env=environment)
        validate_dopfn_source(source, manifest, revision)
        staged = Path(temporary) / "source"
        for filename in json.loads(manifest.read_text()):
            target = staged / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / filename, target)
        try:
            staged.rename(destination)
        except OSError:
            if not destination.exists():
                raise
            validate_dopfn_source(destination, manifest, revision)


def prepare_runtime_files(spec, cache=None):
    for auxiliary in spec.get("runtime_files", []):
        path = ROOT / auxiliary["path"]
        source = spec["source"].rsplit("/", 1)[0] + "/" + path.name
        install({**auxiliary, "source": source}, cache=cache)


def download(spec, destination):
    parsed = urlsplit(spec["source"])
    owner, repository, _, revision, filename = parsed.path.strip("/").split("/", 4)
    if parsed.netloc == "huggingface.co":
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=f"{owner}/{repository}", revision=revision, filename=filename,
        )
        shutil.copyfile(local, destination)
    else:
        target = f"https://github.com/{owner}/{repository}/raw/{revision}/{filename}"
        with urlopen(target, timeout=120) as remote, destination.open("wb") as local:
            shutil.copyfileobj(remote, local)



def install(spec, local=None, *, cache=None):
    try:
        return validate_checkpoint(spec, ROOT)
    except CheckpointUnavailable:
        pass
    destination = ROOT / spec["path"]
    if local is None and cache is not None:
        try:
            local = validate_checkpoint({**spec, "path": destination.name}, cache)
        except CheckpointUnavailable:
            pass
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        stage = Path(temporary)
        pending = stage / destination.name
        if local:
            shutil.copyfile(local, pending)
        else:
            download(spec, pending)
        validate_checkpoint({**spec, "path": pending.name}, stage)
        pending.replace(destination)
    return destination


def main():
    config = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())
    specs = config["checkpoints"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=tuple(specs))
    parser.add_argument("--local", action="append", default=[], metavar="MODEL=FILE")
    parser.add_argument("--dopfn-source", type=Path,
                        help="install Do-PFN from a local copy of the pinned source")
    args = parser.parse_args()
    local = {}
    for value in args.local:
        model, separator, path = value.partition("=")
        if model not in specs or not separator or not Path(path).is_file():
            parser.error(f"invalid local checkpoint: {value}")
        local[model] = Path(path)
    names = args.only or list(local) or list(specs)
    if set(local) - set(names):
        parser.error("local checkpoints must also be selected by --only")
    if args.dopfn_source is not None and ("dopfn" not in names or not args.dopfn_source.is_dir()):
        parser.error("--dopfn-source requires Do-PFN to be selected and an existing source directory")
    for model in names:
        spec = specs[model]
        cache = None
        if model == "dopfn":
            prepare_dopfn_source(spec, args.dopfn_source)
            cache = ROOT / "Baselines" / "CFMs" / "Do-PFN" / "artifacts"
        prepare_runtime_files(spec, cache)
        core = {key: value for key, value in spec.items()
                if key not in {"runtime_files", "source_manifest"}}
        install(core, local.get(model), cache=cache)
        print(f"verified {model}: {validate_checkpoint(spec, ROOT)}")


if __name__ == "__main__":
    main()
