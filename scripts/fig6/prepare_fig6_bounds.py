#!/usr/bin/env python3
"""Prepare the pinned NeuralCSA source and CSA-PFN checkpoint for Figure 6."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "Baselines" / "src")]

from scripts.fig2.prepare_fig2 import install
from baselines.partial_identification.assets import (
    ASSET_ROOT, CSA_PFN_SPEC, NEURALCSA_COMMIT, NEURALCSA_ROOT,
)


def git(repository, *arguments, **options):
    return subprocess.run(
        ["git", "-C", str(repository), *arguments], check=True,
        env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}, **options,
    )


def verify_neuralcsa(repository):
    revision = git(repository, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip()
    patch = ASSET_ROOT / "NeuralCSA.patch"
    changes = git(repository, "diff", "--no-ext-diff", "--no-color", "HEAD", capture_output=True).stdout
    if revision != NEURALCSA_COMMIT or changes != patch.read_bytes():
        raise RuntimeError(f"NeuralCSA checkout at {repository} does not match the pinned revision and patch")


def make_neuralcsa_packages(repository):
    # The upstream imports these as top-level packages. A marker keeps the
    # repository's own data package from shadowing NeuralCSA's data directory.
    for package in ("data", "models", "utils"):
        (repository / package / "__init__.py").touch(exist_ok=True)


def prepare_neuralcsa(source):
    if NEURALCSA_ROOT.exists():
        verify_neuralcsa(NEURALCSA_ROOT)
        make_neuralcsa_packages(NEURALCSA_ROOT)
        return

    with tempfile.TemporaryDirectory(dir=ASSET_ROOT) as directory:
        staging = Path(directory) / "NeuralCSA"
        staging.mkdir()
        git(staging, "init", "--quiet")
        git(staging, "fetch", "--quiet", "--depth=1", source, NEURALCSA_COMMIT)
        git(staging, "sparse-checkout", "set", "--no-cone", "/data/data_structures.py", "/models/", "/utils/")
        git(staging, "checkout", "--quiet", "--detach", "FETCH_HEAD")
        git(staging, "apply", str(ASSET_ROOT / "NeuralCSA.patch"))
        verify_neuralcsa(staging)
        make_neuralcsa_packages(staging)
        try:
            staging.rename(NEURALCSA_ROOT)
        except OSError:
            if not NEURALCSA_ROOT.is_dir():
                raise
            verify_neuralcsa(NEURALCSA_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=("neuralcsa", "csa_pfn_msm"))
    parser.add_argument("--neuralcsa-source", type=Path, help="Local Git checkout of the pinned NeuralCSA source")
    parser.add_argument("--csa-pfn-checkpoint", type=Path, help="Local copy of the pinned CSA-PFN weights")
    args = parser.parse_args()
    models = args.only or ("neuralcsa", "csa_pfn_msm")
    if "neuralcsa" in models:
        source = str(args.neuralcsa_source.resolve()) if args.neuralcsa_source else "https://github.com/DennisFrauen/NeuralCSA.git"
        prepare_neuralcsa(source)
        print(NEURALCSA_ROOT)
    if "csa_pfn_msm" in models:
        print(install(CSA_PFN_SPEC, local=args.csa_pfn_checkpoint))


if __name__ == "__main__":
    main()
