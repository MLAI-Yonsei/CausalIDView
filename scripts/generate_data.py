#!/usr/bin/env python3
"""Generate the 40 deterministic SCM worlds used by Figure 2."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.generate_world import generate_world


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "fig2.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "worlds")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for seed in config["scm"]["world_seeds"]:
        path = generate_world(int(seed), config, args.output)
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    main()
