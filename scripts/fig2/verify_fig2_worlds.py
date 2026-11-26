#!/usr/bin/env python3
"""Check generated Figure 2 world pairing against the saved experiment manifests."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.integrity import array_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", type=Path, default=ROOT / "artifacts" / "worlds")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "results" / "fig2_world_manifest.json").read_text())
    max_delta = 0.0
    for row in manifest:
        seed = row["world_id"]
        with np.load(args.worlds / f"world_{seed:03d}.npz", allow_pickle=False) as world:
            query = world["split"] == 1
            train_ids = world["unit_id"][~query]
            if array_hash(train_ids) != row["train_row_ids_hash"]:
                raise ValueError(f"world {seed} context IDs differ from the archive")
            if array_hash(world["X"][query]) != row["X_query_hash"]:
                raise ValueError(f"world {seed} query X differs from the archive")
            np.testing.assert_array_equal(world["unit_id"][query], row["query_unit_id"])
            tau = world["tau"][query]
            np.testing.assert_allclose(tau, row["tau_true"], atol=1e-12, rtol=0)
            max_delta = max(max_delta, float(np.max(np.abs(tau - row["tau_true"]))))
    print(f"{len(manifest)} world manifests matched; max absolute target delta {max_delta:.3g}")


if __name__ == "__main__":
    main()
