#!/usr/bin/env python3
"""Plot the Figure 3 response plane with world-level standard errors."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fig3-reproduction-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.fig3 import SUMMARY_FILE, load_settings

TITLES = {"BD": "Back-Door", "FD": "Front-Door", "IV": "Instrumental Variable", "PROX": "Proximal"}


def render(source: Path, output: Path) -> None:
    panels, worlds = load_settings(ROOT)
    scores = pd.read_csv(source).set_index(["regime", "model"], verify_integrity=True)
    cells = {(regime, model) for regime, models in panels.items() for model, _ in models}
    if set(scores.index) != cells or not scores.n_worlds.eq(len(worlds)).all():
        raise ValueError("Figure 3 plotting requires all cells and worlds")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for axis, (regime, models) in zip(axes.flat, panels.items()):
        for model, _ in models:
            row = scores.loc[(regime, model)]
            axis.errorbar(row.stay_mean, row.move_mean, xerr=row.stay_se, yerr=row.move_se,
                          fmt="o", capsize=2, label=row.model_label)
        axis.set(title=TITLES[regime], xlim=(0, None), ylim=(0, None),
                 xlabel="Spurious response $E_{stay}$", ylabel="Change-tracking error $E_{move}$")
        axis.grid(alpha=0.2)
        axis.legend(fontsize="small")
    fig.suptitle(f"Structural response across {len(worlds)} paired worlds (error bars show SE)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fresh = ROOT / "artifacts" / "fig3_fresh"
    parser.add_argument("--input", type=Path, default=fresh / SUMMARY_FILE)
    parser.add_argument("--output", type=Path, default=fresh / "fig3.pdf")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"missing aggregate scores: {args.input}. Run scripts/fig3/run_fig3.py first.")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must use different paths")
    render(args.input, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
