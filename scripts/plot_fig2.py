#!/usr/bin/env python3
"""Quick, CPU-only reproduction of Figure 2 from results/fig2.csv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fig2-reproduction-mpl"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("BD", "FD", "IV", "PROX")
TITLES = {"BD": "Back-Door", "FD": "Front-Door", "IV": "Instrumental Variable", "PROX": "Proximal"}
LABELS = {
    "causalpfn": "CausalPFN", "dopfn": "Do-PFN", "causalfm": "CausalFM",
    "tabpfn_x": "TabPFN-v3.5\n+ X-Learner", "tabpfn_s": "TabPFN-v3.5\n+ S-Learner",
    "tabpfn_dr": "TabPFN-v3.5\n+ DR-Learner", "causalpfn_fd_xonly": "CausalPFN",
    "dopfn_fd_xonly": "Do-PFN", "tabpfn_fd": "TabPFN-v3.5\n+ FD Plug-in",
    "fd_nn": "NN\n+ FD Plug-in", "fd_xgboost": "XGBoost\n+ FD Plug-in",
    "wald_tabpfn": "TabPFN-v3.5\n+ Wald", "forestdriv": "ForestDRIV", "kiv": "KIV",
    "p_learner": "ExtraTrees\n+ P-Learner",
    "p_learner_nn_full": "NN\n+ P-Learner",
    "p_learner_tabpfn_v3_full": "TabPFN-v3.5\n+ P-Learner",
}
INK, ATE, CENTERED, EDGE = "#23303A", "#1261F0", "#8ACBFF", "#4F5A62"
GRID, BEST, SPINE = "#DCE0E3", "#F3EEE3", "#92999E"


def configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.family": "DejaVu Sans", "font.weight": 400, "font.size": 7.4,
        "axes.titlesize": 8.6, "axes.titleweight": 400, "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9, "ytick.labelsize": 7.0, "axes.linewidth": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "path", "axes.unicode_minus": False,
    })


def render(source: Path, output: Path, png: Path | None = None) -> None:
    data = pd.read_csv(source)
    required = {"regime", "model", "panel_order", "mean", "ci_low", "ci_high", "ate_error_sq", "centered_sPEHE_sq", "is_best"}
    if not required.issubset(data.columns) or len(data) != 24:
        raise ValueError("results/fig2.csv does not satisfy the 24-cell plotting contract")
    configure_style()
    figure, axes = plt.subplots(1, 4, figsize=(16.5, 7.2))
    for axis, regime, letter in zip(axes, REGIMES, "ABCD"):
        part = data[data.regime == regime].sort_values("panel_order")
        x_max = float(np.ceil(part.ci_high.max() * 1.08 / 0.05) * 0.05)
        for position, row in enumerate(part.itertuples(index=False)):
            if bool(row.is_best):
                axis.axvspan(position - 0.43, position + 0.43, color=BEST, zorder=0.2)
            axis.bar(position, row.ate_error_sq, width=0.82, color=ATE, edgecolor=EDGE, linewidth=0.72, zorder=2)
            axis.bar(position, row.centered_sPEHE_sq, bottom=row.ate_error_sq, width=0.82, color=CENTERED, edgecolor=EDGE, linewidth=0.72, zorder=2)
            axis.errorbar(position, row.mean, yerr=[[row.mean-row.ci_low], [row.ci_high-row.mean]], fmt="none", ecolor=INK, elinewidth=0.85, capsize=1.8, zorder=3)
        axis.set_xticks(np.arange(len(part)), [LABELS[m] for m in part.model], rotation=45, ha="right", rotation_mode="anchor")
        for tick, best in zip(axis.get_xticklabels(), part.is_best):
            tick.set_fontweight("bold" if best else "normal")
        axis.set_xlim(-0.41, len(part) - 0.59)
        axis.set_ylim(0.0, x_max * 1.18)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=4))
        axis.tick_params(axis="x", labelsize=13.5, pad=9, color=SPINE, width=0.6, length=2.5)
        axis.tick_params(axis="y", labelsize=13.5, pad=3.5, length=0)
        axis.set_title(f"({letter.lower()}) {TITLES[regime]}", pad=6, fontsize=15.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines["bottom"].set_color(SPINE)
        axis.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
        axis.grid(axis="x", visible=False)
    figure.subplots_adjust(left=0.060, right=0.995, top=0.885, bottom=0.46, wspace=0.20)
    figure.legend(handles=[Patch(facecolor=ATE, label="ATE error²"), Patch(facecolor=CENTERED, label="c-sPEHE²")], loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=2, frameon=False, fontsize=14.0, handlelength=1.5, columnspacing=1.7)
    box = axes[0].get_position()
    figure.text(0.018, box.y0 + box.height / 2, "Mean sPEHE²  (lower is better)", rotation="vertical", ha="center", va="center", fontsize=16.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=600, bbox_inches="tight", pad_inches=0.025)
    if png:
        png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "fig2.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "fig2.pdf")
    parser.add_argument("--png", type=Path)
    args = parser.parse_args()
    render(args.input, args.output, args.png)
    print(args.output)


if __name__ == "__main__":
    main()
