#!/usr/bin/env python3
"""Reproduce the two CausalIDView paper figures from bundled evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "causalidview-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "figures"
BOOTSTRAP_REPLICATES = 10_000

REGIMES = ("BD", "FD", "IV", "PROX")
PANEL_MODELS = {
    "BD": ("causalpfn", "dopfn", "causalfm", "tabpfn_x", "xgb_x", "dr_learner"),
    "FD": ("causalpfn_fd_xonly", "dopfn_fd_xonly", "causalfm", "tabpfn_fd", "fd_nn", "fd_xgboost"),
    "IV": ("causalpfn", "dopfn", "causalfm", "wald_tabpfn", "forestdriv", "kiv"),
    "PROX": (
        "causalpfn", "dopfn", "causalfm", "p_learner",
        "p_learner_tabpfn_v3_final", "p_learner_xgb_final",
    ),
}
REGIME_TITLES = {
    "BD": "Back-Door",
    "FD": "Front-Door",
    "IV": "Instrumental Variable",
    "PROX": "Proximal",
}
MODEL_LABELS = {
    "causalpfn": "CausalPFN",
    "dopfn": "Do-PFN",
    "causalfm": "CausalFM",
    "tabpfn_x": "TabPFN-v3\n+ X-Learner",
    "xgb_x": "XGBoost\n+ X-Learner",
    "dr_learner": "Cross-fitted DR-\nLearner",
    "causalpfn_fd_xonly": "CausalPFN\n[X-only]",
    "dopfn_fd_xonly": "Do-PFN\n[X-only]",
    "tabpfn_fd": "TabPFN\n+ FD Plug-in",
    "fd_nn": "NN\n+ FD Plug-in",
    "fd_xgboost": "XGBoost\n+ FD Plug-in",
    "wald_tabpfn": "TabPFN-v3\n+ Wald",
    "forestdriv": "ForestDRIV",
    "kiv": "KIV",
    "p_learner": "ExtraTrees\n+ P-Learner",
    "p_learner_tabpfn_v3_final": "TabPFN-v3\n+ P-Learner",
    "p_learner_xgb_final": "XGBoost\n+ P-Learner",
}

INK = "#23303A"
ATE_COMPONENT = "#1261F0"
CENTERED_COMPONENT = "#8ACBFF"
BAR_EDGE = "#4F5A62"
GRID = "#DCE0E3"
BEST_FACE = "#F3EEE3"
SPINE = "#92999E"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.weight": 400,
        "font.size": 7.4,
        "axes.titlesize": 8.6,
        "axes.titleweight": 400,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 6.9,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "path",
        "axes.unicode_minus": False,
    })


def configure_structural_response_style() -> None:
    """Apply the standalone style used by the released response figure."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_figure2_world_metrics() -> pd.DataFrame:
    frame = pd.read_csv(DATA / "figure2_world_metrics.csv")
    expected_columns = {
        "regime", "model", "model_label", "world_id",
        "sPEHE_sq", "ate_error_sq", "centered_sPEHE_sq",
    }
    if set(frame.columns) != expected_columns or len(frame) != 960:
        raise ValueError("invalid Figure 2 source table")
    expected_cells = {(regime, model) for regime in REGIMES for model in PANEL_MODELS[regime]}
    actual_cells = set(zip(frame["regime"], frame["model"]))
    if actual_cells != expected_cells:
        raise ValueError("Figure 2 model selection differs from the 24-cell contract")
    counts = frame.groupby(["regime", "model"])["world_id"].agg(["count", "nunique"])
    if not counts.eq(40).all().all():
        raise ValueError("each Figure 2 cell must contain 40 distinct worlds")
    identity_error = np.abs(
        frame["sPEHE_sq"] - frame["ate_error_sq"] - frame["centered_sPEHE_sq"]
    )
    if float(identity_error.max()) > 1e-10:
        raise ValueError("sPEHE decomposition check failed")
    return frame


def aggregate_figure2(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(20260902)
    indices = rng.integers(0, 40, size=(BOOTSTRAP_REPLICATES, 40))
    rows: list[dict[str, object]] = []
    for regime in REGIMES:
        for panel_order, model in enumerate(PANEL_MODELS[regime]):
            part = frame[(frame["regime"] == regime) & (frame["model"] == model)].sort_values("world_id")
            values = part["sPEHE_sq"].to_numpy(np.float64)
            bootstrap_means = values[indices].mean(axis=1)
            low, high = np.quantile(bootstrap_means, [0.025, 0.975])
            rows.append({
                "regime": regime,
                "model": model,
                "model_label": str(part.iloc[0]["model_label"]),
                "panel_order": panel_order,
                "mean": float(values.mean()),
                "ci_low": float(low),
                "ci_high": float(high),
                "ate_error_sq": float(part["ate_error_sq"].mean()),
                "centered_sPEHE_sq": float(part["centered_sPEHE_sq"].mean()),
                "n_worlds": 40,
            })
    result = pd.DataFrame(rows)
    result["is_best"] = result["mean"].eq(result.groupby("regime")["mean"].transform("min"))
    return result


def finish_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(True)
    axis.spines["bottom"].set_color(SPINE)
    axis.tick_params(axis="x", color=SPINE, width=0.6, length=2.5)
    axis.tick_params(axis="y", length=0, pad=3.5)
    axis.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    axis.grid(axis="x", visible=False)


def render_figure2(output: Path) -> pd.DataFrame:
    world = load_figure2_world_metrics()
    plot = aggregate_figure2(world)
    configure_style()
    figure, axes = plt.subplots(1, 4, figsize=(16.5, 7.2))
    for axis, regime, letter in zip(axes, REGIMES, "ABCD"):
        part = plot[plot["regime"] == regime].set_index("model")
        x_max = float(np.ceil(part["ci_high"].max() * 1.08 / 0.05) * 0.05)
        positions = np.arange(len(PANEL_MODELS[regime]))
        labels: list[str] = []
        best_flags: list[bool] = []
        for position, model in zip(positions, PANEL_MODELS[regime]):
            row = part.loc[model]
            labels.append(MODEL_LABELS[model])
            is_best = bool(row["is_best"])
            best_flags.append(is_best)
            if is_best:
                axis.axvspan(position - 0.43, position + 0.43, color=BEST_FACE, zorder=0.2)
            axis.bar(
                position, row["ate_error_sq"], width=0.82,
                color=ATE_COMPONENT, edgecolor=BAR_EDGE, linewidth=0.72, zorder=2,
            )
            axis.bar(
                position, row["centered_sPEHE_sq"], bottom=row["ate_error_sq"], width=0.82,
                color=CENTERED_COMPONENT, edgecolor=BAR_EDGE, linewidth=0.72, zorder=2,
            )
            axis.errorbar(
                position, row["mean"],
                yerr=[[row["mean"] - row["ci_low"]], [row["ci_high"] - row["mean"]]],
                fmt="none", ecolor=INK, elinewidth=0.85, capsize=1.8, zorder=3,
            )
        axis.set_xticks(positions, labels, rotation=45, ha="right", rotation_mode="anchor")
        for tick, is_best in zip(axis.get_xticklabels(), best_flags):
            tick.set_fontweight("bold" if is_best else "normal")
        axis.set_xlim(-0.41, len(positions) - 0.59)
        axis.set_ylim(0.0, x_max * 1.18)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], min_n_ticks=4))
        axis.tick_params(axis="x", labelsize=13.5, pad=9)
        axis.tick_params(axis="y", labelsize=13.5, pad=3.5)
        axis.set_title(f"({letter.lower()}) {REGIME_TITLES[regime]}", pad=6, fontsize=15.5)
        finish_axis(axis)
    figure.subplots_adjust(left=0.060, right=0.995, top=0.885, bottom=0.46, wspace=0.20)
    figure.legend(
        handles=[
            Patch(facecolor=ATE_COMPONENT, label="ATE error²"),
            Patch(facecolor=CENTERED_COMPONENT, label="c-sPEHE²"),
        ],
        loc="upper center", bbox_to_anchor=(0.50, 0.985), ncol=2,
        frameon=False, fontsize=14.0, handlelength=1.5, columnspacing=1.7,
    )
    panel_box = axes[0].get_position()
    figure.text(
        0.018, panel_box.y0 + panel_box.height / 2,
        "Mean sPEHE²  (lower is better)", rotation="vertical",
        ha="center", va="center", fontsize=16.0,
    )
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "figure2_point_estimation_benchmark"
    for extension in ("pdf", "svg", "png"):
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)
    plot.to_csv(output / "figure2_plot_data.csv", index=False, float_format="%.17g")
    return plot


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left, ddof=0) == 0 or np.std(right, ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def structural_response_metrics() -> tuple[pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    with np.load(DATA / "fd_structural_response.npz", allow_pickle=False) as archive:
        world_ids = np.asarray(archive["world_id"])
        unit_ids = np.asarray(archive["unit_id"])
        tau_true = np.asarray(archive["tau_true"], dtype=np.float64)
        original = np.asarray(archive["pred_original"], dtype=np.float64)
        doubled = np.asarray(archive["pred_causal_x2"], dtype=np.float64)
    if (
        not np.array_equal(world_ids, np.arange(40))
        or unit_ids.shape != (40, 100)
        or tau_true.shape != (40, 100)
        or original.shape != (2, 40, 100)
        or doubled.shape != (2, 40, 100)
        or not all(np.isfinite(value).all() for value in (tau_true, original, doubled))
    ):
        raise ValueError("invalid FD structural-response source arrays")
    rng = np.random.default_rng(20260906)
    indices = rng.integers(0, 40, size=(BOOTSTRAP_REPLICATES, 40), dtype=np.int16)
    model_ids = ("causalfm_fd", "tabpfn_fd")
    labels = ("CausalFM-FD", "TabPFN + FD plug-in")
    rows: list[dict[str, object]] = []
    scatter: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_index, (model, label) in enumerate(zip(model_ids, labels)):
        predicted_change = doubled[model_index] - original[model_index]
        tracking = np.mean(np.abs(predicted_change - tau_true), axis=1) / np.mean(np.abs(tau_true), axis=1) * 100.0
        response = np.asarray([
            correlation(predicted_change[world], tau_true[world]) for world in range(40)
        ])
        bootstrap_tracking = tracking[indices].mean(axis=1)
        bootstrap_response = response[indices].mean(axis=1)
        tracking_low, tracking_high = np.percentile(bootstrap_tracking, [2.5, 97.5])
        response_low, response_high = np.percentile(bootstrap_response, [2.5, 97.5])
        rows.extend({
            "model": model,
            "model_label": label,
            "world_id": world,
            "tracking_error_pct": float(tracking[world]),
            "response_r": float(response[world]),
        } for world in range(40))
        rows.append({
            "model": model,
            "model_label": label,
            "world_id": "aggregate",
            "tracking_error_pct": float(tracking.mean()),
            "tracking_ci_low": float(tracking_low),
            "tracking_ci_high": float(tracking_high),
            "response_r": float(response.mean()),
            "response_ci_low": float(response_low),
            "response_ci_high": float(response_high),
        })
        scatter[model] = tau_true.reshape(-1), predicted_change.reshape(-1)
    return pd.DataFrame(rows), scatter


def render_fd_structural_response(output: Path) -> pd.DataFrame:
    metrics, scatter = structural_response_metrics()
    aggregate = metrics[metrics["world_id"].eq("aggregate")].set_index("model")
    configure_structural_response_style()
    model_order = (("causalfm_fd", "CausalFM-FD"), ("tabpfn_fd", "TabPFN + FD plug-in"))
    colors = {"causalfm_fd": "#B9A0D6", "tabpfn_fd": "#5B2C83"}
    scatter_alpha = {"causalfm_fd": 0.24, "tabpfn_fd": 0.14}
    figure = plt.figure(figsize=(9.35, 3.25))
    grid = figure.add_gridspec(1, 4, width_ratios=(1.12, 0.08, 1.0, 1.0), wspace=0.26)

    axis = figure.add_subplot(grid[0, 0])
    positions = np.asarray([-0.34, 0.34])
    means = np.asarray([aggregate.loc[model, "tracking_error_pct"] for model, _ in model_order])
    lows = np.asarray([aggregate.loc[model, "tracking_ci_low"] for model, _ in model_order])
    highs = np.asarray([aggregate.loc[model, "tracking_ci_high"] for model, _ in model_order])
    bars = axis.bar(positions, means, width=0.58, color=[colors[model] for model, _ in model_order], zorder=2)
    axis.errorbar(
        positions, means, yerr=np.vstack([means - lows, highs - means]),
        fmt="none", ecolor="#252525", elinewidth=0.9, capsize=2.2, zorder=3,
    )
    for bar, value in zip(bars, means):
        axis.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 3.0,
            f"{value:.1f}%", ha="center", va="bottom", fontsize=7.5,
        )
    axis.set_xticks(positions, ["CausalFM-FD", "TabPFN +\nFD plug-in"])
    axis.set_ylabel("2× CATE tracking error (%)")
    axis.set_ylim(0, 116)
    axis.grid(axis="y", alpha=0.24, linewidth=0.6, zorder=0)

    low = float(min(values.min() for pair in scatter.values() for values in pair))
    high = float(max(values.max() for pair in scatter.values() for values in pair))
    padding = 0.04 * max(high - low, 1e-3)
    for column, (model, label) in zip((2, 3), model_order):
        axis = figure.add_subplot(grid[0, column])
        true_change, predicted_change = scatter[model]
        axis.scatter(
            true_change, predicted_change, s=5, alpha=scatter_alpha[model],
            color=colors[model], edgecolors="none", rasterized=True,
        )
        axis.plot(
            [low - padding, high + padding], [low - padding, high + padding],
            color="#333333", linewidth=0.9, linestyle="--",
        )
        axis.set_xlim(low - padding, high + padding)
        axis.set_ylim(low - padding, high + padding)
        axis.set_aspect("equal", adjustable="box")
        row = aggregate.loc[model]
        axis.text(
            0.04, 0.96,
            f"mean world r = {row['response_r']:.2f}\ntracking error = {row['tracking_error_pct']:.1f}%",
            transform=axis.transAxes, ha="left", va="top", fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )
        axis.set_title(label, fontsize=9, pad=5)
        axis.set_xlabel("True CATE change")
        axis.set_ylabel("Predicted change")
        axis.grid(alpha=0.18, linewidth=0.5)

    figure.subplots_adjust(left=0.075, right=0.99, top=0.865, bottom=0.235)
    panel_a = figure.axes[0].get_position()
    panel_b_left = figure.axes[1].get_position()
    panel_b_right = figure.axes[2].get_position()
    figure.text((panel_a.x0 + panel_a.x1) / 2, 0.035, "(a)", ha="center", va="bottom", fontsize=10)
    figure.text((panel_b_left.x0 + panel_b_right.x1) / 2, 0.035, "(b)", ha="center", va="bottom", fontsize=10)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "fig_fd_structural_response_combined"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(stem.with_suffix(".png"), dpi=320, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)
    metrics.to_csv(output / "fd_structural_response_metrics.csv", index=False, float_format="%.17g")
    return metrics


def verify_sources() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    for filename, expected in manifest["sha256"].items():
        observed = sha256(DATA / filename)
        if observed != expected:
            raise ValueError(f"checksum mismatch for {filename}: {observed}")
    figure2 = aggregate_figure2(load_figure2_world_metrics())
    for key, expected in manifest["numeric_checks"]["figure2_mean_spehe_sq"].items():
        regime, model = key.split("/")
        observed = float(figure2[(figure2.regime == regime) & (figure2.model == model)].iloc[0]["mean"])
        if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
            raise ValueError(f"Figure 2 numerical check failed for {key}")
    metrics, _ = structural_response_metrics()
    aggregate = metrics[metrics["world_id"].eq("aggregate")].set_index("model")
    for model, expected in manifest["numeric_checks"]["fd_tracking_error_pct"].items():
        observed = float(aggregate.loc[model, "tracking_error_pct"])
        if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
            raise ValueError(f"FD tracking-error check failed for {model}")
    print("PASS: source checksums, 24×40 Figure 2 cells, and FD response metrics verified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", choices=("all", "figure2", "fd-response"), default="all")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verify_sources()
    if args.verify_only:
        return 0
    if args.figure in {"all", "figure2"}:
        render_figure2(args.output)
    if args.figure in {"all", "fd-response"}:
        render_fd_structural_response(args.output)
    print(f"Wrote reproducible outputs to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
