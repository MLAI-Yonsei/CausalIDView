"""Render the hidden-confounder track from bundled, frozen plot observations."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PANEL_A_ORDER = (
    "causalpfn", "dopfn", "causalfm", "tabpfn_x", "xgb_x", "dr_learner",
)
PANEL_A_CFM = {"causalpfn", "dopfn", "causalfm"}
PANEL_A_LABELS = {
    "causalpfn": "CausalPFN",
    "dopfn": "Do-PFN",
    "causalfm": "CausalFM",
    "tabpfn_x": "TabPFN-X",
    "xgb_x": "XGBoost\n+ X-Learner",
    "dr_learner": "Cross-fitted\nDR-Learner",
}
PANEL_POINT_ORDER = ("causalpfn", "dopfn", "causalfm", "tabpfn_x")
PANEL_POINT_LABELS = {
    "causalpfn": "CausalPFN",
    "dopfn": "Do-PFN",
    "causalfm": "CausalFM",
    "tabpfn_x": "TabPFN-X",
}
PANEL_BOUND_ORDER = ("csa_pfn_msm", "b_learner", "neuralcsa")
PANEL_BOUND_LABELS = {
    "csa_pfn_msm": "CSA-PFN + MSM",
    "b_learner": "B-Learner",
    "neuralcsa": "NeuralCSA",
}

COLORS = {
    "causalpfn": "#3B5B92",
    "dopfn": "#7083A6",
    "causalfm": "#2B8C8C",
    "tabpfn_x": "#A7B0B5",
    "xgb_x": "#B8BFC3",
    "dr_learner": "#7D898F",
    "csa_pfn_msm": "#7765A7",
    "b_learner": "#9CA6AC",
    "neuralcsa": "#5D7E88",
}
CFM_COLOR = "#3B5B92"
MUTED = "#58636A"
GRID = "#E4E8EA"
INK = "#20282D"

BASE_FONTSIZE = 16.0
TITLE_FONTSIZE = 18.5
AXIS_LABEL_FONTSIZE = 17.5
TICK_FONTSIZE = 16.0
CATEGORY_FONTSIZE = 15.5
LEGEND_FONTSIZE = 14.5

PLOT_COLUMNS = (
    "panel", "panel_title", "model_key", "model", "group", "condition",
    "condition_id", "metric", "value", "raw_value", "metric_scale",
    "replicate_id", "replication_unit", "source_artifact", "ci95_low",
    "ci95_high", "n_replicates", "coverage", "gamma",
)


def load_hidden_confounder_data(path: Path) -> pd.DataFrame:
    """Load and validate every independent observation used by the four panels."""
    frame = pd.read_csv(path, dtype={"replicate_id": str})
    if tuple(frame.columns) != PLOT_COLUMNS or len(frame) != 376:
        raise ValueError("invalid hidden-confounder source table")
    if not np.isfinite(frame["value"].to_numpy(float)).all():
        raise ValueError("hidden-confounder source contains non-finite plotted values")
    if frame.duplicated(["panel", "metric", "model_key", "replicate_id"]).any():
        raise ValueError("hidden-confounder source contains duplicate plot observations")

    expected_models = {
        "(a)": set(PANEL_A_ORDER),
        "(b)": set(PANEL_POINT_ORDER),
        "(c)": set(PANEL_POINT_ORDER),
        "(d)": set(PANEL_BOUND_ORDER),
    }
    expected_counts = {"(a)": 40, "(b)": 2, "(c)": 2, "(d)": 40}
    if set(frame["panel"]) != set(expected_models):
        raise ValueError("hidden-confounder source must contain panels (a)--(d)")
    for panel, models in expected_models.items():
        part = frame.loc[frame["panel"] == panel]
        if set(part["model_key"]) != models:
            raise ValueError(f"{panel} model selection differs from the figure contract")
        counts = part.groupby("model_key")["replicate_id"].nunique()
        if not counts.eq(expected_counts[panel]).all():
            raise ValueError(f"{panel} replication counts differ from the figure contract")

    for panel, conditions in {"(b)": {"bd", "hc"}, "(c)": {"full", "hidden"}}.items():
        part = frame.loc[frame["panel"] == panel]
        if set(part["condition_id"].dropna()) != conditions:
            raise ValueError(f"{panel} condition selection differs from the figure contract")
        if not part["n_replicates"].eq(40 if panel == "(b)" else 100).all():
            raise ValueError(f"{panel} aggregate replication count is invalid")
    return frame


def _ordered_values(frame: pd.DataFrame, model_key: str) -> np.ndarray:
    values = frame.loc[frame["model_key"] == model_key, "value"].dropna().to_numpy(float)
    if values.size == 0:
        raise ValueError(f"no observations available for model {model_key!r}")
    return values


def _format_axis(axis: plt.Axes) -> None:
    axis.set_axisbelow(True)
    axis.yaxis.grid(True, color=GRID, linewidth=0.6)
    axis.xaxis.grid(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7B858A")
    axis.spines["bottom"].set_color("#7B858A")
    axis.tick_params(axis="both", colors=INK, width=0.65, length=3)
    axis.tick_params(axis="x", pad=3)


def _draw_boxplot(
    axis: plt.Axes,
    frame: pd.DataFrame,
    order: Sequence[str],
    labels: Mapping[str, str],
) -> None:
    values = [_ordered_values(frame, key) for key in order]
    positions = np.arange(1, len(order) + 1)
    box = axis.boxplot(
        values,
        positions=positions,
        widths=0.48,
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        boxprops={"linewidth": 0.85, "edgecolor": INK},
        whiskerprops={"linewidth": 0.8, "color": "#465057"},
        capprops={"linewidth": 0.8, "color": "#465057"},
        medianprops={"linewidth": 1.45, "color": INK},
    )
    for patch, key in zip(box["boxes"], order):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.84)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [labels[key] for key in order], rotation=30, ha="right",
        rotation_mode="anchor", fontsize=TICK_FONTSIZE,
    )
    axis.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axis.set_ylabel("Endpoint RMSE", fontsize=AXIS_LABEL_FONTSIZE, color=INK, labelpad=2)
    axis.set_xlim(0.35, len(order) + 0.65)
    axis.margins(y=0.08)
    _format_axis(axis)
    axis.text(
        0.0, 1.105, "(d) Sensitivity Bounds", transform=axis.transAxes,
        fontsize=TITLE_FONTSIZE, fontweight=400, color=INK,
        ha="left", va="bottom",
    )


def _draw_grouped_bars(
    axis: plt.Axes,
    frame: pd.DataFrame,
    *,
    panel_label: str,
    title: str,
    conditions: tuple[str, str],
    condition_labels: tuple[str, str],
) -> None:
    order = PANEL_POINT_ORDER
    positions = np.arange(len(order), dtype=float)
    bar_width = 0.34
    offsets = (-bar_width / 2.0, bar_width / 2.0)
    alpha = {conditions[0]: 0.42, conditions[1]: 0.90}
    max_upper = 0.0
    for condition_index, (condition_id, offset) in enumerate(zip(conditions, offsets)):
        selected = frame.loc[frame["condition_id"] == condition_id].set_index("model_key").reindex(order)
        if selected["value"].isna().any():
            raise ValueError(f"{panel_label} is missing rows for {condition_id}")
        values = selected["value"].to_numpy(float)
        lower = values - selected["ci95_low"].to_numpy(float)
        upper = selected["ci95_high"].to_numpy(float) - values
        max_upper = max(max_upper, float(np.max(values + upper)))
        bars = axis.bar(
            positions + offset,
            values,
            width=bar_width,
            color=[COLORS[key] for key in order],
            alpha=alpha[condition_id],
            edgecolor=INK,
            linewidth=0.65,
            yerr=np.vstack([lower, upper]),
            error_kw={"ecolor": INK, "elinewidth": 0.75, "capsize": 2.5, "capthick": 0.75},
            label=condition_labels[condition_index],
            zorder=3,
        )
        if condition_index == 1:
            for bar in bars:
                bar.set_hatch("//")
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [PANEL_POINT_LABELS[key] for key in order], rotation=18.0, ha="right",
        rotation_mode="anchor", fontsize=CATEGORY_FONTSIZE,
    )
    axis.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    axis.set_ylabel(r"sPEHE $\downarrow$", fontsize=AXIS_LABEL_FONTSIZE, color=INK, labelpad=2)
    axis.set_xlim(-0.65, len(order) - 0.35)
    axis.set_ylim(0.0, max_upper * 1.25)
    axis.margins(y=0.0)
    axis.legend(
        handles=[
            Patch(facecolor="#AAB2B6", edgecolor=INK, linewidth=0.6,
                  alpha=alpha[conditions[0]], label=condition_labels[0]),
            Patch(facecolor="#AAB2B6", edgecolor=INK, linewidth=0.6,
                  alpha=alpha[conditions[1]], hatch="//", label=condition_labels[1]),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.995), frameon=False,
        fontsize=LEGEND_FONTSIZE, handlelength=1.1, handletextpad=0.4,
        columnspacing=0.7, ncol=2,
    )
    _format_axis(axis)
    axis.text(
        0.0, 1.105, f"{panel_label} {title}", transform=axis.transAxes,
        fontsize=TITLE_FONTSIZE, fontweight=400, color=INK,
        ha="left", va="bottom",
    )


def _draw_membership(axis: plt.Axes, frame: pd.DataFrame) -> None:
    values = [_ordered_values(frame, key) for key in PANEL_A_ORDER]
    positions = np.arange(len(PANEL_A_ORDER), 0, -1)
    box = axis.boxplot(
        values,
        positions=positions,
        vert=False,
        widths=0.60,
        patch_artist=True,
        whis=1.5,
        showfliers=False,
        boxprops={"linewidth": 0.8, "edgecolor": INK},
        whiskerprops={"linewidth": 0.75, "color": "#465057"},
        capprops={"linewidth": 0.75, "color": "#465057"},
        medianprops={"linewidth": 1.35, "color": INK},
    )
    for patch, key in zip(box["boxes"], PANEL_A_ORDER):
        patch.set_facecolor(COLORS[key])
        patch.set_alpha(0.84)
    axis.axhline(3.5, color="#B4BDC1", linewidth=0.8, linestyle=(0, (2.2, 2.2)))
    axis.set_ylim(0.4, len(PANEL_A_ORDER) + 0.6)
    axis.set_yticks(positions)
    axis.set_yticklabels([PANEL_A_LABELS[key] for key in PANEL_A_ORDER], fontsize=14.5)
    for tick, key in zip(axis.get_yticklabels(), PANEL_A_ORDER):
        tick.set_color(CFM_COLOR if key in PANEL_A_CFM else MUTED)
    axis.tick_params(axis="y", length=0, pad=4)
    low = min(float(np.min(item)) for item in values)
    high = max(float(np.max(item)) for item in values)
    span = max(high - low, 1e-8)
    axis.set_xlim(low - 0.08 * span, high + 0.08 * span)
    axis.set_xlabel(r"Inclusion rate $\uparrow$", labelpad=4)
    axis.tick_params(axis="x", labelsize=TICK_FONTSIZE)
    axis.xaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axis.yaxis.grid(False)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#7B858A")
    axis.spines["bottom"].set_color("#7B858A")
    axis.tick_params(axis="both", colors=INK, width=0.65, length=3)
    axis.text(
        0.0, 1.105, "(a) Oracle-set Inclusion Rate", transform=axis.transAxes,
        fontsize=TITLE_FONTSIZE, fontweight=400, color=INK,
        ha="left", va="bottom",
    )


def _configure_style() -> None:
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": BASE_FONTSIZE,
        "font.weight": 400,
        "axes.titlesize": TITLE_FONTSIZE,
        "axes.titleweight": 400,
        "axes.labelsize": AXIS_LABEL_FONTSIZE,
        "xtick.labelsize": TICK_FONTSIZE,
        "ytick.labelsize": TICK_FONTSIZE,
        "axes.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def render_hidden_confounder(data_path: Path, output: Path) -> pd.DataFrame:
    """Validate the frozen observations and write all public figure artifacts."""
    plot_data = load_hidden_confounder_data(data_path)
    _configure_style()
    output.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(19.0, 5.5))
    grid = figure.add_gridspec(
        1, 4, left=0.055, right=0.995, bottom=0.22, top=0.80,
        width_ratios=(1.15, 1.02, 1.02, 0.92), wspace=0.30,
    )
    axes = [figure.add_subplot(grid[index]) for index in range(4)]
    _draw_membership(axes[0], plot_data.loc[plot_data["panel"] == "(a)"])
    _draw_grouped_bars(
        axes[1], plot_data.loc[plot_data["panel"] == "(b)"],
        panel_label="(b)", title="Synthetic Data",
        conditions=("bd", "hc"), condition_labels=("BD observed", "HC hidden"),
    )
    _draw_grouped_bars(
        axes[2], plot_data.loc[plot_data["panel"] == "(c)"],
        panel_label="(c)", title="IHDP vs IHDP-HC",
        conditions=("full", "hidden"), condition_labels=("Full", "Hidden"),
    )
    _draw_boxplot(
        axes[3], plot_data.loc[plot_data["panel"] == "(d)"],
        PANEL_BOUND_ORDER, PANEL_BOUND_LABELS,
    )

    stem = output / "hidden_confounder_track"
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        stem.with_suffix(".jpg"), dpi=400, bbox_inches="tight", pad_inches=0.04,
        pil_kwargs={"quality": 95},
    )
    shutil.copyfile(stem.with_suffix(".pdf"), output / "fig3_hidden_confounder_track.pdf")
    shutil.copyfile(stem.with_suffix(".jpg"), output / "fig3_hidden_confounder_track.jpg")
    plt.close(figure)

    # Preserve the audited decimal representation and provenance byte-for-byte.
    shutil.copyfile(data_path, output / "hidden_confounder_track_plot_data.csv")
    source_manifest = data_path.with_name("hidden_confounder_track_manifest.json")
    shutil.copyfile(source_manifest, output / "hidden_confounder_track_manifest.json")
    return plot_data


def verify_hidden_confounder(
    data_path: Path, expected_means: Mapping[str, float]
) -> None:
    """Run the hidden-confounder numerical and replication-unit contract."""
    frame = load_hidden_confounder_data(data_path)
    for key, expected in expected_means.items():
        panel, model = key.split("/", maxsplit=1)
        observed = frame.loc[
            (frame["panel"] == panel) & (frame["model_key"] == model), "value"
        ].mean()
        if not np.isclose(observed, expected, atol=1e-12, rtol=1e-12):
            raise ValueError(f"hidden-confounder numerical check failed for {panel}/{model}")
