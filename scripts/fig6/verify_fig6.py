#!/usr/bin/env python3
"""Score new Figure 6 predictions, compare reference values, and render the figure.

Metrics, model labels, and bootstrap protocol adapted from fig6.py and common.py
in the causalidview_39_reproduce archive (see data/fig6/PROVENANCE.md).
"""

from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.fig6.run_fig6b import CFMS, SCORE_FIELDS, score_predictions

ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
ARCHIVE = ROOT / "data" / "fig6"
REFERENCE = RESULTS / "fig6"
IHDP = ROOT / "data" / "ihdp_hc"
POINT_MODELS = (*CFMS, "tabpfn_x", "tabpfn_s", "tabpfn_dr")
POINT_LABELS = dict(zip(POINT_MODELS, (
    "CausalPFN", "Do-PFN", "CausalFM", "TabPFN-v3.5 + X", "TabPFN-v3.5 + S", "TabPFN-v3.5 + DR",
)))
CONDITIONS = ("observed", "hidden")
WORLDS = range(40)
TRIALS = range(100)
SENSITIVITY_GAMMA = 2.0
PARTIAL_MODELS = ("xgboost", "logistic", "tabpfn_v35")
SENSITIVITY_MODELS = ("csa_pfn_msm", "neuralcsa", "b_learner_tabpfn_all")
BOUND_STYLE = {
    "xgboost": ("XGBoost", "#B8BFC3"),
    "logistic": ("Logistic Regression", "#8A959B"),
    "tabpfn_v35": ("TabPFN-v3.5", "#3B5B92"),
    "b_learner_tabpfn_all": ("B-Learner + TabPFN (all)", "#8F6A1E"),
    "csa_pfn_msm": ("CSA-PFN (MSM)", "#7765A7"),
    "neuralcsa": ("NeuralCSA", "#5D7E88"),
}
POINT_CELL = ["model", "replicate", "condition"]
BOUND_QUERY = ["world_id", "unit_id", "leaf", "gamma"]
TOLERANCE = 1e-10


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def check_cells(counts: pd.Series, expected, count: int, description: str) -> None:
    actual = {tuple(None if pd.isna(value) else value for value in key) for key in counts.index}
    if actual != set(expected) or not counts.eq(count).all():
        raise ValueError(f"Incomplete or unexpected {description} coverage (expected {count} rows per cell)")


def check_finite(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Nonfinite {description}")


def join_queries(predictions: pd.DataFrame, targets: pd.DataFrame, key: list[str]) -> pd.DataFrame:
    joined = predictions.merge(targets, on=key, how="left", validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError(f"Missing evaluator target for query key {key}")
    return joined.drop(columns="_merge")


def score_points(predictions, targets, panel, query_id, replicates, query_count, atol, rtol):
    expected = list(product(POINT_MODELS, replicates, CONDITIONS))
    if predictions.duplicated([*POINT_CELL, query_id]).any():
        raise ValueError(f"Duplicate panel {panel} predictions")
    sizes = targets.groupby("replicate").size()
    if set(sizes.index) != set(replicates) or not sizes.eq(query_count).all():
        raise ValueError(f"Invalid panel {panel} evaluator coverage")
    joined = join_queries(predictions, targets, ["replicate", query_id])
    check_finite(joined, ["tau_hat", "tau_true", "target"], f"panel {panel} point data")
    if not np.allclose(joined["tau_true"], joined["target"], atol=atol, rtol=rtol):
        raise ValueError(f"Panel {panel} prediction targets disagree with the evaluator")
    scores = joined.assign(squared_error=(joined["tau_hat"] - joined["target"]) ** 2)
    groups = scores.groupby(POINT_CELL, dropna=False)["squared_error"]
    check_cells(groups.size(), expected, query_count, f"panel {panel} prediction")
    result = np.sqrt(groups.mean()).reindex(expected).reset_index(name="spehe")
    result.insert(0, "panel", panel)
    return result[["panel", "model", "replicate", "condition", "spehe"]]


def load_fig6a_predictions(path: Path) -> tuple[pd.DataFrame, list[Path]]:
    rows = []
    files = []
    for (view, condition), model, world in product(zip(("BD", "HC"), CONDITIONS), POINT_MODELS, WORLDS):
        file = path / view / model / f"world_{world:03d}.npz"
        with np.load(file, allow_pickle=False) as prediction:
            unit_id, tau_hat = prediction["unit_id"], prediction["tau_hat"]
            if unit_id.shape != (100,) or tau_hat.shape != (100,):
                raise ValueError(f"Expected 100 Figure 6(a) query predictions in {file}")
        rows.append(pd.DataFrame(dict(model=model, replicate=world, condition=condition,
                                      unit_id=unit_id, tau_hat=tau_hat)))
        files.append(file)
    return pd.concat(rows, ignore_index=True), files


def score_fig6a(predictions: Path | None) -> tuple[pd.DataFrame, list[Path]]:
    targets = read_csv(ARCHIVE / "synthetic_targets.csv.gz").rename(columns={"world_id": "replicate"})
    if predictions is None:
        synthetic = read_csv(ARCHIVE / "synthetic_predictions.csv.gz").rename(columns={"world_id": "replicate"})
        files = []
    else:
        synthetic, files = load_fig6a_predictions(predictions)
        # Fresh files carry query IDs and estimates; the evaluator supplies their truth.
        truth = targets.rename(columns={"target": "tau_true"})
        synthetic = join_queries(synthetic, truth, ["replicate", "unit_id"])
    return score_points(synthetic, targets, "a", "unit_id", WORLDS, 100, 1e-12, 1e-12), files


def score_fig6b(
    ihdp_evaluators: list[Path], predictions: Path | None, cfm_predictions: Path | None,
) -> tuple[pd.DataFrame, list[Path]]:
    if predictions is not None:
        model_paths = dict.fromkeys(POINT_MODELS, predictions)
        if cfm_predictions is not None:
            model_paths.update(dict.fromkeys(CFMS, cfm_predictions))
        scores, files = score_predictions(IHDP, model_paths)
        panel_b = pd.DataFrame(scores, columns=SCORE_FIELDS)
        panel_b["condition"] = panel_b["condition"].replace({"full": "observed"})
        panel_b.insert(0, "panel", "b")
        return panel_b[["panel", "model", "replicate", "condition", "spehe"]], files

    trial_targets = []
    for trial, path in zip(TRIALS, ihdp_evaluators, strict=True):
        with np.load(path, allow_pickle=False) as evaluator:
            trial_targets.append(pd.DataFrame({
                "replicate": trial, "sample_id": evaluator["test_sample_id"], "target": evaluator["test_tau_true"],
            }))
    ihdp = read_csv(ARCHIVE / "ihdp_predictions.csv.gz").rename(columns={"trial": "replicate"})
    ihdp["condition"] = ihdp["condition"].replace({"full": "observed"})
    panel_b = score_points(ihdp, pd.concat(trial_targets, ignore_index=True), "b", "sample_id", TRIALS, 75, 1e-5, 1e-6)
    return panel_b, []


def figure_points(
    ihdp_evaluators: list[Path], fresh_a: Path | None, fresh_b: Path | None, fresh_cfms: Path | None,
) -> tuple[pd.DataFrame, list[Path], list[Path]]:
    panel_a, fresh_a_files = score_fig6a(fresh_a)
    panel_b, fresh_b_files = score_fig6b(ihdp_evaluators, fresh_b, fresh_cfms)
    return pd.concat([panel_a, panel_b], ignore_index=True), fresh_a_files, fresh_b_files


def overlay_fig6b(points: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, set[str]]:
    scores = read_csv(path)[[*POINT_CELL, "spehe"]].copy()
    models = set(scores["model"])
    if not models or not models.issubset(POINT_MODELS) or not scores["condition"].isin(["full", "hidden"]).all():
        raise ValueError("Figure 6(b) override requires supported models with full and hidden conditions")
    check_finite(scores, ["spehe"], "Figure 6(b) override scores")
    if scores["spehe"].lt(0).any():
        raise ValueError("Figure 6(b) override sPEHE must be nonnegative")
    scores["condition"] = scores["condition"].replace({"full": "observed"})
    counts = scores.groupby(POINT_CELL, dropna=False).size()
    check_cells(counts, product(models, TRIALS, CONDITIONS), 1, "Figure 6(b) override")
    mask = points["panel"].eq("b") & points["model"].isin(models)
    joined = join_queries(points.loc[mask, POINT_CELL], scores, POINT_CELL)
    points.loc[mask, "spehe"] = joined["spehe"].to_numpy()
    return points, models


def summarize_points(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (panel, model, condition), group in points.groupby(["panel", "model", "condition"]):
        values = group.sort_values("replicate")["spehe"].to_numpy(dtype=float)
        key = f"fig6/{panel}/{model}/{condition}"
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "little")
        rng = np.random.default_rng(seed)
        n = len(values)
        means = values[rng.integers(0, n, size=(10_000, n))].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        rows.append(dict(panel=panel, model=model, condition=condition, n=n, mean=values.mean(),
                         ci_low=low, ci_high=high))
    return pd.DataFrame(rows)


def select_gamma2(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["leaf"].ne("sensitivity") | frame["gamma"].eq(SENSITIVITY_GAMMA)]


def select_panel_c_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = select_gamma2(frame)
    previous_panel_d = frame["leaf"].eq("sensitivity") & frame["method"].isin(
        ("b_learner", "b_learner_tabpfn_final")
    )
    return frame.loc[~previous_panel_d]


def score_bounds(override_paths: list[Path], archived: bool) -> tuple[pd.DataFrame, set[tuple[str, str]]]:
    targets = read_csv(ARCHIVE / "bound_targets.csv.gz")
    world_sizes = targets.groupby("world_id").size()
    if set(world_sizes.index) != set(WORLDS) or not world_sizes.eq(100).all():
        raise ValueError("Invalid bound evaluator coverage")
    oracle = select_gamma2(read_csv(ARCHIVE / "bound_oracles.csv.gz"))
    columns = [*BOUND_QUERY, "method", "lower", "upper"]
    predictions = select_panel_c_predictions(read_csv(ARCHIVE / "bound_predictions.csv.gz")[columns]) if archived else None
    replaced_pairs = set()
    if override_paths:
        fresh = select_panel_c_predictions(pd.concat([
            (pd.read_parquet(path) if path.suffix == ".parquet" else read_csv(path))[columns]
            for path in override_paths
        ], ignore_index=True))
        replaced_pairs = set(zip(fresh["leaf"], fresh["method"]))
        if not replaced_pairs:
            raise ValueError("Bound prediction override has no rows for Figure 6(c)")
        if predictions is None:
            predictions = fresh
        else:
            original_pairs = pd.MultiIndex.from_frame(predictions[["leaf", "method"]])
            predictions = pd.concat([predictions[~original_pairs.isin(replaced_pairs)], fresh], ignore_index=True)
    if predictions is None:
        raise ValueError("Fresh bound predictions are required")
    if predictions.duplicated([*BOUND_QUERY, "method"]).any():
        raise ValueError("Duplicate bound prediction keys")
    joined = join_queries(
        predictions.rename(columns={"lower": "lower_hat", "upper": "upper_hat"}),
        oracle.rename(columns={"lower": "lower_true", "upper": "upper_true"}), BOUND_QUERY,
    )
    joined = join_queries(joined, targets, ["world_id", "unit_id"])
    check_finite(joined, ["lower_hat", "upper_hat", "lower_true", "upper_true", "target"], "bound endpoints/targets")
    joined["endpoint_sq"] = ((joined["lower_hat"] - joined["lower_true"]) ** 2
                            + (joined["upper_hat"] - joined["upper_true"]) ** 2) / 2
    joined["containment"] = joined["lower_hat"].le(joined["target"]) & joined["target"].le(joined["upper_hat"])
    groups = joined.groupby(["leaf", "method", "world_id", "gamma"], dropna=False)
    expected = [*product(("manski", "iv"), PARTIAL_MODELS, WORLDS, (None,)),
                *product(("sensitivity",), SENSITIVITY_MODELS, WORLDS, (SENSITIVITY_GAMMA,))]
    check_cells(groups.size(), expected, 100, "bound prediction")
    results = groups.agg(containment=("containment", "mean"), n_query=("unit_id", "size"),
                         endpoint_sq=("endpoint_sq", "mean")).reset_index()
    results["endpoint_rmse"] = np.sqrt(results.pop("endpoint_sq"))
    return results.rename(columns={"leaf": "partial_information"}), replaced_pairs


def summarize_panel_c(bounds: pd.DataFrame) -> pd.DataFrame:
    return bounds.groupby(["partial_information", "method"], as_index=False).agg(
        mean=("endpoint_rmse", "mean"), sd=("endpoint_rmse", "std"), n=("world_id", "size"),
    )


def archived_rows(table: pd.DataFrame, name: str, replaced_a: bool, replaced_b: set[str], replaced_bounds) -> pd.DataFrame:
    if name.startswith("point_"):
        fresh_a = table["panel"].eq("a") & replaced_a
        fresh_b = table["panel"].eq("b") & table["model"].isin(replaced_b)
        return table[~(fresh_a | fresh_b)]
    pairs = pd.MultiIndex.from_frame(table[["partial_information", "method"]])
    return table[~pairs.isin(replaced_bounds)]


REFERENCE_KEYS = {
    "point_per_replicate": ["panel", *POINT_CELL],
    "point_summary": ["panel", "model", "condition"],
    "bounds_per_world": ["partial_information", "method", "world_id", "gamma"],
    "panel_c": ["partial_information", "method"],
}


def read_reference(path: Path, reference_manifest: dict, hashes: dict) -> pd.DataFrame:
    relative = path.relative_to(ROOT).as_posix()
    digest = sha256(path)
    if digest != reference_manifest[relative]["sha256"]:
        raise ValueError(f"Figure 6 reference checksum mismatch for {path.name}")
    hashes[relative] = digest
    return read_csv(path)


def panel_reference(reference: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    if "panel" in reference:
        return reference.loc[reference["panel"].isin(actual["panel"])]
    return reference


def verify_tables(tables, replaced_a, replaced_b, replaced_bounds, reference_manifest):
    checked = {}
    hashes = {}
    for name, actual in tables.items():
        reference = REFERENCE / f"{name}.csv"
        frozen = panel_reference(read_reference(reference, reference_manifest, hashes), actual)
        actual = archived_rows(actual, name, replaced_a, replaced_b, replaced_bounds)
        expected = archived_rows(frozen, name, replaced_a, replaced_b, replaced_bounds)
        try:
            pd.testing.assert_frame_equal(
                actual.sort_values(REFERENCE_KEYS[name]).reset_index(drop=True),
                expected.sort_values(REFERENCE_KEYS[name]).reset_index(drop=True),
                check_dtype=False, check_exact=False, atol=TOLERANCE, rtol=TOLERANCE,
            )
        except AssertionError as exc:
            raise ValueError(f"Figure 6 calculation differs from {reference.name}\n{exc}") from exc
        checked[name] = {"rows_compared": len(actual), "total_rows": len(tables[name])}
    return checked, hashes


def compare_summaries(tables, reference_manifest, *, paper_reference=False):
    checked = {}
    hashes = {}
    reports = []
    aggregates = {
        "point_summary": ["mean", "ci_low", "ci_high"],
        "panel_c": ["mean", "sd"],
    }
    for name, metrics in aggregates.items():
        if name not in tables:
            continue
        if paper_reference:
            metrics = ["mean"]
        keys = REFERENCE_KEYS[name]
        summary = tables[name]
        actual = summary.melt(id_vars=keys, value_vars=metrics, value_name="actual", var_name="metric")
        frozen = panel_reference(read_reference(REFERENCE / f"{name}.csv", reference_manifest, hashes), summary)
        reference = frozen.melt(
            id_vars=keys, value_vars=metrics, value_name="reference", var_name="metric",
        )
        report = join_queries(actual, reference, [*keys, "metric"])
        if len(actual) != len(reference):
            raise ValueError(f"Figure 6 comparison coverage differs for {name}")
        report["reference_source"] = "v2_reference"
        if paper_reference and name == "point_summary":
            paper = read_reference(REFERENCE / "paper_bar_values.csv", reference_manifest, hashes)
            report = report.merge(paper, on=keys, how="left", validate="one_to_one")
            paper_rows = report["paper"].notna()
            report.loc[paper_rows, "reference"] = report.loc[paper_rows, "paper"]
            report.loc[paper_rows, "reference_source"] = "paper_pdf"
            paper_deltas = (report.loc[paper_rows, "actual"] - report.loc[paper_rows, "paper"]).abs()
            thresholds = report.loc[paper_rows, "atol"] + report.loc[paper_rows, "rtol"] * report.loc[paper_rows, "paper"].abs()
            report["within_tolerance"] = pd.Series(pd.NA, index=report.index, dtype="boolean")
            report.loc[paper_rows, "within_tolerance"] = paper_deltas.le(thresholds)
            report = report.drop(columns="paper")
        report.insert(0, "table", name)
        report["delta"] = report["actual"] - report["reference"]
        checked[name] = {"values_compared": len(report)}
        reports.append(report)
    return pd.concat(reports, ignore_index=True), checked, hashes


def render(points, panel_c, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(18, 4.6))
    figure.subplots_adjust(wspace=0.34, top=0.80, bottom=0.27)
    for ax, panel, title in zip(axes, ("a", "b"), ("Synthetic, 40 paired worlds", "IHDP, 100 paired trials")):
        for offset, condition in enumerate(CONDITIONS):
            bars = points[points["panel"].eq(panel) & points["condition"].eq(condition)]
            bars = bars.set_index("model").loc[list(POINT_MODELS)]
            error = np.maximum([bars["mean"] - bars["ci_low"], bars["ci_high"] - bars["mean"]], 0)
            ax.bar(np.arange(len(POINT_MODELS)) + (offset - 0.5) * 0.35, bars["mean"], width=0.33,
                   label=condition.title(), alpha=0.5 if offset == 0 else 1, hatch="//" if offset else "",
                   yerr=error, capsize=2)
        ax.set_xticks(range(len(POINT_MODELS)), [POINT_LABELS[m] for m in POINT_MODELS], rotation=20, ha="right")
        ax.set(title=f"({panel}) {title}", ylabel="sPEHE")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.37, 0.98), ncol=2)

    ax = axes[2]
    panel_c = panel_c.set_index(["partial_information", "method"])
    shown_models = set()
    for index, (leaf, models) in enumerate([("manski", PARTIAL_MODELS), ("iv", PARTIAL_MODELS),
                                            ("sensitivity", SENSITIVITY_MODELS)]):
        for offset, model in enumerate(models):
            bar = panel_c.loc[(leaf, model)]
            label, color = BOUND_STYLE[model]
            ax.bar(index * 4 + offset, bar["mean"], yerr=bar["sd"], color=color,
                   label=label if model not in shown_models else None, capsize=2)
            shown_models.add(model)
    ax.set_xticks([1, 5, 9], ["Manski", "IV", f"Sensitivity (Γ={SENSITIVITY_GAMMA:g})"])
    ax.set(title="(c) Partial identification", ylabel="Endpoint RMSE")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)

    for ext in ("pdf", "png"):
        figure.savefig(output / f"fig6.{ext}", dpi=200, bbox_inches="tight")
    plt.close(figure)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(archive_manifest, filenames, ihdp_evaluators):
    provenance = ARCHIVE / "archive_manifest.json"
    sources = {provenance.relative_to(ROOT).as_posix(): sha256(provenance)}
    for name in filenames:
        path = ARCHIVE / name
        digest = sha256(path)
        if digest != archive_manifest["files"][name]["sha256"]:
            raise ValueError(f"Figure 6 archive input checksum mismatch for {name}")
        sources[path.relative_to(ROOT).as_posix()] = digest
    for path in ihdp_evaluators:
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256(path)
        if digest != archive_manifest["ihdp_evaluators"][relative]["sha256"]:
            raise ValueError(f"Figure 6 evaluator checksum mismatch for {relative}")
        sources[relative] = digest
    return sources


def write_outputs(output, tables, comparison, manifest, *, plot=False):
    if output.is_relative_to(ROOT / "data") or output.is_relative_to(RESULTS):
        raise ValueError("Figure 6 output directory must be outside data/ and results/")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        staging = Path(temp)
        for name, table in tables.items():
            table.to_csv(staging / f"{name}.csv", index=False)
        comparison.to_csv(staging / "comparison.csv", index=False)
        if plot:
            render(tables["point_summary"], tables["panel_c"], staging)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        for path in staging.iterdir():
            path.replace(output / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", action="store_true", help="Use saved predictions for stages without overrides")
    parser.add_argument("--output-dir", "--output", type=Path, help="output path (default under artifacts/)")
    parser.add_argument("--paper-reference", action="store_true",
                        help="Use the historical PDF comparison for point-summary means")
    parser.add_argument("--gamma", type=float, choices=(SENSITIVITY_GAMMA,),
                        help="Sensitivity level for panel (c), fixed at Γ=2")
    parser.add_argument("--gamma-mode", choices=["single"], help=argparse.SUPPRESS)
    parser.add_argument("--fig6a-predictions", type=Path, help="complete panel (a) BD/HC per-world prediction directory")
    parser.add_argument("--fig6b-predictions", type=Path, help="complete panel (b) Full/Hidden prediction directory")
    parser.add_argument("--fig6b-cfm-predictions", type=Path, help="optional separate panel (b) CFM prediction directory")
    parser.add_argument("--fig6b-metrics", type=Path, help="override panel (b) with scored trial metrics in --archived mode")
    parser.add_argument("--bound-predictions", type=Path, action="append", default=[],
                        help="replacement endpoints (repeat for multiple model or shard files), CSV/CSV.gz or Parquet with PyArrow")
    args = parser.parse_args()
    if args.fig6b_metrics is not None and (not args.archived or args.fig6b_predictions is not None):
        parser.error("--fig6b-metrics requires --archived and cannot be combined with raw Figure 6(b) predictions")
    if args.archived and args.fig6b_cfm_predictions is not None and args.fig6b_predictions is None:
        parser.error("--fig6b-cfm-predictions also requires --fig6b-predictions in archived mode")
    name = "fig6_archived" if args.archived else "fig6_fresh"
    output = (args.output_dir or ARTIFACTS / name).resolve()
    if output.is_relative_to(ROOT / "data") or output.is_relative_to(RESULTS):
        raise ValueError("Figure 6 output directory must be outside data/ and results/")

    fresh_a = args.fig6a_predictions
    fresh_b = args.fig6b_predictions
    bound_predictions = args.bound_predictions
    if not args.archived:
        fresh_a = fresh_a or ARTIFACTS / "fig6a_predictions"
        fresh_b = fresh_b or ARTIFACTS / "fig6b_predictions"
        bound_predictions = bound_predictions or [
            ARTIFACTS / "fig6_bounds" / f"{model}.csv" for model in (*PARTIAL_MODELS, *SENSITIVITY_MODELS)
        ]
        for path, command in (
            (fresh_a, "predict_fig6a.py"),
            (fresh_b, "predict_fig6b.py and predict_fig6b_cfms.py"),
        ):
            if not path.is_dir():
                raise FileNotFoundError(f"Missing fresh predictions at {path}. Run {command} first.")

    ihdp_evaluators = [IHDP / f"trial_{trial:03d}" / "evaluator_only.npz" for trial in TRIALS]
    provenance = ARCHIVE / "archive_manifest.json"
    archive_manifest = json.loads(provenance.read_text())
    optional_predictions = {
        "synthetic_predictions.csv.gz": args.archived and fresh_a is None,
        "ihdp_predictions.csv.gz": args.archived and fresh_b is None,
        "bound_predictions.csv.gz": args.archived,
    }
    filenames = [name for name in sorted(archive_manifest["files"]) if optional_predictions.get(name, True)]
    sources = verify_sources(archive_manifest, filenames, ihdp_evaluators)

    points, fresh_a_files, fresh_b_files = figure_points(ihdp_evaluators, fresh_a, fresh_b, args.fig6b_cfm_predictions)
    replaced_models = set(POINT_MODELS) if fresh_b_files else set()
    if args.fig6b_metrics is not None:
        points, replaced_models = overlay_fig6b(points, args.fig6b_metrics)
    point_summary = summarize_points(points)
    bounds, replaced_bounds = score_bounds(bound_predictions, args.archived)
    panel_c = summarize_panel_c(bounds)
    tables = dict(point_per_replicate=points, point_summary=point_summary, bounds_per_world=bounds,
                  panel_c=panel_c)
    reference_manifest = archive_manifest["reference_files"]
    comparison, summaries_compared, references = compare_summaries(
        tables, reference_manifest, paper_reference=args.paper_reference,
    )
    sources.update(references)
    if args.archived:
        compared, references = verify_tables(tables, bool(fresh_a_files), replaced_models, replaced_bounds,
                                             reference_manifest)
        sources.update(references)
    else:
        compared = {}
    for path in (*fresh_a_files, *fresh_b_files, args.fig6b_metrics, *bound_predictions):
        if path is not None:
            sources[str(path.resolve())] = sha256(path)

    manifest = dict(
        implementation_sha256=sha256(Path(__file__)), sources=sources, synthetic_worlds=len(WORLDS),
        ihdp_trials=len(TRIALS), input_mode="archived" if args.archived else "fresh", panel_c_gamma_mode="single",
        panel_c_gamma=SENSITIVITY_GAMMA, panel_c_gamma_grid=[SENSITIVITY_GAMMA],
        fig6a_prediction_override=bool(fresh_a_files), fig6b_replacement_models=sorted(replaced_models),
        bound_replacement_pairs=sorted(replaced_bounds),
        archived_reference_tolerance=TOLERANCE, archived_verification=compared, reference_comparisons=summaries_compared,
    )
    write_outputs(output, tables, comparison, manifest, plot=True)
    if args.archived:
        rows_compared = sum(item["rows_compared"] for item in compared.values())
        print(f"Recomputed Figure 6 and matched {rows_compared} archived reference rows (tolerance {TOLERANCE:g}).")
    print(f"Compared {len(comparison)} values. {output / 'comparison.csv'}")
    if args.paper_reference:
        paper = comparison.loc[comparison["reference_source"].eq("paper_pdf"), "within_tolerance"]
        print(f"{int(paper.sum())}/{len(paper)} PDF bar values within their recorded tolerances.")
    print(output / "fig6.pdf")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, ImportError, KeyError, OSError, ValueError) as exc:
        sys.exit(f"ERROR: {exc}")
