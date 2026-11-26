#!/usr/bin/env python3
"""Rebuild Table 2 from fresh null-CATE scores and compare it to the paper reference."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "table2"
REFERENCE = ROOT / "results" / "table2" / "table2.csv"
PROVENANCE = DATA / "provenance.json"
FROZEN_SPLITS = DATA / "primary_splits.csv"
FRESH_DIR = ROOT / "artifacts" / "table2" / "fresh"
ARCHIVED_DIR = FRESH_DIR.parent / "archived"
RCT_ARCHIVE = DATA / "rct_null_results.csv"
REQUESTED_ARCHIVE = DATA / "requested_baseline_results.csv"
DATASETS = {"actg175": 768, "pennsylvania": 1024, "illinois": 2048}
MODELS = (
    "causalpfn", "dopfn", "causalfm", "causal_tree", "tarnet", "grf", "cfrnet",
    "tabpfn_x", "tabpfn_s", "tabpfn_dr",
)
METRICS = ["null_spehe", "null_ate", "null_cpehe"]
SPLIT_HASHES = ["context_ids_hash", "query_ids_hash", "input_hash"]
SPLIT_KEY = ["dataset", "realization"]
REALIZATION_KEY = ["dataset", "model", "realization"]
CELL_KEY = ["dataset", "model", "metric"]
DATASET_LABELS = {"actg175": "ACTG 175", "pennsylvania": "Pennsylvania", "illinois": "Illinois"}
MODEL_LABELS = dict(zip(MODELS, (
    "CausalPFN", "Do-PFN", "CausalFM", "Causal Tree", "TARNet", "GRF / Causal Forest", "CFRNet",
    "TabPFN-v3.5 + X", "TabPFN-v3.5 + S", "TabPFN-v3.5 + DR",
), strict=True))


def primary_metrics(paths: list[Path]) -> pd.DataFrame:
    columns = [*REALIZATION_KEY, *SPLIT_HASHES, *METRICS, "protocol", "status", "n_context", "n_query"]
    hash_types = dict.fromkeys(SPLIT_HASHES, str)
    rows = pd.concat([pd.read_csv(path, usecols=columns, dtype=hash_types) for path in paths], ignore_index=True)
    if not rows.protocol.isin(("primary", "fixed768")).all():
        raise ValueError("unexpected realization protocol")
    primary = rows.loc[rows.protocol.eq("primary")].copy()
    keys = list(primary[REALIZATION_KEY].itertuples(index=False, name=None))
    expected = set(product(DATASETS, MODELS, range(40)))
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError("expected one primary row per dataset, model, and realization (0–39)")
    if not primary.status.eq("success").all():
        raise ValueError("every primary realization must have status success")
    if not primary.n_context.eq(primary.dataset.map(DATASETS)).all() or not primary.n_query.eq(256).all():
        raise ValueError("primary realization context/query sizes differ from Table 2")
    for column in SPLIT_HASHES:
        if not primary[column].str.fullmatch("[0-9a-f]{64}", na=False).all():
            raise ValueError(f"invalid paired split hash in {column}")
    split_counts = primary.groupby(SPLIT_KEY)[SPLIT_HASHES].nunique()
    if not split_counts.eq(1).to_numpy().all():
        raise ValueError("paired split hashes must match across all models for each realization")

    values = primary[METRICS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("null-CATE metrics must be finite and nonnegative")
    squared = np.square(values)
    if not np.allclose(squared[:, 0], squared[:, 1] + squared[:, 2], atol=1e-9, rtol=0):
        raise ValueError("sPEHE² must equal ATE error² plus centered sPEHE²")
    primary[METRICS] = squared
    return primary


def verify_checksum(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"frozen file checksum mismatch: {path}")


def load_frozen_splits(provenance: dict) -> pd.DataFrame:
    verify_checksum(FROZEN_SPLITS, provenance["primary_splits"]["sha256"])
    splits = pd.read_csv(FROZEN_SPLITS, dtype=dict.fromkeys(SPLIT_HASHES, str))
    return splits.set_index(SPLIT_KEY)[SPLIT_HASHES].sort_index()


def verify_frozen_inputs(primary: pd.DataFrame, frozen: pd.DataFrame) -> None:
    actual = primary[[*SPLIT_KEY, *SPLIT_HASHES]].drop_duplicates().set_index(SPLIT_KEY).sort_index()
    if not actual.equals(frozen):
        raise ValueError("realization hashes differ from the frozen splits")


def rebuild(primary: pd.DataFrame) -> pd.DataFrame:
    groups = primary.groupby(["dataset", "model"])
    cells = []
    for dataset, model in product(DATASETS, MODELS):
        group = groups.get_group((dataset, model)).sort_values("realization")
        for metric in METRICS:
            values = group[metric].to_numpy(dtype=float)
            mean, sd = float(values.mean()), float(values.std(ddof=0))
            cells.append({
                "dataset": dataset, "model": model, "metric": metric,
                "n_realizations": len(values), "n_context": DATASETS[dataset],
                "mean": mean, "sd": sd, "display": f"{mean:.4f} ± {sd:.4f}",
            })
    return pd.DataFrame(cells)


def compare(actual: pd.DataFrame, reference_path: Path, atol: float) -> pd.DataFrame:
    actual = actual.set_index(CELL_KEY).sort_index()
    reference = pd.read_csv(reference_path, float_precision="round_trip").set_index(CELL_KEY).sort_index()
    pd.testing.assert_index_equal(actual.index, reference.index)
    metadata = ["n_realizations", "n_context", "display"]
    columns = [*metadata, "mean", "sd"]
    comparison = actual[columns].add_prefix("actual_").join(reference[columns].add_prefix("reference_"))
    deltas = (actual[["mean", "sd"]] - reference[["mean", "sd"]]).add_suffix("_delta")
    comparison = comparison.join(deltas)
    comparison["display_matches"] = actual["display"].eq(reference["display"])
    comparison["matches"] = actual[metadata].eq(reference[metadata]).all(axis=1) & deltas.abs().le(atol).all(axis=1)
    return comparison.reset_index()


def plot_table(cells: pd.DataFrame, output: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    display = cells.set_index(CELL_KEY)["display"]
    values = [[display[dataset, model, metric] for dataset, metric in product(DATASETS, METRICS)]
              for model in MODELS]
    headings = [f"{DATASET_LABELS[dataset]}\nnC={DATASETS[dataset]}\n{metric}"
                for dataset, metric in product(DATASETS, ("sPEHE²", "ATE error²", "c-sPEHE²"))]
    figure = Figure(figsize=(19, 5))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.set_axis_off()
    table = axis.table(cellText=values, rowLabels=list(MODEL_LABELS.values()),
                       colLabels=headings, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for column in range(len(headings)):
        header = table[0, column]
        header.set_height(0.16)
        header.set_facecolor("#EDF2F6")
        header.set_text_props(weight="bold")
    axis.set_title("RCT negative-control null-CATE · mean ± population SD over 40 primary realizations", pad=16)
    figure.savefig(output, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", action="store_true", help="Use bundled score archives by default")
    parser.add_argument("--rct-results", type=Path, help="Score CSV for the six pretrained models")
    parser.add_argument("--requested-results", type=Path, help="Score CSV for Causal Tree, TARNet, GRF, and CFRNet")
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, help="Output CSV, defaults to artifacts/table2/{fresh,archived}/table2.csv")
    parser.add_argument("--plot", action="store_true", help="Write a PDF next to the CSV")
    parser.add_argument("--aggregate-only", action="store_true", help="Render supplied scores without comparing reference values")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for means and SDs (default: %(default)g)")
    args = parser.parse_args()
    if not np.isfinite(args.atol) or args.atol < 0:
        parser.error("--atol must be finite and nonnegative")

    defaults = DATA if args.archived else FRESH_DIR
    rct_scores = args.rct_results or defaults / RCT_ARCHIVE.name
    requested_scores = args.requested_results or defaults / REQUESTED_ARCHIVE.name
    inputs = [rct_scores, requested_scores]
    archives = {path.resolve(): path for path in (RCT_ARCHIVE, REQUESTED_ARCHIVE)}
    used_archives = [archives[path.resolve()] for path in inputs if path.resolve() in archives]
    if used_archives and not args.archived:
        parser.error("bundled metric archives require --archived")

    output_dir = ARCHIVED_DIR if args.archived else FRESH_DIR
    output = args.output or output_dir / "table2.csv"
    pdf = output.with_suffix(".pdf") if args.plot else None
    comparison_path = None
    if not args.archived and not args.aggregate_only:
        comparison_path = output.with_name(f"{output.stem}_comparison.csv")
    outputs = [path for path in (output, pdf, comparison_path) if path is not None]
    output_paths = {path.resolve() for path in outputs}
    protected = [*inputs, args.reference, PROVENANCE, FROZEN_SPLITS, REFERENCE, RCT_ARCHIVE, REQUESTED_ARCHIVE]
    if len(output_paths) != len(outputs) or output_paths.intersection(path.resolve() for path in protected):
        parser.error("output paths must differ from each other and from the input and reference paths")

    try:
        missing = [str(path) for path in inputs if not path.is_file()]
        if missing:
            advice = " Supply both fresh score CSVs or select --archived to use the bundled scores." if not args.archived else ""
            raise FileNotFoundError(f"missing Table 2 score CSVs: {', '.join(missing)}.{advice}")
        provenance = json.loads(PROVENANCE.read_text())
        frozen = load_frozen_splits(provenance)
        frozen_files = used_archives.copy()
        if not args.aggregate_only and args.reference.resolve() == REFERENCE.resolve():
            frozen_files.append(REFERENCE)
        for path in frozen_files:
            verify_checksum(path, provenance["archived_files"][path.relative_to(ROOT).as_posix()]["sha256"])
        primary = primary_metrics(inputs)
        verify_frozen_inputs(primary, frozen)
        result = rebuild(primary)
        comparison = None if args.aggregate_only else compare(result, args.reference, args.atol)
        failures = 0 if comparison is None else int((~comparison.matches).sum())
        failure_message = f"{failures} of {len(result)} cells differ from the reference (absolute tolerance {args.atol:g})"
        if failures and args.archived:
            raise ValueError(failure_message)

        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False, float_format="%.17g")
        print(f"{len(result)} cells from {len(primary)} primary metrics rows")
        print(f"Wrote {output}")
        if comparison is not None:
            delta = comparison[["mean_delta", "sd_delta"]].abs().max().max()
            print(f"Max absolute delta from the reference: {delta:.3g}")
            if comparison_path is not None:
                comparison.to_csv(comparison_path, index=False, float_format="%.17g")
                print(f"Wrote {comparison_path}")
        if pdf is not None:
            plot_table(result, pdf)
            print(f"Wrote {pdf}")
        if failures:
            raise ValueError(failure_message)
    except (ValueError, AssertionError, KeyError, OSError) as error:
        raise SystemExit(f"Table 2 verification failed: {error}") from error


if __name__ == "__main__":
    main()
