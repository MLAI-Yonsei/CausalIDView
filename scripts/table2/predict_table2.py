#!/usr/bin/env python3
"""Generate Table 2 predictions and score them against zero CATE."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "Baselines" / "src")]

from baselines.table2 import R_MODELS, TRADITIONAL_MODELS, optimization_seeds
from scripts.table2.verify_table2 import (
    DATASETS, FRESH_DIR, METRICS, PROVENANCE, RCT_ARCHIVE, REQUESTED_ARCHIVE, SPLIT_HASHES,
    load_frozen_splits,
)


QUERY_SIZE = 256
CFMS = {"causalpfn": "causalpfn", "dopfn": "dopfn", "causalfm": "causalfm_standard"}
TABPFN = {"tabpfn_x": "BD", "tabpfn_s": "BD_S", "tabpfn_dr": "BD_DR"}
MODELS = tuple(CFMS) + tuple(TABPFN) + TRADITIONAL_MODELS
FEATURES = {
    "actg175": ("age", "wtkg", "hemo", "homo", "drugs", "karnof", "oprior", "z30", "zprior",
                "preanti", "race", "gender", "str2", "symptom", "cd80"),
    "pennsylvania": ("r_age", "sex", "black", "hispanic", "othrace", "dep", "sep", "lo", "recall", "retire"),
    "illinois": ("age", "male", "black", "hispanic", "natvamer", "otherace", "white"),
}
OUTCOME = {"actg175": "cd40", "pennsylvania": "wages", "illinois": "avprearn"}


def load_dataset(dataset: str, raw_dir: Path, source: dict) -> pd.DataFrame:
    path = raw_dir / source["filename"]
    if not path.is_file():
        raise FileNotFoundError(f"missing {path}; obtain it from {source['source']}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"SHA-256 mismatch for {path}: {digest}")
    if dataset == "actg175":
        raw = pd.read_csv(path, sep=r"\s+")
        frame = raw.loc[raw.arms.isin((0, 1))].assign(treatment=lambda part: part.arms)
    else:
        raw = pd.read_stata(path, convert_categoricals=False)
        if dataset == "pennsylvania":
            eligible = raw.dem_inel.eq(1) & raw.revasamp.eq(1) & raw.tg.isin((0, 6))
            frame = raw.loc[eligible].assign(treatment=lambda part: part.tg.eq(6))
        else:
            frame = raw.loc[raw.control.eq(1) | raw.jsie.eq(1)].assign(treatment=lambda part: part.jsie)
    return frame.dropna(subset=["treatment", OUTCOME[dataset]])


def archived_hash(values: np.ndarray) -> str:
    """Match the RCT runner's dtype and text shape encoding."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def prepare_input(frame: pd.DataFrame, dataset: str, realization: int):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    treatment = frame.treatment.to_numpy(dtype=np.int64)
    n_context = DATASETS[dataset]
    rng = np.random.default_rng(realization)
    n_treated = int(round(n_context * float(treatment.mean())))
    treated = rng.permutation(np.flatnonzero(treatment == 1))[:n_treated]
    control = rng.permutation(np.flatnonzero(treatment == 0))[:n_context - n_treated]
    context = rng.permutation(np.concatenate([treated, control]))
    remaining = np.setdiff1d(np.arange(len(frame)), context, assume_unique=True)
    query = rng.choice(remaining, size=QUERY_SIZE, replace=False)

    categorical = ["sep", "lo"] if dataset == "pennsylvania" else []
    numeric = [column for column in FEATURES[dataset] if column not in categorical]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    transform = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", make_pipeline(SimpleImputer(strategy="most_frequent"), encoder), categorical),
    ], sparse_threshold=0)
    xc = np.asarray(transform.fit_transform(frame.iloc[context]), dtype=np.float64)
    xq = np.asarray(transform.transform(frame.iloc[query]), dtype=np.float64)
    raw_y = frame.iloc[context][OUTCOME[dataset]].to_numpy(dtype=np.float64)
    y = (raw_y - float(raw_y.mean())) / float(raw_y.std(ddof=0))
    tc = treatment[context]
    context_ids = frame.index.to_numpy(dtype=np.int64)[context]
    query_ids = frame.index.to_numpy(dtype=np.int64)[query]
    hashes = dict(zip(SPLIT_HASHES, (
        archived_hash(context_ids), archived_hash(query_ids),
        archived_hash(np.concatenate([xc.ravel(), tc.astype(float), y, xq.ravel()])),
    ), strict=True))
    return {"X_context": xc, "treatment": tc, "outcome": y, "X_query": xq}, query_ids, hashes


def run_worker(model: str, realization: int, input_path: Path, output_path: Path, checkpoint: Path):
    with np.load(input_path, allow_pickle=False) as archive:
        arrays = dict(archive)
    if model in CFMS:
        from baselines.cfm import _run_dopfn_v2, _run_frontier_v2

        if model == "dopfn":
            prediction, _, _, metadata = _run_dopfn_v2(
                arrays, "cuda:0", QUERY_SIZE, model_root=checkpoint.parents[1],
            )
        else:
            prediction, _, _, metadata = _run_frontier_v2(
                model, arrays, "cuda:0", QUERY_SIZE, compute_uq=False, graph_regime="backdoor",
            )
        if metadata.get("n_context_effective") != len(arrays["X_context"]):
            raise ValueError(f"{model} did not use the full Table 2 context")
    else:
        from baselines.registry import create_estimator
        from baselines.tabpfn.fig2 import _model_seed
        from baselines.tabpfn.fig2_config import estimator_config

        config = estimator_config(_model_seed(TABPFN[model], realization), "v3.5")
        config["tabpfn"].update(checkpoint_root=str(checkpoint.parent), device="cuda:0")
        estimator = create_estimator(model, config)
        estimator.fit(arrays["X_context"], arrays["treatment"], arrays["outcome"], outcome_type="continuous")
        prediction = estimator.predict_cate(arrays["X_query"])
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if prediction.shape != (QUERY_SIZE,) or not np.isfinite(prediction).all():
        raise ValueError(f"invalid query-aligned predictions for {model}")
    np.save(output_path, prediction, allow_pickle=False)


def check_models(models: list[str], runtime: Path):
    import yaml

    from scripts.fig2.predict_fig2 import check_runtime
    from baselines.checkpoints import validate_checkpoint

    specs = yaml.safe_load((ROOT / "configs" / "fig2.yaml").read_text())["checkpoints"]
    profiles = {}
    workers = {}
    for model in models:
        profile = model if model in CFMS else "tabpfn"
        if profile not in profiles:
            python = runtime / f".venv-{profile}" / "bin" / "python"
            check_runtime(python, profile, "cuda:0")
            spec = dict(specs[CFMS.get(model, "tabpfn_v35")])
            if "source_manifest" in spec:
                spec["source_manifest"] = str(ROOT / spec["source_manifest"])
            profiles[profile] = python, validate_checkpoint(spec, runtime)
        workers[model] = profiles[profile]
    return workers


def check_traditional_runtime(models, runtime, neural_device, rscript, r_libs):
    from scripts.fig2.predict_fig2 import check_runtime

    neural = runtime / ".venv-neural" / "bin" / "python"
    if any(model not in R_MODELS for model in models):
        check_runtime(neural, "neural", neural_device)
    if any(model in R_MODELS for model in models):
        local_rscript = runtime / ".r_env" / "bin" / "Rscript"
        rscript = rscript or (str(local_rscript) if local_rscript.is_file() else shutil.which("Rscript"))
        if not rscript:
            raise FileNotFoundError("Rscript is required for Causal Tree and GRF. Set --rscript or add it to PATH.")
        if os.sep in rscript:
            rscript = str(Path(rscript).resolve())
        r_libs = r_libs or Path(os.environ.get("R_LIBS_USER") or runtime / ".r_libs")
        if not r_libs.is_dir():
            raise FileNotFoundError(f"R package directory is missing: {r_libs}. See the Table 2 setup in README.md.")
        r_libs = r_libs.resolve()
    return neural, rscript, r_libs


def predict_inputs(
    inputs, models: list[str], runtime: Path, output: Path,
    *, neural_device: str, rscript: str | None, r_libs: Path | None,
) -> None:
    pretrained = [model for model in models if model not in TRADITIONAL_MODELS]
    traditional = [model for model in models if model in TRADITIONAL_MODELS]
    workers = check_models(pretrained, runtime) if pretrained else {}
    neural, rscript, r_libs = check_traditional_runtime(traditional, runtime, neural_device, rscript, r_libs)
    output.mkdir(parents=True, exist_ok=True)
    scores_by_group = {}
    if pretrained:
        scores_by_group[RCT_ARCHIVE.name] = []
    if traditional:
        scores_by_group[REQUESTED_ARCHIVE.name] = []
    for name in scores_by_group:
        if (output / name).exists():
            raise FileExistsError(f"score CSV already exists at {output / name}. Choose another --output-dir.")
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "Baselines" / "src")]),
        "CAUSALARENA_WORKSPACE": str(runtime),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "PYTHONNOUSERSITE": "1", "TABPFN_DISABLE_TELEMETRY": "1",
        "SKB_DATA_DIRECTORY": os.environ.get("SKB_DATA_DIRECTORY", str(ROOT / "artifacts" / "cache" / "skrub")),
    }
    with TemporaryDirectory(dir=output) as temp:
        paths = Path(temp)
        pending = paths / "results"
        pending.mkdir()
        input_path, output_path = paths / "input.npz", paths / "prediction.npy"
        for dataset, realization, values, query_ids, hashes in inputs:
            np.savez_compressed(input_path, **values)
            for model in models:
                if model in TRADITIONAL_MODELS:
                    python = sys.executable if model in R_MODELS else neural
                    command = [str(python), str(Path(__file__).with_name("traditional_worker.py")), model,
                               str(realization), str(input_path), str(output_path), "--device", neural_device]
                    if model in R_MODELS:
                        command.extend(["--rscript", str(rscript), "--r-libs", str(r_libs)])
                    score_name = REQUESTED_ARCHIVE.name
                    metadata = {"seed": optimization_seeds(model, realization)[0]}
                else:
                    python, checkpoint = workers[model]
                    command = [str(python), str(Path(__file__).resolve()), "_worker", model, str(realization),
                               str(input_path), str(output_path), str(checkpoint)]
                    score_name = RCT_ARCHIVE.name
                    metadata = {}
                output_path.unlink(missing_ok=True)
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
                prediction = np.load(output_path, allow_pickle=False)
                result = pending / dataset / model / f"realization_{realization:03d}.npz"
                result.parent.mkdir(parents=True, exist_ok=True)
                with result.open("xb") as prediction_file:
                    np.savez_compressed(prediction_file, unit_id=query_ids, tau_hat=prediction, **hashes)
                # The negative-control target is a CATE of zero for every query unit.
                scores_by_group[score_name].append({
                    "dataset": dataset, "protocol": "primary", "realization": realization, "model": model,
                    "status": "success", "n_context": len(values["X_context"]), "n_query": len(query_ids),
                    **hashes, **metadata, **dict(zip(METRICS, (
                        np.sqrt(np.mean(np.square(prediction))), abs(prediction.mean()), prediction.std(ddof=0),
                    ), strict=True)),
                })
                print(f"Scored {dataset}/{model}/realization_{realization:03d}", flush=True)
        for name, rows in scores_by_group.items():
            pd.DataFrame(rows).to_csv(pending / name, index=False)
        files = [path.relative_to(pending) for path in pending.rglob("*.npz")]
        files.extend(Path(name) for name in scores_by_group)
        with (output / ".table2.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            for path in files:
                if (output / path).exists():
                    raise FileExistsError(f"Table 2 output already exists: {output / path}")
            for path in files:
                (output / path).parent.mkdir(parents=True, exist_ok=True)
                (pending / path).replace(output / path)
    for name, rows in scores_by_group.items():
        print(f"Scored {len(rows)} predictions against zero CATE. Wrote {output / name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--dataset", choices=tuple(DATASETS), action="append")
    parser.add_argument("--realization", choices=range(40), type=int, action="append", metavar="0–39")
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--neural-device", choices=("cpu", "cuda:0"), default="cuda:0",
                        help="Device used to train TARNet and CFRNet")
    parser.add_argument("--rscript", help="Rscript executable for Causal Tree and GRF")
    parser.add_argument("--r-libs", type=Path, help="Directory containing the pinned R packages")
    parser.add_argument("--output-dir", type=Path, default=FRESH_DIR)
    parser.add_argument("--prepare-only", action="store_true", help="Verify input hashes without loading models")
    args = parser.parse_args()
    datasets = list(dict.fromkeys(args.dataset or DATASETS))
    realizations = list(dict.fromkeys(args.realization or range(40)))
    models = list(dict.fromkeys(args.model or MODELS))
    try:
        provenance = json.loads(PROVENANCE.read_text())
        sources = provenance["raw_inputs"]
        frozen = load_frozen_splits(provenance)
        frames = {dataset: load_dataset(dataset, args.raw_dir, sources[dataset]) for dataset in datasets}
        inputs = []
        for dataset, realization in product(datasets, realizations):
            values, query_ids, hashes = prepare_input(frames[dataset], dataset, realization)
            if hashes != frozen.loc[(dataset, realization)].to_dict():
                raise ValueError(f"input or split hash mismatch for {dataset}/{realization}")
            if not args.prepare_only:
                inputs.append((dataset, realization, values, query_ids, hashes))
        print(f"Verified {len(datasets) * len(realizations)} frozen Table 2 inputs", flush=True)
        if not args.prepare_only:
            predict_inputs(
                inputs, models, args.runtime_root.absolute(), args.output_dir.absolute(),
                neural_device=args.neural_device, rscript=args.rscript, r_libs=args.r_libs,
            )
    except (KeyError, OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Table 2 prediction failed: {error}") from error


if __name__ == "__main__":
    if sys.argv[1:2] == ["_worker"]:
        _, model, realization, source, destination, model_checkpoint = sys.argv[1:]
        run_worker(model, int(realization), Path(source), Path(destination), Path(model_checkpoint))
    else:
        main()
