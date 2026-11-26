"""Run one estimator in its own Python runtime."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np

from .fig2 import CFMS, TABPFN, specialist_seed
from data.integrity import array_hash


def run_cfm(model, values, device, graph, *, dopfn_fd_microbatch=False):
    from .cfm import WORKSPACE, _run_dopfn, _run_dopfn_v2, _run_frontier_v2

    microbatch_size = min(25, len(values["X_query"]))
    dopfn_root = WORKSPACE / "Baselines" / "CFMs" / "Do-PFN-fig2"
    if model == "dopfn_fd_xonly" and not dopfn_fd_microbatch:
        prediction, _ = _run_dopfn(values, device, model_root=dopfn_root)
        return prediction
    if model in {"dopfn", "dopfn_fd_xonly"}:
        prediction, _, _, _ = _run_dopfn_v2(values, device, microbatch_size, model_root=dopfn_root)
        return prediction
    prediction, _, _, _ = _run_frontier_v2(
        CFMS[model], values, device, microbatch_size,
        compute_uq=False, graph_regime=graph,
    )
    return prediction


def run_tabpfn(model, world_id, values, hashes):
    from .tabpfn import fig2 as tabpfn

    route = TABPFN[model]
    runner = {
        "BD": tabpfn._run_bd, "BD_S": tabpfn._run_bd, "BD_DR": tabpfn._run_bd,
        "FD": tabpfn._run_fd, "IV": tabpfn._run_iv, "PX": tabpfn._run_px,
    }[route]
    view = {"payload": values, "route": route, **hashes}
    prediction, _ = runner(view, tabpfn._model_seed(route, world_id), "v3.5")
    return prediction


def run_fd(model, world_id, values):
    from .frontdoor import run

    seed = 20360901 + 1009 * world_id
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        input_path, prediction = directory / "input.npz", directory / "output.npz"
        np.savez_compressed(input_path, **values)
        run(input_path, prediction, directory / "metadata.json", model.removeprefix("fd_"),
            seed, fold_count=5, integration_method="exact", integration_batch_size=8192)
        with np.load(prediction, allow_pickle=False) as archive:
            return archive["tau_hat"]


def run_specialist(model, world_id, values, train_ids, query_ids, train_rows, query_rows):
    train, query = values["X_context"], values["X_query"]
    n_train, n_query = len(train), len(query)
    ordered = np.concatenate([train_rows, query_rows])
    restore_rows = np.argsort(ordered)
    arrays = {
        "X": np.concatenate([train, query])[restore_rows],
        "unit_id": np.concatenate([train_ids, query_ids])[restore_rows],
    }
    for column, key in (("T", "treatment"), ("Y", "outcome"), ("I", "instrument"),
                        ("Z_P", "Z_context"), ("W_P", "W_context")):
        if key in values:
            arrays[column] = np.concatenate([values[key], np.zeros(n_query, dtype=values[key].dtype)])[restore_rows]
    if len(ordered) != n_train + n_query or not np.array_equal(ordered[restore_rows], np.arange(len(ordered))):
        raise ValueError("context/query positions must partition the world")
    if model == "forestdriv":
        from .iv.forestdriv import run as estimator
    else:
        from .proximal.extra_trees import run as estimator
    prediction, _ = estimator(arrays, train_rows, query_rows, specialist_seed(model, world_id))
    return prediction


def run_nn_p_learner(world_id, values, hashes):
    from .proximal.full_nn import compute_full_nn_oof_score, fit_full_nn_final_stage

    seed = specialist_seed("p_learner", world_id)
    score, _ = compute_full_nn_oof_score(
        values["X_context"], values["Z_context"], values["W_context"],
        values["treatment"], values["outcome"], values["X_query"],
        route_seed=seed, **hashes,
    )
    bundle = score.bundle
    final = fit_full_nn_final_stage(
        bundle.x_context, bundle.gamma_oof, bundle.x_query,
        seed=seed + 1_200_000, training_log=[],
    )
    return final.tau_hat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("world", type=int)
    parser.add_argument("graph")
    parser.add_argument("device")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dopfn-fd-microbatch", action="store_true")
    args = parser.parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    if args.model in CFMS or args.model in TABPFN or args.model in {"fd_nn", "p_learner_nn_full"}:
        if args.device == "cuda:0":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError(f"{args.model} requires a visible CUDA GPU for this run")
    with np.load(args.input, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    train_ids, query_ids = values.pop("train_ids"), values.pop("query_ids")
    train_rows, query_rows = values.pop("train_rows"), values.pop("query_rows")
    hashes = {}
    if "Z_context" in values:
        hashes = {
            "input_hash": array_hash(
                train_ids, values["X_context"], values["Z_context"], values["W_context"],
                values["treatment"], values["outcome"], query_ids, values["X_query"],
            ),
            "context_query_split_hash": array_hash(train_rows, query_rows, train_ids, query_ids),
        }
    if args.model in CFMS:
        prediction = run_cfm(args.model, values, args.device, args.graph,
                             dopfn_fd_microbatch=args.dopfn_fd_microbatch)
    elif args.model in TABPFN:
        prediction = run_tabpfn(args.model, args.world, values, hashes)
    elif args.model in {"fd_nn", "fd_xgboost"}:
        prediction = run_fd(args.model, args.world, values)
    elif args.model in {"forestdriv", "p_learner"}:
        prediction = run_specialist(args.model, args.world, values, train_ids, query_ids, train_rows, query_rows)
    elif args.model == "p_learner_nn_full":
        prediction = run_nn_p_learner(args.world, values, hashes)
    elif args.model == "kiv":
        from .iv.kiv import run
        result, _ = run(values, 20360901 + 1009 * args.world)
        prediction = result["tau_hat"]
    else:
        parser.error(f"unknown model {args.model}")
    np.savez_compressed(args.output, tau_hat=np.asarray(prediction, dtype=float).reshape(-1))


if __name__ == "__main__":
    main()
