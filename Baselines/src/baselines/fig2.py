"""Figure 2 input roles and estimator selection shared by the parent and workers."""
from __future__ import annotations

import numpy as np

from data.views import observational_view

CFMS = {
    "causalpfn": "causalpfn", "causalpfn_fd_xonly": "causalpfn",
    "dopfn": "dopfn", "dopfn_fd_xonly": "dopfn", "causalfm": "causalfm",
}
TABPFN = {
    "tabpfn_x": "BD", "tabpfn_s": "BD_S", "tabpfn_dr": "BD_DR",
    "tabpfn_fd": "FD", "wald_tabpfn": "IV", "p_learner_tabpfn_v3_full": "PX",
}
CLASSICAL = {"forestdriv", "kiv", "p_learner", "fd_xgboost"}
NEURAL = {"fd_nn", "p_learner_nn_full"}
MODEL_IDS = set(CFMS) | set(TABPFN) | CLASSICAL | NEURAL


def runtime(model, regime):
    if model in CFMS:
        backend = CFMS[model]
        checkpoint = backend
        if backend == "causalfm":
            checkpoint = {"FD": "causalfm_fd", "IV": "causalfm_iv_binary"}.get(
                regime, "causalfm_standard",
            )
        return backend, checkpoint
    if model in TABPFN:
        return "tabpfn", "tabpfn_v35"
    if model in CLASSICAL:
        return "classical", None
    if model in NEURAL:
        return "neural", None
    raise ValueError(f"unsupported Figure 2 model {model}")


def model_input(world, regime, model):
    view = observational_view(world, regime)
    context, query = view.context, view.query
    graph = "backdoor"
    extra = ()
    if model in {"causalpfn", "dopfn", "causalfm"}:
        extra = {"BD": ("U",), "IV": ("I",), "PROX": ("Z_P", "W_P")}.get(regime, ())
    if model in {"tabpfn_x", "tabpfn_s", "tabpfn_dr"}:
        extra = ("U",)
    if model == "causalfm" and regime in {"FD", "IV"}:
        graph = "frontdoor" if regime == "FD" else "conditional_iv"
        extra = ()
    values = {
        "X_context": np.column_stack([context["X"], *(context[key] for key in extra)]),
        "X_query": np.column_stack([query["X"], *(query[key] for key in extra)]),
        "treatment": context["T"].astype(float),
        "outcome": context["Y"],
    }
    if graph != "backdoor":
        values["graph_aux_context"] = context["M" if regime == "FD" else "I"].astype(float)
    if model in {"fd_nn", "fd_xgboost", "tabpfn_fd"}:
        values["mediator"] = context["M"]
    if model in {"forestdriv", "kiv", "wald_tabpfn"}:
        values["instrument"] = context["I"]
    if regime == "PROX" and model not in CFMS:
        values.update(Z_context=context["Z_P"], W_context=context["W_P"], treatment=context["T"])
    return values, graph, context["unit_id"], query["unit_id"]


# Offsets from the original Figure 2 run.
SPECIALIST_SEED_OFFSETS = {"forestdriv": 2, "p_learner": 4}


def specialist_seed(model: str, world_id: int) -> int:
    if model not in SPECIALIST_SEED_OFFSETS:
        raise ValueError(model)
    return 20260901 + 1009 * int(world_id) + 100000 * SPECIALIST_SEED_OFFSETS[model]
