"""Proximal estimator input contract."""

from data.views import observational_view


MODELS = (
    "causalpfn",
    "dopfn",
    "causalfm",
    "p_learner",
    "p_learner_nn_full",
    "p_learner_tabpfn_v3_full",
)


def inputs(world):
    return observational_view(world, "PROX")
