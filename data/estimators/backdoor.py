"""Back-door estimator input contract."""

from data.views import observational_view


MODELS = ("causalpfn", "dopfn", "causalfm", "tabpfn_x", "xgb_x", "dr_learner")


def inputs(world):
    return observational_view(world, "BD")
