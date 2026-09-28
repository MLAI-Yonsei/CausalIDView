"""Front-door estimator input contract."""

from data.views import observational_view


MODELS = ("causalpfn_fd_xonly", "dopfn_fd_xonly", "causalfm", "fd_nn", "fd_xgboost", "tabpfn_fd")


def inputs(world):
    return observational_view(world, "FD")
