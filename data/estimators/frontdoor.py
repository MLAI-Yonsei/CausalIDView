"""Front-door estimator input contract."""

from data.views import observational_view


MODELS = ("causalpfn_fd_xonly", "dopfn_fd_xonly", "causalfm", "tabpfn_fd", "fd_nn", "fd_xgboost")


def inputs(world):
    return observational_view(world, "FD")
