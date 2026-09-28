"""Back-door estimator input contract."""

from data.views import observational_view


MODELS = ("causalpfn", "dopfn", "causalfm", "tabpfn_x", "tabpfn_s", "tabpfn_dr")


def inputs(world):
    return observational_view(world, "BD")
