"""Instrumental-variable estimator input contract."""

from data.views import observational_view


MODELS = ("causalpfn", "dopfn", "causalfm", "forestdriv", "kiv", "wald_tabpfn")


def inputs(world):
    return observational_view(world, "IV")
