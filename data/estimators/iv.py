"""Instrumental-variable estimator input contract."""

from data.views import observational_view


MODELS = ("causalpfn", "dopfn", "causalfm", "wald_tabpfn", "forestdriv", "kiv")


def inputs(world):
    return observational_view(world, "IV")
