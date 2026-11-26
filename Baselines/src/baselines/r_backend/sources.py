"""The R package pins used by the original Table 2 implementation."""

SOURCES = {
    "causal_tree": {
        "backend": "rscript-causalTree",
        "implementation_variant": "adaptive-non-honest-causal-tree",
        "package_version": "0.0+git.51a8762",
        "commit_sha": "51a8762bff0ab17b971bce852cb37bece2d8240f",
    },
    "grf": {
        "backend": "rscript-grf",
        "implementation_variant": "official-grf-causal-forest",
        "package_version": "2.6.1",
        "commit_sha": "5bee99b51471f76cb2d63acbc8a9b0ffec408ba0",
    },
}


def get_source_record(model_id: str) -> dict[str, str]:
    return SOURCES[model_id].copy()
