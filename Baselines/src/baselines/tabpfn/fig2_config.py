"""Published TabPFN options for the Figure 2 estimator routes."""
from __future__ import annotations

import os
from typing import Any

from ..config import BASELINES_ROOT
from .backend import OfficialTabPFNBackend, normalize_tabpfn_version, tabpfn_version_spec


def estimator_config(seed: int, version: str) -> dict[str, Any]:
    version = normalize_tabpfn_version(version)
    spec = tabpfn_version_spec(version)
    return {
        "profile": "default", "seed": seed, "thread_count": 1,
        "tabpfn": {
            "model_version": version,
            "checkpoint_revision": spec["checkpoint_revision"],
            "checkpoint_root": str(BASELINES_ROOT / "TabPFN" / str(spec["checkpoint_directory"])),
            "n_estimators": 8, "n_preprocessing_jobs": 1,
            "device": os.environ.get("TABPFN_DEVICE", "cuda:0"),
            "ignore_pretraining_limits": True,
        },
    }


def fixed_tabpfn_backend(
    seed: int, *, version: str = "v3", device: str = "cuda:0",
) -> tuple[Any, dict[str, Any]]:
    """Construct a pinned frozen TabPFN backend for a supported model version."""
    import tabpfn

    normalized = normalize_tabpfn_version(version)
    spec = tabpfn_version_spec(normalized)

    backend = OfficialTabPFNBackend({
        "tabpfn": {
            "model_version": normalized,
            "checkpoint_revision": spec["checkpoint_revision"],
            "checkpoint_root": str(
                BASELINES_ROOT / "TabPFN" / str(spec["checkpoint_directory"])
            ),
            "n_estimators": 8,
            "n_preprocessing_jobs": 1,
            "device": device,
            "ignore_pretraining_limits": True,
        },
    })
    provenance = {
        **dict(backend.provenance()),
        "package_version": str(tabpfn.__version__),
        "random_state": int(seed),
        "hpo_performed": False,
        "parameter_updates": False,
        "inference_mode": (
            f"frozen pretrained TabPFN-{normalized.removeprefix('v')} "
            "in-context estimation"
        ),
    }
    return backend, provenance
