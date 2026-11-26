from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol

from ..config import BASELINES_ROOT
from ..exceptions import DependencyUnavailableError


TABPFN_CODE_VERSION = "8.2.0"
TABPFN_CHECKPOINT_REVISION = "24a16a89d245878b846555110985634aa2e656d7"
CLASSIFIER_FILENAME = "tabpfn-v3-classifier-v3_default.ckpt"
REGRESSOR_FILENAME = "tabpfn-v3-regressor-v3_default.ckpt"
TABPFN_V35_CODE_VERSION = "9.0.0"
TABPFN_V35_CHECKPOINT_REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
TABPFN_V35_FILENAME = "tabpfn-v3.5-20260909.safetensors"


TABPFN_VERSION_SPECS: dict[str, dict[str, Any]] = {
    "v3": {
        "package_version": TABPFN_CODE_VERSION,
        "checkpoint_revision": TABPFN_CHECKPOINT_REVISION,
        "model_identifier": "Prior-Labs/tabpfn_3",
        "classifier_filename": CLASSIFIER_FILENAME,
        "regressor_filename": REGRESSOR_FILENAME,
        "checkpoint_directory": "tabpfn_3_weights",
    },
    "v3.5": {
        "package_version": TABPFN_V35_CODE_VERSION,
        "checkpoint_revision": TABPFN_V35_CHECKPOINT_REVISION,
        "model_identifier": "Prior-Labs/tabpfn_3_5",
        "classifier_filename": TABPFN_V35_FILENAME,
        "regressor_filename": TABPFN_V35_FILENAME,
        "checkpoint_directory": "tabpfn_3_5_weights",
    },
}


def normalize_tabpfn_version(value: Any) -> str:
    normalized = str(value).strip().lower().replace("tabpfn-", "")
    aliases = {"3": "v3", "v3": "v3", "3.5": "v3.5", "v3.5": "v3.5", "v3_5": "v3.5"}
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"unsupported TabPFN version: {value!r}") from error


def tabpfn_version_spec(value: Any) -> Mapping[str, Any]:
    return TABPFN_VERSION_SPECS[normalize_tabpfn_version(value)]


def resolve_tabpfn_checkpoint_root(config: Mapping[str, Any]) -> Path:
    """Resolve the pinned checkpoint directory used by production and tests."""

    tabpfn_config = config.get("tabpfn", {})
    configured_root = tabpfn_config.get("checkpoint_root") or os.environ.get(
        "TABPFN_CHECKPOINT_ROOT"
    )
    if configured_root:
        return Path(configured_root)
    spec = tabpfn_version_spec(tabpfn_config.get("model_version", "v3"))
    return BASELINES_ROOT / "TabPFN" / str(spec["checkpoint_directory"])


class TabPFNBackend(Protocol):
    backend_name: str

    def create_classifier(self, *, random_state: int, role: str) -> Any:
        ...

    def create_regressor(self, *, random_state: int, role: str) -> Any:
        ...

    def effective_feature_budget(self, estimator: Any) -> int:
        ...

    def provenance(self) -> Mapping[str, Any]:
        ...


class OfficialTabPFNBackend:
    backend_name = "official-tabpfn-8.2.0-v3"

    def __init__(
        self, config: Mapping[str, Any], *, package_version: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        tabpfn_config = config.get("tabpfn", {})
        self.model_version = normalize_tabpfn_version(
            tabpfn_config.get("model_version", "v3")
        )
        self.version_spec = dict(tabpfn_version_spec(self.model_version))
        self.package_version = package_version or str(self.version_spec["package_version"])
        self.model_identifier = str(self.version_spec["model_identifier"])
        self.classifier_filename = str(self.version_spec["classifier_filename"])
        self.regressor_filename = str(self.version_spec["regressor_filename"])
        if checkpoint_path is not None and self.classifier_filename != self.regressor_filename:
            raise ValueError("An explicit checkpoint path requires a shared classifier/regressor checkpoint")
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            self.classifier_filename = self.regressor_filename = checkpoint_path.name
        self.backend_name = f"official-tabpfn-{self.package_version}-{self.model_version}"
        self.checkpoint_root = resolve_tabpfn_checkpoint_root(config)
        self.checkpoint_revision = str(
            tabpfn_config.get(
                "checkpoint_revision", self.version_spec["checkpoint_revision"]
            )
        )
        self.device = str(tabpfn_config.get("device", "cpu"))
        self.n_estimators = int(tabpfn_config.get("n_estimators", 8))
        self.n_preprocessing_jobs = int(tabpfn_config.get("n_preprocessing_jobs", 1))
        self.ignore_pretraining_limits = bool(
            tabpfn_config.get("ignore_pretraining_limits", False)
        )

    def effective_feature_budget(self, estimator: Any) -> int:
        """Read the per-estimator feature cap from the pinned checkpoint config."""

        inference_config = estimator.get_inference_config()
        transforms = inference_config.PREPROCESS_TRANSFORMS
        budgets = {
            int(transform.max_features_per_estimator)
            for transform in transforms
            if transform.max_features_per_estimator is not None
        }
        if not transforms or min(budgets, default=sys.maxsize) < 2:
            raise DependencyUnavailableError(
                "Pinned TabPFN inference config has no usable feature budget"
            )
        # None means that the preprocessing transform does not subsample features.
        return min(budgets, default=sys.maxsize)

    def provenance(self) -> Mapping[str, Any]:
        return {
            "base_learner": {
                "family": "tabpfn",
                "backend": self.backend_name,
                "package": "tabpfn",
                "package_version": self.package_version,
            },
            "tabpfn_version": self.model_version.removeprefix("v"),
            "tabpfn_model_identifier": self.model_identifier,
            "tabpfn_revision": self.checkpoint_revision,
            "tabpfn_ensemble_size": self.n_estimators,
            "tabpfn_device": self.device,
            "tabpfn_checkpoint_filenames": {
                "classifier": self.classifier_filename,
                "regressor": self.regressor_filename,
            },
        }

    def _check_runtime(self) -> None:
        try:
            installed_version = version("tabpfn")
        except PackageNotFoundError as error:
            raise DependencyUnavailableError(
                f"TabPFN {self.package_version} is not installed for {self.model_version}"
            ) from error
        if installed_version != self.package_version:
            raise DependencyUnavailableError(
                f"TabPFN {self.package_version} is required for {self.model_version}, "
                f"found {installed_version}"
            )
        expected_revision = str(self.version_spec["checkpoint_revision"])
        if self.checkpoint_revision != expected_revision:
            raise DependencyUnavailableError(
                f"TabPFN {self.model_version} checkpoint revision must be "
                + expected_revision
            )

    def _checkpoint(self, filename: str) -> Path:
        self._check_runtime()
        path = self.checkpoint_path or self.checkpoint_root / filename
        if not path.is_file():
            raise DependencyUnavailableError(
                f"Missing pinned TabPFN {self.model_version} checkpoint "
                f"{filename}; fetch {self.model_identifier} revision "
                f"{self.version_spec['checkpoint_revision']}"
            )
        return path

    def _prepare_device_environment(self) -> None:
        if self.device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")

    def create_classifier(self, *, random_state: int, role: str) -> Any:
        self._prepare_device_environment()
        try:
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
        except ImportError as error:
            raise DependencyUnavailableError(
                f"TabPFN {self.package_version} import failed"
            ) from error
        return TabPFNClassifier.create_default_for_version(
            ModelVersion(self.model_version),
            model_path=self._checkpoint(self.classifier_filename),
            device=self.device,
            n_estimators=self.n_estimators,
            auto_scale_n_estimators=False,
            random_state=random_state,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            show_progress_bar=False,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )

    def create_regressor(self, *, random_state: int, role: str) -> Any:
        self._prepare_device_environment()
        try:
            from tabpfn import TabPFNRegressor
            from tabpfn.constants import ModelVersion
        except ImportError as error:
            raise DependencyUnavailableError(
                f"TabPFN {self.package_version} import failed"
            ) from error
        return TabPFNRegressor.create_default_for_version(
            ModelVersion(self.model_version),
            model_path=self._checkpoint(self.regressor_filename),
            device=self.device,
            n_estimators=self.n_estimators,
            auto_scale_n_estimators=False,
            random_state=random_state,
            n_preprocessing_jobs=self.n_preprocessing_jobs,
            show_progress_bar=False,
            ignore_pretraining_limits=self.ignore_pretraining_limits,
        )
