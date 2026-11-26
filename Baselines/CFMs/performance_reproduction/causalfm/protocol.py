"""Read-only adapter and evidence audit for the official CausalFM release."""

from __future__ import annotations

import json
import math
import random
import sys
import time
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from importlib import import_module, invalidate_caches
from pathlib import Path
from threading import RLock
from types import ModuleType
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from ..common.artifacts import sha256_file
from ..common.schemas import (
    PaperTarget,
    TaskMetric,
    TaskSpec,
    load_paper_targets,
    stable_task_id,
)


PAPER_PROTOCOL = "five_seed_repetitions_fixed_split"
RELEASE_NOTEBOOK_PROTOCOL = "ordered_80_20_single_run"
PAPER_TARGET_STATUS = "not_reproducible_from_release"

_TARGET_PATH = Path(__file__).resolve().parents[1] / "paper_targets.json"
_SYNTHETIC_RELATIVE_PATHS = tuple(
    Path("DATA_standard/standard_cate_TEST") / f"cate_test_dataset_{index}.csv"
    for index in range(1, 11)
)
_JOBS_RELATIVE_PATH = Path("DATA_standard/jobs_data/jobs_data.csv")
_NOTEBOOK_RELATIVE_PATHS = (
    Path("evaluation/notebook/test_standard_cate.ipynb"),
    Path("evaluation/notebook/test_jobs.ipynb"),
)
_CHECKPOINT_RELATIVE_PATH = Path("checkpoints_standard/best_model.pth")
_OFFICIAL_MODULE_PREFIXES = ("src", "DATA_standard", "tabpfn")
_OFFICIAL_IMPORT_LOCK = RLock()


def set_causalfm_seed(seed: int = 0) -> None:
    """Seed every RNG used by the released CausalFM inference path."""

    if seed != 0:
        raise ValueError("CausalFM failure-search seed must be 0")
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch = import_module("torch")
    except ModuleNotFoundError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class CausalFMBatchMetric:
    """Metric calculated from one official loader batch."""

    sample_count: int
    mse: float
    pehe: float
    context_row_count: int
    cate: np.ndarray
    ite: np.ndarray


@dataclass(frozen=True)
class CausalFMReleaseAudit:
    """Immutable evidence describing paper/release protocol availability."""

    paper_protocol: str
    release_notebook_protocol: str
    protocols_match: bool
    paper_target_status: str
    paper_evaluation_seeds: tuple[int, ...]
    paper_seed_evidence: tuple[str, ...]
    file_sha256: Mapping[str, str]
    jobs_notebook_path_matches_release: bool


@dataclass(frozen=True)
class _OfficialImports:
    root: Path

    def import_module(self, name: str) -> ModuleType:
        if not _is_official_module_name(name):
            raise ValueError(f"not an official CausalFM module name: {name}")
        module = import_module(name)
        _validate_official_module_origins(self.root)
        return module


def load_causalfm_targets(path: str | Path = _TARGET_PATH) -> tuple[PaperTarget, ...]:
    """Load only CausalFM targets from the shared strict paper manifest."""

    return tuple(
        target for target in load_paper_targets(path) if target.model == "causalfm"
    )


def build_causalfm_tasks(root: str | Path) -> tuple[TaskSpec, ...]:
    """Build the ten synthetic and one Jobs tasks released by CausalFM."""

    release_root = _release_root(root)
    tasks = [
        _task("synthetic", index, release_root / relative_path)
        for index, relative_path in enumerate(_SYNTHETIC_RELATIVE_PATHS, start=1)
    ]
    tasks.append(_task("jobs", 1, release_root / _JOBS_RELATIVE_PATH))
    missing = [
        task.dataset_path for task in tasks if not Path(task.dataset_path).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"missing official CausalFM dataset: {missing[0]}")
    return tuple(tasks)


def load_causalfm_model(root: str | Path, device: str) -> Any:
    """Strictly load the official standard-CATE model without copying internals."""

    release_root = _release_root(root)
    checkpoint_path = release_root / _CHECKPOINT_RELATIVE_PATH
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"missing official CausalFM checkpoint: {checkpoint_path}"
        )
    torch = import_module("torch")
    with _isolated_official_imports(release_root) as official:
        model_module = official.import_module("src.tabpfn.model.causalFM")
        model = model_module.PerFeatureTransformerCATE(seed=0).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state = (
            checkpoint.get("model_state_dict", checkpoint)
            if isinstance(checkpoint, dict)
            else checkpoint
        )
        model.load_state_dict(state, strict=True)
        model.eval()
    return model


def estimate_causalfm_cate_arrays(
    model: Any,
    X_support: np.ndarray,
    treatment_support: np.ndarray,
    outcome_support: np.ndarray,
    X_query: np.ndarray,
    *,
    device: str = "cuda:0",
    torch_module: Any | None = None,
) -> np.ndarray:
    """Run the released standard-CATE API on explicit float32 CUDA arrays."""

    if device != "cuda:0":
        raise ValueError("failure-search CausalFM must use logical cuda:0")
    torch = import_module("torch") if torch_module is None else torch_module
    X_fit = np.asarray(X_support, dtype=np.float64)
    treatment = np.asarray(treatment_support, dtype=np.float64)
    outcome = np.asarray(outcome_support, dtype=np.float64)
    query = np.asarray(X_query, dtype=np.float64)
    if (
        X_fit.ndim != 2
        or query.ndim != 2
        or X_fit.shape[1] != query.shape[1]
    ):
        raise ValueError(
            "CausalFM support/query covariates must be feature-aligned matrices"
        )
    if treatment.shape != (len(X_fit),) or outcome.shape != (len(X_fit),):
        raise ValueError("CausalFM treatment/outcome must align with support rows")
    if set(np.unique(treatment)) != {0.0, 1.0}:
        raise ValueError("CausalFM support treatment must contain both binary arms")
    if not all(
        np.isfinite(value).all()
        for value in (X_fit, treatment, outcome, query)
    ):
        raise ValueError("CausalFM support/query arrays must be finite")
    tensors = (
        torch.as_tensor(X_fit, dtype=torch.float32, device=device),
        torch.as_tensor(treatment[:, None], dtype=torch.float32, device=device),
        torch.as_tensor(outcome[:, None], dtype=torch.float32, device=device),
        torch.as_tensor(query, dtype=torch.float32, device=device),
    )
    with torch.inference_mode():
        output = model.estimate_cate(*tensors)
    if not isinstance(output, Mapping) or "cate" not in output:
        raise ValueError("CausalFM estimate_cate must return a mapping containing cate")
    prediction = np.asarray(_to_numpy(output["cate"]), dtype=np.float64).reshape(-1)
    if prediction.shape != (len(query),) or not np.isfinite(prediction).all():
        raise ValueError("CausalFM CATE must be a finite query-aligned vector")
    return prediction


def evaluate_causalfm_batch(
    model: Any, batch: Mapping[str, Any], device: str
) -> CausalFMBatchMetric:
    """Apply the released ordered 80/20 protocol to one official loader batch."""

    required = {"X", "a", "y", "ite"}
    missing = sorted(required - set(batch))
    if missing:
        raise ValueError(f"official CausalFM batch is missing: {', '.join(missing)}")
    x = _to_device(batch["X"], device)
    treatment = _to_device(batch["a"], device)
    outcome = _to_device(batch["y"], device)
    ite = _to_device(batch["ite"], device)
    row_count = len(outcome)
    split = int(0.8 * row_count)
    if split <= 0 or split >= row_count:
        raise ValueError("ordered 80/20 split requires at least two rows")

    x_train = x[:split].squeeze(1)
    treatment_train = treatment[:split].squeeze(1)
    outcome_train = outcome[:split].squeeze(1)
    x_test = x[split:].squeeze(1)
    ite_test = ite[split:].squeeze()
    output = model.estimate_cate(x_train, treatment_train, outcome_train, x_test)
    if not isinstance(output, Mapping) or "cate" not in output:
        raise ValueError("estimate_cate must return a mapping containing 'cate'")
    predicted = _to_numpy(output["cate"])
    expected = _to_numpy(ite_test)
    if predicted.shape != expected.shape:
        raise ValueError(
            f"CATE/ITE shape mismatch: predicted {predicted.shape}, expected {expected.shape}"
        )
    if (
        predicted.size == 0
        or not np.isfinite(predicted).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("CATE and held-out ITE must be finite and non-empty")
    squared_error = (predicted - expected) ** 2
    mse = float(np.mean(squared_error))
    return CausalFMBatchMetric(
        sample_count=int(predicted.size),
        mse=mse,
        pehe=math.sqrt(mse),
        context_row_count=split,
        cate=np.array(predicted, copy=True),
        ite=np.array(expected, copy=True),
    )


def evaluate_causalfm_task(model: Any, task: TaskSpec, device: str) -> TaskMetric:
    """Evaluate one task with the official loader and return the released PEHE."""

    return evaluate_causalfm_task_with_predictions(model, task, device)[0]


def evaluate_causalfm_task_with_predictions(
    model: Any, task: TaskSpec, device: str
) -> tuple[TaskMetric, Mapping[str, np.ndarray]]:
    """Evaluate one released task and preserve its real held-out CATE outputs."""

    dataset_path = Path(task.dataset_path).resolve()
    release_root = _find_release_root(dataset_path)
    torch = import_module("torch")
    with _isolated_official_imports(release_root) as official:
        loader_module = official.import_module("DATA_standard.causaldatasettest")
        loader = loader_module.create_test_data_loader(
            str(dataset_path), batch_size=1, shuffle=False, num_workers=0
        )
        inference_context = (
            torch.inference_mode()
            if hasattr(torch, "inference_mode")
            else nullcontext()
        )
        started = time.perf_counter()
        metrics: list[CausalFMBatchMetric] = []
        with inference_context:
            for batch in loader:
                metrics.append(evaluate_causalfm_batch(model, batch, device))
    if len(metrics) != 1:
        raise ValueError(
            f"one CausalFM task must load exactly one batch, got {len(metrics)}"
        )
    result = metrics[0]
    metric = TaskMetric(
        task_id=task.task_id,
        metric="pehe",
        value=result.pehe,
        sample_count=result.sample_count,
        runtime_seconds=time.perf_counter() - started,
    )
    return metric, {"cate": result.cate, "ite": result.ite}


def audit_causalfm_release(
    root: str | Path, targets: Sequence[PaperTarget]
) -> CausalFMReleaseAudit:
    """Hash official inputs and classify the unavailable paper repetition protocol."""

    release_root = _release_root(root)
    _validate_targets(targets)
    required_paths = (
        *_NOTEBOOK_RELATIVE_PATHS,
        *_SYNTHETIC_RELATIVE_PATHS,
        _JOBS_RELATIVE_PATH,
        _CHECKPOINT_RELATIVE_PATH,
    )
    missing = [
        relative
        for relative in required_paths
        if not (release_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing official CausalFM release file: {missing[0].as_posix()}"
        )

    notebook_sources = tuple(
        _notebook_code(release_root / relative) for relative in _NOTEBOOK_RELATIVE_PATHS
    )
    if not all(_is_ordered_80_20_notebook(source) for source in notebook_sources):
        raise ValueError(
            "official CausalFM notebooks do not implement the audited ordered 80/20 protocol"
        )
    hashes = {
        relative.as_posix(): sha256_file(release_root / relative)
        for relative in required_paths
    }
    seed_evidence = _find_paper_seed_evidence(release_root)
    jobs_notebook = notebook_sources[1]
    jobs_notebook_path_matches = (
        "DATA_standard/jobs_data/jobs_data.csv" in jobs_notebook
    )
    return CausalFMReleaseAudit(
        paper_protocol=PAPER_PROTOCOL,
        release_notebook_protocol=RELEASE_NOTEBOOK_PROTOCOL,
        protocols_match=False,
        paper_target_status=PAPER_TARGET_STATUS,
        paper_evaluation_seeds=(),
        paper_seed_evidence=seed_evidence,
        file_sha256=dict(sorted(hashes.items())),
        jobs_notebook_path_matches_release=jobs_notebook_path_matches,
    )


def _task(suite: str, index: int, dataset_path: Path) -> TaskSpec:
    return TaskSpec(
        task_id=stable_task_id("causalfm", suite, index, None),
        model="causalfm",
        suite=suite,
        index=index,
        fold=None,
        dataset_path=str(dataset_path.resolve()),
    )


def _release_root(root: str | Path) -> Path:
    release_root = Path(root).expanduser().resolve()
    if not release_root.is_dir():
        raise FileNotFoundError(
            f"official CausalFM root is not a directory: {release_root}"
        )
    return release_root


def _find_release_root(path: Path) -> Path:
    for candidate in path.parents:
        if (candidate / "DATA_standard").is_dir() and (
            candidate / "checkpoints_standard"
        ).is_dir():
            return candidate
    raise ValueError(f"dataset is not inside an official CausalFM release: {path}")


@contextmanager
def _isolated_official_imports(root: Path) -> Iterator[_OfficialImports]:
    with _OFFICIAL_IMPORT_LOCK:
        previous_path = list(sys.path)
        previous_bytecode_setting = sys.dont_write_bytecode
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if _is_official_module_name(name)
        }
        try:
            _remove_official_modules()
            official_paths = (str(root / "src"), str(root))
            sys.path[:] = [
                *official_paths,
                *(entry for entry in previous_path if entry not in official_paths),
            ]
            sys.dont_write_bytecode = True
            invalidate_caches()
            yield _OfficialImports(root=root)
        finally:
            _remove_official_modules()
            sys.modules.update(saved_modules)
            sys.path[:] = previous_path
            sys.dont_write_bytecode = previous_bytecode_setting
            invalidate_caches()


def _remove_official_modules() -> None:
    for name in tuple(sys.modules):
        if _is_official_module_name(name):
            del sys.modules[name]


def _is_official_module_name(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in _OFFICIAL_MODULE_PREFIXES
    )


def _validate_official_module_origins(root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if not _is_official_module_name(name):
            continue
        origins = _module_origins(module)
        if not origins or any(not _is_relative_to(origin, root) for origin in origins):
            rendered = ", ".join(str(origin) for origin in origins) or "<unknown>"
            raise ImportError(
                f"official CausalFM module {name!r} resolved outside {root}: {rendered}"
            )


def _module_origins(module: ModuleType) -> tuple[Path, ...]:
    file_name = getattr(module, "__file__", None)
    if file_name:
        return (Path(file_name).resolve(),)
    module_path = getattr(module, "__path__", ())
    return tuple(Path(entry).resolve() for entry in module_path)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _to_device(value: Any, device: str) -> Any:
    return value.to(device) if hasattr(value, "to") else np.asarray(value)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _notebook_code(path: Path) -> str:
    try:
        notebook = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"invalid official CausalFM notebook {path}: {error}"
        ) from error
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )


def _is_ordered_80_20_notebook(source: str) -> bool:
    compact = "".join(source.split())
    literal_batch_size = (
        "create_test_data_loader(csv_path,batch_size=1,shuffle=False)" in compact
    )
    configured_batch_size = all(
        marker in compact
        for marker in (
            "batch_size=1",
            "create_test_data_loader(csv_path,batch_size=batch_size,shuffle=False)",
        )
    )
    return (literal_batch_size or configured_batch_size) and all(
        marker in compact
        for marker in (
            "ratio=0.8",
            "int(y.shape[0]*ratio)",
            "x[:train_text_split]",
            "x[train_text_split:]",
            "estimate_cate(",
        )
    )


def _find_paper_seed_evidence(root: Path) -> tuple[str, ...]:
    evidence: list[str] = []
    excluded_parts = {".git", ".venv", "__pycache__"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(part in excluded_parts for part in path.parts):
            continue
        if path.suffix.lower() not in {".py", ".ipynb", ".md", ".txt"}:
            continue
        try:
            text = path.read_text(encoding="utf-8").lower()
        except (OSError, UnicodeDecodeError):
            continue
        if (
            "5 repetitions" in text or "five repetitions" in text
        ) and "random seed" in text:
            evidence.append(path.relative_to(root).as_posix())
    return tuple(evidence)


def _validate_targets(targets: Sequence[PaperTarget]) -> None:
    by_id = {
        target.target_id: target for target in targets if target.model == "causalfm"
    }
    required = {"causalfm.synthetic.pehe", "causalfm.jobs.pehe"}
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(f"missing CausalFM paper target: {missing[0]}")
    for target_id in sorted(required):
        target = by_id[target_id]
        if target.availability != PAPER_TARGET_STATUS:
            raise ValueError(
                f"{target_id} must be unavailable from the official release"
            )
