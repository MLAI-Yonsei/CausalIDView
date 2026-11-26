"""Read-only Do-PFN synthetic datasets and immutable task manifests."""

from __future__ import annotations

import hashlib
import pickle
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from ..common.schemas import TaskSpec, stable_task_id


DOPFN_MAIN_CASES = (
    "Observed_Confounder",
    "Backdoor_Criterion",
    "Frontdoor_Criterion",
    "Observed_Mediator",
    "Observed_Mediator_and_Confounder",
    "Unobserved_Confounder",
)
DOPFN_ROOT = Path(__file__).resolve().parents[2] / "Do-PFN"


def build_dopfn_synthetic_tasks(root: str | Path = DOPFN_ROOT) -> tuple[TaskSpec, ...]:
    """Build exactly the six 100-realization main-figure suites."""

    release = Path(root).resolve()
    if not release.is_dir():
        raise FileNotFoundError(f"missing Do-PFN release root: {release}")
    tasks: list[TaskSpec] = []
    for suite in DOPFN_MAIN_CASES:
        case_dir = release / "data" / "prior_sampling" / suite
        expected = [case_dir / f"{suite}_{index}.pkl" for index in range(1, 101)]
        if not case_dir.is_dir() or any(not path.is_file() for path in expected):
            raise FileNotFoundError(f"Do-PFN suite does not contain exactly 1..100: {suite}")
        extras = sorted(case_dir.glob("*.pkl"))
        if set(extras) != set(expected):
            raise ValueError(f"Do-PFN suite contains unexpected pickle files: {suite}")
        for index, path in enumerate(expected, start=1):
            dataset_path = str(path.relative_to(release))
            tasks.append(
                TaskSpec(
                    task_id=stable_task_id("dopfn", suite, index, None),
                    model="dopfn",
                    suite=suite,
                    index=index,
                    fold=None,
                    dataset_path=dataset_path,
                )
            )
    return tuple(tasks)


def load_official_pickle(task: TaskSpec, root: str | Path = DOPFN_ROOT) -> Any:
    """Load one vendored pickle without rewriting or repickling it."""

    if task.model != "dopfn":
        raise ValueError(f"expected Do-PFN task, got {task.model!r}")
    path = Path(root).resolve() / task.dataset_path
    if not path.is_file():
        raise FileNotFoundError(f"missing Do-PFN input: {path}")
    with _release_context(Path(root).resolve()):
        with path.open("rb") as handle:
            dataset = pickle.load(handle)
    validate_interventional_dataset(dataset)
    return dataset


def pickle_sha256(task: TaskSpec, root: str | Path = DOPFN_ROOT) -> str:
    path = Path(root).resolve() / task.dataset_path
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_interventional_dataset(dataset: Any) -> None:
    """Require finite, aligned observational/interventional arrays."""

    fields = ("x_obs", "x_int", "y_obs", "y_int", "cate")
    for field in fields:
        if not hasattr(dataset, field):
            raise ValueError(f"Do-PFN dataset is missing {field}")
    arrays = {field: _as_numpy(getattr(dataset, field), field) for field in fields}
    if arrays["x_obs"].ndim != 2 or arrays["x_int"].ndim != 2:
        raise ValueError("Do-PFN x_obs/x_int must be two-dimensional")
    if arrays["x_obs"].shape != arrays["x_int"].shape or arrays["x_obs"].shape[1] < 1:
        raise ValueError("Do-PFN observational/interventional features are misaligned")
    n_rows = arrays["x_obs"].shape[0]
    for field in ("y_obs", "y_int", "cate"):
        if arrays[field].reshape(-1).shape[0] != n_rows:
            raise ValueError(f"Do-PFN {field} is not aligned with features")
    treatment = arrays["x_obs"][:, 0]
    if not np.isfinite(treatment).all() or not np.isin(treatment, (0.0, 1.0)).all():
        raise ValueError("Do-PFN treatment column must be finite binary values")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Do-PFN dataset contains non-finite values")


def _as_numpy(value: Any, field: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Do-PFN {field} must be numeric") from error


@contextmanager
def _release_context(root: Path):
    previous_cwd = Path.cwd()
    root_text = str(root)
    sys.path.insert(0, root_text)
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass
