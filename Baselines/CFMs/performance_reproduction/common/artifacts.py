"""Atomic, inspectable result artifacts for paper-reproduction tasks."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import TaskMetric, TaskSpec, stable_task_id


_RESUME_HASH_KEYS = (
    "code_sha256",
    "checkpoint_sha256",
    "data_sha256",
    "command_sha256",
    "task_manifest_sha256",
)
_ARTIFACT_FILES = ("predictions.npz", "task.json", "metrics.json", "provenance.json", "run.log")
_COMPLETION_FILE = "completion.json"
_PAYLOAD_ROOT = ".payloads"
_LOCK_ROOT = ".locks"


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_publish_task(
    output_root: str | Path,
    task: TaskSpec,
    predictions: Mapping[str, Any],
    metrics: Sequence[TaskMetric],
    provenance: Mapping[str, Any],
    log_text: str,
    *,
    overwrite: bool = False,
) -> Path:
    """Publish a validated immutable payload through an atomic completion pointer.

    A reader only follows ``completion.json``. Replacing that small file is atomic
    on the supported filesystem, so an overwrite leaves readers on either the old
    complete payload or the new complete payload—never an empty task directory.
    """

    _validate_task(task)
    ordered_predictions = _validate_predictions(predictions)
    metrics_payload = _validate_metrics(metrics, task.task_id)
    provenance_payload = _validate_provenance(provenance)
    if not isinstance(log_text, str):
        raise ValueError("log_text must be a string")

    tasks_root = Path(output_root) / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    task_dir = tasks_root / task.task_id
    completion_path = task_dir / _COMPLETION_FILE
    with _task_claim(tasks_root, task.task_id):
        if task_dir.exists() and not task_dir.is_dir():
            raise ValueError(f"task artifact path is not a directory: {task_dir}")
        if completion_path.exists() and not overwrite:
            raise FileExistsError(f"task artifact already exists: {task_dir}")

        token = f"{os.getpid()}-{uuid.uuid4().hex}"
        staged_payload = tasks_root / f".{task.task_id}.tmp-{token}"
        payload_dir = tasks_root / _PAYLOAD_ROOT / task.task_id / token
        pointer_temp = task_dir / f".{_COMPLETION_FILE}.tmp-{token}"
        created_task_dir = False
        moved_payload = False
        try:
            staged_payload.mkdir()
            np.savez_compressed(staged_payload / "predictions.npz", **ordered_predictions)
            _write_json(staged_payload / "task.json", asdict(task))
            _write_json(staged_payload / "metrics.json", {"metrics": metrics_payload})
            _write_json(staged_payload / "provenance.json", provenance_payload)
            (staged_payload / "run.log").write_text(log_text, encoding="utf-8")
            _validate_serialized_artifacts(staged_payload, task.task_id)

            payload_dir.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_payload, payload_dir)
            moved_payload = True
            _validate_serialized_artifacts(payload_dir, task.task_id)
            if not task_dir.exists():
                task_dir.mkdir()
                created_task_dir = True
            _write_json(pointer_temp, {"version": token})
            os.replace(pointer_temp, completion_path)
        finally:
            if pointer_temp.exists():
                pointer_temp.unlink()
            if staged_payload.exists():
                shutil.rmtree(staged_payload)
            if moved_payload and payload_dir.exists() and not _payload_is_current(task_dir, payload_dir):
                shutil.rmtree(payload_dir)
            if created_task_dir and task_dir.exists() and not any(task_dir.iterdir()):
                task_dir.rmdir()

    return task_dir


def task_is_resumable(task_dir: str | Path, expected_hashes: Mapping[str, str]) -> bool:
    """Return whether the current payload is complete and matches all evidence."""

    if set(expected_hashes) != set(_RESUME_HASH_KEYS):
        return False
    if not all(_is_sha256(value) for value in expected_hashes.values()):
        return False
    try:
        directory = Path(task_dir)
        payload_dir = _resolve_payload(directory)
        _, _, provenance = _validate_serialized_artifacts(payload_dir, directory.name)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return all(provenance.get(key) == value for key, value in expected_hashes.items())


@contextmanager
def _task_claim(tasks_root: Path, task_id: str):
    """Serialize publication decisions; flock releases automatically on process death."""

    lock_dir = tasks_root / _LOCK_ROOT
    lock_dir.mkdir(exist_ok=True)
    descriptor = os.open(lock_dir / f"{task_id}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _payload_is_current(task_dir: Path, payload_dir: Path) -> bool:
    try:
        return _resolve_payload(task_dir) == payload_dir
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _resolve_payload(task_dir: Path) -> Path:
    if not task_dir.is_dir():
        raise ValueError("task directory is missing")
    pointer = _read_json_object(task_dir / _COMPLETION_FILE, "completion pointer")
    if set(pointer) != {"version"} or not isinstance(pointer["version"], str):
        raise ValueError("completion pointer must contain only a version")
    version = pointer["version"]
    if not version or Path(version).name != version:
        raise ValueError("completion pointer version is invalid")
    payload_dir = task_dir.parent / _PAYLOAD_ROOT / task_dir.name / version
    if not payload_dir.is_dir():
        raise ValueError("completion pointer payload is missing")
    return payload_dir


def _validate_serialized_artifacts(
    payload_dir: Path, expected_task_id: str
) -> tuple[TaskSpec, tuple[TaskMetric, ...], dict[str, Any]]:
    if not payload_dir.is_dir() or not all((payload_dir / name).is_file() for name in _ARTIFACT_FILES):
        raise ValueError("serialized artifact set is incomplete")
    task = _decode_task(_read_json_object(payload_dir / "task.json", "task"))
    if task.task_id != expected_task_id:
        raise ValueError("serialized task_id does not match task directory")
    metrics = _decode_metrics(_read_json_object(payload_dir / "metrics.json", "metrics"), task.task_id)
    predictions = _read_predictions(payload_dir / "predictions.npz")
    sample_count = next(iter(predictions.values())).shape[0]
    if any(metric.sample_count != sample_count for metric in metrics):
        raise ValueError("metric sample_count does not match predictions")
    provenance = _validate_provenance(
        _read_json_object(payload_dir / "provenance.json", "provenance")
    )
    try:
        (payload_dir / "run.log").read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("run.log must be UTF-8 text") from error
    return task, metrics, provenance


def _decode_task(payload: dict[str, Any]) -> TaskSpec:
    if set(payload) != set(TaskSpec.__dataclass_fields__):
        raise ValueError("serialized task schema is invalid")
    try:
        task = TaskSpec(**payload)
    except TypeError as error:
        raise ValueError("serialized task schema is invalid") from error
    _validate_task(task)
    return task


def _decode_metrics(payload: dict[str, Any], task_id: str) -> tuple[TaskMetric, ...]:
    if set(payload) != {"metrics"} or not isinstance(payload["metrics"], list):
        raise ValueError("serialized metrics schema is invalid")
    try:
        metrics = tuple(TaskMetric(**item) for item in payload["metrics"] if isinstance(item, dict))
    except TypeError as error:
        raise ValueError("serialized metrics schema is invalid") from error
    if len(metrics) != len(payload["metrics"]):
        raise ValueError("serialized metrics schema is invalid")
    _validate_metrics(metrics, task_id)
    return metrics


def _read_predictions(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            predictions = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, EOFError) as error:
        raise ValueError("serialized predictions artifact is invalid") from error
    return _validate_predictions(predictions)


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"serialized {label} is invalid") from error
    if not isinstance(payload, dict):
        raise ValueError(f"serialized {label} must be an object")
    return payload


def _validate_task(task: TaskSpec) -> None:
    if not isinstance(task, TaskSpec):
        raise ValueError("task must be a TaskSpec")
    for field in ("task_id", "model", "suite", "dataset_path"):
        value = getattr(task, field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"task {field} must be a non-empty string")
    if Path(task.task_id).name != task.task_id:
        raise ValueError("task_id must be a single path component")
    if not _is_nonnegative_int(task.index):
        raise ValueError("task index must be a non-negative integer")
    if task.fold is not None and not _is_nonnegative_int(task.fold):
        raise ValueError("task fold must be a non-negative integer or None")
    if task.task_id != stable_task_id(task.model, task.suite, task.index, task.fold):
        raise ValueError("task_id is inconsistent with task fields")


def _validate_predictions(predictions: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(predictions, Mapping) or not predictions:
        raise ValueError("predictions must be a non-empty mapping")
    arrays: dict[str, np.ndarray] = {}
    sample_count: int | None = None
    for name in sorted(predictions):
        if not isinstance(name, str) or not name:
            raise ValueError("prediction names must be non-empty strings")
        array = np.asarray(predictions[name])
        if array.ndim == 0 or array.shape[0] == 0:
            raise ValueError("prediction arrays must have a non-empty sample dimension")
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("prediction arrays must contain only finite numeric values")
        if sample_count is None:
            sample_count = int(array.shape[0])
        elif array.shape[0] != sample_count:
            raise ValueError("prediction arrays must share their sample dimension")
        arrays[name] = array
    return arrays


def _validate_metrics(metrics: Sequence[TaskMetric], task_id: str) -> list[dict[str, Any]]:
    if isinstance(metrics, (str, bytes)) or not isinstance(metrics, Sequence) or not metrics:
        raise ValueError("metrics must be a non-empty sequence of TaskMetric records")
    payload: list[dict[str, Any]] = []
    for metric in metrics:
        if not isinstance(metric, TaskMetric):
            raise ValueError("metrics must contain TaskMetric records")
        if metric.task_id != task_id:
            raise ValueError("metric task_id must match task")
        if not isinstance(metric.metric, str) or not metric.metric:
            raise ValueError("metric name must be non-empty")
        if not _is_finite_number(metric.value):
            raise ValueError("metric value must be finite")
        if not _is_nonnegative_int(metric.sample_count):
            raise ValueError("metric sample_count must be a non-negative integer")
        if not _is_finite_number(metric.runtime_seconds) or metric.runtime_seconds < 0:
            raise ValueError("metric runtime_seconds must be finite and non-negative")
        payload.append(asdict(metric))
    return payload


def _validate_provenance(provenance: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")
    missing = [key for key in _RESUME_HASH_KEYS if key not in provenance]
    if missing:
        raise ValueError(f"provenance is missing required keys: {', '.join(missing)}")
    if not all(_is_sha256(provenance[key]) for key in _RESUME_HASH_KEYS):
        raise ValueError("required provenance hashes must be lowercase SHA-256 hex digests")
    payload = dict(provenance)
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"provenance must be finite JSON data: {error}") from error
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
