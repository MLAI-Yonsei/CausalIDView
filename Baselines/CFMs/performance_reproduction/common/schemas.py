"""Immutable records and strict JSON loading for paper-reproduction evidence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any


class ReproductionStatus(str, Enum):
    """The fixed outcomes used to classify a published-result comparison."""

    EXACT_MATCH = "exact_match"
    STATISTICALLY_COMPATIBLE = "statistically_compatible"
    MISMATCH = "mismatch"
    NOT_REPRODUCIBLE_FROM_RELEASE = "not_reproducible_from_release"
    DEFERRED_TO_NON_CFM = "deferred_to_non_cfm"


@dataclass(frozen=True)
class PaperTarget:
    target_id: str
    model: str
    paper_version: str
    source_url: str
    result_locator: str
    suite: str
    metric: str
    target_display: str | None
    target_value: float | None
    displayed_decimals: int | None
    uncertainty_value: float | None
    uncertainty_kind: str | None
    unit_scale: float
    availability: str
    source_sha256: str | None = None
    table_number: int | None = None
    dataset_column: str | None = None
    uncertainty_display: str | None = None
    comparison_availability: str | None = None
    table_caption: str | None = None


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    model: str
    suite: str
    index: int
    fold: int | None
    dataset_path: str


@dataclass(frozen=True)
class TaskMetric:
    task_id: str
    metric: str
    value: float
    sample_count: int
    runtime_seconds: float


@dataclass(frozen=True)
class Comparison:
    target_id: str
    observed_value: float | None
    absolute_difference: float | None
    status: ReproductionStatus
    reason: str


_PAPER_TARGET_KEYS = frozenset(PaperTarget.__dataclass_fields__)
_PAPER_TARGET_REQUIRED_KEYS = _PAPER_TARGET_KEYS - {
    "source_sha256",
    "table_number",
    "dataset_column",
    "uncertainty_display",
    "comparison_availability",
    "table_caption",
}
_TASK_SPEC_KEYS = frozenset(TaskSpec.__dataclass_fields__)
_UNCERTAINTY_KINDS = frozenset(
    {"standard_error", "standard_deviation", "confidence_interval_95"}
)


def stable_task_id(model: str, suite: str, index: int, fold: int | None) -> str:
    """Create the stable ID shared by task manifests and result artifacts."""

    task_id = f"{model}__{suite}__{index:03d}"
    return task_id if fold is None else f"{task_id}__fold-{fold:02d}"


def load_paper_targets(path: str | Path) -> tuple[PaperTarget, ...]:
    """Load a strict target manifest without silently accepting changed evidence."""

    payload = _load_manifest(path, "targets")
    targets = tuple(_parse_paper_target(item, position) for position, item in enumerate(payload))
    _reject_duplicate_ids((target.target_id for target in targets), "target_id")
    return targets


def load_task_specs(path: str | Path) -> tuple[TaskSpec, ...]:
    """Load a strict task manifest and verify every declared stable task ID."""

    payload = _load_manifest(path, "tasks")
    tasks = tuple(_parse_task_spec(item, position) for position, item in enumerate(payload))
    _reject_duplicate_ids((task.task_id for task in tasks), "task_id")
    return tasks


def _load_manifest(path: str | Path, collection_key: str) -> list[Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON manifest {path!s}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    _reject_unknown_keys(payload, {collection_key}, "manifest")
    if collection_key not in payload:
        raise ValueError(f"manifest requires {collection_key!r}")
    collection = payload[collection_key]
    if not isinstance(collection, list):
        raise ValueError(f"manifest {collection_key!r} must be a list")
    return collection


def _parse_paper_target(value: Any, position: int) -> PaperTarget:
    item = _require_object(value, f"targets[{position}]")
    _reject_unknown_keys(item, _PAPER_TARGET_KEYS, f"targets[{position}]")

    target_value = _optional_finite_number(item.get("target_value"), "target_value")
    target_display = _optional_string(item.get("target_display"), "target_display")
    displayed_decimals = _optional_nonnegative_int(
        item.get("displayed_decimals"), "displayed_decimals"
    )
    if target_value is not None and target_display is None:
        raise ValueError("numeric target_value requires target_display")
    if target_value is not None and displayed_decimals is None:
        raise ValueError("numeric target_value requires displayed_decimals")
    if target_value is None and displayed_decimals is not None:
        raise ValueError("displayed_decimals requires numeric target_value")

    _require_keys(item, _PAPER_TARGET_REQUIRED_KEYS, f"targets[{position}]")
    uncertainty_value = _optional_finite_number(
        item["uncertainty_value"], "uncertainty_value"
    )
    if uncertainty_value is not None and uncertainty_value < 0:
        raise ValueError("uncertainty_value must be non-negative")
    uncertainty_kind = _optional_string(item["uncertainty_kind"], "uncertainty_kind")
    if uncertainty_kind is not None and uncertainty_kind not in _UNCERTAINTY_KINDS:
        raise ValueError(f"unsupported uncertainty_kind: {uncertainty_kind!r}")
    if uncertainty_value is None and uncertainty_kind is not None:
        raise ValueError("uncertainty_kind requires uncertainty_value")
    if uncertainty_value is not None and uncertainty_kind is None:
        raise ValueError("uncertainty_value requires uncertainty_kind")
    uncertainty_display = _optional_string(
        item.get("uncertainty_display"), "uncertainty_display"
    )
    if uncertainty_display is not None:
        if uncertainty_value is None:
            raise ValueError("uncertainty_display requires uncertainty_value")
        _validate_numeric_display(
            uncertainty_display, uncertainty_value, "uncertainty_display"
        )

    comparison_availability = _optional_string(
        item.get("comparison_availability"), "comparison_availability"
    )
    if comparison_availability is not None and comparison_availability not in {
        ReproductionStatus.NOT_REPRODUCIBLE_FROM_RELEASE.value,
        ReproductionStatus.DEFERRED_TO_NON_CFM.value,
    }:
        raise ValueError(
            f"unsupported comparison_availability: {comparison_availability!r}"
        )

    unit_scale = _finite_number(item["unit_scale"], "unit_scale")
    if unit_scale <= 0:
        raise ValueError("unit_scale must be positive")
    return PaperTarget(
        target_id=_required_string(item["target_id"], "target_id"),
        model=_required_string(item["model"], "model"),
        paper_version=_required_string(item["paper_version"], "paper_version"),
        source_url=_required_string(item["source_url"], "source_url"),
        result_locator=_required_string(item["result_locator"], "result_locator"),
        suite=_required_string(item["suite"], "suite"),
        metric=_required_string(item["metric"], "metric"),
        target_display=target_display,
        target_value=target_value,
        displayed_decimals=displayed_decimals,
        uncertainty_value=uncertainty_value,
        uncertainty_kind=uncertainty_kind,
        unit_scale=unit_scale,
        availability=_required_string(item["availability"], "availability"),
        source_sha256=_optional_sha256(item.get("source_sha256"), "source_sha256"),
        table_number=_optional_positive_int(item.get("table_number"), "table_number"),
        dataset_column=_optional_string(item.get("dataset_column"), "dataset_column"),
        uncertainty_display=uncertainty_display,
        comparison_availability=comparison_availability,
        table_caption=_optional_string(item.get("table_caption"), "table_caption"),
    )


def _parse_task_spec(value: Any, position: int) -> TaskSpec:
    item = _require_object(value, f"tasks[{position}]")
    _reject_unknown_keys(item, _TASK_SPEC_KEYS, f"tasks[{position}]")
    _require_keys(item, _TASK_SPEC_KEYS, f"tasks[{position}]")
    model = _required_string(item["model"], "model")
    suite = _required_string(item["suite"], "suite")
    index = _nonnegative_int(item["index"], "index")
    fold = _optional_nonnegative_int(item["fold"], "fold")
    task_id = _required_string(item["task_id"], "task_id")
    expected_task_id = stable_task_id(model, suite, index, fold)
    if task_id != expected_task_id:
        raise ValueError(
            f"task_id {task_id!r} is inconsistent with fields; expected {expected_task_id!r}"
        )
    return TaskSpec(
        task_id=task_id,
        model=model,
        suite=suite,
        index=index,
        fold=fold,
        dataset_path=_required_string(item["dataset_path"], "dataset_path"),
    )


def _require_object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    return value


def _reject_unknown_keys(item: dict[str, Any], allowed: frozenset[str] | set[str], location: str) -> None:
    unknown_keys = sorted(set(item) - set(allowed))
    if unknown_keys:
        raise ValueError(f"{location} contains unknown keys: {', '.join(unknown_keys)}")


def _require_keys(item: dict[str, Any], required: frozenset[str], location: str) -> None:
    missing_keys = sorted(required - set(item))
    if missing_keys:
        raise ValueError(f"{location} is missing required keys: {', '.join(missing_keys)}")


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    return numeric


def _optional_finite_number(value: Any, field: str) -> float | None:
    return None if value is None else _finite_number(value, field)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, field: str) -> int | None:
    return None if value is None else _nonnegative_int(value, field)


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    numeric = _nonnegative_int(value, field)
    if numeric == 0:
        raise ValueError(f"{field} must be positive")
    return numeric


def _optional_sha256(value: Any, field: str) -> str | None:
    if value is None:
        return None
    digest = _required_string(value, field)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _validate_numeric_display(display: str, value: float, field: str) -> None:
    try:
        displayed_value = Decimal(display)
    except InvalidOperation as error:
        raise ValueError(f"{field} must be a finite non-negative number") from error
    if not displayed_value.is_finite() or displayed_value < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    if displayed_value != Decimal(str(value)):
        raise ValueError(f"{field} does not match uncertainty_value")


def _reject_duplicate_ids(ids: Any, field: str) -> None:
    seen: set[str] = set()
    for value in ids:
        if value in seen:
            raise ValueError(f"duplicate {field}: {value!r}")
        seen.add(value)
