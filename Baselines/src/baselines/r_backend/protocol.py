from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ..exceptions import CATEValidationError


@dataclass(frozen=True)
class RJobPaths:
    job_dir: Path
    train_csv: Path
    predict_csv: Path
    config_json: Path
    output_csv: Path
    stdout_log: Path
    stderr_log: Path
    artifact_dir: Path
    lock_file: Path


@dataclass(frozen=True)
class RBackendResult:
    row_ids: NDArray[np.int64]
    cate: NDArray[np.float64]
    mu0: NDArray[np.float64] | None = None
    mu1: NDArray[np.float64] | None = None
    cate_lower: NDArray[np.float64] | None = None
    cate_upper: NDArray[np.float64] | None = None
    stdout: str = ""
    stderr: str = ""
    command: tuple[str, ...] = ()
    paths: RJobPaths | None = None


def _row_id_array(row_ids: Sequence[int] | None, n_rows: int) -> NDArray[np.int64]:
    if row_ids is None:
        values = np.arange(n_rows, dtype=np.int64)
    else:
        raw = np.asarray(row_ids)
        if raw.ndim != 1 or len(raw) != n_rows:
            raise CATEValidationError(f"row_ids must have shape ({n_rows},)")
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise CATEValidationError("row_ids must be integers") from error
        if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
            raise CATEValidationError("row_ids must be finite integers")
        values = numeric.astype(np.int64)
    if len(np.unique(values)) != n_rows:
        raise CATEValidationError("row_ids must be unique")
    return values


def _optional_vector(
    value: object | None,
    *,
    name: str,
    n_rows: int,
) -> NDArray[np.float64] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CATEValidationError(f"{name} must be numeric") from error
    if array.shape != (n_rows,) or not np.isfinite(array).all():
        raise CATEValidationError(f"{name} must be finite with shape ({n_rows},)")
    return array


def write_r_input(
    path: Path,
    X: object,
    *,
    treatment: object | None = None,
    y: object | None = None,
    row_ids: Sequence[int] | None = None,
) -> None:
    """Write the stable CSV boundary consumed by all R scripts."""

    try:
        X_array = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CATEValidationError("R input X must be numeric") from error
    if X_array.ndim != 2 or not np.isfinite(X_array).all():
        raise CATEValidationError("R input X must be a finite two-dimensional array")
    ids = _row_id_array(row_ids, len(X_array))
    treatment_array = _optional_vector(
        treatment, name="treatment", n_rows=len(X_array)
    )
    y_array = _optional_vector(y, name="y", n_rows=len(X_array))
    if (treatment_array is None) != (y_array is None):
        raise CATEValidationError("treatment and y must either both be present or absent")
    if treatment_array is not None and not np.isin(treatment_array, [0.0, 1.0]).all():
        raise CATEValidationError("treatment may contain only 0 and 1")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["_row_id"]
    if treatment_array is not None:
        header.extend(["_treatment", "_outcome"])
    header.extend(f"x{column}" for column in range(X_array.shape[1]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for index, row in enumerate(X_array):
            values: list[int | float] = [int(ids[index])]
            if treatment_array is not None and y_array is not None:
                values.extend([int(treatment_array[index]), float(y_array[index])])
            values.extend(float(value) for value in row)
            writer.writerow(values)


def _finite_column(
    rows_by_id: dict[int, dict[str, str]],
    ordered_ids: NDArray[np.int64],
    column: str,
) -> NDArray[np.float64]:
    try:
        values = np.asarray(
            [float(rows_by_id[int(row_id)][column]) for row_id in ordered_ids],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CATEValidationError(f"R output column {column} must be numeric") from error
    if not np.isfinite(values).all():
        raise CATEValidationError(f"R output column {column} must be finite")
    return values


def read_r_output(
    path: Path,
    *,
    expected_row_ids: Sequence[int],
) -> RBackendResult:
    """Read R output and restore the exact caller-supplied row order."""

    if not path.is_file():
        raise CATEValidationError(f"R output file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "_row_id" not in fieldnames or "cate" not in fieldnames:
            raise CATEValidationError("R output requires _row_id and cate columns")
        rows = list(reader)
    parsed_ids: list[int] = []
    for row in rows:
        try:
            parsed_ids.append(int(row["_row_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise CATEValidationError("R output row IDs must be integers") from error
    if len(set(parsed_ids)) != len(parsed_ids):
        raise CATEValidationError("R output contains duplicate row IDs")
    expected = _row_id_array(expected_row_ids, len(expected_row_ids))
    if set(parsed_ids) != set(int(value) for value in expected):
        raise CATEValidationError("R output row IDs do not match requested row IDs")
    rows_by_id = {row_id: row for row_id, row in zip(parsed_ids, rows)}
    optional: dict[str, NDArray[np.float64] | None] = {}
    for column in ("mu0", "mu1", "cate_lower", "cate_upper"):
        optional[column] = (
            _finite_column(rows_by_id, expected, column)
            if column in fieldnames
            else None
        )
    return RBackendResult(
        row_ids=expected,
        cate=_finite_column(rows_by_id, expected, "cate"),
        mu0=optional["mu0"],
        mu1=optional["mu1"],
        cate_lower=optional["cate_lower"],
        cate_upper=optional["cate_upper"],
    )
