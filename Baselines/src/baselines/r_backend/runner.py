from __future__ import annotations

from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping, Sequence
from uuid import uuid4

from numpy.typing import NDArray

from ..exceptions import CATEValidationError, DependencyUnavailableError, RBackendError
from .protocol import RBackendResult, RJobPaths, read_r_output, write_r_input


from ..config import BASELINES_ROOT


WORKSPACE_ROOT = BASELINES_ROOT.parent


class RBackendRunner:
    def __init__(
        self,
        *,
        temp_root: Path = WORKSPACE_ROOT / "artifacts" / "r_jobs",
        artifact_root: Path = WORKSPACE_ROOT / "artifacts" / "r_models",
        r_libs_user: Path = WORKSPACE_ROOT / ".r_libs",
        rscript: str | None = None,
        retain_success: bool = False,
    ) -> None:
        self.temp_root = Path(temp_root)
        self.artifact_root = Path(artifact_root)
        self.r_libs_user = Path(r_libs_user)
        self.rscript = rscript or shutil.which("Rscript")
        self.retain_success = retain_success

    def _require_rscript(self) -> str:
        if not self.rscript:
            raise DependencyUnavailableError(
                "Rscript is unavailable; install workspace-local R and expose it on PATH"
            )
        return self.rscript

    def _paths(self, model_id: str, artifact_dir: Path | None = None) -> RJobPaths:
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.r_libs_user.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix=f"{model_id}-", dir=self.temp_root))
        resolved_artifact_dir = (
            Path(artifact_dir)
            if artifact_dir is not None
            else self.artifact_root / f"{model_id}-{uuid4().hex}"
        )
        resolved_artifact_dir.mkdir(parents=True, exist_ok=artifact_dir is not None)
        return RJobPaths(
            job_dir=job_dir,
            train_csv=job_dir / "train.csv",
            predict_csv=job_dir / "predict.csv",
            config_json=job_dir / "config.json",
            output_csv=job_dir / "output.csv",
            stdout_log=job_dir / "stdout.log",
            stderr_log=job_dir / "stderr.log",
            artifact_dir=resolved_artifact_dir,
            lock_file=resolved_artifact_dir / ".lock",
        )

    @staticmethod
    def _tail(value: str, limit: int = 4000) -> str:
        return value[-limit:]

    def _error(
        self,
        *,
        model_id: str,
        command: tuple[str, ...],
        paths: RJobPaths,
        exit_code: int | str,
        stdout: str,
        stderr: str,
        reason: str = "R backend execution failed",
    ) -> RBackendError:
        return RBackendError(
            f"{reason} for model {model_id!r} with exit code {exit_code}; "
            f"command={' '.join(command)!r}; job_dir={paths.job_dir}; "
            f"artifact_dir={paths.artifact_dir}; stdout_tail={self._tail(stdout)!r}; "
            f"stderr_tail={self._tail(stderr)!r}"
        )

    def run(
        self,
        *,
        model_id: str,
        script: Path,
        mode: str,
        config: Mapping[str, Any],
        train_X: NDArray[Any],
        train_treatment: NDArray[Any],
        train_y: NDArray[Any],
        predict_X: NDArray[Any],
        predict_row_ids: Sequence[int] | None = None,
        artifact_dir: Path | None = None,
    ) -> RBackendResult:
        rscript = self._require_rscript()
        script_path = Path(script).resolve()
        if not script_path.is_file():
            raise DependencyUnavailableError(f"R backend script is missing: {script_path}")
        if mode not in {"fit", "predict", "fit_predict"}:
            raise CATEValidationError("R backend mode must be fit, predict, or fit_predict")
        paths = self._paths(model_id, artifact_dir=artifact_dir)
        write_r_input(
            paths.train_csv,
            train_X,
            treatment=train_treatment,
            y=train_y,
        )
        write_r_input(paths.predict_csv, predict_X, row_ids=predict_row_ids)
        expected_ids = (
            list(range(len(predict_X)))
            if predict_row_ids is None
            else list(predict_row_ids)
        )
        resolved_config = dict(config)
        resolved_config.update(
            model_id=model_id,
            mode=mode,
            artifact_dir=str(paths.artifact_dir),
        )
        paths.config_json.write_text(
            json.dumps(resolved_config, sort_keys=True, indent=2), encoding="utf-8"
        )
        command = (
            rscript,
            "--vanilla",
            str(script_path),
            "--mode",
            mode,
            "--train",
            str(paths.train_csv),
            "--predict",
            str(paths.predict_csv),
            "--config",
            str(paths.config_json),
            "--output",
            str(paths.output_csv),
            "--artifact-dir",
            str(paths.artifact_dir),
        )
        environment = os.environ.copy()
        environment.update(
            R_LIBS_USER=str(self.r_libs_user),
            TMPDIR=str(paths.job_dir),
        )
        timeout = float(config.get("r_timeout_seconds", 3600))
        try:
            with paths.lock_file.open("w", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                completed = subprocess.run(
                    command,
                    text=True,
                    capture_output=True,
                    env=environment,
                    timeout=timeout,
                    check=False,
                )
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or ""
            stderr = error.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            paths.stdout_log.write_text(stdout, encoding="utf-8")
            paths.stderr_log.write_text(stderr, encoding="utf-8")
            raise self._error(
                model_id=model_id,
                command=command,
                paths=paths,
                exit_code="timeout",
                stdout=stdout,
                stderr=stderr,
                reason=f"R backend exceeded {timeout:g}s timeout",
            ) from error
        paths.stdout_log.write_text(completed.stdout, encoding="utf-8")
        paths.stderr_log.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise self._error(
                model_id=model_id,
                command=command,
                paths=paths,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        try:
            parsed = read_r_output(paths.output_csv, expected_row_ids=expected_ids)
        except CATEValidationError as error:
            raise self._error(
                model_id=model_id,
                command=command,
                paths=paths,
                exit_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                reason=f"Invalid R output: {error}",
            ) from error
        result = replace(
            parsed,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=command,
            paths=paths,
        )
        if not self.retain_success:
            shutil.rmtree(paths.job_dir)
        return result
