"""Read-only CausalPFN benchmark catalogs and offline ACIC source routing."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd

from ..common.schemas import TaskSpec, stable_task_id


VENDOR_ROOT = Path(__file__).resolve().parents[2] / "CausalPFN"
ACIC_X_URL = (
    "https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
    "causallib/datasets/data/acic_challenge_2016/x.csv"
)
ACIC_SOURCE_URLS: dict[str, str] = {ACIC_X_URL: "x.csv"}
ACIC_SOURCE_URLS.update(
    {
        f"https://raw.githubusercontent.com/BiomedSciAI/causallib/master/"
        f"causallib/datasets/data/acic_challenge_2016/zymu_{index}.csv": f"zymu_{index}.csv"
        for index in range(1, 11)
    }
)
_SOURCE_MANIFEST = "source_manifest.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_utc_acquired_at(value: object, url: str) -> datetime:
    """Parse one acquisition timestamp and require an exact UTC offset."""

    if not isinstance(value, str):
        raise ValueError(f"ACIC source manifest has invalid acquired_at: {url}")
    try:
        acquired_at = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"ACIC source manifest has invalid acquired_at: {url}"
        ) from error
    if acquired_at.tzinfo is None or acquired_at.utcoffset() != timedelta(0):
        raise ValueError(f"ACIC source manifest has invalid acquired_at: {url}")
    return acquired_at


def ensure_acic_sources(cache_dir: str | Path) -> dict[str, Path]:
    """Acquire the official ACIC bytes once and return URL-to-cache paths.

    Existing files are accepted only when the recorded manifest agrees with the
    URL, size, and SHA-256.  The downloaded content is checked for empty/HTML
    responses before publication.
    """

    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / _SOURCE_MANIFEST
    prior: dict[str, Any] = {}
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        prior = payload.get("sources", {})
        if set(prior) != set(ACIC_SOURCE_URLS):
            raise ValueError("ACIC source manifest URLs changed")
        for url, record in prior.items():
            if not isinstance(record, dict):
                raise ValueError(f"ACIC source manifest is invalid: {url}")
            _parse_utc_acquired_at(record.get("acquired_at"), url)
    sources: dict[str, dict[str, Any]] = {}
    result: dict[str, Path] = {}
    for url, filename in ACIC_SOURCE_URLS.items():
        path = root / filename
        previous = prior.get(url)
        if path.exists():
            data = path.read_bytes()
            digest = _sha256_bytes(data)
            if not data or digest == "" or data.lstrip().lower().startswith((b"<html", b"<!doctype")):
                raise ValueError(f"invalid cached ACIC source: {path}")
            if previous is not None and (
                previous.get("filename") != filename
                or previous.get("sha256") != digest
                or previous.get("size") != len(data)
            ):
                raise ValueError(f"cached ACIC source changed: {url}")
        else:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if not data or data.lstrip().lower().startswith((b"<html", b"<!doctype")):
                raise ValueError(f"invalid downloaded ACIC source: {url}")
            path.write_bytes(data)
            digest = _sha256_bytes(data)
        sources[url] = {
            "filename": filename,
            "sha256": digest,
            "size": path.stat().st_size,
            "acquired_at": previous["acquired_at"] if previous is not None else datetime.now(timezone.utc).isoformat(),
        }
        result[url] = path
    manifest_path.write_text(
        json.dumps({"sources": sources}, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _validate_acic_manifest(cache_dir: Path) -> None:
    """Reject ACIC bytes unless their complete provenance record verifies."""

    manifest_path = cache_dir / _SOURCE_MANIFEST
    if not manifest_path.is_file():
        raise ValueError("ACIC source manifest is missing")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("ACIC source manifest is invalid") from error
    if not isinstance(payload, dict) or set(payload) != {"sources"}:
        raise ValueError("ACIC source manifest is invalid")
    sources = payload["sources"]
    if not isinstance(sources, dict) or set(sources) != set(ACIC_SOURCE_URLS):
        raise ValueError("ACIC source manifest URLs changed")
    for url, filename in ACIC_SOURCE_URLS.items():
        record = sources[url]
        if not isinstance(record, dict) or set(record) != {
            "filename", "sha256", "size", "acquired_at"
        }:
            raise ValueError(f"ACIC source manifest is invalid: {url}")
        path = cache_dir / filename
        try:
            data = path.read_bytes()
        except OSError as error:
            raise ValueError(f"ACIC source manifest is invalid: {url}") from error
        if (
            record["filename"] != filename
            or not isinstance(record["sha256"], str)
            or record["sha256"] != _sha256_bytes(data)
            or type(record["size"]) is not int
            or record["size"] != len(data)
            or not isinstance(record["acquired_at"], str)
        ):
            raise ValueError(f"ACIC source manifest does not match cached bytes: {url}")
        _parse_utc_acquired_at(record["acquired_at"], url)


@contextmanager
def _route_acic_reads(cache_dir: Path) -> Iterator[None]:
    """Temporarily map only the eleven official ACIC URLs to cached bytes."""

    routes = {url: cache_dir / filename for url, filename in ACIC_SOURCE_URLS.items()}
    original = pd.read_csv

    def routed(source: Any, *args: Any, **kwargs: Any):
        return original(routes.get(str(source), source), *args, **kwargs)

    pd.read_csv = routed  # type: ignore[assignment]
    try:
        yield
    finally:
        pd.read_csv = original  # type: ignore[assignment]


def build_acic_catalog(cache_dir: str | Path, causalpfn_root: str | Path = VENDOR_ROOT):
    """Instantiate the official ACIC catalog while forbidding live URL reads."""

    cache = Path(cache_dir).resolve()
    _validate_acic_manifest(cache)
    root = Path(causalpfn_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing official CausalPFN root: {root}")
    sys.path.insert(0, str(root))
    try:
        with _route_acic_reads(cache):
            module = _load_benchmark_module(root, "acic2016")
            return module.ACIC2016Dataset(test_ratio=0.1, seed=42, n_tables=10)
    finally:
        _purge_vendor_modules()
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


class _StableCatalog:
    """Cache each vendor realization after its first official construction.

    Several vendor catalogs keep a per-realization NumPy generator and advance
    it when ``__getitem__`` is called.  Caching the first value preserves the
    notebook's first-read bytes while making the required serial/isolated
    equivalence check independent of read order.
    """

    def __init__(self, catalog: Any):
        self._catalog = catalog
        self._entries: dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self._catalog)

    def __getitem__(self, index: int) -> Any:
        if index not in self._entries:
            self._entries[index] = self._catalog[index]
        return self._entries[index]


def build_catalogs(causalpfn_root: str | Path = VENDOR_ROOT, acic_cache: str | Path | None = None):
    """Build the four catalogs with the official notebook defaults."""

    if acic_cache is None:
        raise ValueError("acic_cache is required; run ensure_acic_sources before building catalogs")
    cache = Path(acic_cache).resolve()
    _validate_acic_manifest(cache)
    root = Path(causalpfn_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing official CausalPFN root: {root}")
    sys.path.insert(0, str(root))
    try:
        with _route_acic_reads(cache):
            ihdp = _load_benchmark_module(root, "ihdp").IHDPDataset(n_tables=100, seed=42)
            acic = build_acic_catalog(cache, root)
            realcause = _load_benchmark_module(root, "realcause")
            return {
                "ihdp": _StableCatalog(ihdp),
                "acic2016": _StableCatalog(acic),
                "lalonde_cps": _StableCatalog(
                    realcause.RealCauseLalondeCPSDataset(n_tables=100, test_ratio=0.1, seed=42)
                ),
                "lalonde_psid": _StableCatalog(
                    realcause.RealCauseLalondePSIDDataset(n_tables=100, test_ratio=0.1, seed=42)
                ),
            }
    finally:
        _purge_vendor_modules()
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _purge_vendor_modules() -> None:
    for name in tuple(sys.modules):
        if name == "_causalpfn_benchmarks" or name.startswith("_causalpfn_benchmarks."):
            sys.modules.pop(name, None)


def _load_benchmark_module(root: Path, name: str):
    """Load one official benchmark module without executing benchmarks/__init__.py."""

    package_name = "_causalpfn_benchmarks"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(root / "benchmarks")]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    module_name = f"{package_name}.{name}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    source = root / "benchmarks" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(module_name, source, submodule_search_locations=[])
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official benchmark module: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_causalpfn_tasks(catalogs: Mapping[str, Any]) -> tuple[TaskSpec, ...]:
    """Create the fixed 310-task manifest in paper order."""

    expected = ("ihdp", "acic2016", "lalonde_cps", "lalonde_psid")
    if tuple(catalogs) != expected:
        raise ValueError(f"catalog order must be {expected}")
    tasks: list[TaskSpec] = []
    for suite in expected:
        count = len(catalogs[suite])
        expected_count = {"ihdp": 100, "acic2016": 10, "lalonde_cps": 100, "lalonde_psid": 100}[suite]
        if count != expected_count:
            raise ValueError(f"{suite} catalog must contain {expected_count} tasks, got {count}")
        for index in range(count):
            tasks.append(
                TaskSpec(
                    task_id=stable_task_id("causalpfn", suite, index, None),
                    model="causalpfn",
                    suite=suite,
                    index=index,
                    fold=None,
                    dataset_path=f"{suite}/{index:03d}",
                )
            )
    return tuple(tasks)
