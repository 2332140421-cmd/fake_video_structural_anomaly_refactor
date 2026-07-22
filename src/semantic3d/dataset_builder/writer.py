"""Atomic Parquet, JSON, and array writers for P4-B datasets."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by environment check
    raise RuntimeError(
        "P4-B Parquet output requires pyarrow. Install it in the project .venv."
    ) from exc


def sha256_file(path: str | Path) -> str:
    """Compute a file SHA-256 without loading the whole artifact into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def json_text(value: Any) -> str:
    """Serialize nested fields deterministically for storage in Parquet cells."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    """Write bytes through a sibling temporary file and atomic rename."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_write_json(path: str | Path, value: Any) -> Path:
    """Write indented UTF-8 JSON atomically."""

    payload = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
    ).encode("utf-8")
    return atomic_write_bytes(path, payload)


def _normalize_row(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for column in columns:
        value = row.get(column)
        if isinstance(value, (dict, list, tuple, set)):
            value = json_text(value)
        elif hasattr(value, "value"):
            value = value.value
        elif isinstance(value, Path):
            value = str(value)
        elif isinstance(value, np.generic):
            value = value.item()
        normalized[column] = value
    return normalized


def write_parquet(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    columns: Sequence[str],
) -> Path:
    """Write rows atomically as real Apache Parquet, including empty tables."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    normalized = [_normalize_row(row, columns) for row in rows]
    if normalized:
        table = pa.Table.from_pylist(normalized)
        missing_columns = [name for name in columns if name not in table.column_names]
        for name in missing_columns:
            table = table.append_column(name, pa.nulls(len(normalized)))
        table = table.select(list(columns))
    else:
        table = pa.table({name: pa.array([], type=pa.null()) for name in columns})
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_npz_array(path: str | Path, **arrays: np.ndarray) -> dict[str, Any]:
    """Write compressed arrays atomically and return a table-safe reference."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    shapes = {name: list(np.asarray(value).shape) for name, value in arrays.items()}
    dtypes = {name: str(np.asarray(value).dtype) for name, value in arrays.items()}
    return {
        "path": str(target),
        "shape": shapes,
        "dtype": dtypes,
        "sha256": sha256_file(target),
    }


class DatasetWriter:
    """Path-scoped writer that never stores labels in structural tables."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def parquet(
        self, relative_path: str, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]
    ) -> Path:
        if "label" in relative_path.lower():
            raise ValueError("Structural DatasetWriter cannot write label manifests")
        return write_parquet(self.root / relative_path, rows, columns=columns)

    def json(self, relative_path: str, value: Any) -> Path:
        return atomic_write_json(self.root / relative_path, value)

    def array(self, relative_path: str, **arrays: np.ndarray) -> dict[str, Any]:
        return write_npz_array(self.root / relative_path, **arrays)
