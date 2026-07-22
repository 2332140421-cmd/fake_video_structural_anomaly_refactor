"""Batch artifact registration and checksum validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from semantic3d.dataset_builder.writer import sha256_file


def register_artifacts(paths: Iterable[str | Path]) -> dict[str, str]:
    """Register only complete existing files by SHA-256."""

    output = {}
    for value in sorted(str(Path(path)) for path in paths):
        path = Path(value)
        if not path.is_file() or path.name.endswith((".tmp", ".partial")):
            raise ValueError(f"Incomplete or missing artifact cannot be registered: {path}")
        output[value] = sha256_file(path)
    return output


def validate_artifacts(artifacts: dict[str, str]) -> dict[str, Any]:
    """Validate every artifact before a batch may become completed."""

    checks = {
        path: Path(path).is_file() and sha256_file(path) == digest
        for path, digest in sorted(artifacts.items())
    }
    return {"valid": bool(checks) and all(checks.values()), "checks": checks}

