"""Storage, filesystem, permission, SHA, and atomic-rename probes."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .runtime_config import RuntimeConfig


def _nearest_existing(path: Path) -> Path:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def probe_storage(runtime: RuntimeConfig) -> dict[str, Any]:
    """Probe configured roots without deleting or processing project data."""

    roots = []
    for name in (
        "project_root",
        "data_root",
        "download_root",
        "temporary_root",
        "cache_root",
        "model_root",
        "output_root",
        "log_root",
    ):
        path = Path(getattr(runtime, name))
        parent = _nearest_existing(path)
        usage = shutil.disk_usage(parent)
        roots.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
                "is_directory": path.is_dir(),
                "can_create": os.access(parent, os.W_OK | os.X_OK),
                "readable": path.exists() and os.access(path, os.R_OK),
                "writable": (path.exists() and os.access(path, os.W_OK))
                or (not path.exists() and os.access(parent, os.W_OK | os.X_OK)),
                "filesystem_device": int(os.stat(parent).st_dev),
                "filesystem_probe_root": str(parent),
                "total_bytes": int(usage.total),
                "used_bytes": int(usage.used),
                "free_bytes": int(usage.free),
            }
        )
    by_name = {row["name"]: row for row in roots}
    data_free = int(by_name["data_root"]["free_bytes"])
    enough_safety = data_free >= runtime.storage_safety_margin

    atomic = {
        "passed": False,
        "sha256_passed": False,
        "directory": str(runtime.temporary_root),
        "error": "",
    }
    try:
        runtime.temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="p4c3a_preflight_", dir=runtime.temporary_root) as name:
            directory = Path(name)
            source = directory / "source.tmp"
            target = directory / "target.json"
            payload = b"semantic3d-p4c3a-atomic-probe"
            source.write_bytes(payload)
            os.replace(source, target)
            atomic["passed"] = target.read_bytes() == payload and not source.exists()
            atomic["sha256_passed"] = hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    except Exception as exc:
        atomic["error"] = f"{type(exc).__name__}: {exc}"
    return {
        "roots": roots,
        "all_roots_accessible_or_creatable": all(
            row["is_directory"] or (not row["exists"] and row["can_create"]) for row in roots
        ),
        "all_roots_writable": all(row["writable"] for row in roots),
        "data_free_bytes": data_free,
        "storage_safety_margin_bytes": runtime.storage_safety_margin,
        "batch_storage_limit_bytes": runtime.batch_storage_limit,
        "safety_margin_available": enough_safety,
        "batch_plus_safety_available": data_free
        >= runtime.storage_safety_margin + runtime.batch_storage_limit,
        "temporary_and_output_same_filesystem": by_name["temporary_root"]["filesystem_device"]
        == by_name["output_root"]["filesystem_device"],
        "atomic_write_and_rename": atomic,
        "files_deleted": False,
    }

