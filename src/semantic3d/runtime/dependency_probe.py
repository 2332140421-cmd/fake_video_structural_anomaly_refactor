"""Python import and executable probes for server preflight."""

from __future__ import annotations

import importlib
import importlib.metadata
import shutil
from typing import Any, Iterable


def probe_dependencies(
    python_modules: Iterable[str], executables: Iterable[str]
) -> dict[str, Any]:
    """Probe configured dependencies without installing or downloading anything."""

    modules = []
    for name in sorted(set(python_modules)):
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "")
            if not version:
                try:
                    version = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    version = "unknown"
            modules.append({"name": name, "importable": True, "version": str(version), "error": ""})
        except Exception as exc:  # environment audit must preserve the failure
            modules.append(
                {"name": name, "importable": False, "version": "", "error": f"{type(exc).__name__}: {exc}"}
            )
    programs = [
        {"name": name, "available": bool(shutil.which(name)), "path": shutil.which(name) or ""}
        for name in sorted(set(executables))
    ]
    return {
        "python_modules": modules,
        "executables": programs,
        "all_python_modules_importable": all(row["importable"] for row in modules),
        "all_executables_available": all(row["available"] for row in programs),
        "install_performed": False,
    }

