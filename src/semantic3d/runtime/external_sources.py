"""Controlled activation of pinned external source trees.

External research repositories stay outside this Git checkout.  Callers must
provide their location through the environment, and both the Git revision and
worktree cleanliness are checked before a package becomes importable.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

UNIDEPTH_SOURCE_ROOT_ENV = "SEMANTIC3D_UNIDEPTH_SOURCE_ROOT"
UNIDEPTH_SOURCE_REVISION = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"


class ExternalSourceError(RuntimeError):
    """A pinned external source tree is unavailable or fails verification."""


@dataclass(frozen=True)
class ExternalSourceDescriptor:
    """Verified source provenance without distribution metadata."""

    package: str
    source_root: str
    revision: str
    worktree_clean: bool
    import_available: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe descriptor."""

        return asdict(self)


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExternalSourceError(
            f"Could not verify external source Git metadata at {root}"
        ) from exc
    return result.stdout.strip()


def verify_unidepth_source(
    *,
    environment: Mapping[str, str] | None = None,
    expected_revision: str = UNIDEPTH_SOURCE_REVISION,
) -> ExternalSourceDescriptor:
    """Verify an explicitly configured, clean UniDepth source checkout."""

    values = os.environ if environment is None else environment
    configured = str(values.get(UNIDEPTH_SOURCE_ROOT_ENV, "")).strip()
    if not configured:
        raise ExternalSourceError(
            f"{UNIDEPTH_SOURCE_ROOT_ENV} must name the pinned UniDepth source root"
        )
    root = Path(configured).expanduser().resolve()
    package_root = root / "unidepth"
    if not package_root.is_dir() or not (package_root / "models" / "__init__.py").is_file():
        raise ExternalSourceError(
            f"Configured UniDepth source root does not contain the unidepth package: {root}"
        )
    revision = _git_output(root, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise ExternalSourceError(
            f"UniDepth source revision mismatch: expected {expected_revision}, got {revision}"
        )
    status = _git_output(root, "status", "--short", "--untracked-files=normal")
    if status:
        raise ExternalSourceError("UniDepth source worktree is not clean")
    return ExternalSourceDescriptor(
        package="unidepth",
        source_root=str(root),
        revision=revision,
        worktree_clean=True,
        import_available=True,
    )


def activate_unidepth_source(
    *,
    environment: Mapping[str, str] | None = None,
    expected_revision: str = UNIDEPTH_SOURCE_REVISION,
) -> ExternalSourceDescriptor:
    """Import pinned UniDepth without retaining a new ``sys.path`` entry."""

    descriptor = verify_unidepth_source(
        environment=environment,
        expected_revision=expected_revision,
    )
    source_root = Path(descriptor.source_root)
    existing = sys.modules.get("unidepth")
    if existing is not None and not _module_belongs_to(existing, source_root):
        raise ExternalSourceError(
            "An unrelated unidepth module is already loaded; refusing to overwrite it"
        )
    previous_modules = set(sys.modules)
    inserted = descriptor.source_root not in {str(item) for item in sys.path}
    if inserted:
        sys.path.insert(0, descriptor.source_root)
    try:
        importlib.invalidate_caches()
        models = importlib.import_module("unidepth.models")
        package = importlib.import_module("unidepth")
        if not _module_belongs_to(package, source_root) or not _module_belongs_to(
            models, source_root
        ):
            raise ExternalSourceError(
                "Imported UniDepth modules do not originate from the verified source root"
            )
    except ExternalSourceError:
        for name in set(sys.modules).difference(previous_modules):
            if name == "unidepth" or name.startswith("unidepth."):
                sys.modules.pop(name, None)
        raise
    except Exception as exc:
        for name in set(sys.modules).difference(previous_modules):
            if name == "unidepth" or name.startswith("unidepth."):
                sys.modules.pop(name, None)
        raise ExternalSourceError(
            "Verified UniDepth source could not be imported"
        ) from exc
    finally:
        if inserted and descriptor.source_root in sys.path:
            sys.path.remove(descriptor.source_root)
            importlib.invalidate_caches()
    return descriptor


def _module_belongs_to(module: object, source_root: Path) -> bool:
    """Return whether every import location is inside one verified checkout."""

    spec = getattr(module, "__spec__", None)
    if spec is None:
        return False
    locations = [
        Path(value).resolve()
        for value in (getattr(spec, "submodule_search_locations", None) or ())
    ]
    origin = getattr(spec, "origin", None)
    if origin and origin not in {"built-in", "frozen"}:
        locations.append(Path(origin).resolve())
    return bool(locations) and all(
        location == source_root or source_root in location.parents
        for location in locations
    )


__all__ = [
    "ExternalSourceDescriptor",
    "ExternalSourceError",
    "UNIDEPTH_SOURCE_REVISION",
    "UNIDEPTH_SOURCE_ROOT_ENV",
    "activate_unidepth_source",
    "verify_unidepth_source",
]
