"""Resolve runtime paths without contaminating logical content hashes."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

_VARIABLE = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")
_PATH_FIELDS = (
    "project_root",
    "data_root",
    "download_root",
    "temporary_root",
    "cache_root",
    "model_root",
    "output_root",
    "log_root",
)
_ENV_BY_FIELD = {name: name.upper() for name in _PATH_FIELDS}
_NON_SEMANTIC_KEYS = frozenset(
    {
        "absolute_path",
        "hostname",
        "username",
        "mtime",
        "mtime_ns",
        "run_time",
        "created_at",
        "updated_at",
        "location_metadata",
    }
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved machine-local paths and execution limits."""

    runtime_profile: str
    project_root: Path
    data_root: Path
    download_root: Path
    temporary_root: Path
    cache_root: Path
    model_root: Path
    output_root: Path
    log_root: Path
    device: str
    num_workers: int
    storage_safety_margin: int
    batch_storage_limit: int
    required_python_dependencies: tuple[str, ...]
    required_executables: tuple[str, ...]
    require_cuda_for_formal_batch: bool
    source_config: Path

    def location_metadata(self) -> dict[str, Any]:
        """Return machine-specific data that must stay outside semantic hashes."""

        return {
            "runtime_profile": self.runtime_profile,
            **{name: str(getattr(self, name)) for name in _PATH_FIELDS},
            "device": self.device,
            "num_workers": self.num_workers,
            "source_config": str(self.source_config),
        }


def unresolved_variables(value: Any) -> tuple[str, ...]:
    """Return unresolved ${NAME} placeholders anywhere in a nested value."""

    found: set[str] = set()
    if isinstance(value, str):
        found.update(_VARIABLE.findall(value))
    elif isinstance(value, Mapping):
        for item in value.values():
            found.update(unresolved_variables(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(unresolved_variables(item))
    return tuple(sorted(found))


def _substitute(value: Any, environment: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _VARIABLE.sub(lambda match: environment.get(match.group(1), match.group(0)), value)
    if isinstance(value, dict):
        return {key: _substitute(item, environment) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, environment) for item in value]
    return value


def load_runtime_config(
    config_path: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    require_resolved: bool = True,
) -> RuntimeConfig:
    """Load YAML, apply environment overrides, and resolve all paths."""

    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = dict(os.environ if environment is None else environment)
    substituted = _substitute(raw, env)
    for field, variable in _ENV_BY_FIELD.items():
        if variable in env:
            substituted[field] = env[variable]
    missing = unresolved_variables(substituted)
    if require_resolved and missing:
        raise ValueError(f"Unresolved runtime variables: {', '.join(missing)}")

    project_value = Path(str(substituted["project_root"]))
    project = (
        project_value.resolve()
        if project_value.is_absolute()
        else (path.parent / project_value).resolve()
    )
    resolved_paths: dict[str, Path] = {"project_root": project}
    for field in _PATH_FIELDS[1:]:
        value = Path(str(substituted[field]))
        resolved_paths[field] = value.resolve() if value.is_absolute() else (project / value).resolve()
    return RuntimeConfig(
        runtime_profile=str(substituted["runtime_profile"]),
        project_root=project,
        device=str(substituted["device"]),
        num_workers=int(substituted["num_workers"]),
        storage_safety_margin=int(substituted["storage_safety_margin"]),
        batch_storage_limit=int(substituted["batch_storage_limit"]),
        required_python_dependencies=tuple(
            str(value) for value in substituted.get("required_python_dependencies", ())
        ),
        required_executables=tuple(
            str(value) for value in substituted.get("required_executables", ())
        ),
        require_cuda_for_formal_batch=bool(
            substituted.get("require_cuda_for_formal_batch", True)
        ),
        source_config=path,
        **{name: resolved_paths[name] for name in _PATH_FIELDS[1:]},
    )


def _normalize_logical(value: Any, runtime: RuntimeConfig) -> Any:
    roots = sorted(
        ((str(getattr(runtime, field)), f"${{{field.upper()}}}") for field in _PATH_FIELDS),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    if isinstance(value, str):
        output = value
        for root, token in roots:
            if output == root or output.startswith(root + os.sep):
                output = token + output[len(root) :]
                break
        return output
    if isinstance(value, Mapping):
        return {
            key: _normalize_logical(item, runtime)
            for key, item in sorted(value.items())
            if key not in _NON_SEMANTIC_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_logical(item, runtime) for item in value]
    return value


def canonical_logical_sha256(value: Any, runtime: RuntimeConfig) -> str:
    """Hash logical content after replacing location roots with stable tokens."""

    payload = json.dumps(
        _normalize_logical(value, runtime),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

