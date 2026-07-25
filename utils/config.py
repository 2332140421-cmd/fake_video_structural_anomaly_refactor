"""YAML configuration loading with environment and relative-path resolution."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"Required environment variable {name} is not set.")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping.")
    config = _expand(payload)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def resolve_path(config: Mapping[str, Any], value: str | Path) -> Path:
    raw = Path(os.path.expandvars(str(value))).expanduser()
    if raw.is_absolute():
        return raw
    return Path(str(config["_project_root"])) / raw


def validate_config(config: Mapping[str, Any]) -> None:
    required = {"video", "providers", "object_semantic", "fusion", "training"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {missing}.")
    if int(config["video"]["clip_length"]) < 2:
        raise ValueError("video.clip_length must be at least 2.")
    if int(config["video"]["clip_stride"]) < 1:
        raise ValueError("video.clip_stride must be positive.")


__all__ = ["load_config", "resolve_path", "validate_config"]
