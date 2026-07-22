"""Model registry loading and SHA-256 verification without implicit downloads."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED = (
    "model_id",
    "filename",
    "purpose",
    "source",
    "license_status",
    "redistribution_allowed",
    "expected_library",
    "expected_library_version",
    "size_bytes",
    "sha256",
    "server_target_path",
    "acquisition_method",
    "required_for_server_smoke",
)


def sha256_file(path: str | Path) -> str:
    """Hash a model file without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_registry(path: str | Path) -> dict[str, Any]:
    """Load a YAML model registry and preserve its deterministic order."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError("Model registry must contain a models list")
    return payload


def resolve_server_target(value: str, environment: Mapping[str, str]) -> Path:
    """Resolve the registered ${MODEL_ROOT} target without guessing a path."""

    root = environment.get("MODEL_ROOT")
    if "${MODEL_ROOT}" in value:
        if not root:
            raise ValueError("MODEL_ROOT is required to resolve model target paths")
        value = value.replace("${MODEL_ROOT}", root)
    if "${" in value:
        raise ValueError(f"Unresolved model target path: {value}")
    return Path(value)


def validate_model_registry(
    registry: Mapping[str, Any],
    *,
    local_model_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate schema, smoke hashes, redistribution policy, and optional files."""

    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in registry.get("models", []):
        model = dict(raw)
        missing = [name for name in _REQUIRED if name not in model]
        model_id = str(model.get("model_id", ""))
        if missing:
            errors.append(f"{model_id or '<unknown>'}:missing_fields:{','.join(missing)}")
        if not model_id or model_id in seen:
            errors.append(f"duplicate_or_empty_model_id:{model_id}")
        seen.add(model_id)
        digest = str(model.get("sha256", ""))
        smoke_required = bool(model.get("required_for_server_smoke", False))
        if smoke_required and not _SHA256.fullmatch(digest):
            errors.append(f"{model_id}:server_smoke_requires_sha256")
        if bool(model.get("redistribution_allowed", False)) and "not_reviewed" in str(
            model.get("license_status", "")
        ):
            errors.append(f"{model_id}:unreviewed_license_cannot_allow_redistribution")
        local_path = None
        exists = False
        size_matches = None
        sha_matches = None
        if local_model_root is not None:
            local_path = Path(local_model_root) / str(model.get("filename", ""))
            exists = local_path.is_file()
            if exists:
                size_matches = local_path.stat().st_size == int(model.get("size_bytes", -1))
                sha_matches = sha256_file(local_path) == digest
                if not size_matches:
                    errors.append(f"{model_id}:local_size_mismatch")
                if not sha_matches:
                    errors.append(f"{model_id}:local_sha256_mismatch")
        rows.append(
            {
                "model_id": model_id,
                "filename": str(model.get("filename", "")),
                "sha256": digest,
                "redistribution_allowed": bool(model.get("redistribution_allowed", False)),
                "required_for_server_smoke": smoke_required,
                "local_path": str(local_path) if local_path is not None else "",
                "local_exists": exists,
                "local_size_matches": size_matches,
                "local_sha256_matches": sha_matches,
                "git_distribution_allowed": bool(model.get("redistribution_allowed", False)),
            }
        )
    smoke_hashes_complete = all(
        not row["required_for_server_smoke"] or _SHA256.fullmatch(row["sha256"])
        for row in rows
    )
    return {
        "valid": not errors and bool(rows),
        "errors": sorted(errors),
        "model_count": len(rows),
        "server_smoke_hashes_complete": smoke_hashes_complete,
        "all_local_files_verified": bool(rows)
        and all(row["local_exists"] and row["local_sha256_matches"] for row in rows),
        "models": sorted(rows, key=lambda row: row["model_id"]),
        "downloads_performed": False,
    }


def current_environment() -> Mapping[str, str]:
    """Expose environment through a small testable adapter."""

    return os.environ
