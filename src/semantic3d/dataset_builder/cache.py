"""Content-addressed stage cache with corruption and dependency tracking."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .writer import atomic_write_json, sha256_file


def build_cache_key(
    *,
    source_video_sha256: str,
    stage_name: str,
    stage_config_sha256: str,
    upstream_artifact_sha256: Iterable[str],
    provider_weight_sha256: Iterable[str],
    schema_version: str,
) -> str:
    """Hash every input that can affect one stage artifact."""

    payload = {
        "source_video_sha256": source_video_sha256,
        "stage_name": stage_name,
        "stage_config_sha256": stage_config_sha256,
        "upstream_artifact_sha256": sorted(str(value) for value in upstream_artifact_sha256),
        "provider_weight_sha256": sorted(str(value) for value in provider_weight_sha256),
        "schema_version": schema_version,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CacheLookup:
    """One cache decision used by stage-status reporting."""

    hit: bool
    reason: str
    cache_key: str
    record_path: str


class StageCache:
    """Store stage completion records only after every artifact is durable."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def record_path(self, stage_name: str, cache_key: str) -> Path:
        return self.root / stage_name / f"{cache_key}.json"

    def lookup(self, stage_name: str, cache_key: str) -> CacheLookup:
        path = self.record_path(stage_name, cache_key)
        if not path.exists():
            return CacheLookup(False, "cache_record_missing", cache_key, str(path))
        if path.name.endswith(".tmp"):
            return CacheLookup(False, "temporary_cache_record", cache_key, str(path))
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return CacheLookup(False, "cache_record_corrupt", cache_key, str(path))
        if record.get("cache_key") != cache_key or not record.get("complete", False):
            return CacheLookup(False, "cache_record_incomplete", cache_key, str(path))
        for artifact in record.get("artifacts", []):
            artifact_path = Path(artifact["path"])
            if not artifact_path.exists():
                return CacheLookup(False, "cached_artifact_missing", cache_key, str(path))
            if sha256_file(artifact_path) != artifact.get("sha256"):
                return CacheLookup(False, "cached_artifact_hash_mismatch", cache_key, str(path))
        return CacheLookup(True, "cache_hit", cache_key, str(path))

    def commit(
        self,
        stage_name: str,
        cache_key: str,
        artifacts: Iterable[str | Path],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> Path:
        """Atomically commit a complete cache record after hashing artifacts."""

        entries = []
        for artifact in artifacts:
            path = Path(artifact)
            if not path.exists() or path.name.endswith(".tmp"):
                raise ValueError(f"Cannot cache missing/temporary artifact: {path}")
            entries.append({"path": str(path), "sha256": sha256_file(path)})
        record = {
            "stage_name": stage_name,
            "cache_key": cache_key,
            "complete": True,
            "artifacts": entries,
            "metadata": dict(metadata or {}),
        }
        return atomic_write_json(self.record_path(stage_name, cache_key), record)

    def remove_record(self, stage_name: str, cache_key: str) -> None:
        self.record_path(stage_name, cache_key).unlink(missing_ok=True)


def stage_config_hash(config: Mapping[str, Any]) -> str:
    """Hash a stage's canonical configuration, including explicit NaN values."""

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [normalize(item) for item in value]
        if isinstance(value, float) and math.isnan(value):
            return "NaN"
        return value

    encoded = json.dumps(normalize(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
