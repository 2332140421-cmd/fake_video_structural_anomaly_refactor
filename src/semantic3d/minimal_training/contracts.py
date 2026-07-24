"""Versioned contracts for the P4-C3C-A2 engineering training loop."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from semantic3d.evidence_fusion.contracts import EvidenceBranchGroup

FEATURE_CONTRACT_SCHEMA_VERSION = "semantic3d.p4c3c.a2.feature_contract.v1"
EVIDENCE_MANIFEST_SCHEMA_VERSION = "semantic3d.p4c3c.a2.evidence_manifest.v1"
TRAINING_CONFIG_SCHEMA_VERSION = "semantic3d.p4c3c.a2.training_config.v1"
TRAINING_CHECKPOINT_SCHEMA_VERSION = "semantic3d.p4c3c.a2.checkpoint.v1"


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_manifest_checksum(*paths: str | Path) -> str:
    """Hash path-independent file contents in a stable declared order."""

    digest = hashlib.sha256()
    for path in paths:
        file_path = Path(path)
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


@dataclass(frozen=True)
class FeatureContract:
    """Code-readable fixed branch and tensor contract."""

    schema_version: str
    version: str
    source_schema_version: str
    branch_order: tuple[str, ...]
    feature_names: tuple[str, ...]
    reliability_names: tuple[str, ...]
    allowed_branch_status: frozenset[str]
    raw: Mapping[str, Any]
    sha256: str

    @property
    def branch_count(self) -> int:
        return len(self.branch_order)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def descriptor(self) -> dict[str, Any]:
        """Return checkpoint-safe identity and dimensional metadata."""

        return {
            "schema_version": self.schema_version,
            "feature_contract_version": self.version,
            "source_schema_version": self.source_schema_version,
            "branch_order": list(self.branch_order),
            "feature_names": list(self.feature_names),
            "reliability_names": list(self.reliability_names),
            "branch_count": self.branch_count,
            "feature_dim": self.feature_dim,
            "sha256": self.sha256,
        }


def load_feature_contract(path: str | Path) -> FeatureContract:
    """Load and strictly validate the versioned M6-derived feature contract."""

    contract_path = Path(path)
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Feature contract must be a YAML mapping.")
    if payload.get("schema_version") != FEATURE_CONTRACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported feature contract schema_version: "
            f"{payload.get('schema_version')!r}"
        )
    version = str(payload.get("feature_contract_version", "")).strip()
    if not version:
        raise ValueError("feature_contract_version must be non-empty.")
    branch_order = tuple(str(value) for value in payload.get("branch_order", ()))
    expected = tuple(group.value for group in EvidenceBranchGroup)
    if branch_order != expected:
        raise ValueError(
            "Feature contract branch_order must exactly match the frozen M6 order."
        )
    feature_fields = payload.get("feature_fields")
    reliability_fields = payload.get("reliability_fields")
    if not isinstance(feature_fields, list) or not feature_fields:
        raise ValueError("Feature contract requires non-empty feature_fields.")
    if not isinstance(reliability_fields, list) or not reliability_fields:
        raise ValueError("Feature contract requires non-empty reliability_fields.")
    feature_names = tuple(str(item["name"]) for item in feature_fields)
    reliability_names = tuple(str(item["name"]) for item in reliability_fields)
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("Feature field names must be unique.")
    if len(reliability_names) != 1 or reliability_names != ("confidence",):
        raise ValueError("A2 permits only M6 confidence as the reliability field.")
    if feature_names != ("bounded_risk",):
        raise ValueError("A2 permits only M6 bounded_risk as a learned feature.")
    allowed_status = frozenset(
        str(value) for value in payload.get("allowed_branch_status", ())
    )
    required_status = {
        "observed",
        "missing",
        "not_applicable",
        "blocked_by_input",
        "provider_failed",
    }
    if allowed_status != required_status:
        raise ValueError("Feature contract branch statuses do not match A2.")
    invariants = payload.get("invariants", {})
    if invariants.get("provider_failure_is_anomaly_evidence") is not False:
        raise ValueError("Provider failure must remain excluded from anomaly evidence.")
    if invariants.get("test_split_allowed") is not False:
        raise ValueError("A2 feature contract must reject test split.")
    return FeatureContract(
        schema_version=str(payload["schema_version"]),
        version=version,
        source_schema_version=str(payload.get("source_schema_version", "")),
        branch_order=branch_order,
        feature_names=feature_names,
        reliability_names=reliability_names,
        allowed_branch_status=allowed_status,
        raw=payload,
        sha256=sha256_file(contract_path),
    )
