"""Validation helpers for metadata-only dataset candidate audits."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

ALLOWED_AUDIT_STATUSES = frozenset(
    {
        "VERIFIED",
        "PARTIAL",
        "UNVERIFIED",
        "BLOCKED",
        "NOT_AVAILABLE",
        "NOT_APPLICABLE",
    }
)
ALLOWED_MAPPING_STATUSES = ALLOWED_AUDIT_STATUSES | {"UNRESOLVED_SCHEMA"}
SCORE_FIELDS = tuple("ABCDEFGHIJ")
FORMAL_MAPPING_FIELDS = (
    "sample_id",
    "video_path",
    "label",
    "split",
    "source_dataset",
    "source_id",
    "source_lineage",
    "generator",
    "is_real",
    "temporal_annotation",
    "spatial_annotation",
    "metadata_status",
)
STATUS_FIELDS = (
    "terms_acceptance_required",
    "access_approval_required",
    "commercial_use",
    "record_metadata_available",
    "official_split_available",
    "temporal_annotation_available",
    "spatial_annotation_available",
    "mask_available",
    "source_lineage_available",
    "checksum_available",
    "single_file_download_supported",
    "include_exclude_supported",
    "streaming_supported",
    "archive_layout_known",
    "extracted_layout_known",
    "small_sample_selection_possible",
    "small_sample_selection_without_test_possible",
    "formal_schema_mapping_possible",
    "license_status",
    "schema_status",
    "download_status",
)


class CandidateAuditError(ValueError):
    """Raised when a candidate registry violates an audit safety invariant."""


def load_candidate_registry(path: str | Path) -> dict[str, Any]:
    """Load and validate a candidate registry without network or media access."""

    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateAuditError("candidate registry must be a YAML object")
    validate_candidate_registry(payload)
    return payload


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CandidateAuditError(f"{context} must be an object")
    return value


def validate_candidate_registry(payload: Mapping[str, Any]) -> None:
    """Enforce metadata-only, candidate-only, non-inference audit semantics."""

    policies = _require_mapping(payload.get("policies"), context="policies")
    expected_policies = {
        "media_download_bytes": 0,
        "metadata_only": True,
        "random_split_allowed": False,
        "official_test_media_allowed_in_small_sample": False,
        "label_from_path_allowed": False,
        "generator_from_filename_allowed": False,
        "external_cache_tracked_in_git": False,
    }
    for key, expected in expected_policies.items():
        if policies.get(key) != expected:
            raise CandidateAuditError(f"unsafe audit policy {key!r}")

    declared_statuses = set(payload.get("allowed_status_values", ()))
    if declared_statuses != ALLOWED_AUDIT_STATUSES:
        raise CandidateAuditError("allowed status vocabulary changed")

    required_fields = payload.get("required_record_fields")
    if not isinstance(required_fields, list) or not required_fields:
        raise CandidateAuditError("required_record_fields must be a non-empty list")

    records = payload.get("datasets")
    if not isinstance(records, list) or not records:
        raise CandidateAuditError("datasets must be a non-empty list")

    dataset_ids: set[str] = set()
    for index, value in enumerate(records):
        record = _require_mapping(value, context=f"datasets[{index}]")
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise CandidateAuditError(
                f"{record.get('dataset_id', index)!r} missing fields: {missing}"
            )

        dataset_id = record.get("dataset_id")
        if not isinstance(dataset_id, str) or not dataset_id:
            raise CandidateAuditError("dataset_id must be a non-empty string")
        if dataset_id in dataset_ids:
            raise CandidateAuditError(f"duplicate dataset_id: {dataset_id}")
        dataset_ids.add(dataset_id)

        if record.get("selection_status") != "candidate":
            raise CandidateAuditError(f"{dataset_id} is not marked candidate")
        for field in (
            "data_downloaded",
            "production_adapter_ready",
            "official_schema_verified",
        ):
            if record.get(field) is not False:
                raise CandidateAuditError(f"{dataset_id}.{field} must remain false")

        for field in STATUS_FIELDS:
            if record.get(field) not in ALLOWED_AUDIT_STATUSES:
                raise CandidateAuditError(
                    f"{dataset_id}.{field} has an invalid audit status"
                )

        if not str(record.get("license", "")).strip():
            raise CandidateAuditError(f"{dataset_id}.license is empty")

        scores = _require_mapping(record.get("scores"), context=f"{dataset_id}.scores")
        if tuple(scores.keys()) != SCORE_FIELDS:
            raise CandidateAuditError(f"{dataset_id} score fields must be A through J")
        if any(
            isinstance(score, bool)
            or not isinstance(score, int)
            or not 0 <= score <= 5
            for score in scores.values()
        ):
            raise CandidateAuditError(f"{dataset_id} scores must be integers in [0, 5]")

        mapping = _require_mapping(
            record.get("formal_mapping"),
            context=f"{dataset_id}.formal_mapping",
        )
        if tuple(mapping.keys()) != FORMAL_MAPPING_FIELDS:
            raise CandidateAuditError(
                f"{dataset_id} formal mapping fields are incomplete or reordered"
            )
        invalid_mapping = set(mapping.values()) - ALLOWED_MAPPING_STATUSES
        if invalid_mapping:
            raise CandidateAuditError(
                f"{dataset_id} has invalid mapping statuses: {sorted(invalid_mapping)}"
            )
        if (
            not record["official_schema_verified"]
            and record["production_adapter_ready"]
        ):
            raise CandidateAuditError(
                f"{dataset_id} unverified schema cannot have a production adapter"
            )

    recommendation = _require_mapping(
        payload.get("recommendation"),
        context="recommendation",
    )
    selected = {
        recommendation.get("primary_detection_dataset"),
        recommendation.get("primary_spatial_localization_dataset"),
        recommendation.get("secondary_generalization_dataset"),
        recommendation.get("deferred_dataset"),
    }
    if selected != dataset_ids:
        raise CandidateAuditError("recommendation must classify every candidate exactly once")
    if recommendation.get("small_sample_download_ready") is not False:
        raise CandidateAuditError("small-sample media download must remain blocked")


def summarize_candidate_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, deterministic audit summary."""

    validate_candidate_registry(payload)
    recommendation = payload["recommendation"]
    return {
        "schema_version": payload["schema_version"],
        "candidate_count": len(payload["datasets"]),
        "dataset_ids": [record["dataset_id"] for record in payload["datasets"]],
        "media_download_bytes": payload["policies"]["media_download_bytes"],
        "primary_detection_dataset": recommendation["primary_detection_dataset"],
        "primary_spatial_localization_dataset": recommendation[
            "primary_spatial_localization_dataset"
        ],
        "secondary_generalization_dataset": recommendation[
            "secondary_generalization_dataset"
        ],
        "deferred_dataset": recommendation["deferred_dataset"],
        "small_sample_source_identified": recommendation[
            "small_sample_source_identified"
        ],
        "small_sample_download_ready": recommendation["small_sample_download_ready"],
    }
