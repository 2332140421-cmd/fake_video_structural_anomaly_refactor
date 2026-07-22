"""Validate P4-C0 protocol outputs and fail on severe leakage."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .schema import ProtocolValidationResult

REQUIRED_FILES = (
    "dataset_catalog.parquet",
    "source_groups.parquet",
    "video_inventory.parquet",
    "split_manifest.parquet",
    "labels_schema.json",
    "annotation_availability.parquet",
    "branch_eligibility.parquet",
    "duplicate_audit.parquet",
    "leakage_audit.json",
    "leakage_conflicts.csv",
    "missingness_bias_by_split.csv",
    "missingness_bias_report.json",
    "coverage_targets.json",
    "storage_estimate.json",
    "experiment_protocol.json",
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def validate_protocol(
    protocol_root: str | Path,
    structural_dataset_root: str | Path,
) -> ProtocolValidationResult:
    """Apply the 16 P4-C0 protocol checks to generated artifacts."""

    root = Path(protocol_root)
    structural = Path(structural_dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            errors.append(f"missing_required_file:{name}")
    if errors:
        return ProtocolValidationResult(False, len(errors), 0, tuple(errors), (), {}, {})

    inventory = _rows(root / "video_inventory.parquet")
    groups = _rows(root / "source_groups.parquet")
    splits = _rows(root / "split_manifest.parquet")
    duplicates = _rows(root / "duplicate_audit.parquet")
    branches = _rows(root / "branch_eligibility.parquet")
    annotations = _rows(root / "annotation_availability.parquet")
    protocol = json.loads((root / "experiment_protocol.json").read_text(encoding="utf-8"))
    storage = json.loads((root / "storage_estimate.json").read_text(encoding="utf-8"))
    structural_manifest = json.loads((structural / "dataset_manifest.json").read_text(encoding="utf-8"))

    video_ids = [str(row["video_id"]) for row in inventory]
    checks["video_id_unique"] = len(video_ids) == len(set(video_ids))
    if not checks["video_id_unique"]:
        errors.append("video_id_not_unique")

    group_by_video = {str(row["video_id"]): str(row["source_group_id"]) for row in groups}
    checks["source_group_complete"] = set(video_ids) == set(group_by_video) and all(group_by_video.values())
    if not checks["source_group_complete"]:
        errors.append("source_group_incomplete")

    split_by_video = {str(row["video_id"]): str(row["split"]) for row in splits}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for video_id, group_id in group_by_video.items():
        group_splits[group_id].add(split_by_video.get(video_id, ""))
    checks["source_group_not_cross_split"] = all(len(values) == 1 for values in group_splits.values())
    if not checks["source_group_not_cross_split"]:
        errors.append("source_group_cross_split")

    exact_conflicts = [
        row for row in duplicates if row["exact_duplicate"] and row["split_conflict"]
    ]
    checks["exact_duplicate_not_cross_split"] = not exact_conflicts
    if exact_conflicts:
        errors.append("exact_duplicate_cross_split")

    near_conflicts = [
        row
        for row in duplicates
        if float(row["near_duplicate_score"]) >= float(protocol["duplicate_policy"]["near_duplicate_threshold"])
        and row["split_conflict"]
    ]
    checks["near_duplicate_conflicts_marked"] = all(row["review_required"] for row in near_conflicts)
    if not checks["near_duplicate_conflicts_marked"]:
        errors.append("near_duplicate_conflict_not_marked")

    official_conflict = any(row["split"] == "official_conflict" for row in splits)
    reported_types = set(protocol["leakage_audit"]["reported_conflict_types"])
    checks["official_split_conflict_reported"] = not official_conflict or "official_split_source_group_conflict" in reported_types
    if not checks["official_split_conflict_reported"]:
        errors.append("official_split_conflict_not_reported")

    checks["labels_structural_isolation"] = bool(
        structural_manifest.get("label_isolation")
        and not structural_manifest.get("truth_labels_used", False)
        and protocol["label_protocol"]["labels_manifest_is_external_to_structural_cache"]
    )
    if not checks["labels_structural_isolation"]:
        errors.append("labels_not_isolated_from_structural_builder")

    normal_splits = set(protocol["modeling_routes"]["normal_reference"]["fit_splits"])
    threshold_splits = set(protocol["modeling_routes"]["supervised_aggregation"]["threshold_tuning_splits"])
    checks["test_not_normalization_fit_source"] = "test" not in normal_splits
    checks["test_not_threshold_tuning_source"] = "test" not in threshold_splits
    if not checks["test_not_normalization_fit_source"]:
        errors.append("test_in_normalization_fit_source")
    if not checks["test_not_threshold_tuning_source"]:
        errors.append("test_in_threshold_tuning_source")

    clips = _rows(structural / "manifests/clips.parquet")
    checks["clip_not_cross_split"] = all(str(row["video_id"]) in split_by_video for row in clips)
    if not checks["clip_not_cross_split"]:
        errors.append("clip_has_no_video_split")

    allowed_status = {"applicable", "not_applicable", "observation_missing", "invalid_geometry", "unsupported_mode"}
    checks["branch_eligibility_semantics"] = all(row["eligibility_status"] in allowed_status for row in branches)
    checks["not_applicable_distinct_from_observation_missing"] = "not_applicable" in {row["eligibility_status"] for row in branches} and "observation_missing" in {row["eligibility_status"] for row in branches}
    if not checks["branch_eligibility_semantics"]:
        errors.append("invalid_branch_eligibility_status")

    checks["temporal_annotation_gates_localization"] = all(
        bool(row["temporal_annotation_available"]) or not bool(row["temporal_localization_eligible"])
        for row in inventory
    )
    checks["spatial_annotation_gates_localization"] = all(
        bool(row["spatial_annotation_available"]) or not bool(row["spatial_localization_eligible"])
        for row in inventory
    )
    if not checks["temporal_annotation_gates_localization"]:
        errors.append("temporal_localization_without_annotation")
    if not checks["spatial_annotation_gates_localization"]:
        errors.append("spatial_localization_without_annotation")

    output_path = storage["paths"]["dataset_output_root"]
    checks["storage_path_checked"] = int(output_path["free_bytes"]) >= 0
    estimated_peak = int(storage["formal_build_estimate"]["temporary_peak_bytes"])
    checks["storage_capacity_sufficient"] = int(output_path["free_bytes"]) >= estimated_peak
    if not checks["storage_capacity_sufficient"]:
        warnings.append("configured_dataset_output_root_has_insufficient_free_space_for_estimated_peak")

    decision_inputs = {value for row in splits for value in json.loads(row["decision_inputs"])}
    checks["split_does_not_read_residuals"] = not any("residual" in value.lower() or "score" in value.lower() for value in decision_inputs)
    if not checks["split_does_not_read_residuals"]:
        errors.append("split_decision_uses_residual_or_score")

    unresolved = sum(bool(row["source_group_review_required"]) for row in inventory)
    if unresolved:
        warnings.append(f"source_group_review_required:{unresolved}")
    if len(inventory) <= 6:
        warnings.append("six_video_protocol_smoke_only_not_formal_performance_data")

    metrics = {
        "video_count": len(inventory),
        "source_group_count": len(set(group_by_video.values())),
        "clip_count": len(clips),
        "duplicate_pair_count": len(duplicates),
        "near_duplicate_conflict_count": len(near_conflicts),
        "source_group_review_required_count": unresolved,
        "branch_eligibility_row_count": len(branches),
        "annotation_availability_row_count": len(annotations),
    }
    return ProtocolValidationResult(
        valid=not errors,
        error_count=len(errors),
        warning_count=len(warnings),
        errors=tuple(errors),
        warnings=tuple(warnings),
        checks=checks,
        metrics=metrics,
    )
