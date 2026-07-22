"""Build and validate the deterministic P4-C1 clip manifest."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import yaml

from semantic3d.dataset_builder.writer import sha256_file

from .data_inventory import AvailabilityIndex, load_p4c1_inventory, project_relative_path
from .exclusion_policy import evaluate_exclusion_reasons, exclusion_reason_text
from .leakage_audit import audit_manifest_leakage
from .manifest_report import manifest_jsonl_bytes
from .manifest_schema import (
    P4C1_MANIFEST_SCHEMA_VERSION,
    ExperimentSampleRecord,
    ManifestBuildResult,
    stable_sample_id,
)


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_manifest_config(config_path: str | Path) -> dict[str, Any]:
    """Load a P4-C1 YAML configuration and check its non-modeling scope."""

    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("manifest", {}).get("version") != P4C1_MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported P4-C1 manifest version")
    if config.get("manifest", {}).get("sample_unit") != "clip":
        raise ValueError("P4-C1 currently supports clip samples only")
    prohibited = {
        "train_model",
        "fit_parameters",
        "fit_distribution",
        "select_threshold",
        "compute_performance",
    }
    enabled = [name for name in prohibited if bool(config.get("forbidden_operations", {}).get(name))]
    if enabled:
        raise ValueError(f"P4-C1 forbidden operation enabled: {sorted(enabled)}")
    return config


def build_p4c1_manifest(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> ManifestBuildResult:
    """Build samples from frozen P4-C0 splits and existing P4-B.5 indexes."""

    project = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_manifest_config(config_file)
    inputs = config["inputs"]
    p4c0_config_path = _resolve(project, inputs["p4c0_config"]).resolve()
    p4c0_config = yaml.safe_load(p4c0_config_path.read_text(encoding="utf-8"))
    protocol_root = _resolve(project, inputs.get("p4c0_protocol_root", p4c0_config["output_root"])).resolve()
    structural_root = _resolve(
        project,
        inputs.get(
            "structural_dataset_root",
            p4c0_config["inputs"]["structural_dataset_root"],
        ),
    ).resolve()
    labels_path = _resolve(project, p4c0_config["inputs"]["labels_manifest"]).resolve()
    protocol_path = protocol_root / "experiment_protocol.json"
    validation_path = protocol_root / "protocol_validation.json"
    if bool(config["integrity"].get("require_p4c0_protocol_valid", True)):
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if not bool(validation.get("valid", False)):
            raise ValueError("P4-C0 protocol validation is not valid")

    data = load_p4c1_inventory(protocol_root, structural_root)
    expected_protocol_version = str(p4c0_config["protocol"]["version"])
    if str(data.experiment_protocol.get("protocol_version")) != expected_protocol_version:
        raise ValueError("P4-C0 protocol version does not match its frozen configuration")
    protocol_sha256 = sha256_file(protocol_path)
    required_modalities = tuple(config["availability"]["required_for_usable"])
    availability_index = AvailabilityIndex(structural_root, project)
    videos_by_id = {str(row["video_id"]): row for row in data.videos}

    table_paths = {
        "frame_manifest_path": structural_root / "manifests/frames.parquet",
        "object_observations_path": structural_root / "observations/objects.parquet",
        "depth_observations_path": structural_root / "observations/depth.parquet",
        "pose_observations_path": structural_root / "observations/camera.parquet",
        "track_observations_path": structural_root / "observations/point_tracks_2d.parquet",
        "semantic3d_observations_path": structural_root / "observations/shared_3d_frames.parquet",
    }
    portable_paths = {
        name: project_relative_path(path, project) for name, path in table_paths.items()
    }
    records: list[ExperimentSampleRecord] = []
    availability_rows = []
    sorted_clips = sorted(
        data.clips,
        key=lambda row: (
            str(row["video_id"]),
            int(row["start_frame_index"]),
            int(row["end_frame_index"]),
            str(row["clip_id"]),
        ),
    )
    for clip in sorted_clips:
        video_id = str(clip["video_id"])
        if video_id not in videos_by_id:
            raise ValueError(f"Clip references an unknown source video: {video_id}")
        if video_id not in data.inventory_by_video:
            raise ValueError(f"P4-C0 inventory row is missing for video: {video_id}")
        if video_id not in data.split_by_video:
            raise ValueError(f"P4-C0 split row is missing for video: {video_id}")
        if video_id not in data.source_group_by_video:
            raise ValueError(f"P4-C0 source-group row is missing for video: {video_id}")
        video = videos_by_id[video_id]
        inventory = data.inventory_by_video[video_id]
        split_row = data.split_by_video[video_id]
        group = data.source_group_by_video[video_id]
        availability = availability_index.audit_clip(clip, video)
        reasons = evaluate_exclusion_reasons(
            availability,
            clip_valid=bool(clip.get("valid", True)),
            clip_missing_reason=str(clip.get("missing_reason", "")),
            authenticity_label=inventory.get("binary_label"),
            split=str(split_row["split"]),
            required_modalities=required_modalities,
            require_complete_decoded_frames=bool(
                config["availability"].get("require_complete_decoded_frames", True)
            ),
        )
        start = int(clip["start_frame_index"])
        end = int(clip["end_frame_index"])
        record = ExperimentSampleRecord(
            sample_id=stable_sample_id(
                str(inventory["dataset_name"]), video_id, str(clip["clip_id"]), start, end
            ),
            manifest_schema_version=P4C1_MANIFEST_SCHEMA_VERSION,
            dataset_name=str(inventory["dataset_name"]),
            source_video_id=video_id,
            source_video_name=str(video["source_name"]),
            source_group_id=str(group["source_group_id"]),
            source_group_review_required=bool(group["source_group_review_required"]),
            source_sha256=str(video["source_sha256"]),
            video_path=str(video["source_relative_path"]),
            clip_id=str(clip["clip_id"]),
            clip_start=start,
            clip_end=end,
            core_clip_start=int(clip["core_start_frame_index"]),
            core_clip_end=int(clip["core_end_frame_index"]),
            num_frames=end - start + 1,
            split=str(split_row["split"]),
            authenticity_label=(
                int(inventory["binary_label"])
                if inventory.get("binary_label") is not None
                else None
            ),
            authenticity_label_name=str(inventory["label_name"]),
            manipulation_type=str(inventory["manipulation_type"]),
            scene_id=int(clip["scene_id"]),
            camera_id="",
            camera_identity_status="camera_identity_unavailable",
            coordinate_system_id=str(clip["coordinate_system_id"]),
            valid_object_count=availability.valid_object_count,
            valid_depth_count=availability.valid_depth_count,
            valid_pose_count=availability.valid_pose_count,
            valid_track_point_count=availability.valid_track_point_count,
            valid_semantic3d_count=availability.valid_semantic3d_count,
            usable=not reasons,
            exclusion_reason=exclusion_reason_text(reasons),
            protocol_sha256=protocol_sha256,
            metadata={
                "split_source": str(split_row["split_source"]),
                "source_grouping_basis": str(group["grouping_basis"]),
                "source_group_review_reason": str(group["review_reason"]),
                "geometry_mode": str(clip["geometry_mode"]),
                "sequence_scale_status": str(clip["sequence_scale_status"]),
                "pose_graph_id": str(clip["pose_graph_id"]),
                "availability_missing_modalities": list(availability.missing_modalities),
                "geometry_validation_only": bool(inventory["geometry_validation_only"]),
                "overlapping_clip_inherits_video_split": True,
                "residual_values_read": False,
            },
            **portable_paths,
        )
        records.append(record)
        availability_rows.append(availability)

    paired = sorted(zip(records, availability_rows), key=lambda item: item[0].sample_id)
    records = [item[0] for item in paired]
    availability_rows = [item[1] for item in paired]
    near_threshold = float(config["leakage_audit"]["near_duplicate_threshold"])
    findings, leakage_summary = audit_manifest_leakage(
        records,
        data.source_group_by_video,
        data.duplicate_rows,
        near_duplicate_threshold=near_threshold,
    )
    return ManifestBuildResult(
        records=tuple(records),
        availability=tuple(availability_rows),
        leakage_findings=tuple(findings),
        leakage_summary=leakage_summary,
        protocol_sha256=protocol_sha256,
        p4c0_config_sha256=sha256_file(p4c0_config_path),
        source_manifest_sha256=sha256_file(labels_path),
        config_sha256=sha256_file(config_file),
        structural_dataset_id=str(data.structural_manifest["dataset_id"]),
    )


def manifest_sha256(result: ManifestBuildResult) -> str:
    """Return the deterministic hash of the canonical JSONL manifest."""

    return hashlib.sha256(manifest_jsonl_bytes(result.records)).hexdigest()


def validate_manifest_artifacts(
    output_root: str | Path,
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Validate hashes, ordering, schema, split inheritance, and reproducibility."""

    root = Path(output_root)
    metadata = json.loads((root / "manifest_metadata.json").read_text(encoding="utf-8"))
    payload = (root / "experiment_manifest.jsonl").read_bytes()
    rows = [json.loads(line) for line in payload.splitlines() if line]
    errors: list[str] = []
    checks: dict[str, bool] = {}
    actual_hash = hashlib.sha256(payload).hexdigest()
    checks["manifest_hash_matches"] = actual_hash == metadata["manifest_sha256"]
    checks["sample_ids_unique"] = len(rows) == len({row["sample_id"] for row in rows})
    checks["sample_order_deterministic"] = [row["sample_id"] for row in rows] == sorted(
        row["sample_id"] for row in rows
    )
    checks["excluded_samples_have_reason"] = all(
        row["usable"] or bool(row["exclusion_reason"]) for row in rows
    )
    checks["usable_samples_have_no_reason"] = all(
        not row["usable"] or not row["exclusion_reason"] for row in rows
    )
    checks["protocol_hash_present"] = all(
        row["protocol_sha256"] == metadata["protocol_sha256"] for row in rows
    )
    checks["camera_identity_not_fabricated"] = all(
        bool(row["camera_id"]) or row["camera_identity_status"] == "camera_identity_unavailable"
        for row in rows
    )
    video_splits: dict[str, set[str]] = {}
    for row in rows:
        video_splits.setdefault(row["source_video_id"], set()).add(row["split"])
    checks["source_video_not_cross_split"] = all(len(values) == 1 for values in video_splits.values())
    checks["leakage_error_free"] = not bool(metadata.get("leakage_error_count", 0))
    checks["no_training_or_fitting"] = all(
        metadata.get(name) is False
        for name in (
            "model_training_performed",
            "statistical_fitting_performed",
            "threshold_selection_performed",
            "classification_performance_computed",
        )
    )
    rebuilt = build_p4c1_manifest(config_path, project_root=project_root)
    rebuilt_hash = manifest_sha256(rebuilt)
    checks["deterministic_rebuild_matches"] = rebuilt_hash == actual_hash
    checks["metadata_rebuild_hash_matches"] = (
        metadata["deterministic_rebuild_sha256"] == rebuilt_hash
    )
    with (root / "experiment_manifest.csv").open(encoding="utf-8", newline="") as handle:
        csv_count = sum(1 for _ in csv.DictReader(handle))
    checks["csv_jsonl_row_count_matches"] = csv_count == len(rows)
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
        "sample_count": len(rows),
        "manifest_sha256": actual_hash,
        "protocol_sha256": metadata["protocol_sha256"],
        "deterministic_rebuild_sha256": rebuilt_hash,
        "training_or_fitting_performed": False,
    }

