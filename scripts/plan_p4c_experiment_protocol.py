#!/usr/bin/env python3
"""Plan the P4-C0 protocol from metadata, isolated labels, and coverage only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.writer import (  # noqa: E402
    atomic_write_bytes,
    atomic_write_json,
    sha256_file,
    write_parquet,
)
from semantic3d.experiment_protocol.branch_eligibility import (  # noqa: E402
    branch_rows,
    build_branch_eligibility,
)
from semantic3d.experiment_protocol.coverage_planning import (  # noqa: E402
    build_coverage_targets,
    build_missingness_bias_report,
)
from semantic3d.experiment_protocol.duplicate_audit import audit_duplicates  # noqa: E402
from semantic3d.experiment_protocol.inventory import (  # noqa: E402
    build_dataset_catalog,
    build_video_inventory,
    inventory_rows,
    load_isolated_labels,
    read_parquet_rows,
)
from semantic3d.experiment_protocol.leakage_audit import audit_leakage  # noqa: E402
from semantic3d.experiment_protocol.schema import (  # noqa: E402
    PROTOCOL_VERSION,
    SourceGroupRecord,
    SplitAssignment,
)
from semantic3d.experiment_protocol.source_grouping import assign_source_groups  # noqa: E402
from semantic3d.experiment_protocol.split_planner import (  # noqa: E402
    plan_group_aware_split,
    split_rows,
)
from semantic3d.experiment_protocol.storage_planning import estimate_storage  # noqa: E402
from semantic3d.experiment_protocol.validation import validate_protocol  # noqa: E402


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _csv_bytes(rows: Iterable[Mapping[str, Any]], columns: tuple[str, ...]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        normalized = {
            key: json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in row.items()
        }
        writer.writerow(normalized)
    return handle.getvalue().encode("utf-8")


def _columns(dataclass_type: type[Any]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(dataclass_type))


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def build_protocol(config_path: Path) -> dict[str, Any]:
    """Build all P4-C0 artifacts without reading residual magnitudes."""

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    structural_root = _resolve(config["inputs"]["structural_dataset_root"])
    labels_path = _resolve(config["inputs"]["labels_manifest"])
    output_root = _resolve(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    structural_manifest = json.loads((structural_root / "dataset_manifest.json").read_text(encoding="utf-8"))
    video_rows = read_parquet_rows(structural_root / "manifests/videos.parquet")
    labels = load_isolated_labels(labels_path)
    catalog_config = config["dataset_catalog"]
    source_groups = assign_source_groups(
        video_rows,
        labels,
        dataset_name=str(catalog_config["dataset_name"]),
    )
    inventory = build_video_inventory(
        structural_root,
        PROJECT_ROOT,
        labels,
        source_groups,
        dataset_name=str(catalog_config["dataset_name"]),
        protocol_smoke_only=bool(config["protocol"]["protocol_smoke_only"]),
    )
    catalog = build_dataset_catalog(
        inventory,
        dataset_name=str(catalog_config["dataset_name"]),
        dataset_version=str(catalog_config["dataset_version"]),
        source_root=str(structural_root),
        license_or_usage_note=str(catalog_config["license_or_usage_note"]),
        expected_storage_bytes=_directory_size(structural_root),
    )
    split_config = config["split"]
    assignments = plan_group_aware_split(
        inventory,
        ratios={name: float(split_config[name]) for name in ("train", "validation", "test")},
        random_seed=int(config["protocol"]["random_seed"]),
        preserve_declared_smoke_split=bool(config["protocol"]["preserve_declared_smoke_split"]),
    )
    duplicate_config = config["duplicate_audit"]
    duplicate_rows = audit_duplicates(
        inventory,
        assignments,
        near_duplicate_threshold=float(duplicate_config["near_duplicate_threshold"]),
        sample_count=int(duplicate_config["perceptual_sample_count"]),
    )
    conflicts, leakage_summary = audit_leakage(
        inventory,
        assignments,
        source_groups.values(),
        duplicate_rows,
        near_duplicate_threshold=float(duplicate_config["near_duplicate_threshold"]),
    )
    eligibility = build_branch_eligibility(structural_root)
    acceptance = json.loads(
        (structural_root / "reports/p4b5_acceptance_summary.json").read_text(encoding="utf-8")
    )
    coverage_targets = build_coverage_targets(inventory, eligibility, acceptance)
    missingness_rows, missingness_summary = build_missingness_bias_report(
        inventory,
        assignments,
        structural_root,
        eligibility,
        shortcut_difference_threshold=float(
            config["coverage_planning"]["shortcut_difference_threshold"]
        ),
    )
    storage_config = config["storage"]
    storage = estimate_storage(
        structural_root,
        frame_count=sum(row.frame_count for row in inventory),
        source_video_bytes=sum(row.file_size for row in inventory),
        planned_frame_count=int(storage_config["planned_frame_count"]),
        source_root=storage_config["source_root"],
        dataset_output_root=storage_config["dataset_output_root"],
        cache_root=storage_config["cache_root"],
        archive_root=storage_config["archive_root"],
    )

    write_parquet(
        output_root / "dataset_catalog.parquet",
        [catalog.to_dict()],
        columns=tuple(catalog.to_dict()),
    )
    group_rows = [asdict(row) for row in sorted(source_groups.values(), key=lambda item: item.video_id)]
    write_parquet(
        output_root / "source_groups.parquet",
        group_rows,
        columns=_columns(SourceGroupRecord),
    )
    inventory_table_rows = inventory_rows(inventory)
    write_parquet(
        output_root / "video_inventory.parquet",
        inventory_table_rows,
        columns=tuple(inventory_table_rows[0]),
    )
    split_table_rows = split_rows(assignments)
    write_parquet(
        output_root / "split_manifest.parquet",
        split_table_rows,
        columns=_columns(SplitAssignment),
    )
    labels_schema = {
        "schema_version": "p4c0_label_schema_v1",
        "label_inference_from_filename_allowed": False,
        "binary_video_label": {"values": {"0": "real", "1": "fake"}, "unknown": None},
        "manipulation_type": {"unknown_value": "unknown"},
        "temporal_intervals": {"unit": "seconds_or_frame_range", "not_available": []},
        "spatial_mask": {"not_available": ""},
        "spatial_bbox": {"not_available": []},
        "object_annotation": {"not_available": ""},
        "source_group_id_required": True,
        "annotation_quality_required": True,
        "annotation_source_required": True,
        "source_manifest": str(labels_path),
    }
    atomic_write_json(output_root / "labels_schema.json", labels_schema)
    annotation_rows = [
        {
            "video_id": row.video_id,
            "binary_video_label_available": row.binary_label is not None,
            "manipulation_type_available": row.manipulation_type != "unknown",
            "temporal_annotation_available": row.temporal_annotation_available,
            "spatial_annotation_available": row.spatial_annotation_available,
            "object_annotation_available": row.object_annotation_available,
            "annotation_quality": row.annotation_quality,
            "annotation_source": labels[row.source_name].annotation_source,
            "temporal_localization_eligible": row.temporal_localization_eligible,
            "spatial_localization_eligible": row.spatial_localization_eligible,
        }
        for row in inventory
    ]
    write_parquet(
        output_root / "annotation_availability.parquet",
        annotation_rows,
        columns=tuple(annotation_rows[0]),
    )
    eligibility_rows = branch_rows(eligibility)
    write_parquet(
        output_root / "branch_eligibility.parquet",
        eligibility_rows,
        columns=tuple(eligibility_rows[0]),
    )
    write_parquet(
        output_root / "duplicate_audit.parquet",
        duplicate_rows,
        columns=tuple(duplicate_rows[0]) if duplicate_rows else (
            "video_id_a", "video_id_b", "exact_duplicate", "near_duplicate_score",
            "perceptual_hash_score", "metadata_similarity", "same_source_group",
            "split_conflict", "review_required", "hash_sample_count_a",
            "hash_sample_count_b", "embedding_provider", "residual_fields_used",
        ),
    )
    conflict_columns = (
        "conflict_type", "severity", "video_id_a", "video_id_b", "source_group_id",
        "split_a", "split_b", "details", "review_required",
    )
    atomic_write_bytes(
        output_root / "leakage_conflicts.csv", _csv_bytes(conflicts, conflict_columns)
    )
    atomic_write_json(
        output_root / "leakage_audit.json",
        {
            **leakage_summary,
            "reported_conflict_types": sorted({row["conflict_type"] for row in conflicts}),
            "conflicts": conflicts,
        },
    )
    missingness_columns = tuple(missingness_rows[0]) if missingness_rows else (
        "split", "binary_label", "video_count",
    )
    atomic_write_bytes(
        output_root / "missingness_bias_by_split.csv",
        _csv_bytes(missingness_rows, missingness_columns),
    )
    atomic_write_json(output_root / "missingness_bias_report.json", missingness_summary)
    atomic_write_json(output_root / "coverage_targets.json", coverage_targets)
    atomic_write_json(output_root / "storage_estimate.json", storage)

    split_by_video = {row.video_id: row.split for row in assignments}
    normal_reference_ids = [
        row.video_id
        for row in inventory
        if split_by_video[row.video_id] == "train" and row.binary_label == 0 and not row.geometry_validation_only
    ]
    supervised_train_ids = [
        row.video_id
        for row in inventory
        if split_by_video[row.video_id] == "train" and row.binary_label is not None and not row.geometry_validation_only
    ]
    task_counts = {
        "detection_training_eligible": sum(row.detection_training_eligible for row in inventory),
        "video_classification_eligible": sum(row.video_classification_eligible for row in inventory),
        "temporal_localization_eligible": sum(row.temporal_localization_eligible for row in inventory),
        "spatial_localization_eligible": sum(row.spatial_localization_eligible for row in inventory),
        "object_localization_eligible": sum(row.object_localization_eligible for row in inventory),
        "occlusion_validation_eligible": sum(row.occlusion_validation_eligible for row in inventory),
        "geometry_validation_only": sum(row.geometry_validation_only for row in inventory),
    }
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "structural_dataset_id": structural_manifest["dataset_id"],
        "structural_pipeline_version": structural_manifest["pipeline_version"],
        "strict_prior_hashes": structural_manifest.get("strict_prior_hashes", {}),
        "six_video_protocol_smoke_only": True,
        "formal_performance_claim_allowed": False,
        "statistical_parameters_fitted": False,
        "classification_metrics_computed": False,
        "residual_values_read_for_split": False,
        "dataset_roles": list(catalog.roles),
        "split_policy": {
            "official_split_priority": True,
            "group_aware_when_no_official_split": True,
            "frame_or_clip_random_split_allowed": False,
            "algorithm_version": assignments[0].algorithm_version,
            "random_seed": assignments[0].random_seed,
            "decision_inputs": list(assignments[0].decision_inputs),
        },
        "label_protocol": {
            "source_manifest": str(labels_path),
            "labels_manifest_is_external_to_structural_cache": True,
            "filename_label_inference": False,
        },
        "branch_tiers": {
            "S": "frame static 3D eligible",
            "D1": "static_camera_3d eligible",
            "D2": "rotation_compensated eligible",
            "D3": "full_se3_3d eligible",
            "O": "occlusion observable",
        },
        "task_eligibility_counts": task_counts,
        "modeling_routes": {
            "normal_reference": {
                "fit_splits": ["train"],
                "fit_class": "real_only",
                "fit_video_ids": normal_reference_ids,
                "group_by": ["branch", "geometry_mode", "evidence_level"],
                "parameters_fitted_now": False,
            },
            "supervised_aggregation": {
                "training_splits": ["train"],
                "training_video_ids": supervised_train_ids,
                "validation_splits": ["validation"],
                "threshold_tuning_splits": ["validation"],
                "final_evaluation_splits": ["test"],
                "parameters_fitted_now": False,
            },
        },
        "duplicate_policy": {
            "near_duplicate_threshold": float(duplicate_config["near_duplicate_threshold"]),
            "visual_embedding_provider": "optional_not_configured",
        },
        "leakage_audit": {
            **leakage_summary,
            "reported_conflict_types": sorted({row["conflict_type"] for row in conflicts}),
        },
        "missingness_bias": missingness_summary,
        "source_group_review_required_count": sum(
            row.source_group_review_required for row in inventory
        ),
        "ready_for_p4b_scale_formal_build": False,
        "readiness_reason": "source identities, formal datasets, annotations, and branch/event coverage remain incomplete",
    }
    atomic_write_json(output_root / "experiment_protocol.json", protocol)
    validation = validate_protocol(output_root, structural_root)
    atomic_write_json(output_root / "protocol_validation.json", asdict(validation))
    return {
        "output_root": str(output_root),
        "video_count": len(inventory),
        "source_group_count": len(source_groups),
        "split_counts": {
            split: sum(row.split == split for row in assignments)
            for split in ("train", "validation", "test")
        },
        "task_eligibility_counts": task_counts,
        "source_group_review_required_count": protocol["source_group_review_required_count"],
        "duplicate_pair_count": len(duplicate_rows),
        "leakage_conflict_count": len(conflicts),
        "shortcut_risk": missingness_summary["shortcut_risk"],
        "protocol_valid": validation.valid,
        "validation_errors": list(validation.errors),
        "validation_warnings": list(validation.warnings),
        "statistical_parameters_fitted": False,
        "classification_metrics_computed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/p4c0_experiment_protocol_v1.yaml",
        help="P4-C0 planning configuration",
    )
    args = parser.parse_args()
    result = build_protocol(_resolve(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["protocol_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
