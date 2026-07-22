"""Build deterministic P4-C2 formal-data onboarding readiness artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq
import yaml

from semantic3d.dataset_builder.writer import sha256_file

from .formal_registry import build_formal_dataset_registry
from .formal_schema import (
    P4C2BuildResult,
    P4C2_SCHEMA_VERSION,
    SourceLineageRecord,
    TaskEligibilityRecord,
)
from .source_lineage import (
    audit_formal_split_leakage,
    build_smoke_source_lineage,
    plan_verified_formal_splits,
)
from .storage_readiness import build_storage_batch_plan
from .task_eligibility import TASK_NAMES, build_task_eligibility_matrix


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required P4-C2 metadata table is missing: {path}")
    return pq.read_table(path).to_pylist()


def _jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required P4-C2 JSONL input is missing: {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_p4c2_config(path: str | Path) -> dict[str, Any]:
    """Load P4-C2 config and enforce the metadata-only stage boundary."""

    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("p4c2", {}).get("version") != P4C2_SCHEMA_VERSION:
        raise ValueError("Unsupported P4-C2 schema version")
    prohibited = (
        "download_datasets",
        "run_model_inference",
        "extract_large_scale_features",
        "train_model",
        "fit_parameters",
        "fit_distribution",
        "select_threshold",
        "read_test_performance",
        "compute_authenticity_performance",
        "start_formal_batch_build",
    )
    enabled = [name for name in prohibited if bool(config["forbidden_operations"].get(name))]
    if enabled:
        raise ValueError(f"P4-C2 forbidden operation enabled: {sorted(enabled)}")
    return config


def _load_additional_lineage(
    config: Mapping[str, Any], project_root: Path
) -> tuple[SourceLineageRecord, ...]:
    output = []
    for value in config.get("inputs", {}).get("formal_lineage_jsonl", []):
        for row in _jsonl_rows(_resolve(project_root, str(value))):
            output.append(SourceLineageRecord(**row))
    return tuple(output)


def _build_data_inventory(
    videos: list[Mapping[str, Any]],
    p4c1_rows: list[Mapping[str, Any]],
    task_rows: tuple[Any, ...],
    project_root: Path,
) -> tuple[Mapping[str, Any], ...]:
    samples_by_video: dict[str, list[Mapping[str, Any]]] = {}
    for row in p4c1_rows:
        samples_by_video.setdefault(str(row["source_video_id"]), []).append(row)
    output = []
    for video in sorted(videos, key=lambda row: str(row["video_id"])):
        video_id = str(video["video_id"])
        source = Path(str(video["source_path"]))
        if not source.is_absolute():
            source = project_root / source
        exists = source.is_file()
        hash_matches = exists and sha256_file(source) == str(video["source_sha256"])
        samples = samples_by_video.get(video_id, [])
        tasks = [row for row in task_rows if row.derived_video_id == video_id]
        output.append(
            {
                "dataset_name": str(video["dataset_name"]),
                "derived_video_id": video_id,
                "source_video_name": str(video["source_name"]),
                "video_path": str(video["source_path"]),
                "video_exists": exists,
                "video_size_bytes": int(source.stat().st_size) if exists else 0,
                "source_sha256": str(video["source_sha256"]),
                "source_sha256_matches": bool(hash_matches),
                "frame_count": int(video["frame_count"]),
                "clip_count": len(samples),
                "legacy_usable_clip_count": sum(bool(row.get("usable")) for row in samples),
                "object_available_clip_count": sum(int(row.get("valid_object_count", 0)) > 0 for row in samples),
                "depth_available_clip_count": sum(int(row.get("valid_depth_count", 0)) > 0 for row in samples),
                "pose_available_clip_count": sum(int(row.get("valid_pose_count", 0)) > 0 for row in samples),
                "track_available_clip_count": sum(int(row.get("valid_track_point_count", 0)) > 0 for row in samples),
                "semantic3d_available_clip_count": sum(int(row.get("valid_semantic3d_count", 0)) > 0 for row in samples),
                "eligible_task_count_for_smoke": sum(row.eligibility == "eligible" for row in tasks),
                "eligible_task_count_for_formal": sum(row.eligible_for_formal_experiment for row in tasks),
                "metadata_scan_only": True,
                "video_decoded": False,
                "model_inference_performed": False,
            }
        )
    return tuple(output)


def _add_unscanned_formal_task_rows(
    smoke_tasks: tuple[TaskEligibilityRecord, ...],
    lineage: tuple[SourceLineageRecord, ...],
) -> tuple[TaskEligibilityRecord, ...]:
    """Represent unscanned formal candidates explicitly instead of omitting tasks."""

    output = list(smoke_tasks)
    known = {row.derived_video_id for row in smoke_tasks}
    for source in lineage:
        if source.derived_video_id in known:
            continue
        declared = source.metadata.get("task_eligibility", {})
        for task in TASK_NAMES:
            task_metadata = declared.get(task, {}) if isinstance(declared, Mapping) else {}
            status = str(task_metadata.get("eligibility", "unknown"))
            reason = str(task_metadata.get("ineligibility_reason", "lightweight_inventory_not_scanned"))
            eligible = status == "eligible"
            output.append(
                TaskEligibilityRecord(
                    dataset_name=source.dataset_name,
                    derived_video_id=source.derived_video_id,
                    task_name=task,
                    eligibility=status,
                    eligible_for_declared_role=eligible,
                    eligible_for_formal_experiment=eligible and source.formal_split_eligible,
                    evaluation_scope="formal_candidate_inventory",
                    ineligibility_reason="" if eligible else reason,
                    provider_name=str(task_metadata.get("provider_name", "not_scanned")),
                    available_observation_count=int(
                        task_metadata.get("available_observation_count", 0)
                    ),
                    expected_observation_count=int(
                        task_metadata.get("expected_observation_count", 0)
                    ),
                    metadata={
                        "provider_failure_is_anomaly_evidence": False,
                        "lightweight_inventory_only": True,
                    },
                )
            )
    return tuple(
        sorted(output, key=lambda row: (row.dataset_name, row.derived_video_id, row.task_name))
    )


def evaluate_formal_build_readiness(
    datasets: tuple[Any, ...],
    lineage: tuple[SourceLineageRecord, ...],
    formal_splits: tuple[Any, ...],
    leakage: Mapping[str, Any],
    storage: Mapping[str, Any],
    task_rows: tuple[Any, ...],
) -> dict[str, Any]:
    blockers = []
    formal_datasets = [
        row
        for row in datasets
        if row.dataset_role != "geometry_validation_smoke"
        and row.registry_status == "verified_ready"
        and row.metadata.get("license_verification_status") == "verified"
        and not row.missing_requirements
    ]
    if not formal_datasets:
        blockers.append("no_verified_formal_dataset_registered")
    if not any(row.verification_status == "verified" for row in lineage):
        blockers.append("no_verified_original_source_lineage")
    if not any(row.formal_split for row in formal_splits):
        blockers.append("no_formal_split_assignments")
    split_counts = {
        split: sum(row.formal_split == split for row in formal_splits)
        for split in ("train", "validation", "test")
    }
    for split, count in split_counts.items():
        if count == 0:
            blockers.append(f"formal_{split}_split_empty")
    if leakage["cross_split_leakage_detected"]:
        blockers.append("formal_cross_split_leakage_detected")
    if storage["build_blocked_insufficient_storage"]:
        blockers.append("insufficient_storage_for_planned_formal_build")
    if not any(row.task_name == "reprojection_D2" and row.eligible_for_formal_experiment for row in task_rows):
        blockers.append("formal_reprojection_D2_coverage_unavailable")
    if not any(row.task_name == "reprojection_D3" and row.eligible_for_formal_experiment for row in task_rows):
        blockers.append("formal_reprojection_D3_coverage_unavailable")
    if not any(
        row.task_name == "occlusion_and_reappearance" and row.eligible_for_formal_experiment
        for row in task_rows
    ):
        blockers.append("formal_occlusion_and_reappearance_coverage_unavailable")
    if not any(row.task_name == "temporal_localization" and row.eligible_for_formal_experiment for row in task_rows):
        blockers.append("formal_temporal_localization_annotations_unavailable")
    if not any(row.task_name == "spatial_localization" and row.eligible_for_formal_experiment for row in task_rows):
        blockers.append("formal_spatial_localization_annotations_unavailable")
    blockers = sorted(set(blockers))
    return {
        "ready_for_formal_batch_build": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "registered_dataset_count": len(datasets),
        "verified_formal_dataset_count": len(formal_datasets),
        "lineage_record_count": len(lineage),
        "verified_lineage_count": sum(row.verification_status == "verified" for row in lineage),
        "unverified_lineage_count": sum(row.verification_status == "unverified" for row in lineage),
        "formal_split_counts": split_counts,
        "next_stage_conditions": [
            "register an official, licensed formal dataset without using residual results",
            "verify original_source_id and every real-to-derived relationship",
            "freeze official or group-aware train/validation/test assignments",
            "provide task annotations and provider coverage required by the intended experiment",
            "provide storage for full temporary peak plus safety margin",
            "rerun P4-C2 validation until all blockers are cleared",
        ],
        "formal_build_started": False,
        "model_training_performed": False,
        "statistical_fitting_performed": False,
        "threshold_selection_performed": False,
        "test_performance_read": False,
        "authenticity_performance_computed": False,
    }


# Backward-compatible private alias retained for callers from the initial P4-C2 implementation.
_readiness = evaluate_formal_build_readiness


def build_p4c2_readiness(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> P4C2BuildResult:
    """Build P4-C2 readiness entirely from existing metadata and frozen snapshots."""

    project = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = load_p4c2_config(config_file)
    inputs = config["inputs"]
    protocol_root = _resolve(project, inputs["p4c0_protocol_root"])
    p4c1_root = _resolve(project, inputs["p4c1_manifest_root"])
    storage_path = _resolve(project, inputs["p4c0_storage_estimate"])
    experiment_protocol_path = protocol_root / "experiment_protocol.json"
    p4c1_metadata_path = p4c1_root / "manifest_metadata.json"
    p4c1_manifest_path = p4c1_root / "experiment_manifest.jsonl"

    protocol = json.loads(experiment_protocol_path.read_text(encoding="utf-8"))
    if not bool(protocol.get("six_video_protocol_smoke_only", False)):
        raise ValueError("P4-C2 expected the current P4-C0 six-video smoke protocol")
    p4c1_metadata = json.loads(p4c1_metadata_path.read_text(encoding="utf-8"))
    p4c1_rows = _jsonl_rows(p4c1_manifest_path)
    videos = _rows(protocol_root / "video_inventory.parquet")
    source_groups_list = _rows(protocol_root / "source_groups.parquet")
    source_groups = {str(row["video_id"]): row for row in source_groups_list}
    branches = _rows(protocol_root / "branch_eligibility.parquet")

    datasets = build_formal_dataset_registry(config["datasets"], project_root=project)
    smoke_lineage = build_smoke_source_lineage(videos, source_groups)
    lineage = tuple(
        sorted(
            (*smoke_lineage, *_load_additional_lineage(config, project)),
            key=lambda row: (row.dataset_name, row.derived_video_id),
        )
    )
    if len({row.derived_video_id for row in lineage}) != len(lineage):
        raise ValueError("Duplicate derived_video_id in P4-C2 source lineage")
    formal_splits = plan_verified_formal_splits(
        lineage,
        official_split_by_video=config["formal_split"].get("official_split_by_video", {}),
        ratios=config["formal_split"]["ratios"],
        random_seed=int(config["p4c2"]["random_seed"]),
    )
    leakage = audit_formal_split_leakage(lineage, formal_splits)
    tasks = _add_unscanned_formal_task_rows(
        build_task_eligibility_matrix(videos, p4c1_rows, branches),
        lineage,
    )
    inventory = _build_data_inventory(videos, p4c1_rows, tasks, project)
    storage_estimate = json.loads(storage_path.read_text(encoding="utf-8"))
    storage = build_storage_batch_plan(storage_estimate, config["storage"])
    missingness = {
        **dict(config["missingness_audit_plan"]),
        "provider_failure_is_anomaly_evidence": False,
        "test_performance_values_read": False,
        "planning_only": True,
    }
    readiness = evaluate_formal_build_readiness(
        datasets, lineage, formal_splits, leakage, storage, tasks
    )
    return P4C2BuildResult(
        datasets=datasets,
        lineage=lineage,
        formal_splits=formal_splits,
        task_eligibility=tasks,
        data_inventory=inventory,
        leakage_audit=leakage,
        storage_plan=storage,
        missingness_plan=missingness,
        readiness=readiness,
        protocol_sha256=sha256_file(experiment_protocol_path),
        p4c1_manifest_sha256=str(p4c1_metadata["manifest_sha256"]),
        config_sha256=sha256_file(config_file),
    )


def readiness_manifest_sha256(result: P4C2BuildResult) -> str:
    """Hash all semantic P4-C2 records, excluding presentation artifacts."""

    payload = {
        "datasets": [row.to_dict() for row in result.datasets],
        "lineage": [row.to_dict() for row in result.lineage],
        "formal_splits": [row.to_dict() for row in result.formal_splits],
        "task_eligibility": [row.to_dict() for row in result.task_eligibility],
        "data_inventory": list(result.data_inventory),
        "leakage_audit": result.leakage_audit,
        "storage_plan": result.storage_plan,
        "missingness_plan": result.missingness_plan,
        "readiness": result.readiness,
        "protocol_sha256": result.protocol_sha256,
        "p4c1_manifest_sha256": result.p4c1_manifest_sha256,
        "config_sha256": result.config_sha256,
    }
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
