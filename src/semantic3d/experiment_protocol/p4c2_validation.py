"""Validate P4-C2 schema, hashes, lineage isolation, and stage boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from semantic3d.dataset_builder.writer import sha256_file

from .formal_schema import ELIGIBILITY_STATES
from .p4c2_builder import build_p4c2_readiness, readiness_manifest_sha256


def validate_p4c2_artifacts(
    output_root: str | Path,
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    """Rebuild P4-C2 metadata and validate persisted primary artifacts."""

    root = Path(output_root)
    project = Path(project_root)
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    rebuilt = build_p4c2_readiness(config_path, project_root=project)
    rebuilt_hash = readiness_manifest_sha256(rebuilt)
    metadata = json.loads((root / "build_metadata.json").read_text(encoding="utf-8"))
    artifact_hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))[
        "artifacts"
    ]
    lineage = [
        json.loads(line)
        for line in (root / "source_lineage.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    splits = [
        json.loads(line)
        for line in (root / "formal_split_plan.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    with (root / "task_eligibility_matrix.csv").open(encoding="utf-8", newline="") as handle:
        tasks = list(csv.DictReader(handle))
    registry = json.loads((root / "formal_dataset_registry.json").read_text(encoding="utf-8"))
    split_by_video = {row["derived_video_id"]: row["formal_split"] for row in splits}
    checks = {
        "readiness_manifest_hash_matches": metadata["p4c2_readiness_manifest_sha256"] == rebuilt_hash,
        "deterministic_rebuild_matches": metadata["deterministic_rebuild_sha256"] == rebuilt_hash,
        "artifact_hashes_match": all(
            (root / name).is_file() and sha256_file(root / name) == digest
            for name, digest in artifact_hashes.items()
        ),
        "lineage_sorted": [row["derived_video_id"] for row in lineage]
        == sorted(row["derived_video_id"] for row in lineage),
        "derived_video_ids_unique": len(lineage)
        == len({row["derived_video_id"] for row in lineage}),
        "unverified_lineage_has_no_formal_split": all(
            row["verification_status"] == "verified"
            or not split_by_video.get(row["derived_video_id"], "")
            for row in lineage
        ),
        "source_group_split_isolation": not rebuilt.leakage_audit[
            "cross_split_leakage_detected"
        ],
        "task_statuses_controlled": all(row["eligibility"] in ELIGIBILITY_STATES for row in tasks),
        "task_eligibility_scope_is_video": all(
            row.get("eligibility_scope") == "video" for row in tasks
        ),
        "task_matrix_complete": len(tasks) == len(lineage) * 10,
        "smoke_role_is_not_formal_eligible": all(
            row["dataset_role"] != "geometry_validation_smoke"
            or not any(
                bool(row[name])
                for name in (
                    "eligible_for_training",
                    "eligible_for_model_selection",
                    "eligible_for_threshold_selection",
                    "eligible_for_final_evaluation",
                    "eligible_for_formal_split",
                    "is_formal_dataset",
                )
            )
            for row in registry["datasets"]
        ),
        "dataset_readiness_counts_present": all(
            name in metadata
            for name in (
                "registered_dataset_entries",
                "registered_formal_datasets",
                "registered_smoke_datasets",
            )
        ),
        "provider_failure_not_anomaly_evidence": not bool(
            rebuilt.missingness_plan["provider_failure_is_anomaly_evidence"]
        ),
        "formal_build_not_started": not bool(metadata["formal_build_started"]),
        "model_training_not_performed": not bool(metadata["model_training_performed"]),
        "statistical_fitting_not_performed": not bool(metadata["statistical_fitting_performed"]),
        "threshold_selection_not_performed": not bool(metadata["threshold_selection_performed"]),
        "test_performance_not_read": not bool(metadata["test_performance_read"]),
        "strict_v1_hash_unchanged": sha256_file(project / "configs/scale_priors_strict_v1.yaml")
        == config["compatibility"]["strict_prior_hashes"]["scale_priors_strict_v1.yaml"],
        "strict_v2_hash_unchanged": sha256_file(project / "configs/scale_priors_strict_v2.yaml")
        == config["compatibility"]["strict_prior_hashes"]["scale_priors_strict_v2.yaml"],
    }
    errors = sorted(name for name, passed in checks.items() if not passed)
    return {
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors,
        "checks": checks,
        "p4c2_readiness_manifest_sha256": rebuilt_hash,
        "protocol_sha256": rebuilt.protocol_sha256,
        "p4c1_manifest_sha256": rebuilt.p4c1_manifest_sha256,
        "p4c2_config_sha256": rebuilt.config_sha256,
        "ready_for_formal_batch_build": rebuilt.readiness["ready_for_formal_batch_build"],
        "blockers": rebuilt.readiness["blockers"],
        "formal_split_counts": rebuilt.readiness["formal_split_counts"],
        "lineage_record_count": len(rebuilt.lineage),
        "task_eligibility_record_count": len(rebuilt.task_eligibility),
        "downloads_performed": False,
        "model_inference_performed": False,
        "formal_build_started": False,
        "model_training_performed": False,
        "statistical_fitting_performed": False,
        "threshold_selection_performed": False,
        "test_performance_read": False,
        "authenticity_performance_computed": False,
    }
