"""Tests for P4-C2 formal data registration and build readiness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from semantic3d.experiment_protocol.p4c2_builder import (
    build_p4c2_readiness,
    readiness_manifest_sha256,
)
from semantic3d.experiment_protocol.p4c2_report import write_p4c2_artifacts
from semantic3d.experiment_protocol.p4c2_validation import validate_p4c2_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/p4c2_formal_data_readiness_v1.yaml"


def test_current_registry_marks_six_videos_as_smoke_only() -> None:
    result = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert len(result.datasets) == 1
    dataset = result.datasets[0]
    assert dataset.dataset_name == "local_six_video_protocol_smoke"
    assert dataset.dataset_role == "geometry_validation_smoke"
    assert not dataset.eligible_for_training
    assert not dataset.eligible_for_model_selection
    assert not dataset.eligible_for_threshold_selection
    assert not dataset.eligible_for_final_evaluation
    assert dataset.metadata["metadata_inventory_only"]


def test_unverified_smoke_lineage_never_enters_formal_split() -> None:
    result = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert len(result.lineage) == 6
    assert all(row.verification_status == "unverified" for row in result.lineage)
    assert all(not row.original_source_id for row in result.lineage)
    assert all(not row.formal_split_eligible for row in result.lineage)
    assert all(not row.formal_split for row in result.formal_splits)
    assert all(row.blocked_reason == "unverified_source_lineage" for row in result.formal_splits)


def test_task_matrix_has_explicit_state_for_all_ten_tasks() -> None:
    result = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert len(result.task_eligibility) == 60
    assert {row.eligibility for row in result.task_eligibility} <= {
        "eligible",
        "ineligible",
        "unknown",
        "not_applicable",
        "provider_failed",
    }
    assert all(not row.eligible_for_formal_experiment for row in result.task_eligibility)
    assert any(row.eligibility == "not_applicable" for row in result.task_eligibility)
    assert any(row.eligibility == "provider_failed" for row in result.task_eligibility)
    assert all(
        not row.metadata["provider_failure_is_anomaly_evidence"]
        for row in result.task_eligibility
    )


def test_inventory_is_lightweight_and_records_every_source_video() -> None:
    result = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert len(result.data_inventory) == 6
    assert all(row["video_exists"] for row in result.data_inventory)
    assert all(row["source_sha256_matches"] for row in result.data_inventory)
    assert all(row["metadata_scan_only"] for row in result.data_inventory)
    assert all(not row["video_decoded"] for row in result.data_inventory)
    assert all(not row["model_inference_performed"] for row in result.data_inventory)


def test_storage_snapshot_blocks_full_formal_build() -> None:
    result = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    plan = result.storage_plan
    assert plan["batch_count"] == 10
    assert plan["feasible_batch_count_without_archiving"] == 1
    assert plan["build_blocked_insufficient_storage"]
    assert not plan["full_build_fits_audited_snapshot"]
    assert "insufficient_storage_for_planned_formal_build" in result.readiness["blockers"]
    assert not result.readiness["ready_for_formal_batch_build"]


def test_missingness_plan_forbids_provider_failure_as_evidence() -> None:
    result = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    plan = result.missingness_plan
    assert plan["states"] == [
        "available",
        "unknown",
        "not_applicable",
        "provider_failed",
    ]
    assert not plan["provider_failure_is_anomaly_evidence"]
    assert plan["never_impute_missing_as_zero"]
    assert not plan["test_performance_values_read"]


def test_repeated_build_hash_and_written_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    second = build_p4c2_readiness(CONFIG_PATH, project_root=PROJECT_ROOT)
    digest = readiness_manifest_sha256(first)
    assert digest == readiness_manifest_sha256(second)
    write_p4c2_artifacts(tmp_path, first, deterministic_rebuild_sha256=digest)
    first_hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    write_p4c2_artifacts(tmp_path, second, deterministic_rebuild_sha256=digest)
    second_hashes = json.loads((tmp_path / "artifact_hashes.json").read_text(encoding="utf-8"))
    assert first_hashes == second_hashes
    validation = validate_p4c2_artifacts(tmp_path, CONFIG_PATH, project_root=PROJECT_ROOT)
    assert validation["valid"]


def test_stage_config_forbids_training_download_and_test_performance() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert not any(config["forbidden_operations"].values())


def test_p4c0_p4c1_and_strict_prior_inputs_are_unchanged() -> None:
    expected = {
        "configs/p4c0_experiment_protocol_v1.yaml": "8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304",
        "configs/p4c1_experiment_manifest_v1.yaml": "ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086",
        "configs/scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "configs/scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest

