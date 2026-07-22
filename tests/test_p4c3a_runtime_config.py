"""Tests for P4-C3A runtime portability and P4-C2 policy reinforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from semantic3d.experiment_protocol.evidence_eligibility_policy import (
    EvidenceUse,
    can_contribute_evidence,
)
from semantic3d.experiment_protocol.formal_schema import (
    FormalDatasetRecord,
    FormalSplitRecord,
    SourceLineageRecord,
    TaskEligibilityRecord,
)
from semantic3d.experiment_protocol.p4c2_builder import (
    build_p4c2_readiness,
    evaluate_formal_build_readiness,
)
from semantic3d.runtime.runtime_config import (
    canonical_logical_sha256,
    load_runtime_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _runtime_yaml(path: Path, project_root: str) -> None:
    path.write_text(
        "\n".join(
            [
                "runtime_profile: unit",
                f"project_root: {project_root}",
                "data_root: data",
                "download_root: downloads",
                "temporary_root: tmp",
                "cache_root: cache",
                "model_root: models",
                "output_root: outputs",
                "log_root: logs",
                "device: cpu",
                "num_workers: 1",
                "storage_safety_margin: 10",
                "batch_storage_limit: 20",
                "required_python_dependencies: []",
                "required_executables: []",
                "require_cuda_for_formal_batch: true",
            ]
        ),
        encoding="utf-8",
    )


def test_runtime_environment_overrides_paths(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    _runtime_yaml(config, "${PROJECT_ROOT}")
    project = tmp_path / "server-project"
    runtime = load_runtime_config(
        config,
        environment={"PROJECT_ROOT": str(project), "DATA_ROOT": str(tmp_path / "dataset")},
    )
    assert runtime.project_root == project
    assert runtime.data_root == tmp_path / "dataset"
    assert runtime.output_root == project / "outputs"


def test_canonical_hash_is_independent_of_absolute_machine_path(tmp_path: Path) -> None:
    first_root = tmp_path / "user-a/project"
    second_root = tmp_path / "srv/project"
    first_config = tmp_path / "first.yaml"
    second_config = tmp_path / "second.yaml"
    _runtime_yaml(first_config, str(first_root))
    _runtime_yaml(second_config, str(second_root))
    first = load_runtime_config(first_config)
    second = load_runtime_config(second_config)
    payload_a = {"video": str(first.data_root / "x.mp4"), "sample_id": "same"}
    payload_b = {"video": str(second.data_root / "x.mp4"), "sample_id": "same"}
    assert canonical_logical_sha256(payload_a, first) == canonical_logical_sha256(payload_b, second)


def test_unresolved_server_variable_fails_validation() -> None:
    with pytest.raises(ValueError, match="Unresolved runtime variables"):
        load_runtime_config(
            PROJECT_ROOT / "configs/runtime/server_template.yaml",
            environment={},
        )


def test_smoke_dataset_has_all_explicit_formal_flags_false() -> None:
    registry = json.loads(
        (PROJECT_ROOT / "outputs/p4c2_formal_data_readiness/formal_dataset_registry.json").read_text(
            encoding="utf-8"
        )
    )["datasets"][0]
    assert registry["dataset_role"] == "geometry_validation_smoke"
    assert not registry["is_formal_dataset"]
    assert not registry["eligible_for_formal_split"]
    assert not registry["eligible_for_training"]
    assert not registry["eligible_for_model_selection"]
    assert not registry["eligible_for_threshold_selection"]
    assert not registry["eligible_for_final_evaluation"]


@pytest.mark.parametrize(
    "purpose",
    [
        EvidenceUse.ANOMALY_SCORE,
        EvidenceUse.AUTHENTICITY_LABEL,
        EvidenceUse.NORMAL_REFERENCE_FIT,
        EvidenceUse.SUPERVISED_AGGREGATION,
    ],
)
def test_provider_failed_cannot_be_converted_to_model_evidence(purpose: EvidenceUse) -> None:
    with pytest.raises(ValueError, match="provider_failed cannot be used"):
        can_contribute_evidence("provider_failed", purpose)
    assert not can_contribute_evidence("provider_failed", EvidenceUse.QUALITY_CONTROL)
    assert not can_contribute_evidence("provider_failed", EvidenceUse.MISSINGNESS_AUDIT)


def test_current_readiness_remains_blocked() -> None:
    result = build_p4c2_readiness(
        PROJECT_ROOT / "configs/p4c2_formal_data_readiness_v1.yaml",
        project_root=PROJECT_ROOT,
    )
    assert not result.readiness["ready_for_formal_batch_build"]


def test_synthetic_complete_readiness_has_positive_path() -> None:
    dataset = FormalDatasetRecord(
        dataset_name="formal",
        version="v1",
        official_source="official",
        license="verified-license",
        citation="citation",
        download_method="official-checksummed",
        official_split="official",
        annotation_types=("all",),
        expected_size={"videos": 3},
        checksum_policy="sha256",
        local_root="data/formal",
        dataset_role="formal_training_and_evaluation",
        eligible_for_training=True,
        eligible_for_model_selection=True,
        eligible_for_threshold_selection=True,
        eligible_for_final_evaluation=True,
        registry_status="verified_ready",
        metadata={"license_verification_status": "verified"},
    )
    lineage = tuple(
        SourceLineageRecord(
            "formal",
            f"origin-{index}",
            f"group-{index}",
            f"video-{index}",
            f"video-{index}",
            f"data/video-{index}.mp4",
            str(index) * 64,
            index % 2,
            "known",
            "verified",
            "official_metadata",
            "verified",
            True,
            "",
            "formal_training_and_evaluation",
        )
        for index in range(3)
    )
    splits = tuple(
        FormalSplitRecord(
            "formal",
            f"video-{index}",
            f"origin-{index}",
            f"group-{index}",
            split,
            "official",
            True,
            "",
            True,
            "unit",
            1,
        )
        for index, split in enumerate(("train", "validation", "test"))
    )
    required_tasks = (
        "reprojection_D2",
        "reprojection_D3",
        "occlusion_and_reappearance",
        "temporal_localization",
        "spatial_localization",
    )
    tasks = tuple(
        TaskEligibilityRecord(
            "formal",
            "video-0",
            task,
            "eligible",
            True,
            True,
            "video",
            "",
            "verified_provider",
            1,
            1,
        )
        for task in required_tasks
    )
    readiness = evaluate_formal_build_readiness(
        (dataset,),
        lineage,
        splits,
        {"cross_split_leakage_detected": False},
        {"build_blocked_insufficient_storage": False},
        tasks,
    )
    assert readiness["ready_for_formal_batch_build"]
    assert readiness["blockers"] == []

