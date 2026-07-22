"""Tests for deterministic P4-C1 manifest construction and integrity."""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from semantic3d.experiment_protocol.exclusion_policy import evaluate_exclusion_reasons
from semantic3d.experiment_protocol.manifest_builder import (
    build_p4c1_manifest,
    manifest_sha256,
    validate_manifest_artifacts,
)
from semantic3d.experiment_protocol.manifest_report import write_manifest_artifacts
from semantic3d.experiment_protocol.manifest_schema import (
    P4C1_MANIFEST_SCHEMA_VERSION,
    SampleAvailability,
    stable_sample_id,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/p4c1_experiment_manifest_v1.yaml"


def _availability(**overrides: object) -> SampleAvailability:
    values: dict[str, object] = {
        "video_exists": True,
        "video_readable": True,
        "video_hash_matches": True,
        "expected_frame_count": 8,
        "indexed_frame_count": 8,
        "decoded_frame_count": 8,
        "valid_object_count": 2,
        "valid_depth_count": 8,
        "camera_observation_count": 8,
        "valid_pose_count": 0,
        "valid_track_point_count": 0,
        "valid_semantic3d_count": 0,
        "camera_identity_available": False,
    }
    values.update(overrides)
    return SampleAvailability(**values)  # type: ignore[arg-type]


def test_stable_sample_id_is_deterministic_and_range_sensitive() -> None:
    first = stable_sample_id("dataset", "video", "clip", 0, 7)
    assert first == stable_sample_id("dataset", "video", "clip", 0, 7)
    assert first != stable_sample_id("dataset", "video", "clip", 1, 8)


def test_exclusion_policy_records_missing_required_object_data() -> None:
    reasons = evaluate_exclusion_reasons(
        _availability(valid_object_count=0),
        clip_valid=True,
        clip_missing_reason="",
        authenticity_label=0,
        split="validation",
        required_modalities=("video", "frames", "objects", "depth", "camera"),
    )
    assert reasons == ("required_objects_unavailable",)


def test_optional_pose_and_tracks_do_not_exclude_static_sample() -> None:
    reasons = evaluate_exclusion_reasons(
        _availability(valid_pose_count=0, valid_track_point_count=0),
        clip_valid=True,
        clip_missing_reason="",
        authenticity_label=1,
        split="validation",
        required_modalities=("video", "frames", "objects", "depth", "camera"),
    )
    assert reasons == ()


def test_current_manifest_covers_every_existing_clip_without_silent_skip() -> None:
    result = build_p4c1_manifest(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert len(result.records) == 59
    assert len(result.availability) == 59
    assert len({row.sample_id for row in result.records}) == 59
    assert all(row.manifest_schema_version == P4C1_MANIFEST_SCHEMA_VERSION for row in result.records)
    assert {row.split for row in result.records} == {"validation"}
    assert sum(row.usable for row in result.records) == 50
    assert sum(not row.usable for row in result.records) == 9
    assert {
        row.exclusion_reason for row in result.records if not row.usable
    } == {"required_objects_unavailable"}


def test_manifest_records_required_paths_and_does_not_fabricate_camera_id() -> None:
    result = build_p4c1_manifest(CONFIG_PATH, project_root=PROJECT_ROOT)
    row = result.records[0]
    assert row.video_path.startswith("data/tests_videos/")
    assert row.object_observations_path.endswith("observations/objects.parquet")
    assert row.depth_observations_path.endswith("observations/depth.parquet")
    assert row.pose_observations_path.endswith("observations/camera.parquet")
    assert row.track_observations_path.endswith("observations/point_tracks_2d.parquet")
    assert row.semantic3d_observations_path.endswith("observations/shared_3d_frames.parquet")
    assert row.camera_id == ""
    assert row.camera_identity_status == "camera_identity_unavailable"


def test_manifest_hash_is_reproducible_for_identical_inputs() -> None:
    first = build_p4c1_manifest(CONFIG_PATH, project_root=PROJECT_ROOT)
    second = build_p4c1_manifest(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert manifest_sha256(first) == manifest_sha256(second)


def test_artifacts_save_hashes_and_validate(tmp_path: Path) -> None:
    result = build_p4c1_manifest(CONFIG_PATH, project_root=PROJECT_ROOT)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    digest = manifest_sha256(result)
    metadata = write_manifest_artifacts(
        tmp_path,
        result,
        required_modalities=config["availability"]["required_for_usable"],
        deterministic_rebuild_sha256=digest,
    )
    validation = validate_manifest_artifacts(
        tmp_path,
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    assert metadata["manifest_sha256"] == digest
    assert metadata["protocol_sha256"] == result.protocol_sha256
    assert metadata["deterministic_rebuild_matches"]
    assert validation["valid"]
    assert validation["manifest_sha256"] == digest


def test_manifest_scope_flags_forbid_training_and_performance() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert not config["forbidden_operations"]["train_model"]
    assert not config["forbidden_operations"]["fit_parameters"]
    assert not config["forbidden_operations"]["fit_distribution"]
    assert not config["forbidden_operations"]["select_threshold"]
    assert not config["forbidden_operations"]["compute_performance"]


def test_p4c0_protocol_config_was_not_modified_by_p4c1() -> None:
    p4c0 = PROJECT_ROOT / "configs/p4c0_experiment_protocol_v1.yaml"
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    expected = hashlib.sha256(p4c0.read_bytes()).hexdigest()
    result = build_p4c1_manifest(CONFIG_PATH, project_root=PROJECT_ROOT)
    assert config["inputs"]["p4c0_config"] == "configs/p4c0_experiment_protocol_v1.yaml"
    assert result.p4c0_config_sha256 == expected
