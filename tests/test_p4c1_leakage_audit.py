"""Tests for source-video, derivative, duplicate, and split leakage checks."""

from __future__ import annotations

from semantic3d.experiment_protocol.leakage_audit import audit_manifest_leakage
from semantic3d.experiment_protocol.manifest_schema import ExperimentSampleRecord, stable_sample_id


def _record(
    video_id: str,
    group_id: str,
    split: str,
    *,
    clip_id: str | None = None,
    digest: str | None = None,
) -> ExperimentSampleRecord:
    clip = clip_id or f"clip-{video_id}"
    return ExperimentSampleRecord(
        sample_id=stable_sample_id("unit", video_id, clip, 0, 7),
        manifest_schema_version="p4c1_experiment_manifest_v1",
        dataset_name="unit",
        source_video_id=video_id,
        source_video_name=video_id,
        source_group_id=group_id,
        source_group_review_required=False,
        source_sha256=digest or (video_id * 64)[:64],
        video_path=f"data/{video_id}.mp4",
        clip_id=clip,
        clip_start=0,
        clip_end=7,
        core_clip_start=0,
        core_clip_end=7,
        num_frames=8,
        split=split,
        authenticity_label=0,
        authenticity_label_name="real",
        manipulation_type="unknown",
        scene_id=0,
        camera_id="",
        camera_identity_status="camera_identity_unavailable",
        coordinate_system_id="coord",
        frame_manifest_path="frames.parquet",
        object_observations_path="objects.parquet",
        depth_observations_path="depth.parquet",
        pose_observations_path="camera.parquet",
        track_observations_path="tracks.parquet",
        semantic3d_observations_path="shared.parquet",
        valid_object_count=1,
        valid_depth_count=8,
        valid_pose_count=0,
        valid_track_point_count=0,
        valid_semantic3d_count=8,
        usable=True,
        exclusion_reason="",
        protocol_sha256="a" * 64,
    )


def _group(identity: str = "", review: bool = False) -> dict[str, object]:
    return {
        "original_source_identity": identity,
        "source_group_review_required": review,
    }


def test_overlapping_clips_from_one_video_in_one_split_are_safe() -> None:
    rows = [
        _record("v1", "g1", "validation", clip_id="c1"),
        _record("v1", "g1", "validation", clip_id="c2"),
    ]
    findings, summary = audit_manifest_leakage(rows, {"v1": _group("origin-1")}, [])
    assert not [row for row in findings if row.severity == "error"]
    assert summary["unique_source_video_count"] == 1


def test_one_source_video_cannot_cross_splits() -> None:
    rows = [
        _record("v1", "g1", "train", clip_id="c1"),
        _record("v1", "g1", "test", clip_id="c2"),
    ]
    findings, summary = audit_manifest_leakage(rows, {"v1": _group("origin-1")}, [])
    assert "source_video_cross_split" in {row.finding_type for row in findings}
    assert summary["cross_split_leakage_detected"]


def test_known_original_and_derivative_group_cannot_cross_splits() -> None:
    rows = [_record("real", "g", "train"), _record("fake", "g", "test")]
    groups = {"real": _group("origin-x"), "fake": _group("origin-x")}
    findings, _ = audit_manifest_leakage(rows, groups, [])
    kinds = {row.finding_type for row in findings}
    assert "source_group_cross_split" in kinds
    assert "original_source_identity_cross_split" in kinds


def test_exact_hash_duplicate_cannot_cross_splits() -> None:
    digest = "f" * 64
    rows = [
        _record("v1", "g1", "train", digest=digest),
        _record("v2", "g2", "test", digest=digest),
    ]
    findings, _ = audit_manifest_leakage(
        rows,
        {"v1": _group("a"), "v2": _group("b")},
        [],
    )
    assert "exact_video_hash_cross_split" in {row.finding_type for row in findings}


def test_near_duplicate_audit_is_rechecked_against_frozen_split() -> None:
    rows = [_record("v1", "g1", "train"), _record("v2", "g2", "test")]
    duplicates = [
        {
            "video_id_a": "v1",
            "video_id_b": "v2",
            "exact_duplicate": False,
            "near_duplicate_score": 0.95,
        }
    ]
    findings, _ = audit_manifest_leakage(
        rows,
        {"v1": _group("a"), "v2": _group("b")},
        duplicates,
        near_duplicate_threshold=0.92,
    )
    assert "near_duplicate_cross_split" in {row.finding_type for row in findings}


def test_unresolved_original_identity_is_warning_not_silent_success() -> None:
    row = _record("v1", "g1", "validation")
    findings, summary = audit_manifest_leakage([row], {"v1": _group(review=True)}, [])
    unresolved = [
        finding
        for finding in findings
        if finding.finding_type == "unresolved_original_derivative_identity"
    ]
    assert len(unresolved) == 1
    assert unresolved[0].severity == "warning"
    assert summary["unresolved_original_derivative_identity_count"] == 1
    assert not summary["cross_split_leakage_detected"]
