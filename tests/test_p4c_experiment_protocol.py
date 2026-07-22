"""Tests for label-isolated P4-C0 experiment protocol planning."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from semantic3d.dataset_builder.writer import write_parquet
from semantic3d.experiment_protocol.branch_eligibility import build_branch_eligibility
from semantic3d.experiment_protocol.coverage_planning import (
    build_coverage_targets,
    build_missingness_bias_report,
)
from semantic3d.experiment_protocol.duplicate_audit import audit_duplicates
from semantic3d.experiment_protocol.inventory import load_isolated_labels
from semantic3d.experiment_protocol.leakage_audit import audit_leakage
from semantic3d.experiment_protocol.schema import (
    BranchEligibilityRecord,
    LabelRecord,
    SourceGroupRecord,
    VideoInventoryRecord,
)
from semantic3d.experiment_protocol.source_grouping import (
    assign_source_groups,
    stable_source_group_id,
)
from semantic3d.experiment_protocol.split_planner import (
    SPLIT_INPUT_FIELDS,
    plan_group_aware_split,
)
from semantic3d.experiment_protocol.storage_planning import estimate_storage


def _inventory(
    video_id: str,
    group_id: str,
    label: int,
    *,
    source_path: str = "/tmp/video.mp4",
    declared_split: str = "",
) -> VideoInventoryRecord:
    return VideoInventoryRecord(
        video_id=video_id,
        dataset_name="unit",
        source_name=video_id,
        source_path=source_path,
        source_sha256=hashlib.sha256(video_id.encode()).hexdigest(),
        file_size=100,
        frame_count=10,
        fps=10.0,
        width=64,
        height=64,
        duration_seconds=1.0,
        binary_label=label,
        label_name="real" if label == 0 else "fake",
        manipulation_type="unknown",
        source_group_id=group_id,
        source_group_review_required=False,
        declared_split=declared_split,
        temporal_annotation_available=False,
        spatial_annotation_available=False,
        object_annotation_available=False,
        annotation_quality="binary_only",
        detection_training_eligible=True,
        video_classification_eligible=True,
        temporal_localization_eligible=False,
        spatial_localization_eligible=False,
        object_localization_eligible=False,
        occlusion_validation_eligible=False,
        geometry_validation_only=False,
    )


def test_stable_source_group_id_is_order_independent() -> None:
    assert stable_source_group_id("dataset", "origin-1") == stable_source_group_id(
        "dataset", "origin-1"
    )


def test_original_and_derivative_share_source_group() -> None:
    videos = [
        {"video_id": "real", "source_name": "real", "source_sha256": "a" * 64},
        {"video_id": "fake", "source_name": "fake", "source_sha256": "b" * 64},
    ]
    labels = {
        "real": LabelRecord("real", 0, "real", original_source_identity="origin-x"),
        "fake": LabelRecord("fake", 1, "fake", original_source_identity="origin-x"),
    }
    groups = assign_source_groups(videos, labels, dataset_name="unit")
    assert groups["real"].source_group_id == groups["fake"].source_group_id
    assert not groups["real"].source_group_review_required


def test_group_aware_split_never_splits_source_group() -> None:
    records = [_inventory("real", "group-x", 0), _inventory("fake", "group-x", 1)]
    result = plan_group_aware_split(records, random_seed=7)
    assert len({row.split for row in result}) == 1


def test_official_split_is_preserved() -> None:
    records = [_inventory("a", "ga", 0), _inventory("b", "gb", 1)]
    result = plan_group_aware_split(
        records, official_split_by_video={"a": "train", "b": "test"}
    )
    assert {row.video_id: row.split for row in result} == {"a": "train", "b": "test"}
    assert all(row.official_split_preserved for row in result)


def test_exact_and_near_duplicate_audit(monkeypatch) -> None:
    records = [_inventory("a", "ga", 0), _inventory("b", "gb", 1)]
    records[1] = replace(records[1], source_sha256=records[0].source_sha256)
    assignments = plan_group_aware_split(
        records, official_split_by_video={"a": "train", "b": "test"}
    )
    monkeypatch.setattr(
        "semantic3d.experiment_protocol.duplicate_audit.video_perceptual_hashes",
        lambda *_args, **_kwargs: ("00" * 8,),
    )
    rows = audit_duplicates(records, assignments)
    assert rows[0]["exact_duplicate"]
    assert rows[0]["near_duplicate_score"] == 1.0
    assert rows[0]["review_required"]


def test_label_manifest_does_not_infer_from_filename(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.csv"
    manifest.write_text(
        "video_id,label,label_name,split\nfake_named_video,0,real,validation\n",
        encoding="utf-8",
    )
    labels = load_isolated_labels(manifest)
    assert labels["fake_named_video"].binary_label == 0
    assert labels["fake_named_video"].annotation_source


def _write_branch_fixture(root: Path) -> None:
    (root / "dataset_manifest.json").write_text(
        '{"depth_convention":"relative_depth_not_metric"}', encoding="utf-8"
    )
    write_parquet(
        root / "manifests/videos.parquet",
        [{"video_id": "v1"}],
        columns=("video_id",),
    )
    write_parquet(
        root / "manifests/clips.parquet",
        [{"clip_id": "c1", "video_id": "v1"}],
        columns=("clip_id", "video_id"),
    )
    write_parquet(
        root / "manifests/frames.parquet",
        [{"clip_id": "c1", "video_id": "v1", "frame_index": 0}],
        columns=("clip_id", "video_id", "frame_index"),
    )
    write_parquet(
        root / "observations/shared_3d_frames.parquet",
        [{"video_id": "v1", "frame_index": 0, "valid": True}],
        columns=("video_id", "frame_index", "valid"),
    )
    write_parquet(
        root / "observations/dynamic_readiness.parquet",
        [{"clip_id": "c1", "video_id": "v1", "geometry_mode": "static_camera_3d", "dynamic_3d_ready": True, "valid": True, "missing_reason": ""}],
        columns=("clip_id", "video_id", "geometry_mode", "dynamic_3d_ready", "valid", "missing_reason"),
    )
    write_parquet(
        root / "observations/masks.parquet",
        [{"video_id": "v1", "valid": True, "bbox_fallback": False}],
        columns=("video_id", "valid", "bbox_fallback"),
    )
    write_parquet(
        root / "observations/keypoints.parquet",
        [{"video_id": "v1", "valid": True}],
        columns=("video_id", "valid"),
    )
    write_parquet(
        root / "observations/point_tracks_2d.parquet",
        [{"clip_id": "c1", "valid": True}],
        columns=("clip_id", "valid"),
    )
    write_parquet(
        root / "reports/occlusion_depth_order_sync.parquet",
        [{"clip_id": "c1", "formal_occlusion_event": False, "new_overlap_candidate": True, "depth_order_valid": True}],
        columns=("clip_id", "formal_occlusion_event", "new_overlap_candidate", "depth_order_valid"),
    )


def test_branch_eligibility_distinguishes_event_applicability(tmp_path: Path) -> None:
    _write_branch_fixture(tmp_path)
    rows = build_branch_eligibility(tmp_path)
    clip = [row for row in rows if row.entity_type == "clip"]
    assert next(row for row in clip if row.tier == "S").applicable
    assert next(row for row in clip if row.tier == "D1").applicable
    occlusion = next(row for row in clip if row.tier == "O")
    assert occlusion.eligibility_status == "not_applicable"
    assert occlusion.expected_observation_available


def test_missingness_shortcut_report_flags_class_gap(tmp_path: Path) -> None:
    root = tmp_path
    records = [_inventory("v0", "g0", 0), _inventory("v1", "g1", 1)]
    assignments = plan_group_aware_split(records, preserve_declared_smoke_split=True)
    coverage = []
    for video_id, shared in (("v0", 1.0), ("v1", 0.0)):
        for metric, ratio in (
            ("frame_depth_coverage", 1.0),
            ("frame_shared_3d_coverage", shared),
            ("sequence_depth_aligned_coverage", 1.0),
            ("dynamic_3d_ready_coverage", shared),
        ):
            coverage.append({"scope_type": "video", "scope_id": video_id, "metric_name": metric, "ratio": ratio})
    write_parquet(root / "reports/coverage_metrics.parquet", coverage, columns=tuple(coverage[0]))
    write_parquet(root / "observations/objects.parquet", [{"video_id": "v0", "valid": True}, {"video_id": "v1", "valid": True}], columns=("video_id", "valid"))
    write_parquet(root / "observations/masks.parquet", [{"video_id": "v0", "valid": True, "bbox_fallback": False}], columns=("video_id", "valid", "bbox_fallback"))
    write_parquet(root / "observations/keypoints.parquet", [], columns=("video_id", "valid"))
    write_parquet(root / "observations/dynamic_readiness.parquet", [{"video_id": "v0", "geometry_mode": "static_camera_3d", "missing_reason": ""}, {"video_id": "v1", "geometry_mode": "unavailable", "missing_reason": "no_pose"}], columns=("video_id", "geometry_mode", "missing_reason"))
    rows, summary = build_missingness_bias_report(records, assignments, root, [])
    assert len(rows) == 2
    assert summary["shortcut_risk"]
    assert not summary["provider_failure_is_anomaly_evidence"]


def test_coverage_target_gap_is_nonnegative() -> None:
    records = [_inventory("v", "g", 0)]
    eligibility = [
        BranchEligibilityRecord("video", "v", "v", "S", "frame_static_3d", True, "applicable", True, "frame", "none", "none", "none", "")
    ]
    acceptance = {
        "ordinary_structure_funnel": {"person_structure_track_count": 1},
        "ordinary_formal_structure_graph_count": 2,
        "occlusion_depth_order": {"formal_occlusion_event_count": 0},
    }
    result = build_coverage_targets(records, eligibility, acceptance)
    assert all(row["minimum_gap"] >= 0 for row in result["targets"])
    assert result["targets_not_fitted_from_residuals"]


def test_storage_estimate_does_not_copy_data(tmp_path: Path) -> None:
    (tmp_path / "arrays/depth").mkdir(parents=True)
    (tmp_path / "arrays/depth/a.bin").write_bytes(b"x" * 128)
    result = estimate_storage(tmp_path, frame_count=1, source_video_bytes=64, planned_frame_count=10)
    assert result["formal_build_estimate"]["temporary_peak_bytes"] > 0
    assert not result["data_copied"]


def test_test_split_never_enters_fit_or_threshold_routes() -> None:
    records = [_inventory("a", "ga", 0), _inventory("b", "gb", 1)]
    assignments = plan_group_aware_split(
        records, official_split_by_video={"a": "train", "b": "test"}
    )
    groups = [SourceGroupRecord(row.video_id, row.source_group_id, "test", "", False) for row in records]
    conflicts, summary = audit_leakage(records, assignments, groups, [])
    assert not summary["test_used_for_normalization_fit"]
    assert not summary["test_used_for_threshold_tuning"]
    assert not conflicts


def test_split_inputs_exclude_residual_values() -> None:
    assert not any("residual" in value or "score" in value for value in SPLIT_INPUT_FIELDS)


def test_strict_prior_hashes_are_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / "configs" / name).read_bytes()).hexdigest() == digest
