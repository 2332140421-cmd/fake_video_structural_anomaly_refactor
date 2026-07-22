"""P4-B.5 full-observation contracts and non-leakage tests."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from semantic3d.dataset_builder.p4b5_contracts import (
    ClipTrackHandoffObservation,
    CoverageMetric,
    adaptive_structure_point_target,
    build_fixed_structure_edges,
    classify_clip_geometry,
    synchronized_depth_order,
)
from semantic3d.dataset_builder.p4b5_pipeline import (
    P4B5StructuralEnhancementDatasetBuilder,
)
from semantic3d.dataset_builder.schema import P4B5_PIPELINE_VERSION
from semantic3d.occlusion.mask_observation import InstanceMaskObservation
from semantic3d.occlusion.mask_structure_points import (
    track_formal_mask_internal_points,
)


def _formal_mask(frame_index: int, mask: np.ndarray) -> InstanceMaskObservation:
    return InstanceMaskObservation.from_visible_mask(
        video_id="v",
        frame_index=frame_index,
        object_track_id="object-1",
        semantic_label="cup",
        mask=mask,
        confidence=1.0,
        source_provider="formal_test_segmentation",
        metadata={"formal_mask_evidence": True},
    )


def test_five_coverage_semantics_are_explicit() -> None:
    metric = CoverageMetric("frame_depth_coverage", "dataset", "d", 8, 10, 10, 2, 0, 0, "frame")
    assert metric.ratio == 0.8
    assert "formal_dynamic_evidence_coverage" in inspect.getsource(
        P4B5StructuralEnhancementDatasetBuilder._coverage_metrics
    )


def test_all_frame_depth_is_scheduled_without_video_whitelist() -> None:
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder._stage_06_depth)
    assert "full_video_source_names" not in source
    assert "for frame in self._owned_frames()" in source
    assert "raw_model_output" in source and "depth_map" in source


def test_frame_relative_and_sequence_geometry_are_separate() -> None:
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder._stage_09_shared_3d)
    assert "frame_camera:" in source
    assert '"relative_per_frame"' in source
    assert '"cross_frame_subtraction_allowed": False' in source


def test_unavailable_clip_does_not_remove_frame_relative_contract() -> None:
    decision = classify_clip_geometry(
        median_pixel_motion=25.0,
        tracked_transition_ratio=0.9,
        homography_inlier_ratio=0.2,
        depth_aligned_ratio=1.0,
    )
    assert decision.geometry_mode == "unavailable"
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder._stage_09_shared_3d)
    assert "sequence_geometry_independent" in source


def test_full_person_keypoints_are_not_legacy_migration() -> None:
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder._stage_05_keypoints)
    assert "RealHumanKeypointProvider" in source
    assert '"migrated_coverage_only": False' in source
    assert P4B5_PIPELINE_VERSION in source or "P4B5_PIPELINE_VERSION" in source


def test_formal_mask_clip_tracking_keeps_stable_ids() -> None:
    images = {}
    masks = []
    for frame_index, offset in enumerate((0, 1, 2)):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(image, (15 + offset, 15), (45 + offset, 45), (255, 255, 255), -1)
        mask = np.zeros((64, 64), dtype=bool)
        mask[12:50, 12 + offset : 50 + offset] = True
        images[frame_index] = image
        masks.append(_formal_mask(frame_index, mask))
    points = track_formal_mask_internal_points(images, masks, max_points=8, erosion_pixels=None)
    assert points
    ids_by_frame = {}
    for point in points:
        if point.valid:
            ids_by_frame.setdefault(point.frame_index, set()).add(point.point_id)
    assert set(ids_by_frame[0]) & set(ids_by_frame[1])
    assert all(point.metadata.get("current_mask_used_for_prediction") is False for point in points if point.valid)


def test_adaptive_erosion_budget_uses_area_not_label() -> None:
    assert adaptive_structure_point_target(16.0) == 4
    assert adaptive_structure_point_target(1_000_000.0) == 24


def test_fixed_graph_uses_stable_point_ids() -> None:
    ids = ("a", "b", "c", "d")
    xyz = np.asarray([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    edges = build_fixed_structure_edges(ids, xyz)
    assert edges
    assert all(first in ids and second in ids for first, second in edges)


def test_clip_handoff_never_implies_3d_alignment() -> None:
    handoff = ClipTrackHandoffObservation(
        handoff_id="h", video_id="v", source_clip_id="a", target_clip_id="b",
        global_object_track_id="o", source_local_track_id="oa",
        target_local_track_id="ob", overlap_frame_ids=("f",), mask_iou=1.0,
        point_overlap_ratio=0.5, appearance_similarity=math.nan, handoff_quality=0.5,
        alignment_id="", allows_cross_clip_3d=False, valid=True,
    )
    assert handoff.valid and not handoff.allows_cross_clip_3d


def test_synchronized_depth_order_prefers_boundary_neighbourhood() -> None:
    depth = np.full((30, 30), 8.0)
    first = np.zeros_like(depth, dtype=bool)
    second = np.zeros_like(depth, dtype=bool)
    first[5:20, 5:17] = True
    second[5:20, 14:26] = True
    depth[first] = 3.0
    depth[second & ~first] = 7.0
    result = synchronized_depth_order(depth, first, second, minimum_pixels=4)
    assert result.valid
    assert result.depth_source == "overlap_boundary_neighbourhood"
    assert result.foreground == "a"


def test_uncertain_depth_order_is_not_forced_event() -> None:
    depth = np.ones((20, 20), dtype=float) * 5.0
    first = np.zeros_like(depth, dtype=bool)
    second = np.zeros_like(depth, dtype=bool)
    first[2:15, 2:12] = True
    second[2:15, 9:19] = True
    result = synchronized_depth_order(depth, first, second, minimum_pixels=4)
    assert not result.valid and result.uncertain
    assert result.missing_reason == "depth_order_within_uncertainty"


def test_new_config_writes_incremental_dataset_directory() -> None:
    config = yaml.safe_load(Path("configs/p4b5_six_video_full_observation.yaml").read_text())
    assert config["dataset"]["pipeline_profile"] == P4B5_PIPELINE_VERSION
    assert config["dataset"]["output_root"].endswith("p4b5_six_video_full_observation")
    assert config["dataset"]["previous_dataset_root"].endswith("p4b_six_video_smoke")


def test_builder_source_does_not_read_truth_labels() -> None:
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder)
    assert "labels_manifest" not in source
    assert "is_fake" not in source
    assert "AUC" not in source and "accuracy" not in source


def test_formal_dynamic_evidence_is_owned_only() -> None:
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder._stage_11_dynamic)
    assert 'if not current.get("is_owned_frame")' in source
    assert '"owned_frame_only": True' in source


def test_occlusion_prediction_does_not_use_current_mask() -> None:
    source = inspect.getsource(P4B5StructuralEnhancementDatasetBuilder._stage_12_occlusion)
    score_position = source.index("event = bool")
    history_update_position = source.index("Update history only after")
    assert score_position < history_update_position
    assert '"history_prediction_uses_current_frame": False' in source
