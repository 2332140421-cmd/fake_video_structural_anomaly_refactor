from __future__ import annotations

from dataclasses import replace

import numpy as np

from semantic3d.sequence_geometry import (
    HistogramFeatureSceneCutDetector,
    build_foreground_mask,
)
from semantic3d.sequence_geometry.provider import estimate_background_relative_pose

from synthetic_geometry import synthetic_object_3d, synthetic_shared_3d_frame
from synthetic_sequence_geometry import dotted_background_scene


def test_bbox_foreground_is_dilated_and_marked_low_quality() -> None:
    obj = synthetic_object_3d("person_1", label="person", metric=False)
    frame = synthetic_shared_3d_frame((obj,), metric=False)
    result = build_foreground_mask(frame, bbox_dilation_pixels=5)
    assert result.valid
    assert result.bbox_fallback_count == 1
    assert result.mask_object_count == 0
    assert result.excluded_foreground_ratio > 0.0
    assert result.quality == 0.35
    assert result.metadata["bbox_is_low_quality_foreground_fallback"] is True


def test_near_full_frame_bbox_does_not_remove_all_background_support() -> None:
    obj = synthetic_object_3d("table_1", label="dining_table", metric=False)
    obj = replace(
        obj,
        metadata={**dict(obj.metadata), "source_bbox": [0.0, 0.0, 127.0, 127.0]},
    )
    frame = synthetic_shared_3d_frame((obj,), metric=False)
    result = build_foreground_mask(frame)
    assert result.excluded_foreground_ratio == 0.0
    assert result.metadata["skipped_oversized_bbox_ids"] == ["table_1"]


def test_moving_foreground_does_not_pollute_static_background_identity_pose() -> None:
    source, source_mask = dotted_background_scene(moving_box_x=70)
    target, target_mask = dotted_background_scene(moving_box_x=150)
    K = np.asarray([[260.0, 0.0, 159.5], [0.0, 260.0, 119.5], [0.0, 0.0, 1.0]])
    estimate = estimate_background_relative_pose(
        source,
        target,
        K,
        source_frame_index=0,
        target_frame_index=1,
        source_foreground_mask=source_mask,
        target_foreground_mask=target_mask,
    )
    assert estimate.valid
    assert estimate.is_static_identity
    assert np.allclose(estimate.T_current_from_previous, np.eye(4))
    assert estimate.metadata["foreground_excluded"] is True
    for row in estimate.track_rows:
        sx, sy = int(round(row["source_x"])), int(round(row["source_y"]))
        tx, ty = int(round(row["target_x"])), int(round(row["target_y"]))
        assert not source_mask[sy, sx]
        assert not target_mask[ty, tx]


def test_histogram_content_cut_is_detected() -> None:
    source = np.zeros((120, 160, 3), dtype=np.uint8)
    source[:, :] = (0, 0, 255)
    target = np.zeros((120, 160, 3), dtype=np.uint8)
    target[:, :] = (0, 255, 0)
    decision = HistogramFeatureSceneCutDetector().detect(
        source,
        target,
        source_frame_index=0,
        target_frame_index=1,
    )
    assert decision.is_cut
    assert decision.reason == "histogram_content_discontinuity"


def test_provider_reported_cut_is_always_respected() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    decision = HistogramFeatureSceneCutDetector().detect(
        image,
        image,
        source_frame_index=0,
        target_frame_index=1,
        provider_reported_cut=True,
    )
    assert decision.is_cut
    assert decision.reason == "provider_reported_cut"
