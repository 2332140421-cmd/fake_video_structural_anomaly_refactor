from __future__ import annotations

import math

import numpy as np
import pytest

from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.occlusion import (
    ExistingDetectionMaskAdapter,
    MaskTracker,
    MockInstanceMaskProvider,
    PredictedObjectSupport,
    SyntheticInstanceMaskProvider,
    predict_object_support,
)
from synthetic_occlusion import SHAPE, observed, rectangle, synthetic_occlusion_scenarios


def _frame() -> FrameObservationJSON:
    return FrameObservationJSON(
        frame_index=0, frame_id="f0", width=64, height=64,
        objects=[ObjectObservationJSON("det", "car", 400.0, 4096.0, 5.0, bbox=[10, 10, 30, 30], track_id="car_1")],
    )


def test_visible_and_amodal_masks_are_strictly_distinct() -> None:
    mask = rectangle(10, 10, 20, 20)
    item = observed("o", 0, mask)
    assert item.is_visible_mask and not item.is_amodal_mask
    assert item.amodal_mask is None
    with pytest.raises(ValueError, match="visible or amodal"):
        type(item)("v", 0, "o", "x", SHAPE, mask, mask, 100.0, (10, 10, 20, 20), (), 1.0, "x", True, True, True)


def test_mask_shape_must_match_frame() -> None:
    provider = SyntheticInstanceMaskProvider({(0, "car_1"): np.ones((10, 10), dtype=bool)})
    with pytest.raises(ValueError, match="image shape"):
        provider.predict(video_id="v", frame=_frame())


def test_bbox_fallback_is_explicit_low_quality_diagnostic() -> None:
    mask = ExistingDetectionMaskAdapter(allow_legacy_bbox_fallback=True).predict(video_id="v", frame=_frame())[0]
    assert mask.valid and mask.is_legacy_bbox_fallback
    assert mask.confidence <= 0.25
    assert mask.metadata["formal_mask_evidence"] is False
    missing = ExistingDetectionMaskAdapter(allow_legacy_bbox_fallback=False).predict(video_id="v", frame=_frame())[0]
    assert not missing.valid and missing.visible_mask is None


def test_mock_provider_records_non_real_provenance() -> None:
    result = MockInstanceMaskProvider().predict(video_id="v", frame=_frame())[0]
    assert result.valid and result.metadata["mock_only"] is True
    assert result.metadata["truth_label_used"] is False


def test_support_prediction_uses_history_only() -> None:
    history = [observed("o", 0, rectangle(10, 10, 20, 20)), observed("o", 1, rectangle(12, 10, 22, 20))]
    prediction = predict_object_support(history, target_frame_index=2, geometry_mode="static_camera_3d")
    assert prediction.valid
    assert prediction.history_frames == (0, 1)
    assert prediction.metadata["current_frame_used_for_prediction"] is False
    assert prediction.metadata["translation_uv"] == (2.0, 0.0)


def test_target_mask_changes_validation_not_prediction() -> None:
    history = [observed("o", 0, rectangle(10, 10, 20, 20)), observed("o", 1, rectangle(12, 10, 22, 20))]
    prediction = predict_object_support(history, target_frame_index=2, geometry_mode="static_camera_3d")
    first = MaskTracker().track(prediction, observed("o", 2, rectangle(14, 10, 24, 20)))
    second = MaskTracker().track(prediction, observed("o", 2, rectangle(20, 10, 30, 20)))
    assert first.valid and second.valid
    assert np.array_equal(first.predicted_support_mask, second.predicted_support_mask)
    assert first.mask_iou > second.mask_iou
    assert first.metadata["current_observed_mask_used_for_prediction"] is False


def test_optical_or_point_warp_is_independent_validation_input() -> None:
    history = [observed("o", 0, rectangle(10, 10, 20, 20)), observed("o", 1, rectangle(12, 10, 22, 20))]
    prediction = predict_object_support(history, target_frame_index=2, geometry_mode="static_camera_3d")
    warped = rectangle(15, 10, 25, 20)
    result = MaskTracker().track(
        prediction,
        observed("o", 2, warped),
        optical_flow_or_point_warp=warped,
    )
    assert result.valid and result.mask_iou == 1.0
    assert result.metadata["optical_flow_or_point_warp_used"] is True
    assert result.metadata["current_observed_mask_used_for_prediction"] is False


def test_rotation_only_prediction_is_limited_and_unavailable_is_missing() -> None:
    history = [observed("o", 0, rectangle(10, 10, 20, 20)), observed("o", 1, rectangle(11, 10, 21, 20))]
    rotation = predict_object_support(history, target_frame_index=2, geometry_mode="rotation_compensated")
    assert rotation.valid and rotation.metadata["complete_3d_volume_claimed"] is False
    unavailable = predict_object_support(history, target_frame_index=2, geometry_mode="unavailable")
    assert not unavailable.valid and unavailable.support_mask is None
    assert math.isnan(unavailable.predicted_area)


def test_full_se3_requires_real_projection_support() -> None:
    history = [observed("o", 0, rectangle(10, 10, 20, 20)), observed("o", 1, rectangle(11, 10, 21, 20))]
    missing = predict_object_support(history, target_frame_index=2, geometry_mode="full_se3_3d")
    assert not missing.valid and missing.missing_reason == "full_se3_support_projector_unavailable"
    projected = predict_object_support(
        history,
        target_frame_index=2,
        geometry_mode="full_se3_3d",
        projected_full_se3_mask=rectangle(12, 10, 22, 20),
    )
    assert projected.valid and projected.metadata["full_se3_projection_path"] is True


def test_synthetic_catalogue_covers_required_a_to_n_cases() -> None:
    cases = synthetic_occlusion_scenarios()
    assert len(cases) == 14
    assert "normal_foreground_occlusion" in cases
    assert "bbox_overlap_without_contour_contact" in cases
    assert cases["rotation_only_camera"].geometry_mode.value == "rotation_compensated"
    assert not cases["missing_mask"].has_mask
