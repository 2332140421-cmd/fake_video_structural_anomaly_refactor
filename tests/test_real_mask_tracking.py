from __future__ import annotations

import math

from semantic3d.occlusion import MaskTracker, PredictedObjectSupport, predict_object_support
from synthetic_occlusion import observed, rectangle


def test_tracked_mask_exposes_independent_and_history_fields() -> None:
    history = [observed("o", 0, rectangle(10, 10, 20, 20)), observed("o", 1, rectangle(11, 10, 21, 20))]
    prediction = predict_object_support(history, target_frame_index=2, geometry_mode="static_camera_3d")
    result = MaskTracker().track(prediction, observed("o", 2, rectangle(12, 10, 22, 20)))
    assert result.valid and result.mask_iou == 1.0
    assert result.independently_observed_mask is not result.history_predicted_mask
    assert result.area_change_ratio == 1.0 and result.assignment_consistency == 1.0


def test_scene_cut_prediction_cannot_be_propagated() -> None:
    prediction = PredictedObjectSupport.missing(video_id="v", object_track_id="o", target_frame_index=2, image_shape=(64, 64), geometry_mode="static_camera_3d", reason="scene_cut_breaks_mask_history")
    result = MaskTracker().track(prediction, observed("o", 2, rectangle(12, 10, 22, 20)))
    assert not result.valid and result.propagated_mask is None
    assert math.isnan(result.area_change_ratio)
