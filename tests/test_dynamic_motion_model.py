from __future__ import annotations

import numpy as np

from semantic3d.dynamic_3d import (
    DynamicGeometryMode,
    ObjectMedianTranslationModel,
    PointConstantVelocityModel,
    PointTrack3DObservation,
)
from semantic3d.sequence_geometry import SequenceScaleStatus


def point(point_id: str, frame: int, xyz, *, mode=DynamicGeometryMode.STATIC_CAMERA_3D, object_id="obj") -> PointTrack3DObservation:
    xyz = tuple(float(value) for value in xyz)
    return PointTrack3DObservation(
        point_id=point_id, object_track_id=object_id, frame_index=frame,
        pixel_uv=(20.0 + frame, 20.0), observed_depth=max(xyz[2], 1e-3),
        point_3d_camera=xyz, point_3d_world=(xyz if mode == DynamicGeometryMode.FULL_SE3_3D else None),
        visibility="visible", occlusion_status="visible", tracking_confidence=1.0,
        depth_quality=1.0, reconstruction_quality=1.0,
        source_tracker="synthetic_independent",
        scale_status=SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE,
        geometry_mode=mode, valid=True,
        metadata={"independent_observation": True},
    )


def test_constant_velocity_prediction_does_not_read_current_frame() -> None:
    history = [point("p", 0, (0, 0, 5)), point("p", 1, (1, 0, 5))]
    prediction = PointConstantVelocityModel().predict(history, target_frame_index=2)
    assert prediction.valid
    assert np.allclose(prediction.predicted_point_3d, (2.0, 0.0, 5.0))
    assert prediction.history_frames == (0, 1)
    assert prediction.metadata["target_observation_used"] is False
    assert all(frame < prediction.target_frame_index for frame in prediction.history_frames)


def test_current_observation_cannot_leak_into_prediction() -> None:
    base = [point("p", 0, (0, 0, 5)), point("p", 1, (1, 0, 5))]
    with_current_jump = base + [point("p", 2, (100, 100, 5))]
    first = PointConstantVelocityModel().predict(base, target_frame_index=2)
    second = PointConstantVelocityModel().predict(with_current_jump, target_frame_index=2)
    assert first.predicted_point_3d == second.predicted_point_3d


def test_object_median_translation_is_leave_one_point_out() -> None:
    histories = {
        "target": [point("target", 0, (0, 0, 5)), point("target", 1, (100, 0, 5))],
        "support_a": [point("support_a", 0, (0, 1, 5)), point("support_a", 1, (1, 1, 5))],
        "support_b": [point("support_b", 0, (0, 2, 5)), point("support_b", 1, (1, 2, 5))],
    }
    prediction = ObjectMedianTranslationModel(histories).predict(histories["target"], target_frame_index=2)
    assert prediction.valid
    assert prediction.support_point_ids == ("support_a", "support_b")
    assert np.allclose(prediction.predicted_point_3d, (101.0, 0.0, 5.0))
    assert prediction.metadata["leave_one_point_out"] is True


def test_rotation_only_model_predicts_bearing_not_translation() -> None:
    history = [
        point("p", 0, (0.0, 0.0, 5.0), mode=DynamicGeometryMode.ROTATION_COMPENSATED),
        point("p", 1, (0.1, 0.0, 5.0), mode=DynamicGeometryMode.ROTATION_COMPENSATED),
    ]
    prediction = PointConstantVelocityModel().predict(history, target_frame_index=2)
    assert prediction.valid
    assert prediction.model_type == "point_constant_angular_velocity_history_only"
    assert prediction.metadata["coordinate_kind"] == "bearing"
