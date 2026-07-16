from __future__ import annotations

import math

import numpy as np
import pytest

from semantic3d.dynamic_3d import (
    DynamicGeometryMode,
    PointTrack2DObservation,
    ReprojectionEvidenceType,
    compute_dynamic_reprojection_residual,
)

from synthetic_dynamic_3d import make_synthetic_dynamic_scene


def _moving_scene():
    points = {
        "background": [np.asarray([0.5, 0.1, 6.0])] * 3,
        "object": [np.asarray([0.2 * i, 0.0, 5.0]) for i in range(3)],
    }
    return make_synthetic_dynamic_scene(
        camera_centers=[(0.15 * i, 0.0, 0.0) for i in range(3)],
        world_points=points,
        mode=DynamicGeometryMode.FULL_SE3_3D,
    )


def _point(scene, point_id: str, frame: int):
    return next(
        point for point in scene.points_3d
        if point.point_id == point_id and point.frame_index == frame
    )


def _point2d(scene, point_id: str, frame: int):
    return next(
        point for point in scene.points_2d
        if point.point_id == point_id and point.frame_index == frame
    )


def test_camera_compensation_reduces_background_error() -> None:
    scene = _moving_scene()
    previous = _point(scene, "background", 0)
    current = _point2d(scene, "background", 1)
    true_pose = scene.clip.relative_poses[1].relative_pose_from_previous
    assert true_pose is not None
    compensated = compute_dynamic_reprojection_residual(
        previous,
        current,
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=true_pose,
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
        is_background=True,
    )
    uncompensated = compute_dynamic_reprojection_residual(
        previous,
        current,
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=np.eye(4),
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
        is_background=True,
    )
    assert compensated.pixel_error < 1e-6
    assert uncompensated.pixel_error > compensated.pixel_error + 1.0
    assert compensated.evidence_type == ReprojectionEvidenceType.BACKGROUND_QA
    assert not compensated.residual_evidence.valid


def test_camera_and_object_motion_prediction_recovers_current_observation() -> None:
    scene = _moving_scene()
    previous = _point(scene, "object", 0)
    current = _point2d(scene, "object", 1)
    current_camera_truth = scene.world_points["object"][1] - np.asarray([0.15, 0.0, 0.0])
    result = compute_dynamic_reprojection_residual(
        previous,
        current,
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=scene.clip.relative_poses[1].relative_pose_from_previous,
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
        is_background=False,
        predicted_foreground_point_current_camera=current_camera_truth,
        has_history_motion_model=True,
    )
    assert result.pixel_error < 1e-6
    assert result.evidence_type == ReprojectionEvidenceType.DYNAMIC_RESIDUAL
    assert result.residual_evidence.valid


def test_foreground_camera_only_error_is_diagnostic_not_anomaly() -> None:
    scene = _moving_scene()
    result = compute_dynamic_reprojection_residual(
        _point(scene, "object", 0),
        _point2d(scene, "object", 1),
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=scene.clip.relative_poses[1].relative_pose_from_previous,
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
        is_background=False,
    )
    assert result.valid and result.diagnostic_evidence.valid
    assert not result.residual_evidence.valid
    assert result.evidence_type == ReprojectionEvidenceType.CAMERA_COMPENSATED_MOTION


def test_current_independent_observation_changes_measured_error() -> None:
    scene = _moving_scene()
    original = _point2d(scene, "background", 1)
    assert original.pixel_uv is not None
    offset = PointTrack2DObservation(
        point_id=original.point_id,
        object_track_id=original.object_track_id,
        frame_index=original.frame_index,
        pixel_uv=(original.pixel_uv[0] + 5.0, original.pixel_uv[1]),
        visibility=original.visibility,
        occlusion_status=original.occlusion_status,
        tracking_confidence=original.tracking_confidence,
        source_tracker="independent_offset_tracker",
        valid=True,
        metadata={"independent_observation": True, "generated_from_projection": False},
    )
    result = compute_dynamic_reprojection_residual(
        _point(scene, "background", 0),
        offset,
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=scene.clip.relative_poses[1].relative_pose_from_previous,
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
        is_background=True,
    )
    assert math.isclose(result.pixel_error, 5.0, abs_tol=1e-6)


def test_rotation_only_ignores_unknown_translation() -> None:
    scene = _moving_scene()
    previous = _point(scene, "background", 0)
    assert previous.point_3d_camera is not None
    angle = np.deg2rad(5.0)
    rotation = np.asarray(
        [[np.cos(angle), 0.0, np.sin(angle)], [0.0, 1.0, 0.0], [-np.sin(angle), 0.0, np.cos(angle)]]
    )
    transformed = rotation @ np.asarray(previous.point_3d_camera)
    uv_h = scene.K @ transformed
    uv = uv_h[:2] / uv_h[2]
    current = PointTrack2DObservation(
        point_id=previous.point_id,
        object_track_id=previous.object_track_id,
        frame_index=1,
        pixel_uv=tuple(uv),
        visibility="visible",
        occlusion_status="visible",
        tracking_confidence=1.0,
        source_tracker="synthetic_independent_rotation",
        valid=True,
        metadata={"independent_observation": True, "generated_from_projection": False},
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [100.0, 0.0, 0.0]
    result = compute_dynamic_reprojection_residual(
        previous,
        current,
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=transform,
        geometry_mode=DynamicGeometryMode.ROTATION_COMPENSATED,
        is_background=True,
    )
    assert result.pixel_error < 1e-6
    assert result.metadata["rotation_only_translation_assumed_zero"] is False


def test_missing_input_stays_nan() -> None:
    scene = _moving_scene()
    missing = PointTrack2DObservation.missing(
        point_id="background",
        object_track_id="synthetic_object",
        frame_index=1,
        reason="tracker_lost",
        source_tracker="synthetic",
    )
    result = compute_dynamic_reprojection_residual(
        _point(scene, "background", 0),
        missing,
        K_current=scene.K,
        image_width=160,
        image_height=120,
        relative_pose_current_from_previous=np.eye(4),
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D,
        is_background=True,
    )
    assert not result.valid
    assert math.isnan(result.pixel_error)
    assert math.isnan(result.residual_evidence.value)


def test_same_source_projection_cannot_masquerade_as_observation() -> None:
    with pytest.raises(ValueError, match="Projected points"):
        PointTrack2DObservation(
            point_id="p",
            object_track_id="o",
            frame_index=1,
            pixel_uv=(10.0, 10.0),
            visibility="visible",
            occlusion_status="visible",
            tracking_confidence=1.0,
            source_tracker="project_point",
            valid=True,
            metadata={
                "independent_observation": True,
                "generated_from_projection": True,
            },
        )
