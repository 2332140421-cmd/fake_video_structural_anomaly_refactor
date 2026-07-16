from __future__ import annotations

import math

import numpy as np

from semantic3d.dynamic_3d import (
    DynamicGeometryMode,
    compute_track_3d_continuity_residuals,
)
from semantic3d.sequence_geometry import SequenceScaleStatus

from synthetic_dynamic_3d import make_synthetic_dynamic_scene


def _scene(points, *, scene_cut_frame=None, break_point_id_at=None, scale_status=SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE):
    frame_count = len(next(iter(points.values())))
    return make_synthetic_dynamic_scene(
        camera_centers=[(0.1 * index, 0.0, 0.0) for index in range(frame_count)],
        world_points=points,
        mode=DynamicGeometryMode.FULL_SE3_3D,
        scale_status=scale_status,
        scene_cut_frame=scene_cut_frame,
        break_point_id_at=break_point_id_at,
    )


def test_constant_velocity_residual_is_near_zero() -> None:
    points = {"p": [np.asarray([0.2 * i, 0.0, 5.0]) for i in range(5)]}
    scene = _scene(points)
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    valid = [item for item in residuals if item.valid]
    assert len(valid) == 3
    assert max(item.raw_residual for item in valid) < 1e-6


def test_static_camera_constant_velocity_is_near_zero() -> None:
    points = {"p": [np.asarray([0.15 * i, 0.0, 5.0]) for i in range(5)]}
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 5,
        world_points=points,
        mode=DynamicGeometryMode.STATIC_CAMERA_3D,
    )
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    assert max(item.raw_residual for item in residuals if item.valid) < 1e-6


def test_jump_residual_is_clearly_higher() -> None:
    smooth = [np.asarray([0.2 * i, 0.0, 5.0]) for i in range(5)]
    jumped = [point.copy() for point in smooth]
    jumped[3] = jumped[3] + np.asarray([1.0, 0.0, 0.0])
    scene = _scene({"p": jumped})
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    assert max(item.raw_residual for item in residuals if item.valid) > 0.9


def test_static_camera_jump_residual_is_high() -> None:
    points = [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(5)]
    points[3] = points[3] + np.asarray([1.0, 0.0, 0.0])
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 5,
        world_points={"p": points},
        mode=DynamicGeometryMode.STATIC_CAMERA_3D,
    )
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    assert max(item.raw_residual for item in residuals if item.valid) > 0.9


def test_local_single_point_anomaly_stays_local() -> None:
    smooth = [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(5)]
    jumped = [point.copy() for point in smooth]
    jumped[3] = jumped[3] + np.asarray([0.8, 0.0, 0.0])
    scene = _scene({"bad": jumped, "good": smooth})
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    bad = [item.raw_residual for item in residuals if item.point_id == "bad" and item.valid]
    good = [item.raw_residual for item in residuals if item.point_id == "good" and item.valid]
    assert max(bad) > 0.7
    assert max(good) < 1e-6


def test_point_id_break_returns_invalid_evidence() -> None:
    points = {"p": [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(4)]}
    scene = _scene(points, break_point_id_at=2)
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    assert residuals
    assert all(not item.valid for item in residuals)
    assert all(math.isnan(item.raw_evidence.value) for item in residuals)


def test_scene_cut_blocks_track_residual() -> None:
    points = {"p": [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(4)]}
    scene = _scene(points, scene_cut_frame=2)
    residuals = compute_track_3d_continuity_residuals(
        scene.points_3d, scene_cut_flags=scene.clip.scene_cut_flags
    )
    assert any(item.missing_reason == "scene_cut_breaks_track" for item in residuals)


def test_missing_object_scale_preserves_raw_and_normalized_nan() -> None:
    points = {"p": [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(4)]}
    scene = _scene(points)
    result = next(item for item in compute_track_3d_continuity_residuals(scene.points_3d) if item.valid)
    assert math.isfinite(result.raw_residual)
    assert math.isnan(result.normalized_residual)
    assert not result.normalized_evidence.valid


def test_relative_aligned_and_metric_sequence_are_eligible() -> None:
    points = {"p": [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(4)]}
    relative = _scene(points, scale_status=SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE)
    metric = _scene(points, scale_status=SequenceScaleStatus.METRIC_SEQUENCE)
    assert any(item.valid for item in compute_track_3d_continuity_residuals(relative.points_3d))
    assert any(item.valid for item in compute_track_3d_continuity_residuals(metric.points_3d))


def test_rotation_only_does_not_output_world_3d_continuity() -> None:
    points = {"p": [np.asarray([0.1 * i, 0.0, 5.0]) for i in range(4)]}
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 4,
        world_points=points,
        mode=DynamicGeometryMode.ROTATION_COMPENSATED,
    )
    residuals = compute_track_3d_continuity_residuals(scene.points_3d)
    assert residuals
    assert all(not item.valid for item in residuals)
    assert {item.missing_reason for item in residuals} == {
        "rotation_only_no_world_3d_continuity"
    }
