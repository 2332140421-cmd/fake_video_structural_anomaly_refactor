from __future__ import annotations

import numpy as np
import pytest

from semantic3d.dynamic_3d import (
    DynamicGeometryMode,
    PointTrack2DObservation,
    reconstruct_point_tracks_3d,
)
from semantic3d.sequence_geometry import SequenceScaleStatus

from synthetic_dynamic_3d import constant_velocity_points, make_synthetic_dynamic_scene


def test_full_se3_reconstruction_uses_shared_clip_world_pose() -> None:
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)],
        world_points=constant_velocity_points(3),
        mode=DynamicGeometryMode.FULL_SE3_3D,
    )
    object_points = [point for point in scene.points_3d if point.point_id == "object_p0"]
    assert len(object_points) == 3
    for point in object_points:
        assert point.valid and point.point_3d_world is not None
        assert np.allclose(
            point.point_3d_world,
            scene.world_points["object_p0"][point.frame_index],
            atol=2e-2,
        )
        assert point.metadata["shared_clip_reused"] is True
        assert point.metadata["depth_reestimated"] is False


def test_static_mode_builds_camera_track_without_world_claim() -> None:
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 3,
        world_points=constant_velocity_points(3),
        mode=DynamicGeometryMode.STATIC_CAMERA_3D,
    )
    assert all(point.valid for point in scene.points_3d)
    assert all(point.point_3d_world is None for point in scene.points_3d)


def test_rotation_only_keeps_camera_ray_but_does_not_create_world_trajectory() -> None:
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 3,
        world_points=constant_velocity_points(3),
        mode=DynamicGeometryMode.ROTATION_COMPENSATED,
    )
    assert scene.points_3d
    assert all(point.valid for point in scene.points_3d)
    assert all(point.point_3d_camera is not None for point in scene.points_3d)
    assert all(point.point_3d_world is None for point in scene.points_3d)


def test_relative_per_frame_forbids_3d_tracks() -> None:
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 3,
        world_points=constant_velocity_points(3),
        mode=DynamicGeometryMode.STATIC_CAMERA_3D,
        scale_status=SequenceScaleStatus.RELATIVE_PER_FRAME,
    )
    assert not scene.readiness.dynamic_3d_ready
    assert all(not point.valid for point in scene.points_3d)


def test_current_pixel_must_be_independent_observation() -> None:
    with pytest.raises(ValueError, match="independent"):
        PointTrack2DObservation(
            point_id="p",
            object_track_id="o",
            frame_index=0,
            pixel_uv=(1.0, 2.0),
            visibility="visible",
            occlusion_status="visible",
            tracking_confidence=1.0,
            source_tracker="projector",
            valid=True,
            metadata={"independent_observation": False},
        )


def test_point_id_is_stable_and_unique_per_frame() -> None:
    scene = make_synthetic_dynamic_scene(
        camera_centers=[(0.0, 0.0, 0.0)] * 4,
        world_points=constant_velocity_points(4),
        mode=DynamicGeometryMode.STATIC_CAMERA_3D,
    )
    keys = [(point.point_id, point.frame_index) for point in scene.points_2d]
    assert len(keys) == len(set(keys))
    assert {point.source_tracker for point in scene.points_2d} == {
        "synthetic_ground_truth"
    }
