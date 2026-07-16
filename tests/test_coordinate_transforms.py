from __future__ import annotations

import math

import numpy as np

from semantic3d.geometry.transforms import (
    camera_center_world,
    camera_to_world,
    transform_points,
    world_to_camera,
)
from semantic3d.shared_3d_observation import GeometryScaleStatus, Point3DObservation

from synthetic_geometry import synthetic_camera, synthetic_pose


def _point(x: float, y: float, z: float, frame: str = "camera") -> Point3DObservation:
    return Point3DObservation(
        "point",
        x,
        y,
        z,
        frame,
        GeometryScaleStatus.METRIC_3D,
        1.0,
        True,
    )


def test_identity_pose_keeps_coordinates() -> None:
    output = transform_points(
        (_point(1.0, 2.0, 3.0),),
        np.eye(4),
        source_frame="camera",
        target_frame="world",
        transform_name="T_world_from_camera",
    )[0]
    np.testing.assert_allclose(output.as_array(), [1.0, 2.0, 3.0])


def test_known_rotation_and_translation() -> None:
    angle = math.pi / 2.0
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle), 0.0], [math.sin(angle), math.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    transform = synthetic_pose(rotation, translation=(1.0, 2.0, 3.0))
    output = transform_points(
        (_point(1.0, 0.0, 2.0),),
        transform,
        source_frame="camera",
        target_frame="world",
        transform_name="T_world_from_camera",
    )[0]
    np.testing.assert_allclose(output.as_array(), [1.0, 3.0, 5.0], atol=1e-12)


def test_camera_world_round_trip() -> None:
    pose = synthetic_pose(translation=(2.0, -1.0, 0.5))
    camera = synthetic_camera(T_world_from_camera=pose)
    source = _point(0.5, 1.0, 4.0)
    world = camera_to_world((source,), camera)[0]
    recovered = world_to_camera((world,), camera)[0]
    np.testing.assert_allclose(recovered.as_array(), source.as_array(), atol=1e-12)


def test_camera_center_uses_world_from_camera_translation() -> None:
    pose = synthetic_pose(translation=(3.0, -2.0, 1.5))
    center = camera_center_world(T_world_from_camera=pose, scale_status=GeometryScaleStatus.METRIC_3D)
    np.testing.assert_allclose(center.as_array(), [3.0, -2.0, 1.5])


def test_missing_pose_does_not_claim_world_points() -> None:
    camera = synthetic_camera(with_pose=False)
    world = camera_to_world((_point(0.0, 0.0, 5.0),), camera)[0]
    assert not world.valid
    assert world.x is world.y is world.z is None
    assert world.missing_reason == "no_camera_pose"
