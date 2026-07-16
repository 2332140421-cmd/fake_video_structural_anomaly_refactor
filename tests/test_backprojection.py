from __future__ import annotations

import numpy as np
import pytest

from semantic3d.geometry.backprojection import backproject_pixel, backproject_points
from semantic3d.geometry.projection import project_point, project_points
from semantic3d.shared_3d_observation import GeometryScaleStatus, Point3DObservation

from synthetic_geometry import point3d_observations, synthetic_intrinsics


def test_project_then_backproject_recovers_known_point() -> None:
    K = synthetic_intrinsics()
    source = point3d_observations(np.asarray([[0.75, -0.4, 6.0]]))[0]
    pixel = project_point(source, K)
    assert pixel.valid and pixel.x is not None and pixel.y is not None

    recovered = backproject_pixel(
        pixel.x,
        pixel.y,
        source.z,
        K,
        scale_status=GeometryScaleStatus.METRIC_3D,
    )
    np.testing.assert_allclose(recovered.as_array(), source.as_array(), atol=1e-12)


def test_batch_projection_backprojection_closes() -> None:
    K = synthetic_intrinsics()
    xyz = np.asarray([[-1.0, -0.5, 4.0], [0.5, 1.0, 8.0], [1.5, -1.0, 10.0]])
    source = point3d_observations(xyz)
    projected = project_points(source, K)
    pixels = np.asarray([[point.x, point.y] for point in projected], dtype=float)
    recovered = backproject_points(
        pixels,
        xyz[:, 2],
        K,
        scale_status=GeometryScaleStatus.METRIC_3D,
    )
    np.testing.assert_allclose(
        np.stack([point.as_array() for point in recovered]), xyz, atol=1e-12
    )


def test_non_positive_depth_is_invalid_without_zero_coordinates() -> None:
    point = backproject_pixel(10.0, 20.0, 0.0, synthetic_intrinsics())
    assert not point.valid
    assert point.x is point.y is point.z is None
    assert point.missing_reason == "non_positive_z_depth"


def test_projection_rejects_non_positive_z() -> None:
    point = Point3DObservation(
        "behind",
        1.0,
        2.0,
        -1.0,
        "camera",
        GeometryScaleStatus.RELATIVE_3D,
        1.0,
        True,
    )
    projected = project_point(point, synthetic_intrinsics())
    assert not projected.valid
    assert projected.x is projected.y is None
    assert projected.missing_reason == "non_positive_z_depth"


def test_batch_valid_mask_is_propagated() -> None:
    points = backproject_points(
        np.asarray([[100.0, 100.0], [200.0, 200.0]]),
        np.asarray([5.0, 6.0]),
        synthetic_intrinsics(),
        valid_mask=np.asarray([True, False]),
    )
    assert points[0].valid
    assert not points[1].valid
    assert points[1].x is None
