"""Pinhole projection for explicit camera-frame 3D observations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..shared_3d_observation import (
    Point2DObservation,
    Point3DObservation,
    is_camera_coordinate_frame,
)
from .camera import validate_intrinsics


def _invalid_projection(point: Point3DObservation, reason: str) -> Point2DObservation:
    """Create an invalid image point without fabricated pixel coordinates."""

    return Point2DObservation(
        point_id=point.point_id,
        x=None,
        y=None,
        confidence=0.0,
        valid=False,
        missing_reason=reason,
        source="project_point",
        metadata={"source_point_3d_id": point.point_id},
    )


def project_point(point: Point3DObservation, K: np.ndarray) -> Point2DObservation:
    """Project one camera-frame 3D point to a pixel-centre coordinate."""

    matrix = validate_intrinsics(K)
    if not point.valid:
        return _invalid_projection(point, point.missing_reason or "invalid_3d_point")
    if not is_camera_coordinate_frame(point.coordinate_frame):
        return _invalid_projection(point, "projection_requires_camera_frame")
    xyz = point.as_array()
    if xyz[2] <= 0.0:
        return _invalid_projection(point, "non_positive_z_depth")
    projected = matrix @ xyz
    if not np.isfinite(projected).all() or abs(projected[2]) <= 1e-12:
        return _invalid_projection(point, "invalid_homogeneous_projection")
    return Point2DObservation(
        point_id=point.point_id,
        x=float(projected[0] / projected[2]),
        y=float(projected[1] / projected[2]),
        confidence=point.confidence,
        valid=True,
        source="project_point",
        metadata={
            "source_point_3d_id": point.point_id,
            "pixel_center_convention": "integer_coordinates_are_pixel_centers_no_half_offset",
        },
    )


def project_points(
    points: Sequence[Point3DObservation], K: np.ndarray
) -> tuple[Point2DObservation, ...]:
    """Project a batch while propagating every point's validity."""

    return tuple(project_point(point, K) for point in points)
