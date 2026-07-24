"""Pinhole back-projection under the canonical OpenCV coordinate convention."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..shared_3d_observation import GeometryScaleStatus, Point3DObservation
from .camera import validate_intrinsics


def _invalid_point(
    point_id: str,
    reason: str,
    source_point_2d_id: Optional[str],
    metadata: Optional[dict[str, object]],
    coordinate_frame: str,
) -> Point3DObservation:
    """Create an invalid camera point without zero-valued coordinates."""

    return Point3DObservation(
        point_id=point_id,
        x=None,
        y=None,
        z=None,
        coordinate_frame=coordinate_frame,
        scale_status=GeometryScaleStatus.UNKNOWN,
        confidence=0.0,
        valid=False,
        missing_reason=reason,
        source_point_2d_id=source_point_2d_id,
        metadata=dict(metadata or {}),
    )


def backproject_pixel(
    u: float,
    v: float,
    depth_z: float,
    K: np.ndarray,
    *,
    point_id: str = "point",
    confidence: float = 1.0,
    scale_status: GeometryScaleStatus | str = GeometryScaleStatus.RELATIVE_3D,
    source_point_2d_id: Optional[str] = None,
    valid: bool = True,
    missing_reason: str = "invalid_source_point",
    metadata: Optional[dict[str, object]] = None,
    coordinate_frame: str = "camera",
) -> Point3DObservation:
    """Back-project one pixel using X=(u-cx)Z/fx and Y=(v-cy)Z/fy.

    Integer ``u, v`` coordinates already denote pixel centres; no implicit 0.5
    offset is applied. ``depth_z`` is optical-axis Z depth, not radial range.
    Invalid inputs return an invalid point with ``None`` coordinates.
    """

    matrix = validate_intrinsics(K)
    values = np.asarray([u, v, depth_z, confidence], dtype=float)
    if not valid:
        return _invalid_point(
            point_id, missing_reason, source_point_2d_id, metadata, coordinate_frame
        )
    if not np.isfinite(values).all():
        return _invalid_point(
            point_id,
            "non_finite_pixel_depth_or_confidence",
            source_point_2d_id,
            metadata,
            coordinate_frame,
        )
    if depth_z <= 0.0:
        return _invalid_point(
            point_id,
            "non_positive_z_depth",
            source_point_2d_id,
            metadata,
            coordinate_frame,
        )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1].")

    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    x = (float(u) - cx) * float(depth_z) / fx
    y = (float(v) - cy) * float(depth_z) / fy
    point_metadata = {
        "source_pixel": [float(u), float(v)],
        "depth_definition": "camera_optical_axis_z",
        "pixel_center_convention": "integer_coordinates_are_pixel_centers_no_half_offset",
        **dict(metadata or {}),
    }
    return Point3DObservation(
        point_id=point_id,
        x=x,
        y=y,
        z=float(depth_z),
        coordinate_frame=coordinate_frame,
        scale_status=scale_status,
        confidence=float(confidence),
        valid=True,
        source_point_2d_id=source_point_2d_id,
        metadata=point_metadata,
    )


def backproject_points(
    pixels: np.ndarray,
    depths_z: np.ndarray,
    K: np.ndarray,
    *,
    valid_mask: Optional[np.ndarray] = None,
    confidences: Optional[np.ndarray] = None,
    point_ids: Optional[Sequence[str]] = None,
    scale_status: GeometryScaleStatus | str = GeometryScaleStatus.RELATIVE_3D,
    coordinate_frame: str = "camera",
) -> tuple[Point3DObservation, ...]:
    """Back-project N pixels while preserving per-point validity and confidence."""

    pixel_array = np.asarray(pixels, dtype=float)
    depth_array = np.asarray(depths_z, dtype=float).reshape(-1)
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 2:
        raise ValueError(f"pixels must have shape [N, 2], got {pixel_array.shape}.")
    count = pixel_array.shape[0]
    if depth_array.shape != (count,):
        raise ValueError("depths_z must contain exactly one depth per pixel.")
    mask = (
        np.ones(count, dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    confidence = (
        np.ones(count, dtype=float)
        if confidences is None
        else np.asarray(confidences, dtype=float)
    )
    if mask.shape != (count,) or confidence.shape != (count,):
        raise ValueError("valid_mask and confidences must have shape [N].")
    ids = (
        tuple(point_ids)
        if point_ids is not None
        else tuple(f"point_{index}" for index in range(count))
    )
    if len(ids) != count:
        raise ValueError("point_ids must contain exactly N entries.")
    return tuple(
        backproject_pixel(
            pixel_array[index, 0],
            pixel_array[index, 1],
            depth_array[index],
            K,
            point_id=ids[index],
            confidence=(
                float(confidence[index]) if np.isfinite(confidence[index]) else 0.0
            ),
            scale_status=scale_status,
            source_point_2d_id=ids[index],
            valid=bool(mask[index]),
            missing_reason="masked_invalid_source_point",
            coordinate_frame=coordinate_frame,
        )
        for index in range(count)
    )
