"""Direction-explicit rigid transforms for 3D point observations."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..shared_3d_observation import GeometryScaleStatus, Point3DObservation
from .camera import CameraObservation, validate_rigid_transform


def _invalid_transformed_point(
    point: Point3DObservation,
    target_frame: str,
    reason: str,
    metadata: Optional[dict[str, object]] = None,
) -> Point3DObservation:
    """Create an invalid transformed point with no zero coordinates."""

    return Point3DObservation(
        point_id=point.point_id,
        x=None,
        y=None,
        z=None,
        coordinate_frame=target_frame,
        scale_status=GeometryScaleStatus.UNKNOWN,
        confidence=0.0,
        valid=False,
        missing_reason=reason,
        source_point_2d_id=point.source_point_2d_id,
        metadata={**dict(point.metadata), **dict(metadata or {})},
    )


def transform_points(
    points: Sequence[Point3DObservation],
    transform: np.ndarray,
    *,
    source_frame: str,
    target_frame: str,
    transform_name: str,
) -> tuple[Point3DObservation, ...]:
    """Apply a 4x4 column-vector transform and preserve invalid observations."""

    matrix = validate_rigid_transform(transform, transform_name)
    output: list[Point3DObservation] = []
    for point in points:
        if not point.valid:
            output.append(
                _invalid_transformed_point(
                    point, target_frame, point.missing_reason, {"transform": transform_name}
                )
            )
            continue
        if point.coordinate_frame != source_frame:
            output.append(
                _invalid_transformed_point(
                    point,
                    target_frame,
                    "coordinate_frame_mismatch",
                    {
                        "expected_source_frame": source_frame,
                        "actual_source_frame": point.coordinate_frame,
                    },
                )
            )
            continue
        homogeneous = np.concatenate([point.as_array(), np.asarray([1.0])])
        transformed = matrix @ homogeneous
        if not np.isfinite(transformed).all() or abs(transformed[3]) <= 1e-12:
            output.append(
                _invalid_transformed_point(
                    point,
                    target_frame,
                    "invalid_homogeneous_transform",
                    {"transform": transform_name},
                )
            )
            continue
        xyz = transformed[:3] / transformed[3]
        output.append(
            Point3DObservation(
                point_id=point.point_id,
                x=float(xyz[0]),
                y=float(xyz[1]),
                z=float(xyz[2]),
                coordinate_frame=target_frame,
                scale_status=point.scale_status,
                confidence=point.confidence,
                valid=True,
                source_point_2d_id=point.source_point_2d_id,
                metadata={**dict(point.metadata), "transform": transform_name},
            )
        )
    return tuple(output)


def camera_to_world(
    points: Sequence[Point3DObservation], camera: CameraObservation
) -> tuple[Point3DObservation, ...]:
    """Transform camera-frame points using ``T_world_from_camera``."""

    if not camera.pose_valid or camera.T_world_from_camera is None:
        return tuple(
            _invalid_transformed_point(point, "world", "no_camera_pose")
            for point in points
        )
    return transform_points(
        points,
        camera.T_world_from_camera,
        source_frame="camera",
        target_frame="world",
        transform_name="T_world_from_camera",
    )


def world_to_camera(
    points: Sequence[Point3DObservation], camera: CameraObservation
) -> tuple[Point3DObservation, ...]:
    """Transform world-frame points using ``T_camera_from_world``."""

    if not camera.pose_valid or camera.T_camera_from_world is None:
        return tuple(
            _invalid_transformed_point(point, "camera", "no_camera_pose")
            for point in points
        )
    return transform_points(
        points,
        camera.T_camera_from_world,
        source_frame="world",
        target_frame="camera",
        transform_name="T_camera_from_world",
    )


def camera_center_world(
    *,
    T_world_from_camera: Optional[np.ndarray] = None,
    T_camera_from_world: Optional[np.ndarray] = None,
    scale_status: GeometryScaleStatus | str = GeometryScaleStatus.RELATIVE_3D,
) -> Point3DObservation:
    """Return the world-coordinate camera centre from either transform direction."""

    if T_world_from_camera is None and T_camera_from_world is None:
        template = Point3DObservation(
            point_id="camera_center_source",
            x=None,
            y=None,
            z=None,
            coordinate_frame="camera",
            scale_status=GeometryScaleStatus.UNKNOWN,
            confidence=0.0,
            valid=False,
            missing_reason="no_camera_pose",
        )
        return _invalid_transformed_point(template, "world", "no_camera_pose")
    if T_world_from_camera is None:
        assert T_camera_from_world is not None
        tcw = validate_rigid_transform(T_camera_from_world, "T_camera_from_world")
        twc = np.linalg.inv(tcw)
    else:
        twc = validate_rigid_transform(T_world_from_camera, "T_world_from_camera")
        if T_camera_from_world is not None:
            tcw = validate_rigid_transform(T_camera_from_world, "T_camera_from_world")
            if not (
                np.allclose(twc @ tcw, np.eye(4), atol=1e-6)
                and np.allclose(tcw @ twc, np.eye(4), atol=1e-6)
            ):
                template = Point3DObservation(
                    point_id="camera_center_source",
                    x=None,
                    y=None,
                    z=None,
                    coordinate_frame="camera",
                    scale_status=GeometryScaleStatus.UNKNOWN,
                    confidence=0.0,
                    valid=False,
                    missing_reason="inconsistent_camera_transforms",
                )
                return _invalid_transformed_point(
                    template, "world", "inconsistent_camera_transforms"
                )
    center = twc[:3, 3]
    return Point3DObservation(
        point_id="camera_center",
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2]),
        coordinate_frame="world",
        scale_status=scale_status,
        confidence=1.0,
        valid=True,
        metadata={"computed_from": "T_world_from_camera"},
    )
