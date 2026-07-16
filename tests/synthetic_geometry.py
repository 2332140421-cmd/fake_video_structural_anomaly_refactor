"""Synthetic P1 geometry fixtures; never used as real-video reconstruction evidence."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from semantic3d.depth_provider import (
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from semantic3d.geometry.camera import CameraObservation
from semantic3d.geometry.projection import project_points
from semantic3d.shared_3d_observation import (
    GeometryScaleUnit,
    GeometryScaleStatus,
    Object3DObservation,
    Point2DObservation,
    Point3DObservation,
    ReconstructionFrame,
    Shared3DFrameObservation,
    VisibilityStatus,
)


def synthetic_intrinsics(
    width: int = 640, height: int = 480, fx: float = 500.0, fy: float = 500.0
) -> np.ndarray:
    """Return a known pinhole K with principal point at the image centre."""

    return np.asarray(
        [[fx, 0.0, (width - 1) / 2.0], [0.0, fy, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def synthetic_pose(
    rotation: np.ndarray | None = None,
    translation: Sequence[float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Return a known ``T_world_from_camera`` rigid transform."""

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(translation, dtype=float)
    return transform


def synthetic_camera(
    width: int = 640,
    height: int = 480,
    *,
    with_pose: bool = True,
    T_world_from_camera: np.ndarray | None = None,
) -> CameraObservation:
    """Return calibrated synthetic intrinsics and optional ground-truth pose."""

    pose = (
        (synthetic_pose() if T_world_from_camera is None else T_world_from_camera)
        if with_pose
        else None
    )
    return CameraObservation.from_parameters(
        K=synthetic_intrinsics(width, height),
        image_width=width,
        image_height=height,
        intrinsics_source="synthetic_ground_truth",
        quality=1.0,
        T_world_from_camera=pose,
        pose_source="synthetic_ground_truth" if pose is not None else "",
        metadata={"synthetic_only": True},
    )


def cuboid_xyz(
    center: Sequence[float] = (0.0, 0.0, 8.0),
    size: Sequence[float] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    """Return eight axis-aligned cuboid corners."""

    center_array = np.asarray(center, dtype=float)
    half = np.asarray(size, dtype=float) / 2.0
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    return center_array + signs * half


def point3d_observations(
    xyz: np.ndarray,
    scale_status: GeometryScaleStatus = GeometryScaleStatus.METRIC_3D,
) -> tuple[Point3DObservation, ...]:
    """Wrap known camera-frame coordinates as valid point observations."""

    return tuple(
        Point3DObservation(
            point_id=f"corner_{index}",
            x=float(point[0]),
            y=float(point[1]),
            z=float(point[2]),
            coordinate_frame="camera",
            scale_status=scale_status,
            confidence=1.0,
            valid=True,
            metadata={"synthetic_ground_truth": True},
        )
        for index, point in enumerate(np.asarray(xyz, dtype=float))
    )


def projected_point_observations(
    xyz: np.ndarray, K: np.ndarray
) -> tuple[Point2DObservation, ...]:
    """Project known camera points into valid synthetic image observations."""

    return project_points(point3d_observations(xyz), K)


def rasterize_sparse_depth(
    xyz: np.ndarray,
    K: np.ndarray,
    width: int = 640,
    height: int = 480,
    radius: int = 1,
) -> np.ndarray:
    """Rasterize small constant-Z patches around projected synthetic points."""

    depth = np.full((height, width), np.nan, dtype=np.float32)
    pixels = projected_point_observations(xyz, K)
    for point, coordinate in zip(pixels, np.asarray(xyz, dtype=float), strict=True):
        assert point.x is not None and point.y is not None
        column = int(np.floor(point.x + 0.5))
        row = int(np.floor(point.y + 0.5))
        row_min, row_max = max(0, row - radius), min(height, row + radius + 1)
        col_min, col_max = max(0, column - radius), min(width, column + radius + 1)
        depth[row_min:row_max, col_min:col_max] = float(coordinate[2])
    return depth


def synthetic_depth_observation(
    depth_map: np.ndarray,
    *,
    metric: bool = True,
    frame_index: int = 0,
) -> DepthObservation:
    """Wrap a known Z-depth map with explicit synthetic calibration metadata."""

    array = np.asarray(depth_map, dtype=np.float32)
    valid_mask = np.isfinite(array) & (array > 0.0)
    return DepthObservation(
        depth_map=array,
        raw_model_output=array.copy(),
        visualization_depth=np.where(valid_mask, array, np.nan),
        depth_representation=(
            DepthRepresentation.METRIC_DEPTH
            if metric
            else DepthRepresentation.RELATIVE_DEPTH
        ),
        scale_status=(
            DepthScaleStatus.METRIC_CALIBRATED
            if metric
            else DepthScaleStatus.RELATIVE_PER_FRAME
        ),
        larger_value_means=LargerValueMeans.FARTHER,
        valid_mask=valid_mask,
        confidence_map=np.where(valid_mask, 1.0, 0.0).astype(np.float32),
        provider_name="synthetic_ground_truth",
        frame_index=frame_index,
        valid=bool(np.any(valid_mask)),
        quality=1.0,
        metadata={
            "synthetic_only": True,
            "metric_scale_source": "synthetic_ground_truth" if metric else "none",
        },
    )


def synthetic_object_3d(
    object_id: str,
    *,
    label: str = "object",
    center: Sequence[float] = (0.0, 0.0, 8.0),
    size: Sequence[float] = (1.0, 1.0, 1.0),
    observed_scale: float | None = None,
    metric: bool = True,
    quality: float = 1.0,
    valid: bool = True,
) -> Object3DObservation:
    """Build a contract-valid synthetic sparse object in camera coordinates."""

    if not valid:
        return Object3DObservation.missing(
            "synthetic_video",
            0,
            label,
            object_id,
            canonical_label=label,
            reason="synthetic_missing_object",
        )
    scale_status = (
        GeometryScaleStatus.METRIC_3D
        if metric
        else GeometryScaleStatus.RELATIVE_3D
    )
    scale_unit = (
        GeometryScaleUnit.METER
        if metric
        else GeometryScaleUnit.RELATIVE_UNIT
    )
    points_xyz = cuboid_xyz(center=center, size=size)
    structure = point3d_observations(points_xyz, scale_status=scale_status)
    center_xyz = np.asarray(center, dtype=float)
    center_point = Point3DObservation(
        point_id=f"{object_id}:center",
        x=float(center_xyz[0]),
        y=float(center_xyz[1]),
        z=float(center_xyz[2]),
        coordinate_frame="camera",
        scale_status=scale_status,
        confidence=quality,
        valid=True,
        metadata={"synthetic_ground_truth": True},
    )
    resolved_scale = (
        float(np.linalg.norm(np.asarray(size, dtype=float)))
        if observed_scale is None
        else float(observed_scale)
    )
    normalized = tuple(
        Point3DObservation(
            point_id=f"{point.point_id}:normalized",
            x=float((point.x - center_xyz[0]) / resolved_scale),
            y=float((point.y - center_xyz[1]) / resolved_scale),
            z=float((point.z - center_xyz[2]) / resolved_scale),
            coordinate_frame="object_normalized",
            scale_status=GeometryScaleStatus.NORMALIZED_SHAPE,
            confidence=quality,
            valid=True,
            source_point_2d_id=point.source_point_2d_id,
            metadata={"synthetic_ground_truth": True},
        )
        for point in structure
    )
    return Object3DObservation(
        video_id="synthetic_video",
        frame_index=0,
        track_id=None,
        semantic_label=label,
        canonical_label=label,
        center_3d=center_point,
        boundary_points_3d=structure,
        keypoints_3d=(),
        structure_points_3d=structure,
        observed_scale_3d=resolved_scale,
        normalized_structure_points=normalized,
        scale_status=scale_status,
        visibility=VisibilityStatus.VISIBLE,
        reconstruction_quality=quality,
        valid=True,
        missing_reason="",
        source_object_2d_id=object_id,
        metadata={
            "valid_point_ratio": 1.0,
            "metric_scale_source": "synthetic_ground_truth" if metric else "none",
            "source_bbox": [20.0, 20.0, 60.0, 80.0],
        },
        center_3d_camera=center_point,
        center_3d_world=None,
        scale_method="synthetic_explicit",
        scale_unit=scale_unit,
        reconstruction_frame=ReconstructionFrame.CAMERA,
        scale_quality=quality,
        scale_descriptors={"synthetic_size": list(size)},
        depth_scale_status=(
            DepthScaleStatus.METRIC_CALIBRATED
            if metric
            else DepthScaleStatus.RELATIVE_PER_FRAME
        ),
    )


def synthetic_shared_3d_frame(
    objects: Sequence[Object3DObservation],
    *,
    width: int = 128,
    height: int = 128,
    depth_map: np.ndarray | None = None,
    metric: bool = True,
    approximate_intrinsics: bool = False,
) -> Shared3DFrameObservation:
    """Build one shared frame whose static branches reuse one camera and depth."""

    array = (
        np.full((height, width), 8.0, dtype=np.float32)
        if depth_map is None
        else np.asarray(depth_map, dtype=np.float32)
    )
    depth = synthetic_depth_observation(array, metric=metric)
    if approximate_intrinsics:
        camera = CameraObservation.from_parameters(
            K=synthetic_intrinsics(width, height, fx=150.0, fy=150.0),
            image_width=width,
            image_height=height,
            intrinsics_source="approximate",
            quality=0.5,
            metadata={"synthetic_only": True},
        )
    else:
        camera = synthetic_camera(width, height, with_pose=False)
    return Shared3DFrameObservation(
        video_id="synthetic_video",
        frame_index=0,
        image_width=width,
        image_height=height,
        camera=camera,
        depth=depth,
        objects=tuple(objects),
        valid=True,
        quality=float(np.mean([camera.quality, depth.quality])),
        source_frame_id="synthetic_frame_000",
        metadata={"synthetic_only": True},
    )
