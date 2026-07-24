"""Single-frame metric visible-surface reconstruction in camera coordinates."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..geometry.camera import validate_intrinsics
from ..method_completion.metric_scale import ray_distance_to_z_depth
from ..shared_3d_observation import (
    CoordinateFrame,
    GeometryScaleStatus,
    GeometryScaleUnit,
    Object3DObservation,
    Point3DObservation,
    ReconstructionFrame,
    VisibilityStatus,
)
from .contracts import (
    MetricPointType,
    MetricSurfacePoint,
    ObjectSurfacePointCloud,
    Visibility,
)


def canonicalize_z_depth(
    depth_map: np.ndarray,
    K: np.ndarray,
    *,
    depth_definition: str,
) -> np.ndarray:
    """Return optical-axis z-depth without changing metric units."""

    depth = np.asarray(depth_map, dtype=float)
    matrix = validate_intrinsics(K)
    if depth_definition in {"z_depth", "camera_optical_axis_z"}:
        return depth.copy()
    if depth_definition == "ray_distance":
        rows, columns = np.indices(depth.shape, dtype=float)
        return ray_distance_to_z_depth(depth, rows, columns, matrix)
    raise ValueError(f"Unsupported depth_definition: {depth_definition!r}.")


def backproject_metric_arrays(
    pixels_uv: np.ndarray,
    z_depth_m: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Vectorized metric back-projection into OpenCV camera coordinates."""

    pixels = np.asarray(pixels_uv, dtype=float)
    depth = np.asarray(z_depth_m, dtype=float).reshape(-1)
    matrix = validate_intrinsics(K)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or len(pixels) != len(depth):
        raise ValueError("pixels_uv must be [N,2] with one depth per pixel.")
    fx, fy, cx, cy = (
        float(matrix[0, 0]),
        float(matrix[1, 1]),
        float(matrix[0, 2]),
        float(matrix[1, 2]),
    )
    x = (pixels[:, 0] - cx) * depth / fx
    y = (pixels[:, 1] - cy) * depth / fy
    return np.column_stack((x, y, depth))


def project_metric_arrays(points_xyz_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project camera-frame metric points into integer-centre pixel coordinates."""

    points = np.asarray(points_xyz_m, dtype=float)
    matrix = validate_intrinsics(K)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz_m must have shape [N,3].")
    if np.any(points[:, 2] <= 0.0):
        raise ValueError("Projection requires positive z-depth.")
    homogeneous = (matrix @ points.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _sample_map(
    value_map: Optional[np.ndarray], rows: np.ndarray, columns: np.ndarray, default: float
) -> np.ndarray:
    if value_map is None:
        return np.full(len(rows), default, dtype=float)
    values = np.asarray(value_map, dtype=float)[rows, columns]
    return values


def _metric_points(
    *,
    frame_id: str,
    object_id: Optional[str],
    track_id: Optional[str],
    point_type: MetricPointType,
    rows: np.ndarray,
    columns: np.ndarray,
    z_values: np.ndarray,
    K: np.ndarray,
    confidence_values: np.ndarray,
    uncertainty_values: np.ndarray,
    provider_name: str,
    intrinsics_source: str,
    pose_source: str,
    provenance: dict[str, object],
) -> tuple[MetricSurfacePoint, ...]:
    pixels = np.column_stack((columns, rows)).astype(float)
    xyz = backproject_metric_arrays(pixels, z_values, K)
    output = []
    for index, ((u, v), (x, y, z)) in enumerate(zip(pixels, xyz, strict=True)):
        depth_confidence = float(np.clip(confidence_values[index], 0.0, 1.0))
        output.append(
            MetricSurfacePoint(
                point_id=f"{frame_id}:{object_id or 'scene'}:{point_type.value}:{index:06d}",
                point_type=point_type,
                frame_id=frame_id,
                object_id=object_id,
                track_id=track_id,
                u=float(u),
                v=float(v),
                x_m=float(x),
                y_m=float(y),
                z_m=float(z),
                depth_confidence=depth_confidence,
                confidence=depth_confidence,
                uncertainty=float(uncertainty_values[index]),
                uncertainty_definition="provider_native_uncertainty_not_meter_calibrated",
                visibility=Visibility.VISIBLE,
                valid=True,
                failure_reason="",
                coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC,
                depth_unit="meter",
                depth_definition="z_depth",
                intrinsics_source=intrinsics_source,
                pose_source=pose_source,
                provider_name=provider_name,
                provenance=provenance,
            )
        )
    return tuple(output)


def build_scene_surface_points(
    *,
    frame_id: str,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    K: np.ndarray,
    confidence_map: Optional[np.ndarray],
    uncertainty_map: Optional[np.ndarray],
    provider_name: str,
    intrinsics_source: str,
    stride: int = 16,
) -> tuple[MetricSurfacePoint, ...]:
    """Build a sparse metric visible-surface 2.5D point set for one frame."""

    if stride < 1:
        raise ValueError("stride must be positive.")
    depth = np.asarray(depth_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if depth.shape != valid.shape:
        raise ValueError("depth_map and valid_mask shapes must match.")
    sampled = np.zeros(depth.shape, dtype=bool)
    sampled[::stride, ::stride] = True
    selected = sampled & valid & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(selected)
    confidence = _sample_map(confidence_map, rows, columns, 1.0)
    uncertainty = _sample_map(uncertainty_map, rows, columns, float("nan"))
    return _metric_points(
        frame_id=frame_id,
        object_id=None,
        track_id=None,
        point_type=MetricPointType.SCENE_SURFACE_POINT,
        rows=rows,
        columns=columns,
        z_values=depth[rows, columns],
        K=K,
        confidence_values=confidence,
        uncertainty_values=uncertainty,
        provider_name=provider_name,
        intrinsics_source=intrinsics_source,
        pose_source="unavailable_single_frame",
        provenance={
            "surface_semantics": "visible_surface_2p5d_not_complete_scene",
            "sampling_stride": stride,
            "metric_scale_status": "model_predicted",
            "sensor_ground_truth": False,
        },
    )


def build_object_surface_pointcloud(
    *,
    frame_id: str,
    object_id: str,
    track_id: Optional[str],
    class_name: str,
    mask: np.ndarray,
    mask_quality: float,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    K: np.ndarray,
    confidence_map: Optional[np.ndarray],
    uncertainty_map: Optional[np.ndarray],
    provider_name: str,
    intrinsics_source: str,
    quantile_low: float = 0.05,
    quantile_high: float = 0.95,
    max_recorded_points: int = 2048,
) -> ObjectSurfacePointCloud:
    """Back-project a formal visible mask and estimate robust quantile extents."""

    binary = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_map, dtype=float)
    valid_depth = np.asarray(valid_mask, dtype=bool)
    if binary.shape != depth.shape or depth.shape != valid_depth.shape:
        raise ValueError("Mask, depth, and validity arrays must be aligned.")
    if not 0.0 <= quantile_low < quantile_high <= 1.0:
        raise ValueError("Invalid robust quantile interval.")
    candidate_count = int(np.count_nonzero(binary))
    selected = binary & valid_depth & np.isfinite(depth) & (depth > 0.0)
    rows, columns = np.nonzero(selected)
    valid_ratio = len(rows) / max(candidate_count, 1)
    if not len(rows):
        nan3 = (float("nan"),) * 3
        return ObjectSurfacePointCloud(
            object_id,
            track_id,
            class_name,
            frame_id,
            (),
            0,
            valid_ratio,
            float("nan"),
            float("nan"),
            float("nan"),
            nan3,
            (nan3, nan3, nan3),
            float("nan"),
            float(mask_quality),
            0.0,
            quantile_low,
            quantile_high,
            False,
            "no_valid_metric_depth_in_formal_mask",
        )
    pixels = np.column_stack((columns, rows))
    xyz = backproject_metric_arrays(pixels, depth[rows, columns], K)
    low = np.quantile(xyz, quantile_low, axis=0)
    high = np.quantile(xyz, quantile_high, axis=0)
    inlier = np.all((xyz >= low) & (xyz <= high), axis=1)
    robust = xyz[inlier] if np.count_nonzero(inlier) >= 3 else xyz
    centroid = np.median(robust, axis=0)
    covariance = (
        np.cov(robust, rowvar=False)
        if len(robust) >= 2
        else np.full((3, 3), float("nan"), dtype=float)
    )
    extents = high - low
    conf_all = _sample_map(confidence_map, rows, columns, 1.0)
    unc_all = _sample_map(uncertainty_map, rows, columns, float("nan"))
    finite_uncertainty = unc_all[np.isfinite(unc_all)]
    point_uncertainty = (
        float(np.median(finite_uncertainty))
        if finite_uncertainty.size
        else float("nan")
    )
    if len(rows) > max_recorded_points:
        indices = np.linspace(0, len(rows) - 1, max_recorded_points, dtype=int)
        record_rows, record_columns = rows[indices], columns[indices]
        record_conf, record_unc = conf_all[indices], unc_all[indices]
    else:
        record_rows, record_columns = rows, columns
        record_conf, record_unc = conf_all, unc_all
    points = _metric_points(
        frame_id=frame_id,
        object_id=object_id,
        track_id=track_id,
        point_type=MetricPointType.DENSE_OBJECT_SURFACE_POINT,
        rows=record_rows,
        columns=record_columns,
        z_values=depth[record_rows, record_columns],
        K=K,
        confidence_values=record_conf,
        uncertainty_values=record_unc,
        provider_name=provider_name,
        intrinsics_source=intrinsics_source,
        pose_source="unavailable_single_frame",
        provenance={
            "mask_type": "formal_visible_instance_mask",
            "is_amodal_mask": False,
            "metric_scale_status": "model_predicted",
            "sensor_ground_truth": False,
        },
    )
    finite_conf = conf_all[np.isfinite(conf_all)]
    depth_quality = float(np.median(finite_conf)) if finite_conf.size else 0.0
    return ObjectSurfacePointCloud(
        object_id=object_id,
        track_id=track_id,
        class_name=class_name,
        frame_id=frame_id,
        points=points,
        point_count=int(len(rows)),
        valid_point_ratio=float(valid_ratio),
        x_extent_m=float(extents[0]),
        y_extent_m=float(extents[1]),
        z_extent_m=float(extents[2]),
        robust_centroid_m=tuple(float(value) for value in centroid),
        robust_covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        point_uncertainty=point_uncertainty,
        mask_quality=float(mask_quality),
        depth_quality=depth_quality,
        quantile_low=quantile_low,
        quantile_high=quantile_high,
        valid=True,
        metadata={
            "extent_method": "per_axis_robust_quantiles",
            "candidate_mask_pixels": candidate_count,
            "recorded_point_count": len(points),
            "complete_scene_claim": False,
        },
    )


def to_shared_object_observation(
    pointcloud: ObjectSurfacePointCloud,
    *,
    video_id: str,
    frame_index: int,
) -> Object3DObservation:
    """Adapt an M2 object pointcloud into the existing shared 3D contract."""

    if not pointcloud.valid:
        return Object3DObservation.missing(
            video_id=video_id,
            frame_index=frame_index,
            track_id=pointcloud.track_id,
            semantic_label=pointcloud.class_name,
            source_object_2d_id=pointcloud.object_id,
            reason=pointcloud.failure_reason,
        )
    center = Point3DObservation(
        point_id=f"{pointcloud.frame_id}:{pointcloud.object_id}:robust_centroid",
        x=pointcloud.robust_centroid_m[0],
        y=pointcloud.robust_centroid_m[1],
        z=pointcloud.robust_centroid_m[2],
        coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC.value,
        scale_status=GeometryScaleStatus.METRIC_3D,
        confidence=min(pointcloud.mask_quality, pointcloud.depth_quality),
        valid=True,
        metadata={"centroid_method": "coordinate_wise_median"},
    )
    recorded = tuple(
        Point3DObservation(
            point_id=point.point_id,
            x=point.x_m,
            y=point.y_m,
            z=point.z_m,
            coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC.value,
            scale_status=GeometryScaleStatus.METRIC_3D,
            confidence=point.confidence,
            valid=True,
            source_point_2d_id=point.point_id,
            metadata={"point_type": point.point_type.value, **dict(point.provenance)},
        )
        for point in pointcloud.points
        if point.valid
    )
    extent_diagonal = math.sqrt(
        pointcloud.x_extent_m**2
        + pointcloud.y_extent_m**2
        + pointcloud.z_extent_m**2
    )
    return Object3DObservation(
        video_id=video_id,
        frame_index=frame_index,
        track_id=pointcloud.track_id,
        semantic_label=pointcloud.class_name,
        canonical_label=pointcloud.class_name,
        center_3d=center,
        boundary_points_3d=(),
        keypoints_3d=(),
        structure_points_3d=recorded,
        observed_scale_3d=extent_diagonal,
        normalized_structure_points=(),
        scale_status=GeometryScaleStatus.METRIC_3D,
        visibility=VisibilityStatus.VISIBLE,
        reconstruction_quality=min(pointcloud.mask_quality, pointcloud.depth_quality),
        valid=True,
        missing_reason="",
        source_object_2d_id=pointcloud.object_id,
        center_3d_camera=center,
        scale_method="robust_quantile_extent_diagonal_diagnostic",
        scale_unit=GeometryScaleUnit.METER,
        reconstruction_frame=ReconstructionFrame.CAMERA,
        scale_descriptors={
            "x_extent_m": pointcloud.x_extent_m,
            "y_extent_m": pointcloud.y_extent_m,
            "z_extent_m": pointcloud.z_extent_m,
            "warning": "diagnostic extent diagonal is not a semantic characteristic dimension",
        },
        metadata={
            "coordinate_frame": CoordinateFrame.CAMERA_FRAME_METRIC.value,
            "metric_scale_status": "model_predicted",
            "sensor_ground_truth": False,
            "world_frame_available": False,
        },
    )
