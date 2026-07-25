"""One canonical geometry facade over the existing verified implementations."""

from __future__ import annotations

import numpy as np

from data.schemas import FrameObservation, ObjectObservation
from semantic3d.geometry.backprojection import backproject_points as _legacy_backproject_points
from semantic3d.geometry.projection import project_points as _legacy_project_points
from semantic3d.geometry.transforms import transform_points as _legacy_transform_points
from semantic3d.metric_scene3d.reconstruction import build_object_surface_pointcloud
from semantic3d.reconstruction.object_3d_reconstructor import Object3DReconstructor
from semantic3d.reconstruction.shared_3d_builder import Shared3DFrameBuilder
from semantic3d.shared_3d_observation import GeometryScaleStatus, Point3DObservation


def backproject_points(pixels_xy: np.ndarray, depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    pixels = np.asarray(pixels_xy, dtype=float)
    depths = np.asarray(depth_m, dtype=float).reshape(-1)
    points = _legacy_backproject_points(
        pixels,
        depths,
        K,
        scale_status=GeometryScaleStatus.METRIC_3D,
        coordinate_frame="camera_frame_metric",
    )
    return np.asarray([[point.x, point.y, point.z] for point in points if point.valid], dtype=float)


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = tuple(
        Point3DObservation(
            point_id=f"p{index}",
            x=float(x),
            y=float(y),
            z=float(z),
            coordinate_frame="camera_frame_metric",
            scale_status=GeometryScaleStatus.METRIC_3D,
            confidence=1.0,
            valid=True,
        )
        for index, (x, y, z) in enumerate(np.asarray(points_xyz, dtype=float))
    )
    transformed = _legacy_transform_points(
        points,
        transform,
        source_frame="camera_frame_metric",
        target_frame="camera_frame_metric",
        transform_name="T_target_from_source",
    )
    return np.asarray([[point.x, point.y, point.z] for point in transformed if point.valid])


def project_points(points_xyz: np.ndarray, K: np.ndarray) -> np.ndarray:
    points = tuple(
        Point3DObservation(
            point_id=f"p{index}",
            x=float(x),
            y=float(y),
            z=float(z),
            coordinate_frame="camera_frame_metric",
            scale_status=GeometryScaleStatus.METRIC_3D,
            confidence=1.0,
            valid=True,
        )
        for index, (x, y, z) in enumerate(np.asarray(points_xyz, dtype=float))
    )
    projected = _legacy_project_points(points, K)
    return np.asarray([[point.x, point.y] for point in projected if point.valid])


def predict_target_positions(points_xyz: np.ndarray, relative_pose: np.ndarray, target_K: np.ndarray) -> np.ndarray:
    return project_points(transform_points(points_xyz, relative_pose), target_K)


def build_metric_object_surface(frame: FrameObservation, obj: ObjectObservation):
    if obj.instance_mask is None or frame.metric_depth is None or frame.intrinsics is None:
        return None
    cloud = build_object_surface_pointcloud(
        frame_id=f"{frame.video_id}:{frame.frame_index}",
        object_id=obj.object_id,
        track_id=obj.track_id,
        class_name=obj.category,
        mask=obj.instance_mask,
        mask_quality=obj.mask_quality,
        depth_map=frame.metric_depth,
        valid_mask=(
            np.isfinite(frame.metric_depth) & (frame.metric_depth > 0)
            if frame.depth_valid_mask is None
            else frame.depth_valid_mask
        ),
        K=frame.intrinsics,
        confidence_map=frame.depth_confidence,
        uncertainty_map=None,
        provider_name="paper_core_metric_depth",
        intrinsics_source="model_predicted_or_calibrated",
    )
    if cloud.valid:
        obj.metric_surface_xyz = np.asarray(
            [[point.x_m, point.y_m, point.z_m] for point in cloud.points if point.valid],
            dtype=float,
        )
    return cloud


__all__ = [
    "Object3DReconstructor",
    "Shared3DFrameBuilder",
    "backproject_points",
    "build_metric_object_surface",
    "predict_target_positions",
    "project_points",
    "transform_points",
]
