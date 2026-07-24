"""Visible-mask contour sampling and metric boundary reconstruction."""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from ..shared_3d_observation import CoordinateFrame
from .contracts import (
    BoundaryMetricPoint,
    MetricPointType,
    MetricSurfacePoint,
    Visibility,
)
from .reconstruction import backproject_metric_arrays


def _uniform_closed_polyline_samples(
    contour_xy: np.ndarray,
    *,
    sample_count: int,
    min_spacing_px: float,
) -> np.ndarray:
    """Sample a closed contour at approximately uniform arc-length intervals."""

    points = np.asarray(contour_xy, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        return np.empty((0, 2), dtype=float)
    closed = np.vstack((points, points[0]))
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    keep = lengths > 1e-9
    if not np.any(keep):
        return np.empty((0, 2), dtype=float)
    closed = np.vstack((closed[:-1][keep], closed[:-1][keep][0]))
    segments = np.diff(closed, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    perimeter = float(cumulative[-1])
    target_count = min(sample_count, max(1, int(perimeter / max(min_spacing_px, 1e-6))))
    target = np.linspace(0.0, perimeter, target_count, endpoint=False)
    output = []
    for distance in target:
        segment = min(int(np.searchsorted(cumulative, distance, side="right") - 1), len(lengths) - 1)
        fraction = (distance - cumulative[segment]) / max(lengths[segment], 1e-12)
        output.append(closed[segment] + fraction * segments[segment])
    return np.asarray(output, dtype=float)


def sample_mask_boundary(
    mask: np.ndarray,
    *,
    sample_count: int = 32,
    min_spacing_px: float = 4.0,
    approximation_epsilon_ratio: float = 0.002,
) -> np.ndarray:
    """Extract, simplify, and uniformly sample the largest external contour."""

    binary = np.asarray(mask, dtype=bool)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return np.empty((0, 2), dtype=float)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(
        contour, approximation_epsilon_ratio * max(perimeter, 1.0), True
    )
    return _uniform_closed_polyline_samples(
        simplified[:, 0, :],
        sample_count=sample_count,
        min_spacing_px=min_spacing_px,
    )


def _local_depth_sides(
    depth: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray,
    row: int,
    column: int,
    radius: int,
) -> tuple[float, float]:
    y1, y2 = max(0, row - radius), min(depth.shape[0], row + radius + 1)
    x1, x2 = max(0, column - radius), min(depth.shape[1], column + radius + 1)
    yy, xx = np.ogrid[y1:y2, x1:x2]
    disk = (yy - row) ** 2 + (xx - column) ** 2 <= radius**2
    local_depth = depth[y1:y2, x1:x2]
    local_valid = valid[y1:y2, x1:x2] & np.isfinite(local_depth) & (local_depth > 0.0)
    local_mask = mask[y1:y2, x1:x2]
    foreground = local_depth[disk & local_valid & local_mask]
    background = local_depth[disk & local_valid & ~local_mask]
    foreground_median = (
        float(np.median(foreground)) if foreground.size else float("nan")
    )
    background_median = (
        float(np.median(background)) if background.size else float("nan")
    )
    return foreground_median, background_median


def reconstruct_boundary_points(
    *,
    frame_id: str,
    object_id: str,
    track_id: Optional[str],
    mask: np.ndarray,
    mask_quality: float,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    K: np.ndarray,
    confidence_map: Optional[np.ndarray],
    uncertainty_map: Optional[np.ndarray],
    provider_name: str,
    intrinsics_source: str,
    sample_count: int = 32,
    min_spacing_px: float = 4.0,
    side_radius_px: int = 3,
) -> tuple[BoundaryMetricPoint, ...]:
    """Reconstruct visible boundary samples with foreground/background depths."""

    binary = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if binary.shape != depth.shape or valid.shape != depth.shape:
        raise ValueError("Mask, depth, and valid mask must be aligned.")
    sampled = sample_mask_boundary(
        binary, sample_count=sample_count, min_spacing_px=min_spacing_px
    )
    output = []
    for order, (u, v) in enumerate(sampled):
        row = int(np.clip(round(float(v)), 0, depth.shape[0] - 1))
        column = int(np.clip(round(float(u)), 0, depth.shape[1] - 1))
        foreground, background = _local_depth_sides(
            depth, valid, binary, row, column, side_radius_px
        )
        point_id = f"{frame_id}:{object_id}:boundary:{order:04d}"
        if not math.isfinite(foreground) or foreground <= 0.0:
            point = MetricSurfacePoint.missing(
                point_id=point_id,
                point_type=MetricPointType.BOUNDARY_POINT,
                frame_id=frame_id,
                object_id=object_id,
                track_id=track_id,
                u=float(u),
                v=float(v),
                reason="missing_foreground_boundary_depth",
                provider_name=provider_name,
                provenance={"visible_mask_boundary": True},
            )
            output.append(
                BoundaryMetricPoint(
                    point, foreground, background, float("nan"), order
                )
            )
            continue
        xyz = backproject_metric_arrays(
            np.asarray([[u, v]], dtype=float),
            np.asarray([foreground], dtype=float),
            K,
        )[0]
        depth_confidence = (
            1.0
            if confidence_map is None
            else float(np.clip(confidence_map[row, column], 0.0, 1.0))
        )
        uncertainty = (
            float("nan")
            if uncertainty_map is None
            else float(uncertainty_map[row, column])
        )
        jump = (
            abs(background - foreground)
            if math.isfinite(background)
            else float("nan")
        )
        background_support = 1.0 if math.isfinite(background) else 0.6
        confidence = float(
            np.clip(mask_quality * depth_confidence * background_support, 0.0, 1.0)
        )
        point = MetricSurfacePoint(
            point_id=point_id,
            point_type=MetricPointType.BOUNDARY_POINT,
            frame_id=frame_id,
            object_id=object_id,
            track_id=track_id,
            u=float(u),
            v=float(v),
            x_m=float(xyz[0]),
            y_m=float(xyz[1]),
            z_m=float(xyz[2]),
            depth_confidence=depth_confidence,
            confidence=confidence,
            uncertainty=uncertainty,
            uncertainty_definition="provider_native_uncertainty_not_meter_calibrated",
            visibility=Visibility.VISIBLE,
            valid=True,
            failure_reason="",
            coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC,
            depth_unit="meter",
            depth_definition="z_depth",
            intrinsics_source=intrinsics_source,
            pose_source="unavailable_single_frame",
            provider_name=provider_name,
            provenance={
                "mask_type": "formal_visible_instance_mask",
                "is_amodal_mask": False,
                "foreground_background_window_radius_px": side_radius_px,
                "background_depth_available": math.isfinite(background),
                "sensor_ground_truth": False,
            },
        )
        output.append(BoundaryMetricPoint(point, foreground, background, jump, order))
    return tuple(output)
