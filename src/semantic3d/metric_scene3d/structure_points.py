"""Select auditable geometric track-point candidates inside formal masks."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..occlusion.mask_observation import InstanceMaskObservation
from ..occlusion.mask_structure_points import (
    eroded_mask_interior,
    select_formal_mask_internal_points,
)
from ..shared_3d_observation import CoordinateFrame
from .contracts import MetricPointType, MetricSurfacePoint, Visibility
from .reconstruction import backproject_metric_arrays


def select_geometric_track_points(
    *,
    image: np.ndarray,
    mask_observation: InstanceMaskObservation,
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    K: np.ndarray,
    confidence_map: Optional[np.ndarray],
    uncertainty_map: Optional[np.ndarray],
    frame_id: str,
    object_id: str,
    provider_name: str,
    intrinsics_source: str,
    max_points: int = 24,
    erosion_pixels: int = 4,
    min_distance_px: float = 8.0,
    local_radius_px: int = 2,
    max_local_depth_mad_ratio: float = 0.08,
) -> tuple[MetricSurfacePoint, ...]:
    """Choose texture-supported, depth-smooth, spatially separated mask points.

    These are only single-frame geometric tracking candidates. They are never
    labelled semantic keypoints and are not claimed to be temporally stable
    until a later tracker verifies them.
    """

    binary = mask_observation.selected_mask
    if binary is None or image.shape[:2] != binary.shape:
        return ()
    depth = np.asarray(depth_map, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool)
    if depth.shape != binary.shape or valid.shape != depth.shape:
        raise ValueError("Image, formal mask, depth, and valid mask must be aligned.")
    candidates = select_formal_mask_internal_points(
        image,
        mask_observation,
        max_points=max_points * 4,
        erosion_pixels=erosion_pixels,
        min_distance=max(min_distance_px / 2.0, 1.0),
    )
    if not len(candidates):
        return ()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    texture = cv2.cornerMinEigenVal(gray.astype(np.float32), blockSize=3)
    finite_texture = texture[np.isfinite(texture)]
    texture_scale = float(np.quantile(finite_texture, 0.95)) if finite_texture.size else 1.0
    safe_interior = eroded_mask_interior(binary, erosion_pixels)
    scored = []
    for u, v in candidates:
        row = int(np.clip(round(float(v)), 0, depth.shape[0] - 1))
        column = int(np.clip(round(float(u)), 0, depth.shape[1] - 1))
        if not safe_interior[row, column] or not valid[row, column]:
            continue
        z = float(depth[row, column])
        if not np.isfinite(z) or z <= 0.0:
            continue
        y1, y2 = max(0, row - local_radius_px), min(depth.shape[0], row + local_radius_px + 1)
        x1, x2 = max(0, column - local_radius_px), min(depth.shape[1], column + local_radius_px + 1)
        local = depth[y1:y2, x1:x2]
        local_valid = (
            valid[y1:y2, x1:x2]
            & safe_interior[y1:y2, x1:x2]
            & np.isfinite(local)
            & (local > 0.0)
        )
        values = local[local_valid]
        if values.size < 3:
            continue
        median = float(np.median(values))
        mad_ratio = float(np.median(np.abs(values - median)) / max(median, 1e-12))
        if mad_ratio > max_local_depth_mad_ratio:
            continue
        texture_score = float(np.clip(texture[row, column] / max(texture_scale, 1e-12), 0.0, 1.0))
        smoothness_score = float(np.clip(1.0 - mad_ratio / max_local_depth_mad_ratio, 0.0, 1.0))
        scored.append((texture_score * smoothness_score, float(u), float(v), mad_ratio))
    scored.sort(key=lambda item: (-item[0], item[2], item[1]))
    selected = []
    for item in scored:
        _, u, v, _ = item
        if all(np.hypot(u - old[1], v - old[2]) >= min_distance_px for old in selected):
            selected.append(item)
        if len(selected) >= max_points:
            break
    output = []
    for index, (score, u, v, mad_ratio) in enumerate(selected):
        row, column = int(round(v)), int(round(u))
        z = float(depth[row, column])
        xyz = backproject_metric_arrays(
            np.asarray([[u, v]], dtype=float), np.asarray([z]), K
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
        confidence = float(
            np.clip(mask_observation.confidence * depth_confidence * score, 0.0, 1.0)
        )
        output.append(
            MetricSurfacePoint(
                point_id=f"{frame_id}:{object_id}:geometric_track:{index:04d}",
                point_type=MetricPointType.GEOMETRIC_TRACK_POINT,
                frame_id=frame_id,
                object_id=object_id,
                track_id=mask_observation.object_track_id,
                u=u,
                v=v,
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
                    "point_semantics": "geometric_track_candidate_not_semantic_keypoint",
                    "formal_mask_internal": True,
                    "distance_from_boundary_px": erosion_pixels,
                    "texture_depth_score": score,
                    "local_depth_mad_ratio": mad_ratio,
                    "trackability_verified_across_frames": False,
                    "sensor_ground_truth": False,
                },
            )
        )
    return tuple(output)
