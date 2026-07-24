"""Visibility-aware D2 residuals for real short-baseline geometry smoke."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from ..geometry.camera import validate_intrinsics
from .contracts import (
    D2ResidualObservation,
    D2VisibilityStatus,
    PairwisePoseObservation,
)


def _invalid(
    *,
    evidence_id: str,
    evidence_type: str,
    video_id: str,
    clip_id: str,
    pose: PairwisePoseObservation,
    visibility: D2VisibilityStatus,
    reason: str,
    object_id: str,
    track_id: str,
    point_id: str,
    point_confidence: float,
    metadata: dict[str, object] | None = None,
) -> D2ResidualObservation:
    return D2ResidualObservation(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        video_id=video_id,
        clip_id=clip_id,
        frame_t=pose.frame_t,
        frame_t1=pose.frame_t1,
        object_id=object_id,
        track_id=track_id,
        point_id=point_id,
        point_reprojection_residual=float("nan"),
        boundary_reprojection_residual=float("nan"),
        depth_reprojection_residual=float("nan"),
        object_reprojection_residual=float("nan"),
        visibility_status=visibility,
        pose_confidence=pose.confidence,
        point_confidence=float(np.clip(point_confidence, 0.0, 1.0)),
        valid=False,
        failure_reason=reason,
        provider_status=pose.provider_status,
        metadata=dict(metadata or {}),
    )


def _sample_depth(
    depth: Optional[np.ndarray],
    valid_mask: Optional[np.ndarray],
    u: float,
    v: float,
) -> float:
    if depth is None or valid_mask is None:
        return float("nan")
    row = int(round(v))
    column = int(round(u))
    if (
        row < 0
        or column < 0
        or row >= depth.shape[0]
        or column >= depth.shape[1]
        or not bool(valid_mask[row, column])
    ):
        return float("nan")
    value = float(depth[row, column])
    return value if math.isfinite(value) and value > 0.0 else float("nan")


def compute_d2_projection_residual(
    *,
    evidence_id: str,
    evidence_type: str,
    video_id: str,
    clip_id: str,
    pose: PairwisePoseObservation,
    source_point_camera_m: Sequence[float],
    target_observed_uv: Optional[Sequence[float]],
    K_target: np.ndarray,
    image_width: int,
    image_height: int,
    target_depth_m: Optional[np.ndarray] = None,
    target_depth_valid_mask: Optional[np.ndarray] = None,
    object_id: str = "",
    track_id: str = "",
    point_id: str = "",
    point_confidence: float = 1.0,
    depth_occlusion_margin_m: float = 0.10,
    depth_conflict_relative_threshold: float = 0.10,
) -> D2ResidualObservation:
    """Compare an SE(3)-projected point with an independent target observation.

    Pixel residuals are normalised by image diagonal. A target depth conflict
    is classified before residual creation, so occluded or geometrically
    inconsistent points never become high anomaly values by construction.
    """

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    point = np.asarray(source_point_camera_m, dtype=float).reshape(-1)
    common = {
        "coordinate_convention": "opencv_x_right_y_down_z_forward",
        "pose_convention": pose.pose_convention,
        "residual_normalization": "pixel_error_divided_by_image_diagonal",
        "authenticity_threshold_applied": False,
        "sensor_ground_truth": False,
    }
    if not pose.valid or pose.T_target_from_source is None:
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.INVALID_INPUT,
            reason=f"pose_not_usable:{pose.failure_reason or pose.provider_status.value}",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata=common,
        )
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 0.0:
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.INVALID_INPUT,
            reason="invalid_source_3d_point",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata=common,
        )
    target_xyz = (
        pose.T_target_from_source
        @ np.asarray([point[0], point[1], point[2], 1.0])
    )[:3]
    if not np.isfinite(target_xyz).all() or target_xyz[2] <= 1e-9:
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.BEHIND_CAMERA,
            reason="projected_point_behind_camera",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata={**common, "predicted_xyz_target_m": target_xyz.tolist()},
        )
    K = validate_intrinsics(K_target)
    projected_h = K @ target_xyz
    predicted_uv = projected_h[:2] / projected_h[2]
    u, v = float(predicted_uv[0]), float(predicted_uv[1])
    common.update(
        {
            "predicted_uv": [u, v],
            "predicted_xyz_target_m": target_xyz.tolist(),
        }
    )
    if not (0.0 <= u < image_width and 0.0 <= v < image_height):
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.OUT_OF_FRAME,
            reason="projected_point_out_of_frame",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata=common,
        )
    if target_observed_uv is None:
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.NO_CORRESPONDENCE,
            reason="independent_target_correspondence_missing",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata=common,
        )
    observed_uv = np.asarray(target_observed_uv, dtype=float).reshape(-1)
    if observed_uv.shape != (2,) or not np.isfinite(observed_uv).all():
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.NO_CORRESPONDENCE,
            reason="invalid_target_correspondence",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata=common,
        )

    observed_depth = _sample_depth(
        target_depth_m, target_depth_valid_mask, u, v
    )
    predicted_depth = float(target_xyz[2])
    if target_depth_m is not None and not math.isfinite(observed_depth):
        return _invalid(
            evidence_id=evidence_id,
            evidence_type=evidence_type,
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.NO_CORRESPONDENCE,
            reason="target_depth_unavailable",
            object_id=object_id,
            track_id=track_id,
            point_id=point_id,
            point_confidence=point_confidence,
            metadata=common,
        )
    if math.isfinite(observed_depth):
        depth_delta = predicted_depth - observed_depth
        threshold = max(
            float(depth_occlusion_margin_m),
            float(depth_conflict_relative_threshold) * observed_depth,
        )
        common.update(
            {
                "target_observed_depth_m": observed_depth,
                "predicted_depth_m": predicted_depth,
                "depth_delta_m": depth_delta,
                "depth_consistency_threshold_m": threshold,
            }
        )
        if depth_delta > threshold:
            return _invalid(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                video_id=video_id,
                clip_id=clip_id,
                pose=pose,
                visibility=D2VisibilityStatus.OCCLUDED,
                reason="target_surface_occludes_projected_point",
                object_id=object_id,
                track_id=track_id,
                point_id=point_id,
                point_confidence=point_confidence,
                metadata=common,
            )
        if depth_delta < -threshold:
            return _invalid(
                evidence_id=evidence_id,
                evidence_type=evidence_type,
                video_id=video_id,
                clip_id=clip_id,
                pose=pose,
                visibility=D2VisibilityStatus.DEPTH_CONFLICT,
                reason="projected_point_depth_conflicts_with_target_surface",
                object_id=object_id,
                track_id=track_id,
                point_id=point_id,
                point_confidence=point_confidence,
                metadata=common,
            )

    pixel_error = float(np.linalg.norm(observed_uv - predicted_uv))
    normalized = pixel_error / math.hypot(image_width, image_height)
    depth_residual = (
        abs(predicted_depth - observed_depth) / max(observed_depth, 1e-9)
        if math.isfinite(observed_depth)
        else float("nan")
    )
    point_value = normalized if evidence_type == "point" else float("nan")
    boundary_value = normalized if evidence_type == "boundary" else float("nan")
    common.update(
        {
            "observed_uv": observed_uv.tolist(),
            "pixel_error": pixel_error,
            "independent_target_observation": True,
        }
    )
    return D2ResidualObservation(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        video_id=video_id,
        clip_id=clip_id,
        frame_t=pose.frame_t,
        frame_t1=pose.frame_t1,
        object_id=object_id,
        track_id=track_id,
        point_id=point_id,
        point_reprojection_residual=point_value,
        boundary_reprojection_residual=boundary_value,
        depth_reprojection_residual=depth_residual,
        object_reprojection_residual=float("nan"),
        visibility_status=D2VisibilityStatus.VISIBLE,
        pose_confidence=pose.confidence,
        point_confidence=float(np.clip(point_confidence, 0.0, 1.0)),
        valid=True,
        failure_reason="",
        provider_status=pose.provider_status,
        metadata=common,
    )


def aggregate_object_d2_residual(
    residuals: Iterable[D2ResidualObservation],
    *,
    evidence_id: str,
    video_id: str,
    clip_id: str,
    pose: PairwisePoseObservation,
    object_id: str,
    track_id: str,
) -> D2ResidualObservation:
    """Aggregate visible point/boundary evidence for one object with median."""

    rows = tuple(residuals)
    finite_values: list[float] = []
    qualities: list[float] = []
    source_ids: list[str] = []
    for row in rows:
        if not row.valid:
            continue
        for value in (
            row.point_reprojection_residual,
            row.boundary_reprojection_residual,
        ):
            if math.isfinite(value):
                finite_values.append(value)
                qualities.append(row.point_confidence)
                source_ids.append(row.evidence_id)
                break
    if not finite_values:
        return _invalid(
            evidence_id=evidence_id,
            evidence_type="object",
            video_id=video_id,
            clip_id=clip_id,
            pose=pose,
            visibility=D2VisibilityStatus.NO_CORRESPONDENCE,
            reason="no_valid_visible_object_d2_support",
            object_id=object_id,
            track_id=track_id,
            point_id="",
            point_confidence=0.0,
            metadata={
                "candidate_support_count": len(rows),
                "valid_support_count": 0,
            },
        )
    value = float(np.median(finite_values))
    quality = float(min(pose.confidence, np.median(qualities)))
    return D2ResidualObservation(
        evidence_id=evidence_id,
        evidence_type="object",
        video_id=video_id,
        clip_id=clip_id,
        frame_t=pose.frame_t,
        frame_t1=pose.frame_t1,
        object_id=object_id,
        track_id=track_id,
        point_id="",
        point_reprojection_residual=float("nan"),
        boundary_reprojection_residual=float("nan"),
        depth_reprojection_residual=float("nan"),
        object_reprojection_residual=value,
        visibility_status=D2VisibilityStatus.VISIBLE,
        pose_confidence=pose.confidence,
        point_confidence=quality,
        valid=True,
        failure_reason="",
        provider_status=pose.provider_status,
        metadata={
            "aggregation": "median",
            "source_ids": source_ids,
            "candidate_support_count": len(rows),
            "valid_support_count": len(finite_values),
            "authenticity_threshold_applied": False,
        },
    )
