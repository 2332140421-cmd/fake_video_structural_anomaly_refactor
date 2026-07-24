"""Foreground-aware short-baseline pose estimation for the M4 D2 smoke.

This module estimates a directed transform between adjacent camera frames. It
uses metric model-predicted source depth for translation scale; that depth is
not sensor ground truth. Identity is emitted only after an explicit
multi-evidence static verification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import cv2
import numpy as np

from ..geometry.camera import validate_intrinsics
from ..sequence_geometry.pose_estimation import track_background_correspondences
from .contracts import (
    PairwisePoseObservation,
    PoseProviderStatus,
    StaticVerificationObservation,
)


@dataclass(frozen=True)
class ShortBaselinePoseThresholds:
    """Quality and static-verification thresholds for short frame pairs."""

    minimum_correspondences: int = 20
    minimum_inliers: int = 12
    minimum_inlier_ratio: float = 0.45
    minimum_spatial_coverage: float = 0.08
    maximum_pnp_reprojection_error_px: float = 3.0
    minimum_valid_confidence: float = 0.30
    static_max_median_flow_px: float = 0.60
    static_max_background_flow_px: float = 0.60
    static_max_parallax_px: float = 0.40
    static_max_homography_rotation_degrees: float = 0.25
    static_max_pnp_translation_m: float = 0.010
    static_max_image_difference: float = 0.025
    static_min_homography_inlier_ratio: float = 0.85
    static_required_evidence_count: int = 4
    maximum_corners: int = 1200
    minimum_corner_distance_px: float = 5.0
    forward_backward_threshold_px: float = 1.5

    def __post_init__(self) -> None:
        if self.minimum_correspondences < 8 or self.minimum_inliers < 6:
            raise ValueError("Pose support thresholds are too small.")
        for name in (
            "minimum_inlier_ratio",
            "minimum_spatial_coverage",
            "minimum_valid_confidence",
            "static_min_homography_inlier_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        if self.static_required_evidence_count < 3:
            raise ValueError("Static identity requires at least three evidence checks.")


@dataclass(frozen=True)
class MetricPnPResult:
    """Internal metric PnP estimate and full correspondence inlier mask."""

    transform: Optional[np.ndarray]
    inlier_mask: np.ndarray
    reprojection_error: float
    valid_depth_count: int
    failure_reason: str = ""


def _sample_depth(
    depth_map: np.ndarray,
    valid_mask: np.ndarray,
    points_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_map.shape
    columns = np.rint(points_uv[:, 0]).astype(int)
    rows = np.rint(points_uv[:, 1]).astype(int)
    valid = (
        (columns >= 0)
        & (columns < width)
        & (rows >= 0)
        & (rows < height)
    )
    values = np.full(points_uv.shape[0], np.nan, dtype=float)
    indices = np.flatnonzero(valid)
    if indices.size:
        values[indices] = depth_map[rows[indices], columns[indices]]
        valid[indices] &= valid_mask[rows[indices], columns[indices]]
    valid &= np.isfinite(values) & (values > 0.0)
    return values, valid


def estimate_metric_transform_from_correspondences(
    source_points_uv: np.ndarray,
    target_points_uv: np.ndarray,
    source_depth_m: np.ndarray,
    source_depth_valid_mask: np.ndarray,
    K_source: np.ndarray,
    K_target: np.ndarray,
    *,
    minimum_inliers: int = 6,
    reprojection_threshold_px: float = 3.0,
) -> MetricPnPResult:
    """Estimate ``T_target_from_source`` from source metric depth and 2D tracks.

    Source pixels are backprojected with ``K_source`` and target pixels are
    fitted with ``K_target``. This distinction is necessary when a monocular
    provider predicts slightly different intrinsics in adjacent frames.
    """

    source = np.asarray(source_points_uv, dtype=float).reshape(-1, 2)
    target = np.asarray(target_points_uv, dtype=float).reshape(-1, 2)
    depth = np.asarray(source_depth_m, dtype=float)
    depth_valid = np.asarray(source_depth_valid_mask, dtype=bool)
    source_k = validate_intrinsics(K_source)
    target_k = validate_intrinsics(K_target)
    if source.shape != target.shape:
        raise ValueError("Source and target correspondences must have equal shape.")
    if depth.ndim != 2 or depth_valid.shape != depth.shape:
        raise ValueError("Depth map and valid mask must share a 2D shape.")
    if source.shape[0] < minimum_inliers:
        return MetricPnPResult(
            None,
            np.zeros(source.shape[0], dtype=bool),
            float("nan"),
            0,
            "insufficient_correspondences",
        )

    values, valid = _sample_depth(depth, depth_valid, source)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size < minimum_inliers:
        return MetricPnPResult(
            None,
            np.zeros(source.shape[0], dtype=bool),
            float("nan"),
            int(valid_indices.size),
            "insufficient_valid_metric_depth",
        )
    pixels_h = np.column_stack((source[valid], np.ones(valid_indices.size)))
    rays = (np.linalg.inv(source_k) @ pixels_h.T).T
    object_points = rays * values[valid, None]
    image_points = target[valid]
    success, rvec, translation, local_inliers = cv2.solvePnPRansac(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        target_k,
        None,
        iterationsCount=300,
        reprojectionError=float(reprojection_threshold_px),
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    full_inliers = np.zeros(source.shape[0], dtype=bool)
    if (
        not success
        or local_inliers is None
        or local_inliers.size < minimum_inliers
    ):
        return MetricPnPResult(
            None,
            full_inliers,
            float("nan"),
            int(valid_indices.size),
            "metric_pnp_failed",
        )
    local = local_inliers.reshape(-1)
    full_inliers[valid_indices[local]] = True
    rotation, _ = cv2.Rodrigues(rvec)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    projected, _ = cv2.projectPoints(
        object_points[local].astype(np.float32),
        rvec,
        translation,
        target_k,
        None,
    )
    error = float(
        np.median(
            np.linalg.norm(projected.reshape(-1, 2) - image_points[local], axis=1)
        )
    )
    return MetricPnPResult(
        transform,
        full_inliers,
        error,
        int(valid_indices.size),
    )


def _rotation_degrees(rotation: Optional[np.ndarray]) -> float:
    if rotation is None:
        return float("nan")
    trace = float(np.trace(rotation))
    cosine = float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _homography_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    K_source: np.ndarray,
    K_target: np.ndarray,
) -> tuple[Optional[np.ndarray], np.ndarray, float, float]:
    count = source.shape[0]
    if count < 4:
        return None, np.zeros(count, dtype=bool), 0.0, float("nan")
    homography, mask = cv2.findHomography(source, target, cv2.RANSAC, 2.0)
    if homography is None or mask is None:
        return None, np.zeros(count, dtype=bool), 0.0, float("nan")
    inliers = mask.reshape(-1).astype(bool)
    normalised = np.linalg.inv(K_target) @ homography @ K_source
    scale = float(np.cbrt(max(abs(np.linalg.det(normalised)), 1e-12)))
    u, _, vt = np.linalg.svd(normalised / scale)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    source_h = np.column_stack((source, np.ones(count)))
    projected_h = (homography @ source_h.T).T
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    errors = np.linalg.norm(projected - target, axis=1)
    error = float(np.median(errors[inliers])) if np.any(inliers) else float("nan")
    return rotation, inliers, float(np.mean(inliers)), error


def _essential_inlier_ratio(
    source: np.ndarray,
    target: np.ndarray,
    K_source: np.ndarray,
    K_target: np.ndarray,
) -> float:
    """Return essential-matrix support using per-frame normalised pixels."""

    if source.shape[0] < 8:
        return 0.0
    source_normalised = cv2.undistortPoints(
        source.reshape(-1, 1, 2), K_source, None
    ).reshape(-1, 2)
    target_normalised = cv2.undistortPoints(
        target.reshape(-1, 1, 2), K_target, None
    ).reshape(-1, 2)
    essential, mask = cv2.findEssentialMat(
        source_normalised,
        target_normalised,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.0015,
    )
    if essential is None or mask is None:
        return 0.0
    return float(np.mean(mask.reshape(-1).astype(bool)))


def _invalid_pose(
    frame_t: int,
    frame_t1: int,
    status: PoseProviderStatus,
    reason: str,
    *,
    provider_name: str,
    background_candidates: int = 0,
    foreground_rejected: int = 0,
    inlier_count: int = 0,
    inlier_ratio: float = 0.0,
    static_background_ratio: float = 0.0,
    dynamic_foreground_ratio: float = 0.0,
    reprojection_error: float = float("nan"),
    confidence: float = 0.0,
    degeneracy_status: str = "",
    static_verification: Optional[StaticVerificationObservation] = None,
    metadata: Mapping[str, Any] | None = None,
) -> PairwisePoseObservation:
    return PairwisePoseObservation(
        frame_t=frame_t,
        frame_t1=frame_t1,
        rotation=None,
        translation=None,
        T_target_from_source=None,
        pose_convention="X_target_camera=T_target_from_source@X_source_camera",
        camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
        translation_scale_status="not_available",
        inlier_count=inlier_count,
        inlier_ratio=float(np.clip(inlier_ratio, 0.0, 1.0)),
        reprojection_error=reprojection_error,
        static_background_ratio=float(np.clip(static_background_ratio, 0.0, 1.0)),
        dynamic_foreground_ratio=float(np.clip(dynamic_foreground_ratio, 0.0, 1.0)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        provider_status=status,
        failure_reason=reason,
        background_candidates=background_candidates,
        foreground_rejected=foreground_rejected,
        geometric_inliers=inlier_count,
        degeneracy_status=degeneracy_status or reason,
        provider_name=provider_name,
        valid=False,
        static_verification=static_verification,
        metadata=dict(metadata or {}),
    )


class ShortBaselinePoseProvider:
    """OpenCV LK + metric-depth PnP provider for adjacent frames."""

    provider_name = "opencv_lk_metric_depth_pnp_v1"
    provider_version = "1.0"

    def __init__(
        self,
        thresholds: ShortBaselinePoseThresholds | None = None,
    ) -> None:
        self.thresholds = thresholds or ShortBaselinePoseThresholds()

    def estimate_pair(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        *,
        frame_t: int,
        frame_t1: int,
        K_source: Optional[np.ndarray],
        K_target: Optional[np.ndarray],
        source_depth_m: Optional[np.ndarray],
        source_depth_valid_mask: Optional[np.ndarray],
        source_foreground_mask: Optional[np.ndarray] = None,
        target_foreground_mask: Optional[np.ndarray] = None,
    ) -> PairwisePoseObservation:
        """Estimate one directed adjacent-frame pose without identity fallback."""

        if K_source is None or K_target is None:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.BLOCKED_BY_INTRINSICS,
                "camera_intrinsics_unavailable",
                provider_name=self.provider_name,
            )
        try:
            source_k = validate_intrinsics(K_source)
            target_k = validate_intrinsics(K_target)
        except ValueError as error:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.BLOCKED_BY_INTRINSICS,
                f"invalid_camera_intrinsics:{error}",
                provider_name=self.provider_name,
            )
        if source_image is None or target_image is None:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.PROVIDER_FAILED,
                "frame_decode_failed",
                provider_name=self.provider_name,
            )
        try:
            global_tracks = track_background_correspondences(
                source_image,
                target_image,
                source_frame_index=frame_t,
                target_frame_index=frame_t1,
                maximum_corners=self.thresholds.maximum_corners,
                minimum_distance=self.thresholds.minimum_corner_distance_px,
                forward_backward_threshold=self.thresholds.forward_backward_threshold_px,
            )
            tracks = track_background_correspondences(
                source_image,
                target_image,
                source_frame_index=frame_t,
                target_frame_index=frame_t1,
                source_foreground_mask=source_foreground_mask,
                target_foreground_mask=target_foreground_mask,
                maximum_corners=self.thresholds.maximum_corners,
                minimum_distance=self.thresholds.minimum_corner_distance_px,
                forward_backward_threshold=self.thresholds.forward_backward_threshold_px,
            )
        except (cv2.error, ValueError) as error:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.PROVIDER_FAILED,
                f"background_tracking_failed:{error}",
                provider_name=self.provider_name,
            )

        foreground_rejected = max(
            0, tracks.candidate_count_before_mask - tracks.background_point_count_after_mask
        )
        dynamic_ratio = float(np.clip(tracks.foreground_excluded_ratio, 0.0, 1.0))
        static_ratio = 1.0 - dynamic_ratio
        common_metadata: dict[str, Any] = {
            "track_rows": [dict(row) for row in tracks.track_rows],
            "match_count": tracks.match_count,
            "candidate_count_before_mask": tracks.candidate_count_before_mask,
            "background_point_count_after_mask": tracks.background_point_count_after_mask,
            "spatial_coverage_ratio": tracks.spatial_coverage_ratio,
            "quadrant_support": list(tracks.quadrant_support),
            "feature_concentrated": tracks.feature_concentrated,
            "median_flow_px": tracks.median_flow,
            "median_global_flow_px": global_tracks.median_flow,
            "p90_flow_px": tracks.p90_flow,
            "median_parallax_px": tracks.median_parallax,
            "image_difference": tracks.image_difference,
            "foreground_masks_excluded": True,
            "semantic_scale_prior_used": False,
            "authenticity_label_used": False,
        }
        if (
            tracks.match_count < self.thresholds.minimum_correspondences
            or tracks.spatial_coverage_ratio < self.thresholds.minimum_spatial_coverage
            or tracks.feature_concentrated
        ):
            reason = (
                "insufficient_background_correspondences"
                if tracks.match_count < self.thresholds.minimum_correspondences
                else "degenerate_background_spatial_support"
            )
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE,
                reason,
                provider_name=self.provider_name,
                background_candidates=tracks.background_point_count_after_mask,
                foreground_rejected=foreground_rejected,
                static_background_ratio=static_ratio,
                dynamic_foreground_ratio=dynamic_ratio,
                degeneracy_status=reason,
                metadata=common_metadata,
            )
        if source_depth_m is None or source_depth_valid_mask is None:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.PROVIDER_FAILED,
                "metric_source_depth_unavailable",
                provider_name=self.provider_name,
                background_candidates=tracks.background_point_count_after_mask,
                foreground_rejected=foreground_rejected,
                static_background_ratio=static_ratio,
                dynamic_foreground_ratio=dynamic_ratio,
                degeneracy_status="metric_depth_missing",
                metadata=common_metadata,
            )

        homography_rotation, homography_inliers, homography_ratio, homography_error = (
            _homography_diagnostics(
                tracks.source_points,
                tracks.target_points,
                source_k,
                target_k,
            )
        )
        pnp = estimate_metric_transform_from_correspondences(
            tracks.source_points,
            tracks.target_points,
            source_depth_m,
            source_depth_valid_mask,
            source_k,
            target_k,
            minimum_inliers=self.thresholds.minimum_inliers,
            reprojection_threshold_px=self.thresholds.maximum_pnp_reprojection_error_px,
        )
        pnp_count = int(np.sum(pnp.inlier_mask))
        pnp_ratio = pnp_count / max(tracks.match_count, 1)
        essential_inlier_ratio = _essential_inlier_ratio(
            tracks.source_points,
            tracks.target_points,
            source_k,
            target_k,
        )
        pnp_translation = (
            float(np.linalg.norm(pnp.transform[:3, 3]))
            if pnp.transform is not None
            else float("nan")
        )
        homography_rotation_degrees = _rotation_degrees(homography_rotation)

        global_flow_small = bool(
            math.isfinite(global_tracks.median_flow)
            and global_tracks.median_flow <= self.thresholds.static_max_median_flow_px
        )
        background_flow_small = bool(
            math.isfinite(tracks.median_flow)
            and tracks.median_flow <= self.thresholds.static_max_background_flow_px
        )
        parallax_small = bool(
            math.isfinite(tracks.median_parallax)
            and tracks.median_parallax <= self.thresholds.static_max_parallax_px
        )
        homography_motion_small = bool(
            homography_rotation is not None
            and homography_ratio >= self.thresholds.static_min_homography_inlier_ratio
            and homography_rotation_degrees
            <= self.thresholds.static_max_homography_rotation_degrees
        )
        essential_supports_significant_motion = bool(
            essential_inlier_ratio >= 0.5
            and math.isfinite(global_tracks.median_flow)
            and global_tracks.median_flow
            > self.thresholds.static_max_median_flow_px
            and math.isfinite(pnp_translation)
            and pnp_translation > self.thresholds.static_max_pnp_translation_m
        )
        essential_motion_not_supported = not essential_supports_significant_motion
        image_difference_stable = bool(
            math.isfinite(tracks.image_difference)
            and tracks.image_difference <= self.thresholds.static_max_image_difference
        )
        checks = (
            global_flow_small,
            background_flow_small,
            homography_motion_small,
            essential_motion_not_supported,
            parallax_small,
            image_difference_stable,
        )
        evidence_count = sum(checks)
        verified = evidence_count >= self.thresholds.static_required_evidence_count
        static_confidence = evidence_count / len(checks)
        static_verification = StaticVerificationObservation(
            source_frame_index=frame_t,
            target_frame_index=frame_t1,
            global_flow_small=global_flow_small,
            background_displacement_small=background_flow_small,
            homography_motion_small=homography_motion_small,
            essential_motion_not_supported=essential_motion_not_supported,
            parallax_small=parallax_small,
            image_difference_stable=image_difference_stable,
            evidence_count=evidence_count,
            required_evidence_count=self.thresholds.static_required_evidence_count,
            verified_static=verified,
            median_global_flow=global_tracks.median_flow,
            median_background_flow=tracks.median_flow,
            median_parallax=tracks.median_parallax,
            homography_rotation_degrees=homography_rotation_degrees,
            pnp_translation_norm=pnp_translation,
            confidence=static_confidence,
            failure_reason="" if verified else "multi_evidence_static_check_failed",
            metadata={
                "homography_inlier_ratio": homography_ratio,
                "homography_reprojection_error_px": homography_error,
                "pnp_inlier_ratio": pnp_ratio,
                "essential_inlier_ratio": essential_inlier_ratio,
                "essential_supports_significant_motion": (
                    essential_supports_significant_motion
                ),
                "identity_is_provider_fallback": False,
            },
        )
        common_metadata.update(
            {
                "homography_inlier_count": int(np.sum(homography_inliers)),
                "homography_inlier_ratio": homography_ratio,
                "homography_reprojection_error_px": homography_error,
                "homography_rotation_degrees": homography_rotation_degrees,
                "pnp_valid_depth_count": pnp.valid_depth_count,
                "pnp_translation_norm_m": pnp_translation,
                "essential_inlier_ratio": essential_inlier_ratio,
                "pnp_failure_reason": pnp.failure_reason,
                "pnp_inlier_mask": pnp.inlier_mask.astype(int).tolist(),
            }
        )
        if verified:
            return PairwisePoseObservation(
                frame_t=frame_t,
                frame_t1=frame_t1,
                rotation=np.eye(3),
                translation=np.zeros(3),
                T_target_from_source=np.eye(4),
                pose_convention="X_target_camera=T_target_from_source@X_source_camera",
                camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
                translation_scale_status="zero_static",
                inlier_count=max(pnp_count, int(np.sum(homography_inliers))),
                inlier_ratio=max(pnp_ratio, homography_ratio),
                reprojection_error=float(homography_error),
                static_background_ratio=static_ratio,
                dynamic_foreground_ratio=dynamic_ratio,
                confidence=static_confidence,
                provider_status=PoseProviderStatus.VERIFIED_STATIC,
                failure_reason="",
                background_candidates=tracks.background_point_count_after_mask,
                foreground_rejected=foreground_rejected,
                geometric_inliers=max(pnp_count, int(np.sum(homography_inliers))),
                degeneracy_status="none",
                provider_name=self.provider_name,
                valid=True,
                static_verification=static_verification,
                metadata=common_metadata,
            )

        if pnp.transform is None:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE,
                pnp.failure_reason or "metric_pnp_failed",
                provider_name=self.provider_name,
                background_candidates=tracks.background_point_count_after_mask,
                foreground_rejected=foreground_rejected,
                inlier_count=pnp_count,
                inlier_ratio=pnp_ratio,
                static_background_ratio=static_ratio,
                dynamic_foreground_ratio=dynamic_ratio,
                reprojection_error=pnp.reprojection_error,
                confidence=0.0,
                degeneracy_status=pnp.failure_reason or "metric_pnp_failed",
                static_verification=static_verification,
                metadata=common_metadata,
            )
        quality = float(
            np.clip(
                0.45 * pnp_ratio
                + 0.25 * min(
                    tracks.spatial_coverage_ratio
                    / max(self.thresholds.minimum_spatial_coverage, 1e-9),
                    1.0,
                )
                + 0.30
                * max(
                    0.0,
                    1.0
                    - pnp.reprojection_error
                    / self.thresholds.maximum_pnp_reprojection_error_px,
                ),
                0.0,
                1.0,
            )
        )
        accepted = (
            pnp_count >= self.thresholds.minimum_inliers
            and pnp_ratio >= self.thresholds.minimum_inlier_ratio
            and pnp.reprojection_error
            <= self.thresholds.maximum_pnp_reprojection_error_px
            and quality >= self.thresholds.minimum_valid_confidence
        )
        if not accepted:
            return _invalid_pose(
                frame_t,
                frame_t1,
                PoseProviderStatus.ESTIMATED_LOW_CONFIDENCE,
                "metric_pose_quality_below_threshold",
                provider_name=self.provider_name,
                background_candidates=tracks.background_point_count_after_mask,
                foreground_rejected=foreground_rejected,
                inlier_count=pnp_count,
                inlier_ratio=pnp_ratio,
                static_background_ratio=static_ratio,
                dynamic_foreground_ratio=dynamic_ratio,
                reprojection_error=pnp.reprojection_error,
                confidence=quality,
                degeneracy_status="low_confidence",
                static_verification=static_verification,
                metadata=common_metadata,
            )
        return PairwisePoseObservation(
            frame_t=frame_t,
            frame_t1=frame_t1,
            rotation=pnp.transform[:3, :3],
            translation=pnp.transform[:3, 3],
            T_target_from_source=pnp.transform,
            pose_convention="X_target_camera=T_target_from_source@X_source_camera",
            camera_to_world_or_world_to_camera="camera_t_to_camera_t1",
            translation_scale_status="metric_model_depth",
            inlier_count=pnp_count,
            inlier_ratio=pnp_ratio,
            reprojection_error=pnp.reprojection_error,
            static_background_ratio=static_ratio,
            dynamic_foreground_ratio=dynamic_ratio,
            confidence=quality,
            provider_status=PoseProviderStatus.ESTIMATED_VALID,
            failure_reason="",
            background_candidates=tracks.background_point_count_after_mask,
            foreground_rejected=foreground_rejected,
            geometric_inliers=pnp_count,
            degeneracy_status="none",
            provider_name=self.provider_name,
            valid=True,
            static_verification=static_verification,
            metadata=common_metadata,
        )
