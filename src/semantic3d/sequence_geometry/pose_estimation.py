"""Layered background camera-pose estimation for P3-0.5.

The estimator separates static identity, rotation-only, direction-only SE3,
and depth-assisted relative-scale SE3.  It never uses semantic object-size
priors or anomaly residuals as geometry supervision.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..depth_provider import DepthObservation
from .motion_regime import (
    CameraMotionRegime,
    MotionRegimeObservation,
    MotionRegimeThresholds,
    classify_motion_regime,
)


class PoseModelType(str, Enum):
    """Geometric model represented by one candidate edge."""

    REFERENCE_GAUGE = "reference_gauge"
    STATIC_IDENTITY = "static_identity"
    ROTATION_HOMOGRAPHY = "rotation_homography"
    ESSENTIAL_SE3 = "essential_se3"
    DEPTH_PNP = "depth_pnp"
    MISSING = "missing"


class TranslationScaleStatus(str, Enum):
    """Scale semantics of an estimated relative translation."""

    ZERO_STATIC = "zero_static"
    DIRECTION_ONLY = "direction_only"
    DEPTH_RELATIVE = "depth_relative"
    METRIC = "metric"
    NOT_AVAILABLE = "not_available"

    @property
    def supports_shared_3d_translation(self) -> bool:
        """Return whether translation can share a depth coordinate scale."""

        return self in {
            TranslationScaleStatus.ZERO_STATIC,
            TranslationScaleStatus.DEPTH_RELATIVE,
            TranslationScaleStatus.METRIC,
        }


@dataclass(frozen=True)
class BackgroundTrackDiagnostics:
    """Foreground-filtered LK correspondences and stage-by-stage counts."""

    source_frame_index: int
    target_frame_index: int
    source_points: np.ndarray
    target_points: np.ndarray
    candidate_count_before_mask: int
    background_point_count_after_mask: int
    match_count: int
    spatial_coverage_ratio: float
    quadrant_support: tuple[int, int, int, int]
    feature_concentrated: bool
    median_flow: float
    p90_flow: float
    median_parallax: float
    image_difference: float
    foreground_excluded_ratio: float
    failure_stage: str
    track_rows: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source = np.asarray(self.source_points, dtype=float).reshape(-1, 2)
        target = np.asarray(self.target_points, dtype=float).reshape(-1, 2)
        if source.shape != target.shape:
            raise ValueError("Background source/target points must have matching shape.")
        if source.shape[0] != self.match_count:
            raise ValueError("match_count must equal the correspondence count.")
        object.__setattr__(self, "source_points", source)
        object.__setattr__(self, "target_points", target)
        object.__setattr__(self, "track_rows", tuple(dict(row) for row in self.track_rows))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PoseEstimateCandidate:
    """One pose edge hypothesis with explicit rotation/translation semantics."""

    source_frame_index: int
    target_frame_index: int
    pose_model_type: PoseModelType | str
    T_target_from_source: Optional[np.ndarray]
    rotation_valid: bool
    translation_valid: bool
    translation_scale_status: TranslationScaleStatus | str
    support_count: int
    inlier_count: int
    inlier_ratio: float
    median_parallax: float
    reprojection_error: float
    quality: float
    valid: bool
    missing_reason: str = ""
    selected_reference_frame: Optional[int] = None
    evidence_source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        model = PoseModelType(self.pose_model_type)
        scale_status = TranslationScaleStatus(self.translation_scale_status)
        transform = (
            None
            if self.T_target_from_source is None
            else np.asarray(self.T_target_from_source, dtype=float)
        )
        if self.support_count < 0 or self.inlier_count < 0:
            raise ValueError("Pose support counts must be non-negative.")
        if self.inlier_count > self.support_count:
            raise ValueError("Pose inlier_count cannot exceed support_count.")
        for name in ("inlier_ratio", "quality"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        for name in ("median_parallax", "reprojection_error"):
            value = float(getattr(self, name))
            if not (math.isnan(value) or (math.isfinite(value) and value >= 0.0)):
                raise ValueError(f"{name} must be non-negative or NaN.")
            object.__setattr__(self, name, value)
        if self.valid:
            if transform is None or transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise ValueError("Valid pose candidate requires a finite 4x4 transform.")
            if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
                raise ValueError("Pose transform must be homogeneous.")
            if not self.rotation_valid:
                raise ValueError("A valid pose candidate requires valid rotation.")
            if not self.evidence_source:
                raise ValueError("A valid pose candidate requires evidence_source.")
            if self.missing_reason:
                raise ValueError("A valid pose candidate cannot have missing_reason.")
        else:
            if transform is not None:
                raise ValueError("Invalid pose candidate cannot contain a transform.")
            if not self.missing_reason:
                raise ValueError("Invalid pose candidate requires missing_reason.")
            if not math.isnan(float(self.reprojection_error)):
                raise ValueError("Invalid pose candidate reprojection_error must be NaN.")
        if model == PoseModelType.ROTATION_HOMOGRAPHY and self.translation_valid:
            raise ValueError("Rotation-only pose must not claim valid translation.")
        if model == PoseModelType.STATIC_IDENTITY:
            if scale_status != TranslationScaleStatus.ZERO_STATIC:
                raise ValueError("Static identity requires zero_static translation status.")
            if transform is not None and not np.allclose(transform, np.eye(4), atol=1e-8):
                raise ValueError("Static identity candidate must contain identity transform.")
        object.__setattr__(self, "pose_model_type", model)
        object.__setattr__(self, "translation_scale_status", scale_status)
        object.__setattr__(self, "T_target_from_source", transform)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(
        cls,
        source_frame_index: int,
        target_frame_index: int,
        reason: str,
        *,
        support_count: int = 0,
        metadata: Mapping[str, Any] | None = None,
    ) -> "PoseEstimateCandidate":
        """Create a missing candidate without a fabricated identity transform."""

        return cls(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            pose_model_type=PoseModelType.MISSING,
            T_target_from_source=None,
            rotation_valid=False,
            translation_valid=False,
            translation_scale_status=TranslationScaleStatus.NOT_AVAILABLE,
            support_count=support_count,
            inlier_count=0,
            inlier_ratio=0.0,
            median_parallax=float("nan"),
            reprojection_error=float("nan"),
            quality=0.0,
            valid=False,
            missing_reason=reason,
            selected_reference_frame=source_frame_index,
            metadata=dict(metadata or {}),
        )

    @property
    def full_se3(self) -> bool:
        """Return whether a non-static model estimated rotation and translation."""

        return bool(
            self.valid
            and self.rotation_valid
            and self.translation_valid
            and self.pose_model_type
            in {PoseModelType.ESSENTIAL_SE3, PoseModelType.DEPTH_PNP}
        )

    @property
    def pose_scale_compatible_with_depth(self) -> bool:
        """Return whether the edge translation can share a depth scale."""

        return bool(
            self.valid
            and self.translation_scale_status.supports_shared_3d_translation
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly candidate record."""

        payload = asdict(self)
        payload["pose_model_type"] = self.pose_model_type.value
        payload["translation_scale_status"] = self.translation_scale_status.value
        payload["T_target_from_source"] = (
            None if self.T_target_from_source is None else self.T_target_from_source.tolist()
        )
        payload["full_se3"] = self.full_se3
        payload["pose_scale_compatible_with_depth"] = (
            self.pose_scale_compatible_with_depth
        )
        return payload


@dataclass(frozen=True)
class PosePairEstimation:
    """All candidate models and the selected edge for one frame pair."""

    motion_regime: MotionRegimeObservation
    candidates: tuple[PoseEstimateCandidate, ...]
    selected: PoseEstimateCandidate
    tracks: BackgroundTrackDiagnostics


def _normalised_image_difference(source_gray: np.ndarray, target_gray: np.ndarray) -> float:
    difference = cv2.absdiff(source_gray, target_gray)
    return float(np.mean(difference) / 255.0)


def _quadrant_support(points: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    if points.size == 0:
        return (0, 0, 0, 0)
    right = points[:, 0] >= width / 2.0
    bottom = points[:, 1] >= height / 2.0
    return (
        int(np.sum(~right & ~bottom)),
        int(np.sum(right & ~bottom)),
        int(np.sum(~right & bottom)),
        int(np.sum(right & bottom)),
    )


def _spatial_coverage(points: np.ndarray, width: int, height: int) -> float:
    if points.shape[0] < 2:
        return 0.0
    span = np.ptp(points, axis=0)
    return float(np.clip((span[0] * span[1]) / max(width * height, 1), 0.0, 1.0))


def track_background_correspondences(
    source_image: np.ndarray,
    target_image: np.ndarray,
    *,
    source_frame_index: int,
    target_frame_index: int,
    source_foreground_mask: Optional[np.ndarray] = None,
    target_foreground_mask: Optional[np.ndarray] = None,
    maximum_corners: int = 1200,
    minimum_distance: float = 5.0,
    forward_backward_threshold: float = 1.5,
) -> BackgroundTrackDiagnostics:
    """Track background points and expose where support was lost."""

    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    height, width = source_gray.shape
    if target_gray.shape != source_gray.shape:
        raise ValueError("Source and target frames must have the same image shape.")
    candidate = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=maximum_corners,
        qualityLevel=0.01,
        minDistance=minimum_distance,
    )
    candidate_count = 0 if candidate is None else int(candidate.shape[0])
    feature_mask = np.full(source_gray.shape, 255, dtype=np.uint8)
    excluded_ratio = 0.0
    if source_foreground_mask is not None:
        if source_foreground_mask.shape != source_gray.shape:
            raise ValueError("source_foreground_mask shape must match the image.")
        feature_mask[np.asarray(source_foreground_mask, dtype=bool)] = 0
        excluded_ratio = float(np.mean(source_foreground_mask))
    points = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=maximum_corners,
        qualityLevel=0.01,
        minDistance=minimum_distance,
        mask=feature_mask,
    )
    background_count = 0 if points is None else int(points.shape[0])
    empty = np.empty((0, 2), dtype=float)
    if points is None or background_count == 0:
        return BackgroundTrackDiagnostics(
            source_frame_index,
            target_frame_index,
            empty,
            empty,
            candidate_count,
            background_count,
            0,
            0.0,
            (0, 0, 0, 0),
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            _normalised_image_difference(source_gray, target_gray),
            excluded_ratio,
            "detection",
            (),
        )
    target_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(
        source_gray, target_gray, points, None
    )
    if target_points is None or status_forward is None:
        return BackgroundTrackDiagnostics(
            source_frame_index,
            target_frame_index,
            empty,
            empty,
            candidate_count,
            background_count,
            0,
            0.0,
            (0, 0, 0, 0),
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            _normalised_image_difference(source_gray, target_gray),
            excluded_ratio,
            "matching",
            (),
            {"optical_flow_failure": "forward"},
        )
    backward_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(
        target_gray, source_gray, target_points, None
    )
    if backward_points is None or status_backward is None:
        return BackgroundTrackDiagnostics(
            source_frame_index,
            target_frame_index,
            empty,
            empty,
            candidate_count,
            background_count,
            0,
            0.0,
            (0, 0, 0, 0),
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            _normalised_image_difference(source_gray, target_gray),
            excluded_ratio,
            "matching",
            (),
            {"optical_flow_failure": "backward"},
        )
    source_xy = points.reshape(-1, 2)
    target_xy = target_points.reshape(-1, 2)
    backward_xy = backward_points.reshape(-1, 2)
    valid = status_forward.reshape(-1).astype(bool) & status_backward.reshape(-1).astype(bool)
    valid &= np.linalg.norm(backward_xy - source_xy, axis=1) <= forward_backward_threshold
    if target_foreground_mask is not None:
        if target_foreground_mask.shape != source_gray.shape:
            raise ValueError("target_foreground_mask shape must match the image.")
        columns = np.clip(np.rint(target_xy[:, 0]).astype(int), 0, width - 1)
        rows = np.clip(np.rint(target_xy[:, 1]).astype(int), 0, height - 1)
        valid &= ~np.asarray(target_foreground_mask, dtype=bool)[rows, columns]
        excluded_ratio = float(
            0.5 * (excluded_ratio + np.mean(target_foreground_mask))
        )
    source_xy = source_xy[valid]
    target_xy = target_xy[valid]
    flow = np.linalg.norm(target_xy - source_xy, axis=1)
    median_flow = float(np.median(flow)) if flow.size else float("nan")
    p90_flow = float(np.percentile(flow, 90)) if flow.size else float("nan")
    if flow.size:
        global_displacement = np.median(target_xy - source_xy, axis=0)
        parallax = np.linalg.norm((target_xy - source_xy) - global_displacement, axis=1)
        median_parallax = float(np.median(parallax))
    else:
        median_parallax = float("nan")
    quadrants = _quadrant_support(source_xy, width, height)
    concentrated = bool(source_xy.shape[0] and max(quadrants) / source_xy.shape[0] > 0.8)
    rows = tuple(
        {
            "track_id": f"bg_{source_frame_index}_{target_frame_index}_{index}",
            "source_frame_index": source_frame_index,
            "target_frame_index": target_frame_index,
            "source_x": float(source[0]),
            "source_y": float(source[1]),
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "forward_backward_valid": True,
        }
        for index, (source, target) in enumerate(zip(source_xy, target_xy, strict=True))
    )
    return BackgroundTrackDiagnostics(
        source_frame_index,
        target_frame_index,
        source_xy,
        target_xy,
        candidate_count,
        background_count,
        int(source_xy.shape[0]),
        _spatial_coverage(source_xy, width, height),
        quadrants,
        concentrated,
        median_flow,
        p90_flow,
        median_parallax,
        _normalised_image_difference(source_gray, target_gray),
        excluded_ratio,
        "" if source_xy.shape[0] else "matching",
        rows,
        metadata={
            "forward_backward_threshold_px": forward_backward_threshold,
            "semantic_scale_prior_used": False,
            "anomaly_residual_used": False,
        },
    )


def _homography_model(
    tracks: BackgroundTrackDiagnostics,
    K: np.ndarray,
) -> tuple[Optional[np.ndarray], np.ndarray, float, float]:
    if tracks.match_count < 4:
        return None, np.zeros(tracks.match_count, dtype=bool), 0.0, float("nan")
    H, mask = cv2.findHomography(
        tracks.source_points,
        tracks.target_points,
        cv2.RANSAC,
        2.0,
    )
    if H is None or mask is None:
        return None, np.zeros(tracks.match_count, dtype=bool), 0.0, float("nan")
    inliers = mask.reshape(-1).astype(bool)
    source_h = np.column_stack((tracks.source_points, np.ones(tracks.match_count)))
    projected = (H @ source_h.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - tracks.target_points, axis=1)
    error = float(np.median(errors[inliers])) if np.any(inliers) else float("nan")
    normalised = np.linalg.inv(K) @ H @ K
    scale = float(np.cbrt(max(abs(np.linalg.det(normalised)), 1e-12)))
    approximate_rotation = normalised / scale
    u, _, vt = np.linalg.svd(approximate_rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation, inliers, float(np.mean(inliers)), error


def _essential_model(
    tracks: BackgroundTrackDiagnostics,
    K: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, float]:
    if tracks.match_count < 8:
        return None, None, np.zeros(tracks.match_count, dtype=bool), float("nan")
    E, mask = cv2.findEssentialMat(
        tracks.source_points,
        tracks.target_points,
        K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0,
    )
    if E is None or mask is None:
        return None, None, np.zeros(tracks.match_count, dtype=bool), float("nan")
    if E.shape[0] > 3:
        E = E[:3, :3]
    _, rotation, translation, recovered = cv2.recoverPose(
        E,
        tracks.source_points,
        tracks.target_points,
        K,
        mask=mask,
    )
    inliers = recovered.reshape(-1).astype(bool)
    if not np.any(inliers):
        return None, None, inliers, float("nan")
    source_norm = cv2.undistortPoints(
        tracks.source_points.reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)
    target_norm = cv2.undistortPoints(
        tracks.target_points.reshape(-1, 1, 2), K, None
    ).reshape(-1, 2)
    source_h = np.column_stack((source_norm, np.ones(tracks.match_count)))
    target_h = np.column_stack((target_norm, np.ones(tracks.match_count)))
    residual = np.abs(np.sum(target_h * ((E @ source_h.T).T), axis=1))
    error = float(np.median(residual[inliers]))
    return rotation, translation.reshape(3), inliers, error


def _sample_source_depth_points(
    depth: DepthObservation,
    points: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth_map = depth.require_geometry_depth()
    height, width = depth_map.shape
    columns = np.rint(points[:, 0]).astype(int)
    rows = np.rint(points[:, 1]).astype(int)
    valid = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
    values = np.full(points.shape[0], np.nan, dtype=float)
    selected = np.flatnonzero(valid)
    if selected.size:
        values[selected] = depth_map[rows[selected], columns[selected]]
    valid &= np.isfinite(values) & (values > 0.0)
    rays = (np.linalg.inv(K) @ np.column_stack((points, np.ones(points.shape[0]))).T).T
    points_3d = rays[valid] * values[valid, None]
    return points_3d, valid


def _pnp_model(
    tracks: BackgroundTrackDiagnostics,
    K: np.ndarray,
    source_depth: Optional[DepthObservation],
) -> tuple[Optional[np.ndarray], np.ndarray, float]:
    if source_depth is None or not source_depth.valid or tracks.match_count < 8:
        return None, np.zeros(tracks.match_count, dtype=bool), float("nan")
    try:
        object_points, depth_valid = _sample_source_depth_points(
            source_depth, tracks.source_points, K
        )
    except ValueError:
        return None, np.zeros(tracks.match_count, dtype=bool), float("nan")
    image_points = tracks.target_points[depth_valid]
    if object_points.shape[0] < 8:
        return None, np.zeros(tracks.match_count, dtype=bool), float("nan")
    success, rotation_vector, translation, local_inliers = cv2.solvePnPRansac(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        K,
        None,
        iterationsCount=200,
        reprojectionError=3.0,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    full_inliers = np.zeros(tracks.match_count, dtype=bool)
    if not success or local_inliers is None or local_inliers.size < 6:
        return None, full_inliers, float("nan")
    depth_indices = np.flatnonzero(depth_valid)
    full_inliers[depth_indices[local_inliers.reshape(-1)]] = True
    rotation, _ = cv2.Rodrigues(rotation_vector)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation.reshape(3)
    projected, _ = cv2.projectPoints(
        object_points[local_inliers.reshape(-1)],
        rotation_vector,
        translation,
        K,
        None,
    )
    error = float(
        np.median(
            np.linalg.norm(
                projected.reshape(-1, 2) - image_points[local_inliers.reshape(-1)],
                axis=1,
            )
        )
    )
    return transform, full_inliers, error


def estimate_essential_pose_from_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    K: np.ndarray,
    *,
    source_frame_index: int,
    target_frame_index: int,
) -> PoseEstimateCandidate:
    """Estimate direction-only SE3 from known background point correspondences."""

    source = np.asarray(source_points, dtype=float).reshape(-1, 2)
    target = np.asarray(target_points, dtype=float).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have matching shape.")
    displacement = target - source
    flow = np.linalg.norm(displacement, axis=1)
    global_displacement = np.median(displacement, axis=0) if displacement.size else np.zeros(2)
    parallax = np.linalg.norm(displacement - global_displacement, axis=1)
    tracks = BackgroundTrackDiagnostics(
        source_frame_index=source_frame_index,
        target_frame_index=target_frame_index,
        source_points=source,
        target_points=target,
        candidate_count_before_mask=source.shape[0],
        background_point_count_after_mask=source.shape[0],
        match_count=source.shape[0],
        spatial_coverage_ratio=1.0,
        quadrant_support=(source.shape[0], 0, 0, 0),
        feature_concentrated=False,
        median_flow=float(np.median(flow)) if flow.size else float("nan"),
        p90_flow=float(np.percentile(flow, 90)) if flow.size else float("nan"),
        median_parallax=float(np.median(parallax)) if parallax.size else float("nan"),
        image_difference=0.0,
        foreground_excluded_ratio=0.0,
        failure_stage="",
        track_rows=(),
    )
    rotation, translation, inliers, error = _essential_model(
        tracks, np.asarray(K, dtype=float)
    )
    if rotation is None or translation is None:
        return PoseEstimateCandidate.missing(
            source_frame_index,
            target_frame_index,
            "essential_pose_estimation_failed",
            support_count=source.shape[0],
        )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    inlier_count = int(np.sum(inliers))
    inlier_ratio = inlier_count / max(source.shape[0], 1)
    return PoseEstimateCandidate(
        source_frame_index=source_frame_index,
        target_frame_index=target_frame_index,
        pose_model_type=PoseModelType.ESSENTIAL_SE3,
        T_target_from_source=transform,
        rotation_valid=True,
        translation_valid=True,
        translation_scale_status=TranslationScaleStatus.DIRECTION_ONLY,
        support_count=source.shape[0],
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        median_parallax=tracks.median_parallax,
        reprojection_error=error,
        quality=inlier_ratio,
        valid=True,
        selected_reference_frame=source_frame_index,
        evidence_source="synthetic_or_external_background_correspondences_essential",
    )


def estimate_rotation_pose_from_correspondences(
    source_points: np.ndarray,
    target_points: np.ndarray,
    K: np.ndarray,
    *,
    source_frame_index: int,
    target_frame_index: int,
) -> PoseEstimateCandidate:
    """Estimate a homography rotation without claiming translation validity."""

    source = np.asarray(source_points, dtype=float).reshape(-1, 2)
    target = np.asarray(target_points, dtype=float).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError("source_points and target_points must have matching shape.")
    displacement = target - source
    tracks = BackgroundTrackDiagnostics(
        source_frame_index,
        target_frame_index,
        source,
        target,
        source.shape[0],
        source.shape[0],
        source.shape[0],
        1.0,
        (source.shape[0], 0, 0, 0),
        False,
        float(np.median(np.linalg.norm(displacement, axis=1))),
        float(np.percentile(np.linalg.norm(displacement, axis=1), 90)),
        0.0,
        0.0,
        0.0,
        "",
        (),
    )
    rotation, inliers, ratio, error = _homography_model(
        tracks, np.asarray(K, dtype=float)
    )
    if rotation is None:
        return PoseEstimateCandidate.missing(
            source_frame_index,
            target_frame_index,
            "homography_rotation_estimation_failed",
            support_count=source.shape[0],
        )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    return PoseEstimateCandidate(
        source_frame_index,
        target_frame_index,
        PoseModelType.ROTATION_HOMOGRAPHY,
        transform,
        True,
        False,
        TranslationScaleStatus.NOT_AVAILABLE,
        source.shape[0],
        int(np.sum(inliers)),
        ratio,
        0.0,
        error,
        ratio,
        True,
        selected_reference_frame=source_frame_index,
        evidence_source="synthetic_or_external_background_correspondences_homography",
    )


class LayeredPoseEstimator:
    """Estimate and retain static, rotation, essential, and depth-PnP candidates."""

    def __init__(
        self,
        *,
        motion_thresholds: MotionRegimeThresholds | None = None,
        minimum_quality: float = 0.15,
    ) -> None:
        self.motion_thresholds = motion_thresholds or MotionRegimeThresholds()
        self.minimum_quality = float(minimum_quality)

    def estimate_pair(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        K: np.ndarray,
        *,
        source_frame_index: int,
        target_frame_index: int,
        source_foreground_mask: Optional[np.ndarray] = None,
        target_foreground_mask: Optional[np.ndarray] = None,
        source_depth: Optional[DepthObservation] = None,
        scene_cut: bool = False,
    ) -> PosePairEstimation:
        """Run the model hierarchy and select the richest valid nearby edge."""

        K_array = np.asarray(K, dtype=float)
        tracks = track_background_correspondences(
            source_image,
            target_image,
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            source_foreground_mask=source_foreground_mask,
            target_foreground_mask=target_foreground_mask,
        )
        homography_rotation, homography_inliers, homography_ratio, homography_error = (
            _homography_model(tracks, K_array)
        )
        essential_rotation, essential_translation, essential_inliers, essential_error = (
            _essential_model(tracks, K_array)
        )
        essential_ratio = (
            float(np.mean(essential_inliers)) if essential_inliers.size else 0.0
        )
        geometry_inliers = max(
            int(np.sum(homography_inliers)), int(np.sum(essential_inliers))
        )
        regime = classify_motion_regime(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            background_candidate_count=tracks.candidate_count_before_mask,
            background_point_count=tracks.background_point_count_after_mask,
            background_match_count=tracks.match_count,
            background_inlier_count=geometry_inliers,
            median_background_flow=tracks.median_flow,
            p90_background_flow=tracks.p90_flow,
            median_parallax=tracks.median_parallax,
            homography_inlier_ratio=homography_ratio,
            essential_matrix_inlier_ratio=essential_ratio,
            model_reprojection_error=(
                min(
                    value
                    for value in (homography_error, essential_error)
                    if math.isfinite(value)
                )
                if any(math.isfinite(value) for value in (homography_error, essential_error))
                else float("nan")
            ),
            image_difference=tracks.image_difference,
            foreground_excluded_ratio=tracks.foreground_excluded_ratio,
            spatial_coverage_ratio=tracks.spatial_coverage_ratio,
            quadrant_support=tracks.quadrant_support,
            feature_concentrated=tracks.feature_concentrated,
            scene_cut=scene_cut,
            thresholds=self.motion_thresholds,
            metadata={
                "failure_stage": tracks.failure_stage,
                "semantic_scale_prior_used": False,
            },
        )
        candidates: list[PoseEstimateCandidate] = []
        if regime.supports_identity_pose:
            support = tracks.match_count
            candidates.append(
                PoseEstimateCandidate(
                    source_frame_index,
                    target_frame_index,
                    PoseModelType.STATIC_IDENTITY,
                    np.eye(4),
                    True,
                    True,
                    TranslationScaleStatus.ZERO_STATIC,
                    support,
                    support,
                    1.0,
                    tracks.median_parallax,
                    0.0,
                    float(np.clip(support / 100.0, 0.25, 1.0)),
                    True,
                    selected_reference_frame=source_frame_index,
                    evidence_source=regime.evidence_source,
                    metadata={"identity_evidence": regime.to_dict()},
                )
            )
        if homography_rotation is not None and homography_ratio >= 0.45:
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = homography_rotation
            quality = float(
                np.clip(
                    homography_ratio
                    * tracks.spatial_coverage_ratio
                    / (1.0 + max(homography_error, 0.0)),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                PoseEstimateCandidate(
                    source_frame_index,
                    target_frame_index,
                    PoseModelType.ROTATION_HOMOGRAPHY,
                    transform,
                    True,
                    False,
                    TranslationScaleStatus.NOT_AVAILABLE,
                    tracks.match_count,
                    int(np.sum(homography_inliers)),
                    homography_ratio,
                    tracks.median_parallax,
                    homography_error,
                    quality,
                    True,
                    selected_reference_frame=source_frame_index,
                    evidence_source="foreground_filtered_background_homography",
                )
            )
        if essential_rotation is not None and essential_translation is not None:
            transform = np.eye(4, dtype=float)
            transform[:3, :3] = essential_rotation
            transform[:3, 3] = essential_translation
            quality = float(
                np.clip(
                    essential_ratio * max(tracks.spatial_coverage_ratio, 0.1)
                    / (1.0 + max(essential_error, 0.0)),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                PoseEstimateCandidate(
                    source_frame_index,
                    target_frame_index,
                    PoseModelType.ESSENTIAL_SE3,
                    transform,
                    True,
                    True,
                    TranslationScaleStatus.DIRECTION_ONLY,
                    tracks.match_count,
                    int(np.sum(essential_inliers)),
                    essential_ratio,
                    tracks.median_parallax,
                    essential_error,
                    quality,
                    True,
                    selected_reference_frame=source_frame_index,
                    evidence_source="foreground_filtered_background_essential",
                )
            )
        pnp_transform, pnp_inliers, pnp_error = _pnp_model(
            tracks, K_array, source_depth
        )
        if pnp_transform is not None:
            pnp_count = int(np.sum(pnp_inliers))
            pnp_ratio = pnp_count / max(tracks.match_count, 1)
            quality = float(
                np.clip(
                    pnp_ratio * max(tracks.spatial_coverage_ratio, 0.1)
                    / (1.0 + max(pnp_error, 0.0)),
                    0.0,
                    1.0,
                )
            )
            candidates.append(
                PoseEstimateCandidate(
                    source_frame_index,
                    target_frame_index,
                    PoseModelType.DEPTH_PNP,
                    pnp_transform,
                    True,
                    True,
                    TranslationScaleStatus.DEPTH_RELATIVE,
                    tracks.match_count,
                    pnp_count,
                    pnp_ratio,
                    tracks.median_parallax,
                    pnp_error,
                    quality,
                    True,
                    selected_reference_frame=source_frame_index,
                    evidence_source="foreground_filtered_background_depth_pnp",
                    metadata={
                        "source_depth_scale_status": (
                            None if source_depth is None else source_depth.scale_status.value
                        ),
                    },
                )
            )
        if not candidates:
            candidates.append(
                PoseEstimateCandidate.missing(
                    source_frame_index,
                    target_frame_index,
                    regime.missing_reason or "no_pose_model_passed",
                    support_count=tracks.match_count,
                    metadata={"motion_regime": regime.regime.value},
                )
            )
        valid_candidates = [
            candidate
            for candidate in candidates
            if candidate.valid and candidate.quality >= self.minimum_quality
        ]
        if regime.regime == CameraMotionRegime.STATIC_CAMERA:
            preference = {
                PoseModelType.STATIC_IDENTITY: 4,
                PoseModelType.DEPTH_PNP: 3,
                PoseModelType.ROTATION_HOMOGRAPHY: 2,
                PoseModelType.ESSENTIAL_SE3: 1,
            }
        elif regime.regime == CameraMotionRegime.ROTATION_DOMINANT:
            # Low-parallax homography evidence cannot support translation.  A
            # PnP fit in this regime is retained as a diagnostic candidate but
            # must not silently upgrade rotation-only motion to full SE3.
            preference = {
                PoseModelType.ROTATION_HOMOGRAPHY: 4,
                PoseModelType.DEPTH_PNP: 3,
                PoseModelType.ESSENTIAL_SE3: 2,
                PoseModelType.STATIC_IDENTITY: 1,
            }
        else:
            preference = {
                PoseModelType.DEPTH_PNP: 4,
                PoseModelType.ESSENTIAL_SE3: 3,
                PoseModelType.ROTATION_HOMOGRAPHY: 2,
                PoseModelType.STATIC_IDENTITY: 1,
            }
        selected = (
            max(
                valid_candidates,
                key=lambda item: (preference.get(item.pose_model_type, 0), item.quality),
            )
            if valid_candidates
            else PoseEstimateCandidate.missing(
                source_frame_index,
                target_frame_index,
                "all_pose_candidates_below_quality_threshold",
                support_count=tracks.match_count,
                metadata={
                    "candidate_rejections": [
                        {
                            "model": candidate.pose_model_type.value,
                            "valid": candidate.valid,
                            "quality": candidate.quality,
                            "reason": candidate.missing_reason or "quality_below_threshold",
                        }
                        for candidate in candidates
                    ]
                },
            )
        )
        return PosePairEstimation(regime, tuple(candidates), selected, tracks)


def estimate_adaptive_pose_candidates(
    images: Mapping[int, np.ndarray],
    K: np.ndarray,
    *,
    frame_indices: Sequence[int],
    foreground_masks: Mapping[int, np.ndarray] | None = None,
    depths: Mapping[int, DepthObservation] | None = None,
    scene_cut_flags: Mapping[int, bool] | None = None,
    temporal_strides: Sequence[int] = (1, 2, 4),
    estimator: LayeredPoseEstimator | None = None,
) -> tuple[PosePairEstimation, ...]:
    """Estimate t-1/t-2/t-4 candidate edges without crossing scene cuts."""

    ordered = tuple(int(index) for index in frame_indices)
    positions = {index: position for position, index in enumerate(ordered)}
    cuts = dict(scene_cut_flags or {})
    pose_estimator = estimator or LayeredPoseEstimator()
    results: list[PosePairEstimation] = []
    for target_position, target_index in enumerate(ordered[1:], start=1):
        for stride in sorted(set(int(value) for value in temporal_strides if value > 0)):
            source_position = target_position - stride
            if source_position < 0:
                continue
            source_index = ordered[source_position]
            if any(cuts.get(ordered[pos], False) for pos in range(source_position + 1, target_position + 1)):
                continue
            results.append(
                pose_estimator.estimate_pair(
                    images[source_index],
                    images[target_index],
                    K,
                    source_frame_index=source_index,
                    target_frame_index=target_index,
                    source_foreground_mask=(
                        None if foreground_masks is None else foreground_masks.get(source_index)
                    ),
                    target_foreground_mask=(
                        None if foreground_masks is None else foreground_masks.get(target_index)
                    ),
                    source_depth=None if depths is None else depths.get(source_index),
                )
            )
    return tuple(results)
