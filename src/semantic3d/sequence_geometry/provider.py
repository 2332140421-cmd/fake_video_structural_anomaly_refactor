"""Canonical sequence geometry providers, foreground exclusion, and scene cuts."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..depth_provider import DepthScaleStatus
from ..geometry.camera import CameraObservation, validate_rigid_transform
from ..geometry.transforms import camera_to_world
from ..shared_3d_observation import Object3DObservation, Shared3DFrameObservation
from .observation import (
    DepthAlignmentMode,
    DepthAlignmentObservation,
    RelativePoseObservation,
    SequenceScaleStatus,
    Shared3DClipObservation,
)


@dataclass(frozen=True)
class ForegroundMaskObservation:
    """Foreground exclusion mask and provenance used by background geometry."""

    frame_index: int
    mask: np.ndarray
    excluded_foreground_ratio: float
    object_ids: tuple[str, ...]
    mask_object_count: int
    bbox_fallback_count: int
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mask = np.asarray(self.mask, dtype=bool)
        ratio = float(self.excluded_foreground_ratio)
        quality = float(self.quality)
        if mask.ndim != 2:
            raise ValueError("Foreground mask must be HxW.")
        if not 0.0 <= ratio <= 1.0 or not math.isfinite(ratio):
            raise ValueError("excluded_foreground_ratio must be in [0, 1].")
        if not 0.0 <= quality <= 1.0 or not math.isfinite(quality):
            raise ValueError("foreground mask quality must be in [0, 1].")
        if self.valid and self.missing_reason:
            raise ValueError("Valid foreground mask cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid foreground mask requires missing_reason.")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "excluded_foreground_ratio", ratio)
        object.__setattr__(self, "object_ids", tuple(self.object_ids))
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


def build_foreground_mask(
    frame: Shared3DFrameObservation,
    *,
    bbox_dilation_pixels: int = 8,
    maximum_bbox_area_ratio: float = 0.80,
) -> ForegroundMaskObservation:
    """Build conservative foreground exclusion using masks or dilated bbox fallback."""

    if bbox_dilation_pixels < 0:
        raise ValueError("bbox_dilation_pixels must be non-negative.")
    if not 0.0 < maximum_bbox_area_ratio <= 1.0:
        raise ValueError("maximum_bbox_area_ratio must be in (0, 1].")
    height, width = frame.image_height, frame.image_width
    combined = np.zeros((height, width), dtype=bool)
    mask_count = 0
    bbox_count = 0
    object_ids: list[str] = []
    source_qualities: list[float] = []
    skipped_oversized_bbox_ids: list[str] = []
    for obj in frame.objects:
        object_id = obj.source_object_2d_id
        mask_path = obj.metadata.get("source_mask_path")
        object_mask: Optional[np.ndarray] = None
        if mask_path:
            loaded = cv2.imread(str(Path(str(mask_path))), cv2.IMREAD_GRAYSCALE)
            if loaded is not None and loaded.shape == (height, width):
                object_mask = loaded > 0
        if object_mask is not None and np.any(object_mask):
            combined |= object_mask
            mask_count += 1
            object_ids.append(object_id)
            source_qualities.append(1.0)
            continue
        bbox = obj.metadata.get("source_bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        unclipped_width = max(0, min(width, x2) - max(0, x1))
        unclipped_height = max(0, min(height, y2) - max(0, y1))
        bbox_area_ratio = (unclipped_width * unclipped_height) / float(width * height)
        if bbox_area_ratio > maximum_bbox_area_ratio:
            skipped_oversized_bbox_ids.append(object_id)
            continue
        x1 = max(0, x1 - bbox_dilation_pixels)
        y1 = max(0, y1 - bbox_dilation_pixels)
        x2 = min(width - 1, x2 + bbox_dilation_pixels)
        y2 = min(height - 1, y2 + bbox_dilation_pixels)
        if x2 <= x1 or y2 <= y1:
            continue
        combined[y1 : y2 + 1, x1 : x2 + 1] = True
        bbox_count += 1
        object_ids.append(object_id)
        source_qualities.append(0.35)
    ratio = float(np.mean(combined))
    quality = float(np.mean(source_qualities)) if source_qualities else 1.0
    return ForegroundMaskObservation(
        frame_index=frame.frame_index,
        mask=combined,
        excluded_foreground_ratio=ratio,
        object_ids=tuple(object_ids),
        mask_object_count=mask_count,
        bbox_fallback_count=bbox_count,
        quality=quality,
        valid=True,
        metadata={
            "bbox_dilation_pixels": bbox_dilation_pixels,
            "maximum_bbox_area_ratio": maximum_bbox_area_ratio,
            "bbox_is_low_quality_foreground_fallback": bbox_count > 0,
            "all_detected_objects_excluded_from_background_pose": not skipped_oversized_bbox_ids,
            "skipped_oversized_bbox_ids": skipped_oversized_bbox_ids,
            "oversized_bbox_reason": (
                "bbox_fallback_covering_most_of_frame_is_not_reliable_dynamic_foreground"
                if skipped_oversized_bbox_ids
                else ""
            ),
        },
    )


@dataclass(frozen=True)
class SceneCutDecision:
    """One scene-cut decision with independent diagnostic channels."""

    source_frame_index: int
    target_frame_index: int
    is_cut: bool
    histogram_difference: float
    content_difference: float
    source_feature_count: int
    target_feature_count: int
    feature_match_count: int
    provider_reported_cut: bool
    quality: float
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class BaseSceneCutDetector(ABC):
    """Interface for scene-cut decisions between adjacent frames."""

    @abstractmethod
    def detect(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        *,
        source_frame_index: int,
        target_frame_index: int,
        provider_reported_cut: bool = False,
    ) -> SceneCutDecision:
        """Return a cut decision without crossing the detected boundary."""


class HistogramFeatureSceneCutDetector(BaseSceneCutDetector):
    """Conservative histogram/content/ORB collapse scene-cut detector."""

    def __init__(
        self,
        histogram_threshold: float = 0.35,
        content_threshold: float = 0.45,
        feature_collapse_ratio: float = 0.08,
        minimum_source_features: int = 30,
    ) -> None:
        self.histogram_threshold = float(histogram_threshold)
        self.content_threshold = float(content_threshold)
        self.feature_collapse_ratio = float(feature_collapse_ratio)
        self.minimum_source_features = int(minimum_source_features)

    @staticmethod
    def _histogram(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        return cv2.normalize(histogram, histogram).flatten()

    def detect(
        self,
        source_image: np.ndarray,
        target_image: np.ndarray,
        *,
        source_frame_index: int,
        target_frame_index: int,
        provider_reported_cut: bool = False,
    ) -> SceneCutDecision:
        if source_image.shape != target_image.shape:
            return SceneCutDecision(
                source_frame_index,
                target_frame_index,
                True,
                1.0,
                1.0,
                0,
                0,
                0,
                provider_reported_cut,
                1.0,
                "frame_shape_change",
            )
        correlation = float(
            cv2.compareHist(self._histogram(source_image), self._histogram(target_image), cv2.HISTCMP_CORREL)
        )
        histogram_difference = float(np.clip((1.0 - correlation) / 2.0, 0.0, 1.0))
        content_difference = float(
            np.mean(
                np.abs(source_image.astype(np.float32) - target_image.astype(np.float32))
            )
            / 255.0
        )
        orb = cv2.ORB_create(nfeatures=600)
        source_keypoints, source_descriptors = orb.detectAndCompute(source_image, None)
        target_keypoints, target_descriptors = orb.detectAndCompute(target_image, None)
        matches = []
        if source_descriptors is not None and target_descriptors is not None:
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
                source_descriptors, target_descriptors
            )
        source_count = len(source_keypoints)
        target_count = len(target_keypoints)
        match_count = len(matches)
        collapse = bool(
            source_count >= self.minimum_source_features
            and match_count / max(source_count, 1) < self.feature_collapse_ratio
        )
        appearance_cut = bool(
            histogram_difference >= self.histogram_threshold
            and content_difference >= self.content_threshold
        )
        is_cut = bool(provider_reported_cut or appearance_cut or (collapse and content_difference > 0.2))
        if provider_reported_cut:
            reason = "provider_reported_cut"
        elif appearance_cut:
            reason = "histogram_content_discontinuity"
        elif collapse and is_cut:
            reason = "feature_matching_collapse"
        else:
            reason = "continuous_scene"
        quality = float(
            np.clip(max(histogram_difference, content_difference) if is_cut else 1.0 - content_difference, 0.0, 1.0)
        )
        return SceneCutDecision(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            is_cut=is_cut,
            histogram_difference=histogram_difference,
            content_difference=content_difference,
            source_feature_count=source_count,
            target_feature_count=target_count,
            feature_match_count=match_count,
            provider_reported_cut=provider_reported_cut,
            quality=quality,
            reason=reason,
            metadata={
                "histogram_threshold": self.histogram_threshold,
                "content_threshold": self.content_threshold,
                "feature_collapse_ratio": self.feature_collapse_ratio,
            },
        )


@dataclass(frozen=True)
class BackgroundPoseEstimate:
    """OpenCV background pose estimate before conversion to clip coordinates."""

    source_frame_index: int
    target_frame_index: int
    T_current_from_previous: Optional[np.ndarray]
    support_count: int
    inlier_ratio: float
    reprojection_error: float
    quality: float
    valid: bool
    missing_reason: str
    is_static_identity: bool
    track_rows: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def estimate_background_relative_pose(
    source_image: np.ndarray,
    target_image: np.ndarray,
    K: np.ndarray,
    *,
    source_frame_index: int,
    target_frame_index: int,
    source_foreground_mask: Optional[np.ndarray] = None,
    target_foreground_mask: Optional[np.ndarray] = None,
    minimum_support: int = 12,
    static_flow_threshold_px: float = 0.35,
) -> BackgroundPoseEstimate:
    """Estimate scale-free relative pose from LK background correspondences."""

    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)
    feature_mask = np.full(source_gray.shape, 255, dtype=np.uint8)
    if source_foreground_mask is not None:
        feature_mask[np.asarray(source_foreground_mask, dtype=bool)] = 0
    source_points = cv2.goodFeaturesToTrack(
        source_gray,
        maxCorners=1000,
        qualityLevel=0.01,
        minDistance=7,
        mask=feature_mask,
        blockSize=7,
    )
    candidate_count = 0 if source_points is None else int(source_points.shape[0])
    if source_points is None or candidate_count < minimum_support:
        return BackgroundPoseEstimate(
            source_frame_index,
            target_frame_index,
            None,
            candidate_count,
            0.0,
            float("nan"),
            0.0,
            False,
            "insufficient_background_support",
            False,
            (),
            {"candidate_support_count": candidate_count},
        )
    target_points, status, _ = cv2.calcOpticalFlowPyrLK(
        source_gray,
        target_gray,
        source_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if target_points is None or status is None:
        return BackgroundPoseEstimate(
            source_frame_index,
            target_frame_index,
            None,
            candidate_count,
            0.0,
            float("nan"),
            0.0,
            False,
            "background_optical_flow_failed",
            False,
            (),
            {"candidate_support_count": candidate_count},
        )
    source_xy = source_points.reshape(-1, 2)
    target_xy = target_points.reshape(-1, 2)
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        target_gray,
        source_gray,
        target_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    keep = status.reshape(-1).astype(bool)
    if backward_points is None or backward_status is None:
        keep[:] = False
        forward_backward_error = np.full(source_xy.shape[0], np.inf, dtype=float)
    else:
        backward_xy = backward_points.reshape(-1, 2)
        forward_backward_error = np.linalg.norm(backward_xy - source_xy, axis=1)
        keep &= backward_status.reshape(-1).astype(bool)
        keep &= forward_backward_error <= 1.5
    height, width = source_gray.shape
    target_columns = np.floor(target_xy[:, 0] + 0.5).astype(int)
    target_rows = np.floor(target_xy[:, 1] + 0.5).astype(int)
    keep &= (
        (target_columns >= 0)
        & (target_columns < width)
        & (target_rows >= 0)
        & (target_rows < height)
    )
    if target_foreground_mask is not None:
        valid_indices = np.flatnonzero(keep)
        if valid_indices.size:
            foreground = np.asarray(target_foreground_mask, dtype=bool)
            keep[valid_indices] &= ~foreground[
                target_rows[valid_indices], target_columns[valid_indices]
            ]
    source_xy, target_xy = source_xy[keep], target_xy[keep]
    support_count = int(source_xy.shape[0])
    support_ratio = support_count / max(candidate_count, 1)
    if support_count < minimum_support:
        return BackgroundPoseEstimate(
            source_frame_index,
            target_frame_index,
            None,
            support_count,
            0.0,
            float("nan"),
            0.0,
            False,
            "insufficient_background_support_after_foreground_exclusion",
            False,
            (),
            {
                "candidate_support_count": candidate_count,
                "background_support_ratio": support_ratio,
                "forward_backward_consistency_threshold_px": 1.5,
            },
        )
    flow = np.linalg.norm(target_xy - source_xy, axis=1)
    median_flow = float(np.median(flow))
    if median_flow <= static_flow_threshold_px:
        rows = tuple(
            {
                "track_id": f"bg_{source_frame_index}_{index}",
                "source_frame_index": source_frame_index,
                "target_frame_index": target_frame_index,
                "source_x": float(source_xy[index, 0]),
                "source_y": float(source_xy[index, 1]),
                "target_x": float(target_xy[index, 0]),
                "target_y": float(target_xy[index, 1]),
                "inlier": True,
                "reprojection_error": float(flow[index]),
            }
            for index in range(support_count)
        )
        quality = float(np.clip(support_ratio / (1.0 + median_flow), 0.0, 1.0))
        return BackgroundPoseEstimate(
            source_frame_index,
            target_frame_index,
            np.eye(4),
            support_count,
            1.0,
            median_flow,
            quality,
            True,
            "",
            True,
            rows,
            {
                "candidate_support_count": candidate_count,
                "background_support_ratio": support_ratio,
                "median_background_flow_px": median_flow,
                "identity_evidence": "low_background_optical_flow",
                "translation_scale_status": "identity",
                "foreground_excluded": True,
                "forward_backward_consistency_threshold_px": 1.5,
            },
        )
    essential, essential_mask = cv2.findEssentialMat(
        source_xy,
        target_xy,
        np.asarray(K, dtype=float),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.5,
    )
    if essential is None or essential_mask is None:
        return BackgroundPoseEstimate(
            source_frame_index,
            target_frame_index,
            None,
            support_count,
            0.0,
            float("nan"),
            0.0,
            False,
            "essential_matrix_estimation_failed",
            False,
            (),
            {
                "candidate_support_count": candidate_count,
                "background_support_ratio": support_ratio,
            },
        )
    if essential.shape[0] > 3:
        essential = essential[:3, :3]
    _, rotation, translation, recovered_mask = cv2.recoverPose(
        essential, source_xy, target_xy, np.asarray(K, dtype=float), mask=essential_mask
    )
    inliers = recovered_mask.reshape(-1).astype(bool)
    inlier_count = int(np.sum(inliers))
    inlier_ratio = inlier_count / support_count
    if inlier_count < minimum_support:
        return BackgroundPoseEstimate(
            source_frame_index,
            target_frame_index,
            None,
            support_count,
            inlier_ratio,
            float("nan"),
            0.0,
            False,
            "insufficient_pose_inliers",
            False,
            (),
            {
                "candidate_support_count": candidate_count,
                "background_support_ratio": support_ratio,
                "inlier_count": inlier_count,
                "median_background_flow_px": median_flow,
                "forward_backward_consistency_threshold_px": 1.5,
            },
        )
    inverse_K = np.linalg.inv(np.asarray(K, dtype=float))
    fundamental = inverse_K.T @ essential @ inverse_K
    source_h = np.column_stack((source_xy, np.ones(support_count)))
    target_h = np.column_stack((target_xy, np.ones(support_count)))
    lines = (fundamental @ source_h.T).T
    denominators = np.maximum(np.linalg.norm(lines[:, :2], axis=1), 1e-12)
    epipolar_errors = np.abs(np.sum(target_h * lines, axis=1)) / denominators
    reprojection_error = float(np.median(epipolar_errors[inliers]))
    relative = np.eye(4, dtype=float)
    relative[:3, :3] = rotation
    relative[:3, 3] = translation.reshape(3)
    quality = float(
        np.clip(support_ratio * inlier_ratio / (1.0 + reprojection_error), 0.0, 1.0)
    )
    rows = tuple(
        {
            "track_id": f"bg_{source_frame_index}_{index}",
            "source_frame_index": source_frame_index,
            "target_frame_index": target_frame_index,
            "source_x": float(source_xy[index, 0]),
            "source_y": float(source_xy[index, 1]),
            "target_x": float(target_xy[index, 0]),
            "target_y": float(target_xy[index, 1]),
            "inlier": bool(inliers[index]),
            "reprojection_error": float(epipolar_errors[index]),
        }
        for index in range(support_count)
    )
    return BackgroundPoseEstimate(
        source_frame_index,
        target_frame_index,
        relative,
        support_count,
        inlier_ratio,
        reprojection_error,
        quality,
        True,
        "",
        False,
        rows,
        {
            "candidate_support_count": candidate_count,
            "background_support_ratio": support_ratio,
            "inlier_count": inlier_count,
            "median_background_flow_px": median_flow,
            "translation_scale_status": "monocular_direction_only",
            "foreground_excluded": True,
            "pose_model": "opencv_lk_essential_recover_pose",
            "forward_backward_consistency_threshold_px": 1.5,
        },
    )


def _attach_pose_to_frame(
    frame: Shared3DFrameObservation,
    pose: RelativePoseObservation,
) -> Shared3DFrameObservation:
    """Attach one validated pose and derive world points without changing camera points."""

    if not pose.valid:
        return frame
    assert frame.camera.K is not None
    camera = CameraObservation.from_parameters(
        K=frame.camera.K,
        image_width=frame.image_width,
        image_height=frame.image_height,
        intrinsics_source=frame.camera.intrinsics_source,
        quality=frame.camera.quality,
        distortion=frame.camera.distortion,
        T_world_from_camera=pose.T_world_from_camera,
        T_camera_from_world=pose.T_camera_from_world,
        pose_source=pose.pose_source,
        metadata={
            **dict(frame.camera.metadata),
            "sequence_pose_quality": pose.pose_quality,
            "sequence_pose_source": pose.pose_source,
        },
    )
    objects: list[Object3DObservation] = []
    for obj in frame.objects:
        if not obj.valid:
            objects.append(obj)
            continue
        center_world_points = camera_to_world((obj.center_3d_camera,), camera) if obj.center_3d_camera else ()
        boundary_world = tuple(point for point in camera_to_world(obj.boundary_points_3d, camera) if point.valid)
        keypoints_world = tuple(point for point in camera_to_world(obj.keypoints_3d, camera) if point.valid)
        structure_world = tuple(point for point in camera_to_world(obj.structure_points_3d, camera) if point.valid)
        objects.append(
            replace(
                obj,
                center_3d_world=(center_world_points[0] if center_world_points and center_world_points[0].valid else None),
                boundary_points_3d_world=boundary_world,
                keypoints_3d_world=keypoints_world,
                structure_points_3d_world=structure_world,
                metadata={
                    **dict(obj.metadata),
                    "sequence_pose_attached": True,
                    "sequence_pose_source": pose.pose_source,
                },
            )
        )
    return replace(
        frame,
        camera=camera,
        objects=tuple(objects),
        metadata={
            **dict(frame.metadata),
            "sequence_pose_attached": True,
            "sequence_pose_source": pose.pose_source,
        },
    )


def _sequence_scale_status(
    frames: Sequence[Shared3DFrameObservation],
    alignments: Sequence[DepthAlignmentObservation],
) -> SequenceScaleStatus:
    statuses = {frame.depth.scale_status for frame in frames}
    if statuses == {DepthScaleStatus.METRIC_CALIBRATED}:
        return SequenceScaleStatus.METRIC_SEQUENCE
    if statuses == {DepthScaleStatus.RELATIVE_SHARED_SEQUENCE}:
        return SequenceScaleStatus.RELATIVE_SHARED_SEQUENCE
    required = max(0, len(frames) - 1)
    if (
        statuses == {DepthScaleStatus.RELATIVE_PER_FRAME}
        and len(alignments) == required
        and all(item.valid for item in alignments)
    ):
        return SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE
    if statuses == {DepthScaleStatus.RELATIVE_PER_FRAME}:
        return SequenceScaleStatus.RELATIVE_PER_FRAME
    return SequenceScaleStatus.UNKNOWN


def _make_clip(
    *,
    video_id: str,
    clip_id: str,
    frames: Sequence[Shared3DFrameObservation],
    poses: Sequence[RelativePoseObservation],
    alignments: Sequence[DepthAlignmentObservation],
    scene_cut_flags: Mapping[int, bool],
    background_track_ids: Sequence[str],
    provider_name: str,
    pose_scale_compatible_with_depth: bool,
    valid: bool = True,
    missing_reason: str = "",
    metadata: Optional[Mapping[str, Any]] = None,
) -> Shared3DClipObservation:
    retained = tuple(frames)
    indices = tuple(frame.frame_index for frame in retained)
    pose_records = tuple(poses)
    attached = tuple(
        _attach_pose_to_frame(frame, pose)
        for frame, pose in zip(retained, pose_records, strict=True)
    )
    twc = {
        pose.target_frame_index: pose.T_world_from_camera if pose.valid else None
        for pose in pose_records
    }
    tcw = {
        pose.target_frame_index: pose.T_camera_from_world if pose.valid else None
        for pose in pose_records
    }
    scale_status = _sequence_scale_status(attached, alignments)
    foreground_ids = tuple(
        dict.fromkeys(
            obj.source_object_2d_id for frame in attached for obj in frame.objects
        )
    )
    pose_ratio = sum(pose.valid for pose in pose_records) / len(pose_records)
    alignment_ratio = (
        sum(item.valid for item in alignments) / len(alignments)
        if alignments
        else (1.0 if len(attached) == 1 else 0.0)
    )
    frame_ratio = sum(frame.valid for frame in attached) / len(attached)
    quality = float(np.mean([pose_ratio, alignment_ratio, frame_ratio]))
    return Shared3DClipObservation(
        video_id=video_id,
        clip_id=clip_id,
        frame_indices=indices,
        frames=attached,
        reference_frame_index=indices[0],
        T_world_from_camera_by_frame=twc,
        T_camera_from_world_by_frame=tcw,
        relative_poses=pose_records,
        sequence_scale_status=scale_status,
        depth_alignment_observations=tuple(alignments),
        scene_cut_flags=dict(scene_cut_flags),
        background_track_ids=tuple(dict.fromkeys(background_track_ids)),
        foreground_object_ids=foreground_ids,
        provider_name=provider_name,
        valid=valid,
        quality=quality,
        missing_reason=missing_reason,
        metadata={
            **dict(metadata or {}),
            "pose_scale_compatible_with_depth": pose_scale_compatible_with_depth,
            "sequence_geometry_is_anomaly_score": False,
            "dynamic_residuals_implemented": False,
        },
    )


class BaseSequenceGeometryProvider(ABC):
    """Canonical provider interface for one sequence-level shared geometry clip."""

    @abstractmethod
    def predict_clip(
        self,
        frames: Sequence[Shared3DFrameObservation],
        frame_indices: Sequence[int],
        foreground_masks: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Shared3DClipObservation:
        """Return sequence geometry without producing anomaly residuals."""


class SyntheticSequenceGeometryProvider(BaseSequenceGeometryProvider):
    """Ground-truth provider restricted to deterministic synthetic tests."""

    def __init__(
        self,
        T_world_from_camera_by_frame: Mapping[int, np.ndarray],
        *,
        depth_alignments: Sequence[DepthAlignmentObservation] = (),
        scene_cut_flags: Optional[Mapping[int, bool]] = None,
        background_track_ids: Sequence[str] = (),
        reprojection_error_by_frame: Optional[Mapping[int, float]] = None,
        pose_scale_compatible_with_depth: bool = True,
        video_id: str = "synthetic_video",
        clip_id: str = "synthetic_clip",
    ) -> None:
        self.pose_map = {
            int(index): validate_rigid_transform(matrix, f"synthetic_pose[{index}]")
            for index, matrix in T_world_from_camera_by_frame.items()
        }
        self.depth_alignments = tuple(depth_alignments)
        self.scene_cut_flags = dict(scene_cut_flags or {})
        self.background_track_ids = tuple(background_track_ids)
        self.reprojection_error_by_frame = dict(reprojection_error_by_frame or {})
        self.pose_scale_compatible = bool(pose_scale_compatible_with_depth)
        self.video_id = video_id
        self.clip_id = clip_id

    def predict_clip(
        self,
        frames: Sequence[Shared3DFrameObservation],
        frame_indices: Sequence[int],
        foreground_masks: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Shared3DClipObservation:
        del foreground_masks
        if tuple(frame.frame_index for frame in frames) != tuple(frame_indices):
            raise ValueError("frames and frame_indices must be aligned.")
        retained = list(frames)
        for position, frame in enumerate(frames[1:], start=1):
            if self.scene_cut_flags.get(frame.frame_index, False):
                retained = list(frames[:position])
                break
        poses: list[RelativePoseObservation] = []
        previous_index: Optional[int] = None
        for frame in retained:
            index = frame.frame_index
            twc = self.pose_map.get(index)
            if twc is None:
                poses.append(
                    RelativePoseObservation.missing(
                        index,
                        "missing_synthetic_pose",
                        source_frame_index=previous_index,
                        pose_source="synthetic_ground_truth",
                    )
                )
                previous_index = index
                continue
            if previous_index is None:
                relative = np.eye(4)
            else:
                previous_twc = self.pose_map.get(previous_index)
                relative = (
                    np.linalg.inv(twc) @ previous_twc
                    if previous_twc is not None
                    else np.eye(4)
                )
            poses.append(
                RelativePoseObservation.from_transforms(
                    source_frame_index=previous_index,
                    target_frame_index=index,
                    T_world_from_camera=twc,
                    relative_pose_from_previous=relative,
                    pose_source="synthetic_ground_truth",
                    pose_quality=1.0,
                    background_support_count=max(1, len(self.background_track_ids)),
                    background_inlier_ratio=1.0,
                    reprojection_error=float(self.reprojection_error_by_frame.get(index, 0.0)),
                    metadata={
                        "synthetic_only": True,
                        "background_support_ratio": 1.0,
                        "reference_coordinate_gauge": previous_index is None,
                        "identity_evidence": "synthetic_ground_truth",
                    },
                )
            )
            previous_index = index
        retained_indices = {frame.frame_index for frame in retained}
        alignments = tuple(
            item
            for item in self.depth_alignments
            if item.source_frame in retained_indices and item.target_frame in retained_indices
        )
        return _make_clip(
            video_id=self.video_id,
            clip_id=self.clip_id,
            frames=retained,
            poses=poses,
            alignments=alignments,
            scene_cut_flags=self.scene_cut_flags,
            background_track_ids=self.background_track_ids,
            provider_name="synthetic_sequence_geometry",
            pose_scale_compatible_with_depth=self.pose_scale_compatible,
            metadata={
                "synthetic_only": True,
                "terminated_at_scene_cut": len(retained) < len(frames),
            },
        )


class MockSequenceGeometryProvider(BaseSequenceGeometryProvider):
    """Test placeholder that explicitly returns missing pose/alignment evidence."""

    def predict_clip(
        self,
        frames: Sequence[Shared3DFrameObservation],
        frame_indices: Sequence[int],
        foreground_masks: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Shared3DClipObservation:
        del foreground_masks
        if tuple(frame.frame_index for frame in frames) != tuple(frame_indices):
            raise ValueError("frames and frame_indices must be aligned.")
        poses = tuple(
            RelativePoseObservation.missing(
                frame.frame_index,
                "mock_provider_has_no_pose",
                source_frame_index=(frames[index - 1].frame_index if index else None),
                pose_source="mock_sequence_geometry",
            )
            for index, frame in enumerate(frames)
        )
        alignments = tuple(
            DepthAlignmentObservation.missing(
                frames[index - 1].frame_index,
                frame.frame_index,
                "mock_provider_has_no_depth_alignment",
            )
            for index, frame in enumerate(frames)
            if index
        )
        return _make_clip(
            video_id=frames[0].video_id,
            clip_id="mock_clip",
            frames=frames,
            poses=poses,
            alignments=alignments,
            scene_cut_flags={},
            background_track_ids=(),
            provider_name="mock_sequence_geometry",
            pose_scale_compatible_with_depth=False,
            valid=False,
            missing_reason="mock_sequence_geometry_has_no_real_pose",
            metadata={"mock_only": True},
        )


class LegacyDepthPoseSequenceAdapter(BaseSequenceGeometryProvider):
    """Adapt canonical frame depth and external pose/alignment into the clip contract."""

    def __init__(
        self,
        *,
        T_world_from_camera_by_frame: Mapping[int, np.ndarray],
        relative_pose_observations: Optional[Mapping[int, RelativePoseObservation]] = None,
        depth_alignments: Sequence[DepthAlignmentObservation] = (),
        scene_cut_flags: Optional[Mapping[int, bool]] = None,
        background_track_ids: Sequence[str] = (),
        pose_source: str = "external_pose",
        pose_scale_compatible_with_depth: bool = False,
        video_id: Optional[str] = None,
        clip_id: str = "legacy_sequence_clip",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.pose_map = {
            int(index): validate_rigid_transform(matrix, f"external_pose[{index}]")
            for index, matrix in T_world_from_camera_by_frame.items()
        }
        self.pose_observations = dict(relative_pose_observations or {})
        self.alignments = tuple(depth_alignments)
        self.scene_cut_flags = dict(scene_cut_flags or {})
        self.background_track_ids = tuple(background_track_ids)
        self.pose_source = pose_source
        self.pose_scale_compatible = bool(pose_scale_compatible_with_depth)
        self.video_id = video_id
        self.clip_id = clip_id
        self.metadata = dict(metadata or {})

    def predict_clip(
        self,
        frames: Sequence[Shared3DFrameObservation],
        frame_indices: Sequence[int],
        foreground_masks: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Shared3DClipObservation:
        del foreground_masks
        if tuple(frame.frame_index for frame in frames) != tuple(frame_indices):
            raise ValueError("frames and frame_indices must be aligned.")
        retained = list(frames)
        for position, frame in enumerate(frames[1:], start=1):
            if self.scene_cut_flags.get(frame.frame_index, False):
                retained = list(frames[:position])
                break
        poses: list[RelativePoseObservation] = []
        previous_index: Optional[int] = None
        for frame in retained:
            index = frame.frame_index
            supplied = self.pose_observations.get(index)
            if supplied is not None:
                poses.append(supplied)
            else:
                twc = self.pose_map.get(index)
                previous_twc = self.pose_map.get(previous_index) if previous_index is not None else None
                if twc is None or (previous_index is not None and previous_twc is None):
                    poses.append(
                        RelativePoseObservation.missing(
                            index,
                            "external_pose_missing",
                            source_frame_index=previous_index,
                            pose_source=self.pose_source,
                        )
                    )
                else:
                    relative = np.eye(4) if previous_index is None else np.linalg.inv(twc) @ previous_twc
                    poses.append(
                        RelativePoseObservation.from_transforms(
                            source_frame_index=previous_index,
                            target_frame_index=index,
                            T_world_from_camera=twc,
                            relative_pose_from_previous=relative,
                            pose_source=self.pose_source,
                            pose_quality=0.5,
                            background_support_count=1,
                            background_inlier_ratio=1.0,
                            reprojection_error=0.0,
                            metadata={
                                "adapted_external_pose": True,
                                "background_support_ratio": 1.0,
                                "reference_coordinate_gauge": previous_index is None,
                            },
                        )
                    )
            previous_index = index
        retained_indices = {frame.frame_index for frame in retained}
        alignments = tuple(
            item
            for item in self.alignments
            if item.source_frame in retained_indices and item.target_frame in retained_indices
        )
        return _make_clip(
            video_id=self.video_id or retained[0].video_id,
            clip_id=self.clip_id,
            frames=retained,
            poses=poses,
            alignments=alignments,
            scene_cut_flags=self.scene_cut_flags,
            background_track_ids=self.background_track_ids,
            provider_name="legacy_depth_pose_sequence_adapter",
            pose_scale_compatible_with_depth=self.pose_scale_compatible,
            metadata={
                **self.metadata,
                "legacy_adapter": True,
                "pose_source": self.pose_source,
                "terminated_at_scene_cut": len(retained) < len(frames),
            },
        )


class UnifiedSequenceGeometryProvider(BaseSequenceGeometryProvider, ABC):
    """Future interface for a model jointly producing K, pose, depth, and matches."""

    @abstractmethod
    def predict_clip(
        self,
        frames: Sequence[Shared3DFrameObservation],
        frame_indices: Sequence[int],
        foreground_masks: Optional[Mapping[int, np.ndarray]] = None,
    ) -> Shared3DClipObservation:
        """Return the same canonical clip contract as the legacy adapter."""
