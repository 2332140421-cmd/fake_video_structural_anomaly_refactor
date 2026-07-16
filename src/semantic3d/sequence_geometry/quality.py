"""Sequence geometry quality summaries, never forged-video anomaly scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .observation import Shared3DClipObservation


def _mean_or_nan(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


@dataclass(frozen=True)
class SequenceGeometryQuality:
    """Engineering quality of sequence geometry, not a probability or anomaly."""

    valid_pose_ratio: float
    mean_pose_quality: float
    background_support_ratio: float
    background_inlier_ratio: float
    mean_background_reprojection_error: float
    depth_alignment_valid_ratio: float
    mean_depth_alignment_error: float
    scale_drift: float
    scene_cut_count: int
    valid_shared_3d_frame_ratio: float
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "valid_pose_ratio",
            "background_support_ratio",
            "background_inlier_ratio",
            "depth_alignment_valid_ratio",
            "valid_shared_3d_frame_ratio",
            "quality",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        for name in (
            "mean_pose_quality",
            "mean_background_reprojection_error",
            "mean_depth_alignment_error",
            "scale_drift",
        ):
            value = float(getattr(self, name))
            if not (math.isfinite(value) or math.isnan(value)):
                raise ValueError(f"{name} must be finite or NaN.")
            object.__setattr__(self, name, value)
        if self.scene_cut_count < 0:
            raise ValueError("scene_cut_count must be non-negative.")
        if self.valid and self.missing_reason:
            raise ValueError("Valid quality record cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid quality record requires missing_reason.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_clip(cls, clip: Shared3DClipObservation) -> "SequenceGeometryQuality":
        """Compute reproducible geometry-quality diagnostics from one clip."""

        poses = list(clip.relative_poses)
        transition_poses = [pose for pose in poses if pose.source_frame_index is not None]
        alignments = list(clip.depth_alignment_observations)
        valid_poses = [pose for pose in transition_poses if pose.valid]
        valid_alignments = [item for item in alignments if item.valid]
        pose_ratio = (
            len(valid_poses) / len(transition_poses)
            if transition_poses
            else 1.0
        )
        alignment_ratio = (
            len(valid_alignments) / len(alignments)
            if alignments
            else (1.0 if len(clip.frames) == 1 else 0.0)
        )
        support_ratios = [
            float(pose.metadata.get("background_support_ratio", math.nan))
            for pose in transition_poses
        ]
        mean_pose_quality = _mean_or_nan(
            [pose.pose_quality for pose in transition_poses]
        )
        background_support_ratio = _mean_or_nan(support_ratios)
        if not math.isfinite(background_support_ratio):
            background_support_ratio = 0.0
        background_inlier_ratio = _mean_or_nan(
            [pose.background_inlier_ratio for pose in transition_poses]
        )
        if not math.isfinite(background_inlier_ratio):
            background_inlier_ratio = 0.0
        reprojection_error = _mean_or_nan(
            [pose.reprojection_error for pose in transition_poses if pose.valid]
        )
        alignment_error = _mean_or_nan(
            [item.fitting_error for item in valid_alignments]
        )
        positive_scales = [item.scale for item in valid_alignments if item.scale > 0.0]
        scale_drift = (
            float(np.std(np.log(positive_scales)))
            if len(positive_scales) >= 2
            else float("nan")
        )
        cut_count = sum(bool(value) for value in clip.scene_cut_flags.values())
        frame_ratio = sum(frame.valid for frame in clip.frames) / len(clip.frames)
        normalized_reprojection_quality = (
            1.0 / (1.0 + reprojection_error)
            if math.isfinite(reprojection_error)
            else 0.0
        )
        normalized_alignment_quality = (
            1.0 / (1.0 + alignment_error)
            if math.isfinite(alignment_error)
            else (1.0 if not alignments and len(clip.frames) == 1 else 0.0)
        )
        quality = float(
            np.mean(
                [
                    pose_ratio,
                    mean_pose_quality if math.isfinite(mean_pose_quality) else 0.0,
                    background_support_ratio,
                    background_inlier_ratio,
                    normalized_reprojection_quality,
                    alignment_ratio,
                    normalized_alignment_quality,
                    frame_ratio,
                ]
            )
        )
        valid = bool(clip.frames and any(frame.valid for frame in clip.frames))
        return cls(
            valid_pose_ratio=pose_ratio,
            mean_pose_quality=mean_pose_quality,
            background_support_ratio=background_support_ratio,
            background_inlier_ratio=background_inlier_ratio,
            mean_background_reprojection_error=reprojection_error,
            depth_alignment_valid_ratio=alignment_ratio,
            mean_depth_alignment_error=alignment_error,
            scale_drift=scale_drift,
            scene_cut_count=cut_count,
            valid_shared_3d_frame_ratio=frame_ratio,
            valid=valid,
            quality=quality,
            missing_reason="" if valid else "no_valid_shared_3d_frames",
            metadata={
                "quality_is_probability": False,
                "anomaly_score": False,
                "pose_count": len(transition_poses),
                "valid_pose_count": len(valid_poses),
                "reference_pose_valid": bool(poses and poses[0].valid),
                "alignment_count": len(alignments),
                "valid_alignment_count": len(valid_alignments),
                "sequence_scale_status": clip.sequence_scale_status.value,
                "allows_dynamic_3d": clip.allows_dynamic_3d,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly record while preserving NaN values."""

        return {
            "valid_pose_ratio": self.valid_pose_ratio,
            "mean_pose_quality": self.mean_pose_quality,
            "background_support_ratio": self.background_support_ratio,
            "background_inlier_ratio": self.background_inlier_ratio,
            "mean_background_reprojection_error": self.mean_background_reprojection_error,
            "depth_alignment_valid_ratio": self.depth_alignment_valid_ratio,
            "mean_depth_alignment_error": self.mean_depth_alignment_error,
            "scale_drift": self.scale_drift,
            "scene_cut_count": self.scene_cut_count,
            "valid_shared_3d_frame_ratio": self.valid_shared_3d_frame_ratio,
            "valid": self.valid,
            "quality": self.quality,
            "missing_reason": self.missing_reason,
            "metadata": dict(self.metadata),
        }
