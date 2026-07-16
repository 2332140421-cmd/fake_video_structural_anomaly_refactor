"""Evidence-aware operating-mode gate for dynamic 3D geometry.

The gate grants only the geometric operations supported by the measured
camera/depth evidence.  It is not an anomaly detector or a probability model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..sequence_geometry import SequenceScaleStatus, Shared3DClipObservation


class DynamicGeometryMode(str, Enum):
    """Dynamic geometry operations supported by one stabilized clip."""

    UNAVAILABLE = "unavailable"
    STATIC_CAMERA_3D = "static_camera_3d"
    ROTATION_COMPENSATED = "rotation_compensated"
    FULL_SE3_3D = "full_se3_3d"

    @property
    def allows_camera_frame_3d_tracks(self) -> bool:
        """Return whether shared-scale camera-frame points are meaningful."""

        return self in {
            DynamicGeometryMode.STATIC_CAMERA_3D,
            DynamicGeometryMode.FULL_SE3_3D,
        }

    @property
    def allows_rotation_compensation(self) -> bool:
        """Return whether at least bearing/ray rotation compensation is valid."""

        return self in {
            DynamicGeometryMode.STATIC_CAMERA_3D,
            DynamicGeometryMode.ROTATION_COMPENSATED,
            DynamicGeometryMode.FULL_SE3_3D,
        }

    @property
    def allows_world_3d(self) -> bool:
        """Return whether full camera-compensated world coordinates are valid."""

        return self == DynamicGeometryMode.FULL_SE3_3D


@dataclass(frozen=True)
class Dynamic3DReadinessThresholds:
    """Reproducible engineering smoke thresholds, not paper thresholds."""

    minimum_valid_shared_frame_ratio: float = 0.80
    minimum_pose_graph_connected_ratio: float = 0.80
    minimum_depth_alignment_valid_ratio: float = 0.80
    minimum_independent_track_coverage: float = 0.20
    minimum_mean_track_length: float = 3.0
    minimum_static_pose_ratio: float = 0.75
    maximum_rotation_ratio_for_static: float = 0.25
    minimum_rotation_supported_ratio: float = 0.80
    minimum_full_se3_ratio: float = 0.80
    minimum_reprojection_improvement: float = 0.05
    minimum_depth_stability_improvement: float = 0.0
    minimum_background_3d_stability_improvement: float = 0.0
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        ratio_fields = (
            "minimum_valid_shared_frame_ratio",
            "minimum_pose_graph_connected_ratio",
            "minimum_depth_alignment_valid_ratio",
            "minimum_independent_track_coverage",
            "minimum_static_pose_ratio",
            "maximum_rotation_ratio_for_static",
            "minimum_rotation_supported_ratio",
            "minimum_full_se3_ratio",
        )
        if any(not 0.0 <= float(getattr(self, name)) <= 1.0 for name in ratio_fields):
            raise ValueError("Readiness ratio thresholds must be in [0, 1].")
        if self.minimum_mean_track_length < 1.0 or self.epsilon <= 0.0:
            raise ValueError("Track length and epsilon thresholds must be positive.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Dynamic3DReadinessThresholds":
        """Load thresholds from a YAML mapping."""

        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        values = payload.get("dynamic_3d_readiness", payload)
        if not isinstance(values, Mapping):
            raise ValueError("dynamic_3d_readiness config must be a mapping.")
        return cls(**dict(values))


def relative_improvement(before: float, after: float, epsilon: float = 1e-8) -> float:
    """Return ``(before-after)/(before+epsilon)`` or NaN if incomparable."""

    before_value, after_value = float(before), float(after)
    if (
        not math.isfinite(before_value)
        or not math.isfinite(after_value)
        or before_value < 0.0
        or after_value < 0.0
    ):
        return float("nan")
    return float((before_value - after_value) / (before_value + epsilon))


@dataclass(frozen=True)
class Dynamic3DReadiness:
    """Clip-level geometric readiness and strictly limited operating mode."""

    video_id: str
    clip_id: str
    mode: DynamicGeometryMode | str
    dynamic_3d_ready: bool
    valid_shared_frame_ratio: float
    pose_graph_connected_ratio: float
    static_pose_ratio: float
    rotation_only_ratio: float
    full_se3_ratio: float
    depth_alignment_valid_ratio: float
    sequence_scale_status: SequenceScaleStatus | str
    independent_track_coverage: float
    mean_track_length: float
    reprojection_error_before: float
    reprojection_error_after: float
    reprojection_improvement: float
    depth_stability_before: float
    depth_stability_after: float
    depth_stability_improvement: float
    background_3d_stability_before: float
    background_3d_stability_after: float
    background_3d_stability_improvement: float
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DynamicGeometryMode(self.mode)
        scale_status = SequenceScaleStatus(self.sequence_scale_status)
        for name in (
            "valid_shared_frame_ratio",
            "pose_graph_connected_ratio",
            "static_pose_ratio",
            "rotation_only_ratio",
            "full_se3_ratio",
            "depth_alignment_valid_ratio",
            "independent_track_coverage",
            "quality",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        if not math.isfinite(float(self.mean_track_length)) or self.mean_track_length < 0.0:
            raise ValueError("mean_track_length must be finite and non-negative.")
        for name in (
            "reprojection_error_before",
            "reprojection_error_after",
            "reprojection_improvement",
            "depth_stability_before",
            "depth_stability_after",
            "depth_stability_improvement",
            "background_3d_stability_before",
            "background_3d_stability_after",
            "background_3d_stability_improvement",
        ):
            value = float(getattr(self, name))
            if not (math.isnan(value) or math.isfinite(value)):
                raise ValueError(f"{name} must be finite or NaN.")
            object.__setattr__(self, name, value)
        if self.dynamic_3d_ready:
            if not self.valid or mode == DynamicGeometryMode.UNAVAILABLE:
                raise ValueError("Ready dynamic geometry requires a valid non-unavailable mode.")
            if self.missing_reason:
                raise ValueError("Ready dynamic geometry cannot have missing_reason.")
        else:
            if mode != DynamicGeometryMode.UNAVAILABLE or not self.missing_reason:
                raise ValueError("Not-ready geometry requires unavailable mode and a reason.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "sequence_scale_status", scale_status)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def allows_world_3d(self) -> bool:
        """Return whether complete world-coordinate displacement is authorized."""

        return bool(self.dynamic_3d_ready and self.mode.allows_world_3d)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly readiness report."""

        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["sequence_scale_status"] = self.sequence_scale_status.value
        payload["allows_camera_frame_3d_tracks"] = bool(
            self.dynamic_3d_ready and self.mode.allows_camera_frame_3d_tracks
        )
        payload["allows_rotation_compensation"] = bool(
            self.dynamic_3d_ready and self.mode.allows_rotation_compensation
        )
        payload["allows_world_3d"] = self.allows_world_3d
        return payload


def _finite_min(values: list[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return min(finite) if finite else 0.0


def assess_dynamic_3d_readiness(
    shared_clip: Shared3DClipObservation,
    *,
    valid_shared_frame_ratio: float,
    pose_graph_connected_ratio: float,
    static_pose_ratio: float,
    rotation_only_ratio: float,
    full_se3_ratio: float,
    depth_alignment_valid_ratio: float,
    independent_track_coverage: float,
    mean_track_length: float,
    reprojection_error_before: float,
    reprojection_error_after: float,
    depth_stability_before: float,
    depth_stability_after: float,
    background_3d_stability_before: float,
    background_3d_stability_after: float,
    thresholds: Optional[Dynamic3DReadinessThresholds] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dynamic3DReadiness:
    """Assess a shared clip without re-estimating depth, intrinsics, or pose."""

    limits = thresholds or Dynamic3DReadinessThresholds()
    reprojection_gain = relative_improvement(
        reprojection_error_before, reprojection_error_after, limits.epsilon
    )
    depth_gain = relative_improvement(
        depth_stability_before, depth_stability_after, limits.epsilon
    )
    background_gain = relative_improvement(
        background_3d_stability_before,
        background_3d_stability_after,
        limits.epsilon,
    )
    checks = {
        "shared_frames": valid_shared_frame_ratio >= limits.minimum_valid_shared_frame_ratio,
        "pose_graph": pose_graph_connected_ratio >= limits.minimum_pose_graph_connected_ratio,
        "depth_graph": depth_alignment_valid_ratio >= limits.minimum_depth_alignment_valid_ratio,
        "sequence_scale": shared_clip.sequence_scale_status.allows_dynamic_3d,
        "independent_tracks": independent_track_coverage
        >= limits.minimum_independent_track_coverage,
        "track_length": mean_track_length >= limits.minimum_mean_track_length,
        "reprojection_improved": math.isfinite(reprojection_gain)
        and reprojection_gain >= limits.minimum_reprojection_improvement,
        "depth_stability_improved": math.isfinite(depth_gain)
        and depth_gain >= limits.minimum_depth_stability_improvement,
        "background_3d_improved": math.isfinite(background_gain)
        and background_gain >= limits.minimum_background_3d_stability_improvement,
        "no_scene_cut": not any(shared_clip.scene_cut_flags.values()),
    }
    full_support = full_se3_ratio >= limits.minimum_full_se3_ratio
    static_support = (
        static_pose_ratio >= limits.minimum_static_pose_ratio
        and rotation_only_ratio <= limits.maximum_rotation_ratio_for_static
    )
    rotation_support = (
        rotation_only_ratio > 0.0
        and static_pose_ratio + rotation_only_ratio
        >= limits.minimum_rotation_supported_ratio
    )
    base_ready = all(checks.values())
    if base_ready and full_support:
        mode = DynamicGeometryMode.FULL_SE3_3D
    elif base_ready and static_support:
        mode = DynamicGeometryMode.STATIC_CAMERA_3D
    elif base_ready and rotation_support:
        mode = DynamicGeometryMode.ROTATION_COMPENSATED
    else:
        mode = DynamicGeometryMode.UNAVAILABLE
    ready = mode != DynamicGeometryMode.UNAVAILABLE
    if ready:
        reason = ""
    else:
        failed = [name for name, passed in checks.items() if not passed]
        if not failed and not (full_support or static_support or rotation_support):
            failed = ["unsupported_pose_mix"]
        reason = "+".join(failed) or "dynamic_geometry_not_ready"
    quality_terms = [
        valid_shared_frame_ratio,
        pose_graph_connected_ratio,
        depth_alignment_valid_ratio,
        independent_track_coverage,
        min(mean_track_length / max(limits.minimum_mean_track_length, 1.0), 1.0),
        max(0.0, min(1.0, reprojection_gain)) if math.isfinite(reprojection_gain) else 0.0,
        max(0.0, min(1.0, depth_gain)) if math.isfinite(depth_gain) else 0.0,
        max(0.0, min(1.0, background_gain)) if math.isfinite(background_gain) else 0.0,
    ]
    quality = float(
        max(0.0, min(1.0, _finite_min(quality_terms) if ready else min(quality_terms)))
    )
    return Dynamic3DReadiness(
        video_id=shared_clip.video_id,
        clip_id=shared_clip.clip_id,
        mode=mode,
        dynamic_3d_ready=ready,
        valid_shared_frame_ratio=valid_shared_frame_ratio,
        pose_graph_connected_ratio=pose_graph_connected_ratio,
        static_pose_ratio=static_pose_ratio,
        rotation_only_ratio=rotation_only_ratio,
        full_se3_ratio=full_se3_ratio,
        depth_alignment_valid_ratio=depth_alignment_valid_ratio,
        sequence_scale_status=shared_clip.sequence_scale_status,
        independent_track_coverage=independent_track_coverage,
        mean_track_length=mean_track_length,
        reprojection_error_before=reprojection_error_before,
        reprojection_error_after=reprojection_error_after,
        reprojection_improvement=reprojection_gain,
        depth_stability_before=depth_stability_before,
        depth_stability_after=depth_stability_after,
        depth_stability_improvement=depth_gain,
        background_3d_stability_before=background_3d_stability_before,
        background_3d_stability_after=background_3d_stability_after,
        background_3d_stability_improvement=background_gain,
        quality=quality,
        valid=ready,
        missing_reason=reason,
        metadata={
            **dict(metadata or {}),
            "checks": checks,
            "thresholds": asdict(limits),
            "quality_is_probability": False,
            "anomaly_score": False,
            "shared_clip_reused": True,
            "depth_reestimated": False,
            "intrinsics_reestimated": False,
            "pose_reestimated": False,
        },
    )
