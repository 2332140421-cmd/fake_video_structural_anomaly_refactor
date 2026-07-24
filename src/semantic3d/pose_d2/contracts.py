"""Validity-aware contracts for short-baseline camera pose and D2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..geometry.camera import validate_rigid_transform


class PoseProviderStatus(str, Enum):
    """Unified M4 pose-provider states."""

    VERIFIED_STATIC = "verified_static"
    ESTIMATED_VALID = "estimated_valid"
    ESTIMATED_LOW_CONFIDENCE = "estimated_low_confidence"
    PROVIDER_FAILED = "provider_failed"
    BLOCKED_BY_INTRINSICS = "blocked_by_intrinsics"
    BLOCKED_BY_CORRESPONDENCE = "blocked_by_correspondence"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"

    @property
    def usable_for_geometry(self) -> bool:
        """Return whether the state provides an accepted relative transform."""

        return self in {
            PoseProviderStatus.VERIFIED_STATIC,
            PoseProviderStatus.ESTIMATED_VALID,
        }


class D2VisibilityStatus(str, Enum):
    """Why a transformed point can or cannot support a D2 measurement."""

    VISIBLE = "visible"
    OUT_OF_FRAME = "out_of_frame"
    OCCLUDED = "occluded"
    DEPTH_CONFLICT = "depth_conflict"
    NO_CORRESPONDENCE = "no_correspondence"
    BEHIND_CAMERA = "behind_camera"
    INVALID_INPUT = "invalid_input"


def _optional_array(
    value: Optional[Sequence[float] | np.ndarray],
    shape: tuple[int, ...],
    name: str,
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}.")
    return array


@dataclass(frozen=True)
class StaticVerificationObservation:
    """Independent evidence used before accepting an identity transform."""

    source_frame_index: int
    target_frame_index: int
    global_flow_small: bool
    background_displacement_small: bool
    homography_motion_small: bool
    essential_motion_not_supported: bool
    parallax_small: bool
    image_difference_stable: bool
    evidence_count: int
    required_evidence_count: int
    verified_static: bool
    median_global_flow: float
    median_background_flow: float
    median_parallax: float
    homography_rotation_degrees: float
    pnp_translation_norm: float
    confidence: float
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.required_evidence_count < 2:
            raise ValueError("Static verification requires multiple evidence sources.")
        expected = sum(
            bool(value)
            for value in (
                self.global_flow_small,
                self.background_displacement_small,
                self.homography_motion_small,
                self.essential_motion_not_supported,
                self.parallax_small,
                self.image_difference_stable,
            )
        )
        if self.evidence_count != expected:
            raise ValueError("evidence_count must equal the passed static checks.")
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Static verification confidence must be in [0, 1].")
        if self.verified_static:
            if self.evidence_count < self.required_evidence_count:
                raise ValueError("Verified static requires multiple passed checks.")
            if self.failure_reason:
                raise ValueError("Verified static cannot have failure_reason.")
        elif not self.failure_reason:
            raise ValueError("Unverified static observation requires failure_reason.")
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class PairwisePoseObservation:
    """One directed short-baseline camera pose with full provider diagnostics.

    The transform convention is always
    ``X_target_camera = T_target_from_source @ X_source_camera``.
    """

    frame_t: int
    frame_t1: int
    rotation: Optional[np.ndarray]
    translation: Optional[np.ndarray]
    T_target_from_source: Optional[np.ndarray]
    pose_convention: str
    camera_to_world_or_world_to_camera: str
    translation_scale_status: str
    inlier_count: int
    inlier_ratio: float
    reprojection_error: float
    static_background_ratio: float
    dynamic_foreground_ratio: float
    confidence: float
    provider_status: PoseProviderStatus | str
    failure_reason: str
    background_candidates: int
    foreground_rejected: int
    geometric_inliers: int
    degeneracy_status: str
    provider_name: str
    valid: bool
    static_verification: Optional[StaticVerificationObservation] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = PoseProviderStatus(self.provider_status)
        rotation = _optional_array(self.rotation, (3, 3), "rotation")
        translation = _optional_array(self.translation, (3,), "translation")
        transform = _optional_array(
            self.T_target_from_source, (4, 4), "T_target_from_source"
        )
        if transform is not None:
            transform = validate_rigid_transform(transform, "T_target_from_source")
        for name in ("inlier_ratio", "static_background_ratio", "dynamic_foreground_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Pose confidence must be finite and in [0, 1].")
        error = float(self.reprojection_error)
        if not (math.isnan(error) or (math.isfinite(error) and error >= 0.0)):
            raise ValueError("reprojection_error must be non-negative or NaN.")
        if min(
            self.inlier_count,
            self.background_candidates,
            self.foreground_rejected,
            self.geometric_inliers,
        ) < 0:
            raise ValueError("Pose support counts cannot be negative.")
        usable = status.usable_for_geometry
        if self.valid != usable:
            raise ValueError("valid must match provider_status geometry usability.")
        if usable:
            if rotation is None or translation is None or transform is None:
                raise ValueError("Usable pose requires rotation, translation, and transform.")
            if self.failure_reason:
                raise ValueError("Usable pose cannot have failure_reason.")
            if not math.isfinite(error):
                raise ValueError("Usable pose requires finite reprojection error.")
        elif not self.failure_reason:
            raise ValueError("Unusable pose requires failure_reason.")
        if status == PoseProviderStatus.VERIFIED_STATIC:
            if self.static_verification is None or not self.static_verification.verified_static:
                raise ValueError("Identity pose requires verified static evidence.")
            if transform is None or not np.allclose(transform, np.eye(4), atol=1e-8):
                raise ValueError("verified_static must use an identity transform.")
            if self.static_verification.evidence_count < 3:
                raise ValueError("Identity pose requires at least three static checks.")
        if status == PoseProviderStatus.PROVIDER_FAILED and any(
            value is not None for value in (rotation, translation, transform)
        ):
            raise ValueError("Provider failure cannot contain a fallback pose.")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "T_target_from_source", transform)
        object.__setattr__(self, "provider_status", status)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "reprojection_error", error)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class D2ResidualObservation:
    """One validity-aware point, boundary, or object D2 measurement."""

    evidence_id: str
    evidence_type: str
    video_id: str
    clip_id: str
    frame_t: int
    frame_t1: int
    object_id: str
    track_id: str
    point_id: str
    point_reprojection_residual: float
    boundary_reprojection_residual: float
    depth_reprojection_residual: float
    object_reprojection_residual: float
    visibility_status: D2VisibilityStatus | str
    pose_confidence: float
    point_confidence: float
    valid: bool
    failure_reason: str
    provider_status: PoseProviderStatus | str
    coordinate_frame: str = "clip_local_aligned"
    depth_unit: str = "meter"
    depth_definition: str = "z_depth"
    residual_is_authenticity_decision: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        visibility = D2VisibilityStatus(self.visibility_status)
        provider = PoseProviderStatus(self.provider_status)
        values = (
            float(self.point_reprojection_residual),
            float(self.boundary_reprojection_residual),
            float(self.depth_reprojection_residual),
            float(self.object_reprojection_residual),
        )
        for name in ("pose_confidence", "point_confidence"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        if self.valid:
            if visibility != D2VisibilityStatus.VISIBLE:
                raise ValueError("Valid D2 evidence must be visible.")
            if not any(math.isfinite(value) for value in values):
                raise ValueError("Valid D2 evidence requires a finite residual.")
            if any(math.isinf(value) for value in values):
                raise ValueError("D2 residuals cannot be infinite.")
            if self.failure_reason:
                raise ValueError("Valid D2 evidence cannot have failure_reason.")
            if not provider.usable_for_geometry:
                raise ValueError("Valid D2 requires a usable pose.")
        else:
            if any(math.isfinite(value) for value in values):
                raise ValueError("Invalid D2 evidence must keep all residuals NaN.")
            if not self.failure_reason:
                raise ValueError("Invalid D2 evidence requires failure_reason.")
        if self.coordinate_frame != "clip_local_aligned":
            raise ValueError("M4 D2 output must use clip_local_aligned.")
        if self.residual_is_authenticity_decision:
            raise ValueError("M4 D2 smoke cannot make authenticity decisions.")
        object.__setattr__(self, "visibility_status", visibility)
        object.__setattr__(self, "provider_status", provider)
        object.__setattr__(self, "point_reprojection_residual", values[0])
        object.__setattr__(self, "boundary_reprojection_residual", values[1])
        object.__setattr__(self, "depth_reprojection_residual", values[2])
        object.__setattr__(self, "object_reprojection_residual", values[3])
        object.__setattr__(self, "metadata", dict(self.metadata))
