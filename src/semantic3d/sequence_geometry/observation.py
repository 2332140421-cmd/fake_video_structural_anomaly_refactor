"""Sequence-level shared 3D geometry data contracts.

These contracts distinguish a valid identity pose from missing pose evidence,
and distinguish per-frame relative depth from a sequence-consistent scale.
They do not define forged-video anomaly residuals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..geometry.camera import validate_rigid_transform
from ..shared_3d_observation import Shared3DFrameObservation


class SequenceScaleStatus(str, Enum):
    """Scale consistency available across a clip."""

    METRIC_SEQUENCE = "metric_sequence"
    RELATIVE_SHARED_SEQUENCE = "relative_shared_sequence"
    RELATIVE_ALIGNED_SEQUENCE = "relative_aligned_sequence"
    RELATIVE_PER_FRAME = "relative_per_frame"
    UNKNOWN = "unknown"

    @property
    def allows_dynamic_3d(self) -> bool:
        """Return whether depth scale alone is eligible for dynamic 3D use."""

        return self in {
            SequenceScaleStatus.METRIC_SEQUENCE,
            SequenceScaleStatus.RELATIVE_SHARED_SEQUENCE,
            SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE,
        }


class DepthAlignmentMode(str, Enum):
    """Mathematical domain used to align two depth observations."""

    SCALE_ONLY = "scale_only"
    AFFINE_DEPTH = "affine_depth"
    AFFINE_INVERSE_DEPTH = "affine_inverse_depth"
    PROVIDER_SHARED_SCALE = "provider_shared_scale"
    UNSUPPORTED = "unsupported"


def _optional_matrix(
    value: Optional[np.ndarray], shape: tuple[int, int], name: str
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite matrix with shape {shape}.")
    return array


def _optional_vector(
    value: Optional[np.ndarray], size: int, name: str
) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({size},).")
    return array


@dataclass(frozen=True)
class RelativePoseObservation:
    """One camera pose and its relation to the previous retained frame.

    ``relative_pose_from_previous`` follows
    ``X_current_camera = T_current_from_previous @ X_previous_camera``.
    Absolute transforms follow the P1 camera contract.
    """

    source_frame_index: Optional[int]
    target_frame_index: int
    T_world_from_camera: Optional[np.ndarray]
    T_camera_from_world: Optional[np.ndarray]
    relative_pose_from_previous: Optional[np.ndarray]
    camera_center_world: Optional[np.ndarray]
    rotation: Optional[np.ndarray]
    translation: Optional[np.ndarray]
    pose_source: str
    pose_quality: float
    background_support_count: int
    background_inlier_ratio: float
    reprojection_error: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality = float(self.pose_quality)
        inlier_ratio = float(self.background_inlier_ratio)
        reprojection_error = float(self.reprojection_error)
        if not 0.0 <= quality <= 1.0 or not math.isfinite(quality):
            raise ValueError("pose_quality must be finite and in [0, 1].")
        if not 0.0 <= inlier_ratio <= 1.0 or not math.isfinite(inlier_ratio):
            raise ValueError("background_inlier_ratio must be finite and in [0, 1].")
        if self.background_support_count < 0:
            raise ValueError("background_support_count must be non-negative.")
        twc = _optional_matrix(self.T_world_from_camera, (4, 4), "T_world_from_camera")
        tcw = _optional_matrix(self.T_camera_from_world, (4, 4), "T_camera_from_world")
        relative = _optional_matrix(
            self.relative_pose_from_previous,
            (4, 4),
            "relative_pose_from_previous",
        )
        center = _optional_vector(self.camera_center_world, 3, "camera_center_world")
        rotation = _optional_matrix(self.rotation, (3, 3), "rotation")
        translation = _optional_vector(self.translation, 3, "translation")
        valid = bool(self.valid)
        reason = str(self.missing_reason)
        if valid:
            if any(item is None for item in (twc, tcw, relative, center, rotation, translation)):
                raise ValueError("Valid RelativePoseObservation requires all pose fields.")
            assert twc is not None and tcw is not None and relative is not None
            validate_rigid_transform(twc, "T_world_from_camera")
            validate_rigid_transform(tcw, "T_camera_from_world")
            validate_rigid_transform(relative, "relative_pose_from_previous")
            if not (
                np.allclose(twc @ tcw, np.eye(4), atol=1e-6)
                and np.allclose(tcw @ twc, np.eye(4), atol=1e-6)
            ):
                raise ValueError("World/camera transforms must be mutual inverses.")
            if not self.pose_source.strip():
                raise ValueError("Valid pose requires pose_source.")
            if not math.isfinite(reprojection_error) or reprojection_error < 0.0:
                raise ValueError("Valid pose requires non-negative reprojection_error.")
            if reason:
                raise ValueError("Valid pose cannot have missing_reason.")
            if (
                self.source_frame_index is not None
                and np.allclose(relative, np.eye(4), atol=1e-8)
                and self.background_support_count <= 0
            ):
                raise ValueError(
                    "A valid identity relative pose requires background estimation evidence."
                )
        else:
            if not reason:
                raise ValueError("Invalid pose requires missing_reason.")
            if any(item is not None for item in (twc, tcw, relative, center, rotation, translation)):
                raise ValueError("Missing pose must not contain fabricated matrices or vectors.")
            if not math.isnan(reprojection_error):
                raise ValueError("Invalid pose reprojection_error must be NaN.")
        object.__setattr__(self, "T_world_from_camera", twc)
        object.__setattr__(self, "T_camera_from_world", tcw)
        object.__setattr__(self, "relative_pose_from_previous", relative)
        object.__setattr__(self, "camera_center_world", center)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "pose_quality", quality)
        object.__setattr__(self, "background_inlier_ratio", inlier_ratio)
        object.__setattr__(self, "reprojection_error", reprojection_error)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def is_identity_relative_pose(self) -> bool:
        """Return true only for an observed valid identity relative transform."""

        return bool(
            self.valid
            and self.relative_pose_from_previous is not None
            and np.allclose(self.relative_pose_from_previous, np.eye(4), atol=1e-8)
        )

    @classmethod
    def missing(
        cls,
        target_frame_index: int,
        reason: str,
        *,
        source_frame_index: Optional[int] = None,
        pose_source: str = "",
        background_support_count: int = 0,
        background_inlier_ratio: float = 0.0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RelativePoseObservation":
        """Create missing pose evidence without an identity-matrix placeholder."""

        return cls(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            T_world_from_camera=None,
            T_camera_from_world=None,
            relative_pose_from_previous=None,
            camera_center_world=None,
            rotation=None,
            translation=None,
            pose_source=pose_source,
            pose_quality=0.0,
            background_support_count=background_support_count,
            background_inlier_ratio=background_inlier_ratio,
            reprojection_error=float("nan"),
            valid=False,
            missing_reason=reason,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_transforms(
        cls,
        *,
        source_frame_index: Optional[int],
        target_frame_index: int,
        T_world_from_camera: np.ndarray,
        relative_pose_from_previous: np.ndarray,
        pose_source: str,
        pose_quality: float,
        background_support_count: int,
        background_inlier_ratio: float,
        reprojection_error: float,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "RelativePoseObservation":
        """Create a validated pose from an absolute and previous-relative transform."""

        twc = validate_rigid_transform(T_world_from_camera, "T_world_from_camera")
        tcw = np.linalg.inv(twc)
        relative = validate_rigid_transform(
            relative_pose_from_previous, "relative_pose_from_previous"
        )
        return cls(
            source_frame_index=source_frame_index,
            target_frame_index=target_frame_index,
            T_world_from_camera=twc,
            T_camera_from_world=tcw,
            relative_pose_from_previous=relative,
            camera_center_world=twc[:3, 3],
            rotation=relative[:3, :3],
            translation=relative[:3, 3],
            pose_source=pose_source,
            pose_quality=pose_quality,
            background_support_count=background_support_count,
            background_inlier_ratio=background_inlier_ratio,
            reprojection_error=reprojection_error,
            valid=True,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class DepthAlignmentObservation:
    """Robust mapping between depth representations in two frames.

    The fitted convention is ``target_domain = scale * source_domain + shift``.
    For inverse-depth mode, both domains are reciprocal depth.
    """

    source_frame: int
    target_frame: int
    alignment_mode: DepthAlignmentMode | str
    scale: float
    shift: float
    support_count: int
    inlier_ratio: float
    fitting_error: float
    quality: float
    valid: bool
    missing_reason: str = ""
    holdout_error: float = float("nan")
    holdout_count: int = 0
    physical_valid: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DepthAlignmentMode(self.alignment_mode)
        scale = float(self.scale)
        shift = float(self.shift)
        inlier = float(self.inlier_ratio)
        error = float(self.fitting_error)
        quality = float(self.quality)
        holdout_error = float(self.holdout_error)
        if self.source_frame == self.target_frame:
            raise ValueError("Depth alignment requires distinct source and target frames.")
        if self.support_count < 0:
            raise ValueError("support_count must be non-negative.")
        if self.holdout_count < 0:
            raise ValueError("holdout_count must be non-negative.")
        if not math.isfinite(inlier) or not 0.0 <= inlier <= 1.0:
            raise ValueError("inlier_ratio must be finite and in [0, 1].")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be finite and in [0, 1].")
        if self.valid:
            if mode == DepthAlignmentMode.UNSUPPORTED:
                raise ValueError("Unsupported alignment cannot be valid.")
            if not all(math.isfinite(item) for item in (scale, shift, error)):
                raise ValueError("Valid alignment requires finite parameters and error.")
            if scale <= 0.0 or error < 0.0:
                raise ValueError("Valid alignment requires scale > 0 and error >= 0.")
            if self.missing_reason:
                raise ValueError("Valid alignment cannot have missing_reason.")
            if self.holdout_count > 0 and (
                not math.isfinite(holdout_error) or holdout_error < 0.0
            ):
                raise ValueError(
                    "Valid alignment with holdout samples requires a non-negative "
                    "holdout_error."
                )
        else:
            if not self.missing_reason:
                raise ValueError("Invalid alignment requires missing_reason.")
            if not all(math.isnan(item) for item in (scale, shift, error)):
                raise ValueError("Invalid alignment parameters and error must be NaN.")
            if not math.isnan(holdout_error):
                raise ValueError("Invalid alignment holdout_error must be NaN.")
        object.__setattr__(self, "alignment_mode", mode)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "shift", shift)
        object.__setattr__(self, "inlier_ratio", inlier)
        object.__setattr__(self, "fitting_error", error)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "holdout_error", holdout_error)
        object.__setattr__(self, "physical_valid", bool(self.physical_valid))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def missing(
        cls,
        source_frame: int,
        target_frame: int,
        reason: str,
        *,
        mode: DepthAlignmentMode | str = DepthAlignmentMode.UNSUPPORTED,
        support_count: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "DepthAlignmentObservation":
        """Create invalid alignment with NaN parameters, never identity defaults."""

        return cls(
            source_frame=source_frame,
            target_frame=target_frame,
            alignment_mode=mode,
            scale=float("nan"),
            shift=float("nan"),
            support_count=support_count,
            inlier_ratio=0.0,
            fitting_error=float("nan"),
            quality=0.0,
            valid=False,
            missing_reason=reason,
            holdout_error=float("nan"),
            holdout_count=0,
            physical_valid=False,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True)
class SequenceGeometryObservation:
    """Per-frame sequence geometry and quality diagnostics."""

    frame_index: int
    frame: Shared3DFrameObservation
    relative_pose: RelativePoseObservation
    depth_alignment: Optional[DepthAlignmentObservation]
    scene_cut_before: bool
    excluded_foreground_ratio: float
    background_support_count: int
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index != self.frame.frame_index:
            raise ValueError("Sequence frame_index must match Shared3D frame.")
        ratio = float(self.excluded_foreground_ratio)
        quality = float(self.quality)
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("excluded_foreground_ratio must be in [0, 1].")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("quality must be in [0, 1].")
        if self.background_support_count < 0:
            raise ValueError("background_support_count must be non-negative.")
        if self.valid and self.missing_reason:
            raise ValueError("Valid sequence observation cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid sequence observation requires missing_reason.")
        object.__setattr__(self, "excluded_foreground_ratio", ratio)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class Shared3DClipObservation:
    """One sequence geometry contract shared by future static and dynamic branches."""

    video_id: str
    clip_id: str
    frame_indices: tuple[int, ...]
    frames: tuple[Shared3DFrameObservation, ...]
    reference_frame_index: int
    T_world_from_camera_by_frame: Mapping[int, Optional[np.ndarray]]
    T_camera_from_world_by_frame: Mapping[int, Optional[np.ndarray]]
    relative_poses: tuple[RelativePoseObservation, ...]
    sequence_scale_status: SequenceScaleStatus | str
    depth_alignment_observations: tuple[DepthAlignmentObservation, ...]
    scene_cut_flags: Mapping[int, bool]
    background_track_ids: tuple[str, ...]
    foreground_object_ids: tuple[str, ...]
    provider_name: str
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = tuple(int(item) for item in self.frame_indices)
        frames = tuple(self.frames)
        status = SequenceScaleStatus(self.sequence_scale_status)
        quality = float(self.quality)
        if len(indices) != len(frames) or not indices:
            raise ValueError("frame_indices and frames must be non-empty and aligned.")
        if len(set(indices)) != len(indices):
            raise ValueError("Shared3DClipObservation frame indices must be unique.")
        if tuple(frame.frame_index for frame in frames) != indices:
            raise ValueError("Shared3D frames must follow frame_indices order.")
        if self.reference_frame_index not in indices:
            raise ValueError("reference_frame_index must belong to the clip.")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("clip quality must be finite and in [0, 1].")
        twc_map: dict[int, Optional[np.ndarray]] = {}
        tcw_map: dict[int, Optional[np.ndarray]] = {}
        for index in indices:
            twc = _optional_matrix(
                self.T_world_from_camera_by_frame.get(index),
                (4, 4),
                f"T_world_from_camera[{index}]",
            )
            tcw = _optional_matrix(
                self.T_camera_from_world_by_frame.get(index),
                (4, 4),
                f"T_camera_from_world[{index}]",
            )
            if (twc is None) != (tcw is None):
                raise ValueError("World/camera pose maps must be missing together.")
            if twc is not None and tcw is not None and not (
                np.allclose(twc @ tcw, np.eye(4), atol=1e-6)
                and np.allclose(tcw @ twc, np.eye(4), atol=1e-6)
            ):
                raise ValueError("Pose-map matrices must be mutual inverses.")
            twc_map[index], tcw_map[index] = twc, tcw
        poses = tuple(self.relative_poses)
        if tuple(pose.target_frame_index for pose in poses) != indices:
            raise ValueError("relative_poses must provide one record per retained frame.")
        if self.valid:
            if self.missing_reason:
                raise ValueError("Valid clip cannot have missing_reason.")
        elif not self.missing_reason:
            raise ValueError("Invalid clip requires missing_reason.")
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "frames", frames)
        object.__setattr__(self, "T_world_from_camera_by_frame", twc_map)
        object.__setattr__(self, "T_camera_from_world_by_frame", tcw_map)
        object.__setattr__(self, "relative_poses", poses)
        object.__setattr__(
            self, "depth_alignment_observations", tuple(self.depth_alignment_observations)
        )
        object.__setattr__(self, "scene_cut_flags", dict(self.scene_cut_flags))
        object.__setattr__(self, "background_track_ids", tuple(self.background_track_ids))
        object.__setattr__(self, "foreground_object_ids", tuple(self.foreground_object_ids))
        object.__setattr__(self, "sequence_scale_status", status)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def scale_allows_dynamic_3d(self) -> bool:
        """Return scale-domain eligibility without hiding pose-quality requirements."""

        return self.sequence_scale_status.allows_dynamic_3d

    @property
    def allows_dynamic_3d(self) -> bool:
        """Return the explicit final gate recorded by the sequence provider."""

        return bool(
            self.valid
            and self.scale_allows_dynamic_3d
            and all(pose.valid for pose in self.relative_poses)
            and not any(self.scene_cut_flags.get(index, False) for index in self.frame_indices)
            and bool(self.metadata.get("pose_scale_compatible_with_depth", False))
        )
