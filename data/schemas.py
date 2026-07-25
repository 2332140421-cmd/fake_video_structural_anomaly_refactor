"""Small, shared contracts used by every active paper-core layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np

EvidenceLevel = Literal[
    "point", "track", "boundary", "object", "object_pair", "frame", "clip"
]


@dataclass(frozen=True)
class VideoMetadata:
    video_path: Path
    video_id: str
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class VideoClip:
    clip_id: str
    video_id: str
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    frames: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        count = len(self.frames)
        if not count or len(self.frame_indices) != count or len(self.timestamps) != count:
            raise ValueError("VideoClip frames, indices, and timestamps must align.")


@dataclass
class ObjectObservation:
    object_id: str
    track_id: str
    category: str
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    instance_mask: np.ndarray | None = None
    keypoints_xy: np.ndarray | None = None
    occlusion_ratio: float = 0.0
    truncated: bool = False
    mask_quality: float = 1.0
    track_identity_stable: bool = True
    viewpoint: str = "unknown"
    metric_surface_xyz: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.instance_mask is not None:
            self.instance_mask = np.asarray(self.instance_mask, dtype=bool)
        if self.keypoints_xy is not None:
            self.keypoints_xy = np.asarray(self.keypoints_xy, dtype=float)
        if self.metric_surface_xyz is not None:
            points = np.asarray(self.metric_surface_xyz, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError("metric_surface_xyz must have shape [N,3].")
            self.metric_surface_xyz = points


@dataclass
class FrameObservation:
    video_id: str
    clip_id: str
    frame_index: int
    timestamp: float
    image: np.ndarray
    objects: list[ObjectObservation] = field(default_factory=list)
    metric_depth: np.ndarray | None = None
    depth_valid_mask: np.ndarray | None = None
    depth_confidence: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    relative_pose_from_previous: np.ndarray | None = None
    actual_correspondences: np.ndarray | None = None
    boundary_correspondences: np.ndarray | None = None
    occlusion_states: dict[str, str] = field(default_factory=dict)
    reappearance_states: dict[str, str] = field(default_factory=dict)
    visibility_observations: dict[str, Any] = field(default_factory=dict)
    reappearance_observations: list[dict[str, Any]] = field(default_factory=list)
    availability: dict[str, bool] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        if self.metric_depth is not None:
            self.metric_depth = np.asarray(self.metric_depth, dtype=float)
            if self.metric_depth.shape != self.image.shape[:2]:
                raise ValueError("Metric depth and image geometry must align.")
        if self.depth_valid_mask is not None:
            self.depth_valid_mask = np.asarray(self.depth_valid_mask, dtype=bool)
        if self.intrinsics is not None:
            self.intrinsics = np.asarray(self.intrinsics, dtype=float)
            if self.intrinsics.shape != (3, 3):
                raise ValueError("Intrinsics must be 3x3.")
        for name in ("actual_correspondences", "boundary_correspondences"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.ndim != 2 or array.shape[1] != 4:
                    raise ValueError(f"{name} must have shape [N,4] as source_xy,target_xy.")
                setattr(self, name, array)


@dataclass
class TrackObservation:
    track_id: str
    object_id: str
    frame_indices: tuple[int, ...]
    actual_xy: np.ndarray
    predicted_xy: np.ndarray | None = None
    points_3d: np.ndarray | None = None
    valid_mask: np.ndarray | None = None
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.actual_xy = np.asarray(self.actual_xy, dtype=float)
        if self.actual_xy.shape != (len(self.frame_indices), 2):
            raise ValueError("actual_xy must have shape [T,2].")
        if self.predicted_xy is not None:
            self.predicted_xy = np.asarray(self.predicted_xy, dtype=float)
        if self.points_3d is not None:
            self.points_3d = np.asarray(self.points_3d, dtype=float)
        if self.valid_mask is None:
            self.valid_mask = np.ones(len(self.frame_indices), dtype=bool)
        else:
            self.valid_mask = np.asarray(self.valid_mask, dtype=bool)


@dataclass
class ClipObservation:
    video_id: str
    clip_id: str
    frames: list[FrameObservation]
    tracks: list[TrackObservation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("ClipObservation requires at least one frame.")
        if any(frame.video_id != self.video_id or frame.clip_id != self.clip_id for frame in self.frames):
            raise ValueError("Every frame must belong to the enclosing clip.")


@dataclass(frozen=True)
class ResidualEvidence:
    name: str
    level: EvidenceLevel
    raw_value: float
    normalized_value: float
    availability: str
    confidence: float
    valid_mask: bool
    spatial_support: Mapping[str, Any] = field(default_factory=dict)
    temporal_support: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.level not in {
            "point", "track", "boundary", "object", "object_pair", "frame", "clip"
        }:
            raise ValueError(f"Unsupported evidence level: {self.level}.")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("Evidence confidence must be in [0,1].")
        if self.valid_mask:
            if not math.isfinite(float(self.raw_value)):
                raise ValueError("Available evidence requires a finite raw value.")
            if not math.isfinite(float(self.normalized_value)):
                raise ValueError("Available evidence requires a finite normalized value.")
            if self.reason:
                raise ValueError("Available evidence cannot carry a missing reason.")
        elif not self.reason or not math.isnan(float(self.raw_value)):
            raise ValueError("Unavailable evidence requires NaN and a reason.")

    @classmethod
    def observed(
        cls,
        name: str,
        level: EvidenceLevel,
        value: float,
        *,
        confidence: float = 1.0,
        spatial_support: Mapping[str, Any] | None = None,
        temporal_support: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResidualEvidence":
        value = max(0.0, float(value))
        return cls(
            name=name,
            level=level,
            raw_value=value,
            normalized_value=float(1.0 - math.exp(-value)),
            availability="observed",
            confidence=float(confidence),
            valid_mask=True,
            spatial_support=dict(spatial_support or {}),
            temporal_support=dict(temporal_support or {}),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def unavailable(
        cls,
        name: str,
        level: EvidenceLevel,
        reason: str,
        *,
        availability: str = "blocked_by_input",
        spatial_support: Mapping[str, Any] | None = None,
        temporal_support: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResidualEvidence":
        return cls(
            name=name,
            level=level,
            raw_value=float("nan"),
            normalized_value=float("nan"),
            availability=availability,
            confidence=0.0,
            valid_mask=False,
            spatial_support=dict(spatial_support or {}),
            temporal_support=dict(temporal_support or {}),
            reason=reason,
            metadata=dict(metadata or {}),
        )


@dataclass
class ClipResult:
    video_id: str
    clip_id: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    risk_score: float
    coverage: float
    confidence: float
    residuals: list[ResidualEvidence]
    object_scores: dict[str, float] = field(default_factory=dict)
    track_scores: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)
    dominant_residual: str = ""
    spatial_heatmaps: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class VideoResult:
    video_id: str
    video_path: str
    risk_score: float
    clip_results: list[ClipResult]
    timeline: list[dict[str, Any]]
    suspicious_clips: list[dict[str, Any]]
    object_scores: dict[str, float]
    track_scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
