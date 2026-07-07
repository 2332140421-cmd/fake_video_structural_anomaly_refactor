"""Shared data structures for future visual model integration.

These dataclasses describe the data exchanged between segmentation, depth,
flow, tracking, and residual-analysis modules. They intentionally do not call
or download any large vision model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .scale_depth import ObjectObservation


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class FrameObservation:
    """Observations and optional artifact paths for one video frame.

    Attributes:
        frame_index: Integer frame index in the source video or clip.
        image_path: Optional image path on disk.
        image_id: Optional external image identifier when no path is available.
        width: Frame width in pixels.
        height: Frame height in pixels.
        objects: Object observations detected in this frame.
        depth_map_path: Optional path to a saved depth map for this frame.
        flow_path: Optional path to a saved optical-flow field for this frame.
    """

    frame_index: int
    width: int
    height: int
    objects: List[ObjectObservation] = field(default_factory=list)
    image_path: Optional[str] = None
    image_id: Optional[str] = None
    depth_map_path: Optional[str] = None
    flow_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.image_path is None and self.image_id is None:
            raise ValueError("FrameObservation requires image_path or image_id.")
        if self.width <= 0:
            raise ValueError(f"Frame width must be > 0, got {self.width}.")
        if self.height <= 0:
            raise ValueError(f"Frame height must be > 0, got {self.height}.")

    @property
    def frame_area(self) -> int:
        """Return full frame area in pixels."""

        return self.width * self.height


@dataclass(frozen=True)
class ClipObservation:
    """A short video clip represented by ordered frame observations."""

    clip_id: str
    frames: List[FrameObservation]
    clip_start: int
    clip_end: int

    def __post_init__(self) -> None:
        if self.clip_end < self.clip_start:
            raise ValueError(
                f"clip_end must be >= clip_start, got {self.clip_start}, {self.clip_end}."
            )
        if not self.frames:
            raise ValueError("ClipObservation requires at least one frame.")


@dataclass(frozen=True)
class ObjectTrack:
    """Temporal track of one object across frames.

    Attributes:
        object_id: Stable object or track identifier.
        label: Semantic class label.
        frame_indices: Frame indices where the object is observed.
        centers: Object centers as (x, y) image coordinates.
        depths: Per-frame object median depths.
        mask_areas: Per-frame instance mask areas in pixels.
        projection_scales: Per-frame equivalent projection scales.
    """

    object_id: str
    label: str
    frame_indices: List[int]
    centers: List[Point2D]
    depths: List[float]
    mask_areas: List[float]
    projection_scales: List[float]

    def __post_init__(self) -> None:
        lengths = {
            "frame_indices": len(self.frame_indices),
            "centers": len(self.centers),
            "depths": len(self.depths),
            "mask_areas": len(self.mask_areas),
            "projection_scales": len(self.projection_scales),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(f"ObjectTrack fields must have equal lengths: {lengths}.")
        if not self.frame_indices:
            raise ValueError("ObjectTrack requires at least one observation.")


@dataclass(frozen=True)
class ResidualReport:
    """Residual analysis output for one clip.

    Attributes:
        clip_id: Identifier of the analyzed clip.
        frame_scores: Per-frame or per-segment anomaly scores.
        object_pair_residuals: Residual details keyed by object-pair identifiers.
        total_score: Clip-level fused anomaly score.
        explanations: Human-readable or paper-friendly explanation strings.
    """

    clip_id: str
    frame_scores: List[float]
    object_pair_residuals: Dict[str, Dict[str, float]]
    total_score: float
    explanations: List[str] = field(default_factory=list)
