"""P4-B.5 full-observation contracts and label-free geometry helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


P4B5_PIPELINE_VERSION = "p4b5_full_observation_v1"


@dataclass(frozen=True)
class CoverageMetric:
    """One coverage ratio with explicit missing and applicability counts."""

    metric_name: str
    scope_type: str
    scope_id: str
    numerator: int
    denominator: int
    applicable_count: int
    observation_missing_count: int
    invalid_geometry_count: int
    unsupported_mode_count: int
    unit: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        counts = (
            self.numerator,
            self.denominator,
            self.applicable_count,
            self.observation_missing_count,
            self.invalid_geometry_count,
            self.unsupported_mode_count,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("Coverage counts must be non-negative.")
        if self.numerator > self.denominator:
            raise ValueError("Coverage numerator cannot exceed denominator.")
        if not self.metric_name or not self.scope_type or not self.unit:
            raise ValueError("Coverage metric name, scope type, and unit are required.")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def ratio(self) -> float:
        """Return the observed ratio, preserving an empty denominator as NaN."""

        return self.numerator / self.denominator if self.denominator else math.nan

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["ratio"] = self.ratio
        return row


@dataclass(frozen=True)
class ClipGeometryDecision:
    """Conservative clip geometry state inferred without truth labels."""

    geometry_mode: str
    sequence_scale_status: str
    valid: bool
    quality: float
    missing_reason: str
    median_pixel_motion: float
    tracked_transition_ratio: float
    homography_inlier_ratio: float
    depth_aligned_ratio: float

    def __post_init__(self) -> None:
        if self.geometry_mode not in {
            "static_camera_3d",
            "rotation_compensated",
            "full_se3_3d",
            "unavailable",
        }:
            raise ValueError(f"Unsupported geometry mode: {self.geometry_mode}")
        if not math.isfinite(float(self.quality)) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("Geometry quality must be in [0, 1].")
        if self.valid and (self.geometry_mode == "unavailable" or self.missing_reason):
            raise ValueError("Valid geometry requires a supported mode and no missing reason.")
        if not self.valid and (self.geometry_mode != "unavailable" or not self.missing_reason):
            raise ValueError("Unavailable geometry requires a missing reason.")


@dataclass(frozen=True)
class ClipTrackHandoffObservation:
    """Object identity handoff that does not imply cross-clip 3D alignment."""

    handoff_id: str
    video_id: str
    source_clip_id: str
    target_clip_id: str
    global_object_track_id: str
    source_local_track_id: str
    target_local_track_id: str
    overlap_frame_ids: tuple[str, ...]
    mask_iou: float
    point_overlap_ratio: float
    appearance_similarity: float
    handoff_quality: float
    alignment_id: str
    allows_cross_clip_3d: bool
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("mask_iou", "point_overlap_ratio", "handoff_quality"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        appearance = float(self.appearance_similarity)
        if math.isfinite(appearance) and not 0.0 <= appearance <= 1.0:
            raise ValueError("appearance_similarity must be NaN or in [0, 1].")
        if self.allows_cross_clip_3d and not self.alignment_id:
            raise ValueError("Cross-clip 3D requires an explicit alignment_id.")
        if self.valid and self.missing_reason:
            raise ValueError("Valid handoff cannot have a missing reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid handoff requires a missing reason.")
        object.__setattr__(self, "overlap_frame_ids", tuple(self.overlap_frame_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SynchronizedDepthOrder:
    """Depth order measured near two formal visible masks."""

    depth_a: float
    depth_b: float
    foreground: str
    background: str
    depth_margin: float
    depth_source: str
    quality: float
    valid: bool
    uncertain: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def adaptive_structure_point_target(
    mask_area: float,
    *,
    minimum: int = 4,
    maximum: int = 24,
) -> int:
    """Select a point budget from mask pixels without consulting the class label."""

    if not math.isfinite(float(mask_area)) or mask_area <= 0:
        return 0
    if minimum < 1 or maximum < minimum:
        raise ValueError("Invalid structure point bounds.")
    target = int(round(math.sqrt(float(mask_area)) / 8.0))
    return int(np.clip(target, minimum, maximum))


def build_fixed_structure_edges(
    point_ids: Sequence[str],
    points_xyz: np.ndarray,
    *,
    neighbours: int = 2,
) -> tuple[tuple[str, str], ...]:
    """Build an undirected nearest-neighbour graph once from an anchor frame."""

    ids = tuple(str(value) for value in point_ids)
    points = np.asarray(points_xyz, dtype=float)
    if points.shape != (len(ids), 3):
        raise ValueError("points_xyz must have shape [len(point_ids), 3].")
    if len(ids) != len(set(ids)):
        raise ValueError("point_ids must be unique.")
    if len(ids) < 2 or not np.isfinite(points).all():
        return ()
    neighbour_count = min(max(int(neighbours), 1), len(ids) - 1)
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    edges: set[tuple[str, str]] = set()
    for index, point_id in enumerate(ids):
        order = np.argsort(distances[index])
        for other_index in order[1 : neighbour_count + 1]:
            edges.add(tuple(sorted((point_id, ids[int(other_index)]))))
    return tuple(sorted(edges))


def classify_clip_geometry(
    *,
    median_pixel_motion: float,
    tracked_transition_ratio: float,
    homography_inlier_ratio: float,
    depth_aligned_ratio: float,
    static_motion_threshold: float = 1.5,
    rotation_motion_limit: float = 20.0,
) -> ClipGeometryDecision:
    """Classify only modes supported by observed image/depth evidence.

    Full SE(3) is deliberately not inferred here because the P4-B.5 builder has
    no calibrated metric translation source. Moving clips that cannot be
    justified as rotation-only remain unavailable.
    """

    tracked = float(np.clip(tracked_transition_ratio, 0.0, 1.0))
    inliers = float(np.clip(homography_inlier_ratio, 0.0, 1.0))
    aligned = float(np.clip(depth_aligned_ratio, 0.0, 1.0))
    motion = float(median_pixel_motion)
    if aligned < 0.6:
        return ClipGeometryDecision(
            "unavailable", "relative_per_frame", False, 0.0,
            "insufficient_sequence_depth_alignment", motion, tracked, inliers, aligned,
        )
    if tracked < 0.5 or not math.isfinite(motion):
        return ClipGeometryDecision(
            "unavailable", "relative_shared_sequence", False, 0.0,
            "insufficient_independent_point_tracking", motion, tracked, inliers, aligned,
        )
    if motion <= static_motion_threshold:
        quality = min(aligned, tracked, max(0.0, 1.0 - motion / max(static_motion_threshold, 1e-8)))
        return ClipGeometryDecision(
            "static_camera_3d", "relative_shared_sequence", True, quality, "",
            motion, tracked, inliers, aligned,
        )
    if motion <= rotation_motion_limit and inliers >= 0.65:
        quality = min(aligned, tracked, inliers)
        return ClipGeometryDecision(
            "rotation_compensated", "bearing_only", True, quality, "",
            motion, tracked, inliers, aligned,
        )
    return ClipGeometryDecision(
        "unavailable", "relative_shared_sequence", False, 0.0,
        "full_se3_not_observationally_supported", motion, tracked, inliers, aligned,
    )


def synchronized_depth_order(
    depth_map: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    minimum_pixels: int = 12,
    uncertainty_ratio: float = 0.03,
) -> SynchronizedDepthOrder:
    """Estimate foreground order from overlap-boundary depth when available."""

    depth = np.asarray(depth_map, dtype=float)
    first = np.asarray(mask_a, dtype=bool)
    second = np.asarray(mask_b, dtype=bool)
    if depth.ndim != 2 or first.shape != depth.shape or second.shape != depth.shape:
        raise ValueError("depth_map and masks must share one HxW shape.")
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    kernel = np.ones((5, 5), dtype=np.uint8)
    # Sample each object's independently visible side of the contact boundary.
    # The literal intersection cannot carry two object depths at once and would
    # otherwise force both medians to the same raster value.
    near_first = first & ~second & cv2.dilate(second.astype(np.uint8), kernel).astype(bool)
    near_second = second & ~first & cv2.dilate(first.astype(np.uint8), kernel).astype(bool)
    support_a = near_first & valid_depth
    support_b = near_second & valid_depth
    source = "overlap_boundary_neighbourhood"
    if np.count_nonzero(support_a) < minimum_pixels or np.count_nonzero(support_b) < minimum_pixels:
        support_a = first & valid_depth
        support_b = second & valid_depth
        source = "object_mask_median_low_quality"
    if np.count_nonzero(support_a) < minimum_pixels or np.count_nonzero(support_b) < minimum_pixels:
        return SynchronizedDepthOrder(
            math.nan, math.nan, "", "", math.nan, source, 0.0, False, True,
            "insufficient_valid_depth_support",
        )
    depth_a = float(np.median(depth[support_a]))
    depth_b = float(np.median(depth[support_b]))
    margin = abs(depth_a - depth_b)
    reference = max(float(np.median([depth_a, depth_b])), 1e-8)
    uncertain = margin <= uncertainty_ratio * reference
    if uncertain:
        return SynchronizedDepthOrder(
            depth_a, depth_b, "", "", margin, source,
            0.25 if source == "object_mask_median_low_quality" else 0.5,
            False, True, "depth_order_within_uncertainty",
            {"uncertainty_ratio": uncertainty_ratio},
        )
    foreground, background = ("a", "b") if depth_a < depth_b else ("b", "a")
    quality = 0.5 if source == "object_mask_median_low_quality" else 1.0
    return SynchronizedDepthOrder(
        depth_a, depth_b, foreground, background, margin, source, quality, True, False,
        "", {"uncertainty_ratio": uncertainty_ratio},
    )
