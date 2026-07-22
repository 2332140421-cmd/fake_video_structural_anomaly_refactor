"""Bind independent point tracks to stable object identities.

The module does not estimate detections, masks, depth, intrinsics, or pose.  It
only associates already tracked image points with existing 2D object
observations and carries that provenance into the shared 3D trajectory branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from ..observations import FrameObservationJSON, ObjectObservationJSON
from .track_observation import PointTrack2DObservation, PointTrack3DObservation


class PointRole(str, Enum):
    """Semantic role of a tracked image point."""

    SEMANTIC_KEYPOINT = "semantic_keypoint"
    INTERNAL_STABLE_POINT = "internal_stable_point"
    BOUNDARY_POINT = "boundary_point"
    BACKGROUND_POINT = "background_point"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ObjectPointBinding:
    """Sequence-level ownership decision for one stable point identity."""

    video_id: str
    clip_id: str
    object_track_id: str
    point_id: str
    point_role: PointRole | str
    semantic_keypoint_name: Optional[str]
    frame_indices: tuple[int, ...]
    assignment_source: str
    assignment_quality: float
    mask_support_ratio: float
    bbox_support_ratio: float
    track_consistency_ratio: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        role = PointRole(self.point_role)
        indices = tuple(int(value) for value in self.frame_indices)
        ratios = (
            float(self.assignment_quality),
            float(self.mask_support_ratio),
            float(self.bbox_support_ratio),
            float(self.track_consistency_ratio),
        )
        if not self.video_id.strip() or not self.clip_id.strip() or not self.point_id.strip():
            raise ValueError("Binding video, clip, and point IDs must not be empty.")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in ratios):
            raise ValueError("Binding quality and support ratios must be in [0, 1].")
        if indices != tuple(sorted(set(indices))):
            raise ValueError("Binding frame_indices must be sorted and unique.")
        if self.valid:
            if not self.object_track_id.strip() or not indices or self.missing_reason:
                raise ValueError("A valid binding requires an owner, frames, and no reason.")
        elif not self.missing_reason:
            raise ValueError("An invalid binding requires missing_reason.")
        object.__setattr__(self, "point_role", role)
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "assignment_quality", ratios[0])
        object.__setattr__(self, "mask_support_ratio", ratios[1])
        object.__setattr__(self, "bbox_support_ratio", ratios[2])
        object.__setattr__(self, "track_consistency_ratio", ratios[3])
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObjectPointTrack3D:
    """A bound object point with its independent 2D and shared-depth 3D samples."""

    binding: ObjectPointBinding
    semantic_label: str
    points_2d: tuple[PointTrack2DObservation, ...]
    points_3d: tuple[PointTrack3DObservation, ...]
    object_scale_by_frame: Mapping[int, Optional[float]] = field(default_factory=dict)
    valid: bool = True
    quality: float = 1.0
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Object point track quality must be in [0, 1].")
        if self.valid and (not self.binding.valid or self.missing_reason):
            raise ValueError("A valid object point track requires a valid binding.")
        if not self.valid and not self.missing_reason:
            raise ValueError("An invalid object point track requires missing_reason.")
        keys_2d = {(point.point_id, point.frame_index) for point in self.points_2d}
        keys_3d = {(point.point_id, point.frame_index) for point in self.points_3d}
        if len(keys_2d) != len(self.points_2d) or len(keys_3d) != len(self.points_3d):
            raise ValueError("Point identities must be unique within each frame.")
        object.__setattr__(self, "object_scale_by_frame", dict(self.object_scale_by_frame))
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObjectDynamicObservation:
    """Frame-local dynamic object evidence assembled from stable point IDs."""

    video_id: str
    clip_id: str
    frame_index: int
    object_track_id: str
    semantic_label: str
    point_ids: tuple[str, ...]
    object_scale: Optional[float]
    valid_point_ratio: float
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ratio, quality = float(self.valid_point_ratio), float(self.quality)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (ratio, quality)):
            raise ValueError("Dynamic object quality values must be in [0, 1].")
        if self.object_scale is not None and (
            not math.isfinite(float(self.object_scale)) or float(self.object_scale) <= 0.0
        ):
            raise ValueError("object_scale must be positive when present.")
        if self.valid and (not self.point_ids or self.missing_reason):
            raise ValueError("Valid dynamic observation requires point evidence.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid dynamic observation requires missing_reason.")
        object.__setattr__(self, "point_ids", tuple(self.point_ids))
        object.__setattr__(self, "valid_point_ratio", ratio)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObjectBindingResult:
    """Bindings plus point samples rewritten with stable object ownership."""

    bindings: tuple[ObjectPointBinding, ...]
    points_2d: tuple[PointTrack2DObservation, ...]
    statistics: Mapping[str, int]


@dataclass(frozen=True)
class _Assignment:
    object_track_id: str
    semantic_label: str
    source: str
    quality: float
    role: PointRole
    semantic_keypoint_name: Optional[str] = None


def _load_mask(obj: ObjectObservationJSON, frame: FrameObservationJSON) -> Optional[np.ndarray]:
    if not obj.mask_path:
        return None
    path = Path(obj.mask_path)
    if not path.exists():
        return None
    if path.suffix.lower() == ".npy":
        mask = np.load(path, allow_pickle=False)
    else:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape[:2] != (frame.height, frame.width):
        return None
    return np.asarray(mask, dtype=bool)


def _contains_bbox(bbox: Optional[Sequence[float]], u: float, v: float, shrink: float) -> bool:
    if bbox is None or len(bbox) != 4:
        return False
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        return False
    dx, dy = (x2 - x1) * shrink, (y2 - y1) * shrink
    return x1 + dx <= u <= x2 - dx and y1 + dy <= v <= y2 - dy


def _keypoint_match(
    obj: ObjectObservationJSON, u: float, v: float, radius: float = 6.0
) -> Optional[str]:
    for point in obj.keypoints_2d or ():
        x, y = point.get("x"), point.get("y")
        if x is None or y is None or not bool(point.get("valid", True)):
            continue
        if math.hypot(float(x) - u, float(y) - v) <= radius:
            return str(point.get("keypoint_name", point.get("name", "unknown")))
    return None


def _assign_sample(
    sample: PointTrack2DObservation,
    frame: FrameObservationJSON,
) -> Optional[_Assignment]:
    if not sample.valid or sample.pixel_uv is None:
        return None
    u, v = sample.pixel_uv
    candidates: list[tuple[int, float, _Assignment]] = []
    for obj in frame.objects:
        track_id = str(obj.track_id or obj.person_track_id or "")
        if not track_id:
            continue
        mask = _load_mask(obj, frame)
        row, column = int(round(v)), int(round(u))
        if mask is not None and 0 <= row < mask.shape[0] and 0 <= column < mask.shape[1] and mask[row, column]:
            eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
            internal = bool(eroded[row, column])
            role = PointRole.INTERNAL_STABLE_POINT if internal else PointRole.BOUNDARY_POINT
            tracked_mask = bool(obj.metadata.get("tracked_instance_mask", False))
            source = "tracked_instance_mask" if tracked_mask else "instance_mask"
            priority = 2 if tracked_mask else 0
            quality = (0.80 if tracked_mask else 1.0) * min(1.0, float(obj.confidence))
            candidates.append((priority, -float(obj.confidence), _Assignment(track_id, obj.label, source, quality, role)))
            continue
        keypoint_name = _keypoint_match(obj, u, v)
        if keypoint_name is not None:
            candidates.append((1, -float(obj.confidence), _Assignment(track_id, obj.label, "semantic_keypoint", min(1.0, float(obj.confidence)), PointRole.SEMANTIC_KEYPOINT, keypoint_name)))
            continue
        if _contains_bbox(obj.bbox, u, v, 0.15):
            candidates.append((3, -float(obj.confidence), _Assignment(track_id, obj.label, "shrunk_bbox", 0.65 * min(1.0, float(obj.confidence)), PointRole.INTERNAL_STABLE_POINT)))
        elif _contains_bbox(obj.bbox, u, v, 0.0):
            candidates.append((4, -float(obj.confidence), _Assignment(track_id, obj.label, "bbox_fallback", 0.35 * min(1.0, float(obj.confidence)), PointRole.BOUNDARY_POINT)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1], item[2].object_track_id))
    return candidates[0][2]


def bind_point_tracks_to_objects(
    points: Sequence[PointTrack2DObservation],
    frames: Sequence[FrameObservationJSON],
    *,
    video_id: str,
    clip_id: str,
    minimum_consistency_ratio: float = 0.60,
) -> ObjectBindingResult:
    """Bind stable point IDs without allowing silent object identity switches.

    Existing masks/keypoints are preferred.  Shrunk bboxes and full bboxes are
    explicit lower-quality fallbacks.  A point assigned to two object tracks is
    invalidated rather than converted into a false continuous object track.
    """

    if not 0.0 <= minimum_consistency_ratio <= 1.0:
        raise ValueError("minimum_consistency_ratio must be in [0, 1].")
    frame_map = {int(frame.frame_index): frame for frame in frames}
    if len(frame_map) != len(frames):
        raise ValueError("Object frames must use unique global frame indices.")
    grouped: dict[str, list[PointTrack2DObservation]] = {}
    for point in points:
        grouped.setdefault(point.point_id, []).append(point)
    bindings: list[ObjectPointBinding] = []
    rewritten: list[PointTrack2DObservation] = []
    statistics = {
        "total_points": len(grouped),
        "valid_bindings": 0,
        "background_bindings": 0,
        "mask_bindings": 0,
        "semantic_keypoint_bindings": 0,
        "shrunk_bbox_bindings": 0,
        "bbox_fallback_bindings": 0,
        "assignment_switch_count": 0,
        "assignment_lost_count": 0,
    }
    for point_id, samples in sorted(grouped.items()):
        samples.sort(key=lambda item: item.frame_index)
        assigned = [
            (sample, _assign_sample(sample, frame_map[sample.frame_index]))
            for sample in samples
            if sample.frame_index in frame_map and sample.valid
        ]
        owners = {assignment.object_track_id for _, assignment in assigned if assignment}
        if len(owners) > 1:
            statistics["assignment_switch_count"] += 1
            binding = ObjectPointBinding(
                video_id, clip_id, "", point_id, PointRole.UNKNOWN, None, (),
                "conflicting_assignments", 0.0, 0.0, 0.0, 0.0, False,
                "object_assignment_switched",
                {"candidate_object_track_ids": sorted(owners)},
            )
            bindings.append(binding)
            rewritten.extend(
                PointTrack2DObservation.missing(
                    point_id=sample.point_id,
                    object_track_id="invalid_assignment",
                    frame_index=sample.frame_index,
                    reason="object_assignment_switched",
                    source_tracker=sample.source_tracker,
                )
                for sample in samples
            )
            continue
        if not owners:
            valid_frames = tuple(sample.frame_index for sample in samples if sample.valid)
            binding = ObjectPointBinding(
                video_id, clip_id, "background", point_id,
                PointRole.BACKGROUND_POINT, None, valid_frames, "background_exclusion",
                1.0, 0.0, 0.0, 1.0, bool(valid_frames),
                "" if valid_frames else "no_valid_point_observation",
            )
            bindings.append(binding)
            statistics["background_bindings"] += int(binding.valid)
            rewritten.extend(replace(sample, object_track_id="background") for sample in samples)
            continue
        owner = next(iter(owners))
        owner_assignments = [(sample, assignment) for sample, assignment in assigned if assignment and assignment.object_track_id == owner]
        consistency = len(owner_assignments) / max(1, len([sample for sample in samples if sample.valid]))
        source_counts: dict[str, int] = {}
        for _, assignment in owner_assignments:
            source_counts[assignment.source] = source_counts.get(assignment.source, 0) + 1
        dominant_source = max(source_counts, key=source_counts.get)
        representative = max((assignment for _, assignment in owner_assignments), key=lambda item: item.quality)
        valid_binding = consistency >= minimum_consistency_ratio
        reason = "" if valid_binding else "insufficient_assignment_consistency"
        binding = ObjectPointBinding(
            video_id=video_id,
            clip_id=clip_id,
            object_track_id=owner,
            point_id=point_id,
            point_role=representative.role,
            semantic_keypoint_name=representative.semantic_keypoint_name,
            frame_indices=tuple(sample.frame_index for sample, _ in owner_assignments),
            assignment_source=dominant_source,
            assignment_quality=float(np.mean([item.quality for _, item in owner_assignments])),
            mask_support_ratio=(source_counts.get("instance_mask", 0) + source_counts.get("tracked_instance_mask", 0)) / len(owner_assignments),
            bbox_support_ratio=(source_counts.get("shrunk_bbox", 0) + source_counts.get("bbox_fallback", 0)) / len(owner_assignments),
            track_consistency_ratio=consistency,
            valid=valid_binding,
            missing_reason=reason,
            metadata={"semantic_label": representative.semantic_label, "assignment_source_counts": source_counts},
        )
        bindings.append(binding)
        if valid_binding:
            statistics["valid_bindings"] += 1
            counter_name = {
                "instance_mask": "mask_bindings",
                "tracked_instance_mask": "mask_bindings",
                "semantic_keypoint": "semantic_keypoint_bindings",
                "shrunk_bbox": "shrunk_bbox_bindings",
                "bbox_fallback": "bbox_fallback_bindings",
            }[dominant_source]
            statistics[counter_name] += 1
        assigned_frames = {sample.frame_index for sample, _ in owner_assignments}
        for sample in samples:
            if not valid_binding:
                rewritten.append(PointTrack2DObservation.missing(
                    point_id=sample.point_id, object_track_id=owner,
                    frame_index=sample.frame_index, reason=reason,
                    source_tracker=sample.source_tracker,
                ))
            elif sample.valid and sample.frame_index not in assigned_frames:
                statistics["assignment_lost_count"] += 1
                rewritten.append(PointTrack2DObservation.missing(
                    point_id=sample.point_id, object_track_id=owner,
                    frame_index=sample.frame_index, reason="assignment_lost",
                    source_tracker=sample.source_tracker,
                ))
            else:
                rewritten.append(replace(sample, object_track_id=owner))
    return ObjectBindingResult(tuple(bindings), tuple(rewritten), statistics)


def select_stable_object_point_tracks(
    tracks: Sequence[ObjectPointTrack3D],
    *,
    minimum_track_length: int = 3,
    minimum_tracking_confidence: float = 0.30,
    minimum_depth_quality: float = 0.30,
    minimum_assignment_consistency: float = 0.60,
    maximum_occluded_ratio: float = 0.30,
    allow_boundary_points: bool = False,
) -> tuple[tuple[ObjectPointTrack3D, ...], Mapping[str, str]]:
    """Select stable internal tracks and report deterministic rejection reasons."""

    if minimum_track_length < 1:
        raise ValueError("minimum_track_length must be positive.")
    selected, rejected = [], {}
    for track in tracks:
        reason = ""
        valid_2d = [point for point in track.points_2d if point.valid]
        valid_3d = [point for point in track.points_3d if point.valid]
        if not track.valid or not track.binding.valid:
            reason = track.missing_reason or track.binding.missing_reason
        elif track.binding.point_role == PointRole.BACKGROUND_POINT:
            reason = "background_point_not_object_structure"
        elif track.binding.point_role == PointRole.BOUNDARY_POINT and not allow_boundary_points:
            reason = "unstable_object_boundary_point"
        elif len(valid_2d) < minimum_track_length:
            reason = "insufficient_track_length"
        elif len(valid_3d) < minimum_track_length:
            reason = "insufficient_valid_3d_track_length"
        elif track.binding.track_consistency_ratio < minimum_assignment_consistency:
            reason = "insufficient_assignment_consistency"
        elif not valid_2d or float(np.mean([point.tracking_confidence for point in valid_2d])) < minimum_tracking_confidence:
            reason = "low_tracking_confidence"
        elif valid_3d and float(np.mean([point.depth_quality for point in valid_3d])) < minimum_depth_quality:
            reason = "low_depth_quality"
        elif valid_2d and sum(point.occlusion_status not in {"visible", "unknown"} for point in valid_2d) / len(valid_2d) > maximum_occluded_ratio:
            reason = "frequent_occlusion"
        if not reason and len(valid_3d) >= 3:
            coordinates = []
            for point in valid_3d:
                value = point.point_3d_world or point.point_3d_camera
                if value is not None:
                    coordinates.append(np.asarray(value, dtype=float))
            jumps = np.linalg.norm(np.diff(np.asarray(coordinates), axis=0), axis=1) if len(coordinates) >= 3 else np.asarray([])
            positive = jumps[jumps > 1e-10]
            if len(positive) >= 2 and float(np.max(positive)) > 8.0 * max(float(np.median(positive)), 1e-8):
                reason = "excessive_3d_position_jump"
        if reason:
            rejected[track.binding.point_id] = reason
        else:
            selected.append(track)
    return tuple(selected), rejected


def assemble_object_point_tracks_3d(
    bindings: Sequence[ObjectPointBinding],
    points_2d: Sequence[PointTrack2DObservation],
    points_3d: Sequence[PointTrack3DObservation],
    *,
    object_scale_by_track_and_frame: Optional[Mapping[str, Mapping[int, Optional[float]]]] = None,
) -> tuple[ObjectPointTrack3D, ...]:
    """Join binding, 2D observations, and reconstructed 3D samples by point ID."""

    by_2d: dict[str, list[PointTrack2DObservation]] = {}
    by_3d: dict[str, list[PointTrack3DObservation]] = {}
    for point in points_2d:
        by_2d.setdefault(point.point_id, []).append(point)
    for point in points_3d:
        by_3d.setdefault(point.point_id, []).append(point)
    scales = object_scale_by_track_and_frame or {}
    output = []
    for binding in bindings:
        semantic_label = str(binding.metadata.get("semantic_label", "background"))
        samples_2d = tuple(sorted(by_2d.get(binding.point_id, ()), key=lambda item: item.frame_index))
        samples_3d = tuple(sorted(by_3d.get(binding.point_id, ()), key=lambda item: item.frame_index))
        valid = binding.valid and bool(samples_2d)
        output.append(ObjectPointTrack3D(
            binding=binding,
            semantic_label=semantic_label,
            points_2d=samples_2d,
            points_3d=samples_3d,
            object_scale_by_frame=scales.get(binding.object_track_id, {}),
            valid=valid,
            quality=binding.assignment_quality if valid else 0.0,
            missing_reason="" if valid else binding.missing_reason or "missing_point_samples",
        ))
    return tuple(output)
