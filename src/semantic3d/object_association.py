"""Lightweight cross-frame object association for observation JSON frames."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Optional, Sequence

import yaml

from .observations import FrameObservationJSON, ObjectObservationJSON
from .scale_prior import normalize_label


Box = Sequence[float]


def bbox_iou(box_a: Box | None, box_b: Box | None) -> float:
    """Return intersection-over-union for two [x1, y1, x2, y2] boxes."""

    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(value) for value in box_a]
    bx1, by1, bx2, by2 = [float(value) for value in box_b]
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def _bbox_area(box: Box | None) -> float:
    """Return bbox area or zero for invalid boxes."""

    if box is None or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(value) for value in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_center(box: Box) -> tuple[float, float]:
    """Return bbox center in pixel coordinates."""

    x1, y1, x2, y2 = [float(value) for value in box]
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def normalized_center_distance(
    box_a: Box | None,
    box_b: Box | None,
    frame_width: int,
    frame_height: int,
) -> float:
    """Return center distance normalized by the frame diagonal."""

    if box_a is None or box_b is None or len(box_a) != 4 or len(box_b) != 4:
        return float("inf")
    diagonal = math.hypot(float(frame_width), float(frame_height))
    if diagonal <= 0.0:
        return float("inf")
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    return float(math.hypot(ax - bx, ay - by) / diagonal)


def bbox_area_ratio(box_a: Box | None, box_b: Box | None) -> float:
    """Return max(area_a / area_b, area_b / area_a) for two bboxes."""

    area_a = _bbox_area(box_a)
    area_b = _bbox_area(box_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return float("inf")
    return float(max(area_a / area_b, area_b / area_a))


def _load_default_aliases() -> dict[str, str]:
    """Load label aliases from configs/scale_priors.yaml without requiring priors."""

    config_path = Path(__file__).resolve().parents[2] / "configs" / "scale_priors.yaml"
    try:
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        return {}
    aliases = data.get("aliases", {})
    if not isinstance(aliases, dict):
        return {}
    return {
        normalize_label(str(alias)): normalize_label(str(target))
        for alias, target in aliases.items()
    }


def normalize_object_label(
    label: str,
    aliases: Optional[Mapping[str, str]] = None,
) -> str:
    """Normalize an object label and apply optional alias mapping.

    This canonicalization is intentionally weaker than scale-prior resolution:
    it does not require the label or alias target to have a physical size prior.
    That keeps object association and R_depth_cons independent from R_sd.
    """

    normalized = normalize_label(label)
    if aliases is None:
        aliases = _load_default_aliases()
    alias_map = {
        normalize_label(str(alias)): normalize_label(str(target))
        for alias, target in aliases.items()
    }
    return alias_map.get(normalized, normalized)


def deduplicate_frames_by_index(
    frames: Sequence[FrameObservationJSON],
) -> list[FrameObservationJSON]:
    """Sort frames by frame_index and keep one record for duplicate indices."""

    unique: dict[int, FrameObservationJSON] = {}
    for frame in sorted(frames, key=lambda item: int(item.frame_index)):
        unique.setdefault(int(frame.frame_index), frame)
    return [unique[index] for index in sorted(unique)]


@dataclass
class AssociationDiagnostics:
    """Diagnostics from one association pass."""

    duplicate_frame_count: int = 0
    duplicate_track_frame_count: int = 0
    one_to_many_assignment_count: int = 0


@dataclass
class _TrackState:
    """Internal state for an active object track."""

    track_id: str
    label: str
    last_frame_index: int
    last_object: ObjectObservationJSON


class ObjectAssociator:
    """Assign lightweight cross-frame track ids to detected objects.

    Matching uses canonical labels, bbox IoU, normalized center distance, and
    bbox area ratio. It is designed as a deterministic minimum viable tracker
    for residual analysis, not as a ReID or long-term occlusion tracker.
    """

    def __init__(
        self,
        iou_threshold: float = 0.1,
        center_distance_threshold: float = 0.25,
        max_area_ratio: float = 3.0,
        max_frame_gap: int = 1,
        label_aliases: Optional[Mapping[str, str]] = None,
        w_iou: float = 1.0,
        w_center: float = 1.0,
        w_area: float = 0.2,
    ) -> None:
        """Create an object associator with geometric matching thresholds."""

        if iou_threshold < 0.0:
            raise ValueError("iou_threshold must be >= 0.")
        if center_distance_threshold <= 0.0:
            raise ValueError("center_distance_threshold must be > 0.")
        if max_area_ratio < 1.0:
            raise ValueError("max_area_ratio must be >= 1.")
        if max_frame_gap < 1:
            raise ValueError("max_frame_gap must be >= 1.")

        self.iou_threshold = float(iou_threshold)
        self.center_distance_threshold = float(center_distance_threshold)
        self.max_area_ratio = float(max_area_ratio)
        self.max_frame_gap = int(max_frame_gap)
        self.label_aliases = dict(label_aliases or _load_default_aliases())
        self.w_iou = float(w_iou)
        self.w_center = float(w_center)
        self.w_area = float(w_area)
        self.last_diagnostics = AssociationDiagnostics()

    def associate(
        self,
        frames: Sequence[FrameObservationJSON],
    ) -> list[FrameObservationJSON]:
        """Return deduplicated frames whose objects have track_id fields."""

        sorted_frames = deduplicate_frames_by_index(frames)
        diagnostics = AssociationDiagnostics(
            duplicate_frame_count=max(0, len(frames) - len(sorted_frames))
        )
        active_tracks: dict[str, _TrackState] = {}
        next_track_index = 1
        output_frames: list[FrameObservationJSON] = []

        for frame in sorted_frames:
            canonical_objects = [
                replace(
                    obj,
                    canonical_label=normalize_object_label(
                        obj.canonical_label or obj.label,
                        self.label_aliases,
                    ),
                )
                for obj in frame.objects
            ]
            matches = self._match_frame_objects(frame, canonical_objects, active_tracks)
            assigned_track_ids = set(matches.values())
            if len(assigned_track_ids) != len(matches):
                diagnostics.one_to_many_assignment_count += len(matches) - len(
                    assigned_track_ids
                )

            updated_objects: list[ObjectObservationJSON] = []
            for object_index, obj in enumerate(canonical_objects):
                track_id = matches.get(object_index)
                if track_id is None:
                    track_id = f"trk_{next_track_index:06d}"
                    next_track_index += 1
                updated_obj = replace(obj, track_id=track_id)
                updated_objects.append(updated_obj)
                active_tracks[track_id] = _TrackState(
                    track_id=track_id,
                    label=str(updated_obj.canonical_label or updated_obj.label),
                    last_frame_index=int(frame.frame_index),
                    last_object=updated_obj,
                )

            frame_track_ids = [
                str(obj.track_id) for obj in updated_objects if obj.track_id is not None
            ]
            diagnostics.duplicate_track_frame_count += len(frame_track_ids) - len(
                set(frame_track_ids)
            )

            # Drop matched stale ids only after all updates, then age out old tracks.
            stale_tracks = [
                track_id
                for track_id, state in active_tracks.items()
                if track_id not in assigned_track_ids
                and int(frame.frame_index) - state.last_frame_index > self.max_frame_gap
            ]
            for track_id in stale_tracks:
                active_tracks.pop(track_id, None)

            output_frames.append(replace(frame, objects=updated_objects))

        self.last_diagnostics = diagnostics
        return output_frames

    def _match_frame_objects(
        self,
        frame: FrameObservationJSON,
        objects: Sequence[ObjectObservationJSON],
        active_tracks: Mapping[str, _TrackState],
    ) -> dict[int, str]:
        """Match current objects to active tracks and return object_index -> track_id."""

        candidates: list[tuple[int, str, float]] = []
        for object_index, obj in enumerate(objects):
            for track_id, track in active_tracks.items():
                if not self._is_candidate(frame, obj, track):
                    continue
                cost = self._match_cost(frame, obj, track.last_object)
                candidates.append((object_index, track_id, cost))
        if not candidates:
            return {}

        try:
            return self._linear_assignment(candidates)
        except Exception:
            return self._greedy_assignment(candidates)

    def _is_candidate(
        self,
        frame: FrameObservationJSON,
        obj: ObjectObservationJSON,
        track: _TrackState,
    ) -> bool:
        """Return whether a current object is eligible for an active track."""

        frame_gap = int(frame.frame_index) - track.last_frame_index
        if frame_gap < 1 or frame_gap > self.max_frame_gap:
            return False
        if str(obj.canonical_label or obj.label) != track.label:
            return False

        iou = bbox_iou(obj.bbox, track.last_object.bbox)
        center_distance = normalized_center_distance(
            obj.bbox,
            track.last_object.bbox,
            frame.width,
            frame.height,
        )
        area_ratio = bbox_area_ratio(obj.bbox, track.last_object.bbox)
        if center_distance > self.center_distance_threshold:
            return False
        if area_ratio > self.max_area_ratio:
            return False
        return iou >= self.iou_threshold or center_distance <= (
            self.center_distance_threshold * 0.5
        )

    def _match_cost(
        self,
        frame: FrameObservationJSON,
        current: ObjectObservationJSON,
        previous: ObjectObservationJSON,
    ) -> float:
        """Compute weighted association cost from IoU, center, and area terms."""

        iou = bbox_iou(current.bbox, previous.bbox)
        center_distance = normalized_center_distance(
            current.bbox,
            previous.bbox,
            frame.width,
            frame.height,
        )
        current_area = _bbox_area(current.bbox)
        previous_area = _bbox_area(previous.bbox)
        if current_area <= 0.0 or previous_area <= 0.0:
            area_term = float("inf")
        else:
            area_term = abs(math.log(current_area / previous_area))
        return (
            self.w_iou * (1.0 - iou)
            + self.w_center * center_distance
            + self.w_area * area_term
        )

    @staticmethod
    def _greedy_assignment(candidates: Sequence[tuple[int, str, float]]) -> dict[int, str]:
        """Deterministic fallback assignment by ascending cost."""

        matches: dict[int, str] = {}
        used_tracks: set[str] = set()
        for object_index, track_id, _ in sorted(candidates, key=lambda item: item[2]):
            if object_index in matches or track_id in used_tracks:
                continue
            matches[object_index] = track_id
            used_tracks.add(track_id)
        return matches

    @staticmethod
    def _linear_assignment(candidates: Sequence[tuple[int, str, float]]) -> dict[int, str]:
        """Solve candidate assignment with scipy when available."""

        from scipy.optimize import linear_sum_assignment
        import numpy as np

        object_indices = sorted({item[0] for item in candidates})
        track_ids = sorted({item[1] for item in candidates})
        object_to_row = {object_index: row for row, object_index in enumerate(object_indices)}
        track_to_col = {track_id: col for col, track_id in enumerate(track_ids)}
        large = 1e9
        matrix = np.full((len(object_indices), len(track_ids)), large, dtype=float)
        for object_index, track_id, cost in candidates:
            matrix[object_to_row[object_index], track_to_col[track_id]] = cost

        row_indices, col_indices = linear_sum_assignment(matrix)
        matches: dict[int, str] = {}
        for row, col in zip(row_indices, col_indices):
            if matrix[row, col] >= large:
                continue
            matches[object_indices[row]] = track_ids[col]
        return matches
