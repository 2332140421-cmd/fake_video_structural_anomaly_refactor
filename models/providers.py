"""Thin provider contracts and adapters around the verified legacy providers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from data.schemas import FrameObservation, ObjectObservation, TrackObservation, VideoClip
from models.geometry import backproject_points
from semantic3d.dynamic_3d import DynamicGeometryMode
from semantic3d.keypoint_provider import RealHumanKeypointProvider
from semantic3d.method_completion.metric_depth_adapters import UniDepthV2Adapter
from semantic3d.object_association import ObjectAssociator, bbox_iou
from semantic3d.occlusion.mask_observation import InstanceMaskObservation
from semantic3d.occlusion.mask_provider import RealInstanceMaskProvider
from semantic3d.occlusion.support_prediction import predict_object_support
from semantic3d.occlusion.visibility_state import VisibilityState, infer_visibility_state
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.pose_d2.provider import ShortBaselinePoseProvider
from semantic3d.real_object_provider import RealObjectProvider
from semantic3d.sequence_geometry.pose_estimation import track_background_correspondences


class ObjectProvider(Protocol):
    def predict(self, frame: np.ndarray, frame_index: int) -> Sequence[ObjectObservation]: ...


class DepthIntrinsicsProvider(Protocol):
    def predict(
        self, frame: np.ndarray, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, float]: ...


class PoseProvider(Protocol):
    def estimate(
        self,
        source: np.ndarray,
        target: np.ndarray,
        frame_t: int,
        frame_t1: int,
        source_depth: np.ndarray,
        source_valid: np.ndarray,
        source_k: np.ndarray,
        target_k: np.ndarray,
        source_foreground: np.ndarray | None,
        target_foreground: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float, str]: ...


class TrackProvider(Protocol):
    def track(
        self,
        clip: VideoClip,
        frames: Sequence[FrameObservation],
    ) -> Sequence[TrackObservation]: ...


class LegacyObjectProviderAdapter:
    """Adapt the path-based verified detector without changing its implementation."""

    def __init__(
        self,
        provider: RealObjectProvider,
        mask_provider: RealInstanceMaskProvider | None = None,
    ) -> None:
        self.provider = provider
        self.mask_provider = mask_provider

    def predict(self, frame: np.ndarray, frame_index: int) -> list[ObjectObservation]:
        height, width = frame.shape[:2]
        with tempfile.TemporaryDirectory(
            prefix="paper_core_frame_",
            dir=os.environ.get("CACHE_ROOT"),
        ) as directory:
            path = Path(directory) / "frame.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Failed to prepare provider frame.")
            rows = self.provider.predict(path, frame_index, width, height)
            masks = ()
            if self.mask_provider is not None:
                legacy_frame = FrameObservationJSON(
                    frame_index=frame_index,
                    frame_id=f"frame_{frame_index}",
                    width=width,
                    height=height,
                    objects=rows,
                    image_path=str(path),
                )
                masks = self.mask_provider.predict(
                    video_id="paper_core_video",
                    frame=legacy_frame,
                )
        masks_by_track = {
            mask.object_track_id: mask
            for mask in masks
            if mask.valid and mask.selected_mask is not None
        }
        return [
            ObjectObservation(
                object_id=row.object_id,
                track_id=str(row.track_id or row.person_track_id or row.object_id),
                category=str(
                    row.provenance.get("raw_label", row.label)
                ).strip().lower().replace(" ", "_"),
                bbox_xyxy=tuple(float(value) for value in (row.bbox or (0, 0, width, height))),
                confidence=float(row.confidence),
                instance_mask=(
                    None
                    if str(row.track_id or row.person_track_id or row.object_id)
                    not in masks_by_track
                    else masks_by_track[
                        str(row.track_id or row.person_track_id or row.object_id)
                    ].selected_mask
                ),
                mask_quality=float(
                    masks_by_track[
                        str(row.track_id or row.person_track_id or row.object_id)
                    ].confidence
                    if str(row.track_id or row.person_track_id or row.object_id)
                    in masks_by_track
                    else row.quality or row.confidence
                ),
                metadata={"legacy_provider": type(self.provider).__name__},
            )
            for row in rows
        ]


class LegacyDepthIntrinsicsProviderAdapter:
    """Adapt verified UniDepthV2 frame inference to the in-memory contract."""

    def __init__(self, provider: UniDepthV2Adapter) -> None:
        self.provider = provider

    def predict(
        self, frame: np.ndarray, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, float]:
        with tempfile.TemporaryDirectory(
            prefix="paper_core_depth_",
            dir=os.environ.get("CACHE_ROOT"),
        ) as directory:
            path = Path(directory) / "frame.png"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError("Failed to prepare metric-depth frame.")
            result = self.provider.predict_frame(path, frame_index=frame_index)
        depth = result.depth_observation
        return (
            np.asarray(depth.depth_map, dtype=float),
            np.asarray(depth.valid_mask, dtype=bool),
            None if depth.confidence_map is None else np.asarray(depth.confidence_map, dtype=float),
            np.asarray(result.camera_observation.K, dtype=float),
            float(depth.quality),
        )


class LegacyPoseProviderAdapter:
    def __init__(self, provider: ShortBaselinePoseProvider | None = None) -> None:
        self.provider = provider or ShortBaselinePoseProvider()
        self.pair_metadata: dict[tuple[int, int], dict] = {}

    def estimate(
        self,
        source: np.ndarray,
        target: np.ndarray,
        frame_t: int,
        frame_t1: int,
        source_depth: np.ndarray,
        source_valid: np.ndarray,
        source_k: np.ndarray,
        target_k: np.ndarray,
        source_foreground: np.ndarray | None,
        target_foreground: np.ndarray | None,
    ) -> tuple[np.ndarray | None, float, str]:
        result = self.provider.estimate_pair(
            source,
            target,
            frame_t=frame_t,
            frame_t1=frame_t1,
            K_source=source_k,
            K_target=target_k,
            source_depth_m=source_depth,
            source_depth_valid_mask=source_valid,
            source_foreground_mask=source_foreground,
            target_foreground_mask=target_foreground,
        )
        self.pair_metadata[(frame_t, frame_t1)] = {
            **dict(result.metadata),
            "confidence": float(result.confidence),
            "valid": bool(result.valid),
            "provider_status": result.provider_status.value,
        }
        transform = None if result.T_target_from_source is None else np.asarray(result.T_target_from_source)
        return transform, float(result.confidence), str(result.failure_reason or result.provider_status.value)


class LegacyTrackProviderAdapter:
    """Thin bridge from verified association/KLT/visibility code to paper-core contracts."""

    def __init__(
        self,
        pose_provider: LegacyPoseProviderAdapter,
        *,
        max_frame_gap: int = 2,
        maximum_point_tracks: int = 96,
    ) -> None:
        self.pose_provider = pose_provider
        self.associator = ObjectAssociator(max_frame_gap=max_frame_gap)
        self.maximum_point_tracks = int(maximum_point_tracks)

    @staticmethod
    def _legacy_frame(frame: FrameObservation) -> FrameObservationJSON:
        height, width = frame.image.shape[:2]
        return FrameObservationJSON(
            frame_index=frame.frame_index,
            frame_id=f"{frame.video_id}:{frame.frame_index}",
            width=width,
            height=height,
            objects=[
                ObjectObservationJSON(
                    object_id=obj.object_id,
                    label=obj.category,
                    mask_area=(
                        float(np.count_nonzero(obj.instance_mask))
                        if obj.instance_mask is not None
                        else 0.0
                    ),
                    frame_area=float(height * width),
                    depth=float("nan"),
                    confidence=obj.confidence,
                    bbox=list(obj.bbox_xyxy),
                    quality=obj.mask_quality,
                )
                for obj in frame.objects
            ],
        )

    def _associate_objects(self, frames: Sequence[FrameObservation]) -> None:
        associated = self.associator.associate([self._legacy_frame(frame) for frame in frames])
        last_seen: dict[str, tuple[int, tuple[float, float, float, float]]] = {}
        known_tracks: set[str] = set()
        for frame, legacy in zip(frames, associated, strict=True):
            visible = set()
            height, width = frame.image.shape[:2]
            for obj, linked in zip(frame.objects, legacy.objects, strict=True):
                track_id = f"{frame.clip_id}:{linked.track_id}"
                previous = last_seen.get(track_id)
                gap = 0 if previous is None else frame.frame_index - previous[0]
                association_confidence = (
                    obj.confidence
                    if previous is None
                    else min(obj.confidence, max(bbox_iou(previous[1], obj.bbox_xyxy), 0.05))
                )
                obj.track_id = track_id
                obj.track_identity_stable = True
                x1, y1, x2, y2 = obj.bbox_xyxy
                obj.truncated = bool(x1 <= 0 or y1 <= 0 or x2 >= width or y2 >= height)
                obj.metadata.update(
                    {
                        "frame_index": frame.frame_index,
                        "class_id": obj.metadata.get("class_id"),
                        "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                        "association_confidence": float(association_confidence),
                        "association_scope": "clip",
                        "track_status": "REAPPEARED" if gap > 1 else "ACTIVE",
                    }
                )
                if gap > 1:
                    frame.reappearance_states[track_id] = "UNAVAILABLE_NO_FORMAL_REID"
                last_seen[track_id] = (frame.frame_index, obj.bbox_xyxy)
                known_tracks.add(track_id)
                visible.add(track_id)
            for track_id in known_tracks - visible:
                previous = last_seen.get(track_id)
                if previous is not None and frame.frame_index - previous[0] <= self.associator.max_frame_gap:
                    frame.occlusion_states[track_id] = "UNAVAILABLE_DETECTOR_MISSING"

    @staticmethod
    def _mask_observation(frame: FrameObservation, obj: ObjectObservation) -> InstanceMaskObservation | None:
        if obj.instance_mask is None or not np.any(obj.instance_mask):
            return None
        return InstanceMaskObservation.from_visible_mask(
            video_id=frame.video_id,
            frame_index=frame.frame_index,
            object_track_id=obj.track_id,
            semantic_label=obj.category,
            mask=obj.instance_mask,
            confidence=min(obj.confidence, obj.mask_quality),
            source_provider="paper_core_visible_instance_mask",
        )

    def _visibility(self, frames: Sequence[FrameObservation]) -> None:
        histories: dict[str, list[InstanceMaskObservation]] = {}
        states: dict[str, VisibilityState] = {}
        known: dict[str, str] = {}
        for frame in frames:
            current = {obj.track_id: obj for obj in frame.objects}
            for obj in frame.objects:
                known[obj.track_id] = obj.category
            for track_id in sorted(set(histories) | set(current)):
                prediction = predict_object_support(
                    histories.get(track_id, ()),
                    target_frame_index=frame.frame_index,
                    geometry_mode=DynamicGeometryMode.STATIC_CAMERA_3D,
                )
                observed = (
                    None
                    if track_id not in current
                    else self._mask_observation(frame, current[track_id])
                )
                visibility = infer_visibility_state(
                    prediction,
                    observed,
                    previous_state=states.get(track_id, VisibilityState.UNCERTAIN),
                    # Visible masks alone do not establish depth ordering.
                    nearer_object_masks={},
                    detector_confidence=(current[track_id].confidence if track_id in current else 1.0),
                    detection_confirmed_absent=False,
                )
                frame.visibility_observations[track_id] = visibility
                states[track_id] = visibility.current_state
                frame.occlusion_states[track_id] = visibility.current_state.value.upper()
                if observed is not None:
                    histories.setdefault(track_id, []).append(observed)

    @staticmethod
    def _inside(mask: np.ndarray | None, xy: np.ndarray) -> np.ndarray:
        if mask is None:
            return np.zeros(len(xy), dtype=bool)
        height, width = mask.shape
        columns = np.rint(xy[:, 0]).astype(int)
        rows = np.rint(xy[:, 1]).astype(int)
        valid = (columns >= 0) & (columns < width) & (rows >= 0) & (rows < height)
        output = np.zeros(len(xy), dtype=bool)
        output[valid] = mask[rows[valid], columns[valid]]
        return output

    def _pair_candidates(
        self,
        source: FrameObservation,
        target: FrameObservation,
    ) -> list[tuple[np.ndarray, np.ndarray, str, str, float]]:
        diagnostics = track_background_correspondences(
            source.image,
            target.image,
            source_frame_index=source.frame_index,
            target_frame_index=target.frame_index,
        )
        candidates: list[tuple[np.ndarray, np.ndarray, str, str, float]] = []
        target_objects = {obj.track_id: obj for obj in target.objects}
        for obj in source.objects:
            other = target_objects.get(obj.track_id)
            if other is None:
                continue
            keep = self._inside(obj.instance_mask, diagnostics.source_points)
            keep &= self._inside(other.instance_mask, diagnostics.target_points)
            for source_xy, target_xy in zip(
                diagnostics.source_points[keep], diagnostics.target_points[keep], strict=True
            ):
                candidates.append((source_xy, target_xy, obj.track_id, "OBJECT", min(obj.confidence, other.confidence)))
        if not candidates:
            rows = self.pose_provider.pair_metadata.get(
                (source.frame_index, target.frame_index), {}
            ).get("track_rows", ())
            for row in rows:
                candidates.append(
                    (
                        np.asarray([row["source_x"], row["source_y"]], dtype=float),
                        np.asarray([row["target_x"], row["target_y"]], dtype=float),
                        "_background",
                        "BACKGROUND",
                        float(self.pose_provider.pair_metadata.get(
                            (source.frame_index, target.frame_index), {}
                        ).get("confidence", 1.0)),
                    )
                )
        return sorted(candidates, key=lambda row: (row[2], row[0][1], row[0][0]))[
            : self.maximum_point_tracks
        ]

    def _point_tracks(
        self,
        clip: VideoClip,
        frames: Sequence[FrameObservation],
    ) -> list[TrackObservation]:
        paths: dict[str, dict[str, object]] = {}
        active: dict[str, tuple[str, str, np.ndarray]] = {}
        next_id = 1
        for source, target in zip(frames, frames[1:]):
            candidates = self._pair_candidates(source, target)
            rows = np.asarray(
                [[*source_xy, *target_xy] for source_xy, target_xy, *_ in candidates],
                dtype=float,
            ).reshape(-1, 4)
            target.actual_correspondences = rows if len(rows) else None
            next_active: dict[str, tuple[str, str, np.ndarray]] = {}
            used: set[str] = set()
            for source_xy, target_xy, owner, support_type, confidence in candidates:
                eligible = [
                    (float(np.linalg.norm(endpoint - source_xy)), point_id)
                    for point_id, (active_owner, active_support, endpoint) in active.items()
                    if active_owner == owner and active_support == support_type and point_id not in used
                ]
                distance, point_id = min(eligible, default=(float("inf"), ""))
                if distance > 2.0:
                    point_id = f"pt_{clip.clip_id}_{next_id:04d}"
                    next_id += 1
                    paths[point_id] = {
                        "owner": owner,
                        "support_type": support_type,
                        "frames": [source.frame_index],
                        "xy": [source_xy],
                        "confidence": [confidence],
                    }
                path = paths[point_id]
                if path["frames"][-1] != target.frame_index:
                    path["frames"].append(target.frame_index)
                    path["xy"].append(target_xy)
                    path["confidence"].append(confidence)
                used.add(point_id)
                next_active[point_id] = (owner, support_type, target_xy)
            active = next_active
        by_frame = {frame.frame_index: frame for frame in frames}
        output = []
        for point_id, path in paths.items():
            frame_indices = tuple(path["frames"])
            if len(frame_indices) < 2:
                continue
            actual = np.asarray(path["xy"], dtype=float)
            points_3d = np.full((len(frame_indices), 3), np.nan, dtype=float)
            valid = np.zeros(len(frame_indices), dtype=bool)
            for position, (frame_index, xy) in enumerate(zip(frame_indices, actual, strict=True)):
                frame = by_frame[frame_index]
                x, y = (int(round(value)) for value in xy)
                if (
                    frame.metric_depth is None
                    or frame.intrinsics is None
                    or not (0 <= y < frame.image.shape[0] and 0 <= x < frame.image.shape[1])
                    or (frame.depth_valid_mask is not None and not frame.depth_valid_mask[y, x])
                ):
                    continue
                point = backproject_points(
                    xy[None, :], np.asarray([frame.metric_depth[y, x]]), frame.intrinsics
                )
                if len(point):
                    points_3d[position] = point[0]
                    valid[position] = True
            output.append(
                TrackObservation(
                    track_id=point_id,
                    object_id=str(path["owner"]),
                    frame_indices=frame_indices,
                    actual_xy=actual,
                    points_3d=points_3d,
                    valid_mask=valid,
                    confidence=float(np.mean(path["confidence"])),
                    metadata={
                        "support_type": path["support_type"],
                        "source_tracker": "opencv_lk_forward_backward",
                        "actual_observation": True,
                    },
                )
            )
        return output

    def track(
        self,
        clip: VideoClip,
        frames: Sequence[FrameObservation],
    ) -> Sequence[TrackObservation]:
        self._associate_objects(frames)
        self._visibility(frames)
        return self._point_tracks(clip, frames)


__all__ = [
    "DepthIntrinsicsProvider",
    "LegacyDepthIntrinsicsProviderAdapter",
    "LegacyObjectProviderAdapter",
    "LegacyPoseProviderAdapter",
    "LegacyTrackProviderAdapter",
    "ObjectProvider",
    "PoseProvider",
    "RealHumanKeypointProvider",
    "RealInstanceMaskProvider",
    "RealObjectProvider",
    "ShortBaselinePoseProvider",
    "TrackProvider",
    "UniDepthV2Adapter",
]
