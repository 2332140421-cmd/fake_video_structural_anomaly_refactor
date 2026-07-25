"""Thin provider contracts and adapters around the verified legacy providers."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, Sequence

import cv2
import numpy as np

from data.schemas import ObjectObservation, TrackObservation, VideoClip
from semantic3d.keypoint_provider import RealHumanKeypointProvider
from semantic3d.method_completion.metric_depth_adapters import UniDepthV2Adapter
from semantic3d.occlusion.mask_provider import RealInstanceMaskProvider
from semantic3d.observations import FrameObservationJSON
from semantic3d.pose_d2.provider import ShortBaselinePoseProvider
from semantic3d.real_object_provider import RealObjectProvider


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
    def track(self, clip: VideoClip, objects_by_frame: Sequence[Sequence[ObjectObservation]]) -> Sequence[TrackObservation]: ...


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
                category=row.label,
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
        transform = None if result.T_target_from_source is None else np.asarray(result.T_target_from_source)
        return transform, float(result.confidence), str(result.failure_reason or result.provider_status.value)


__all__ = [
    "DepthIntrinsicsProvider",
    "LegacyDepthIntrinsicsProviderAdapter",
    "LegacyObjectProviderAdapter",
    "LegacyPoseProviderAdapter",
    "ObjectProvider",
    "PoseProvider",
    "RealHumanKeypointProvider",
    "RealInstanceMaskProvider",
    "RealObjectProvider",
    "ShortBaselinePoseProvider",
    "TrackProvider",
    "UniDepthV2Adapter",
]
