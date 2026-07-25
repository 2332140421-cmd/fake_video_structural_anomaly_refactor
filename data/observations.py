"""Build one shared ClipObservation; providers execute at most once per frame."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from models.providers import DepthIntrinsicsProvider, ObjectProvider, PoseProvider, TrackProvider

from .schemas import ClipObservation, FrameObservation, ObjectObservation, VideoClip


def _foreground(objects: Sequence[ObjectObservation], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for obj in objects:
        if obj.instance_mask is not None and obj.instance_mask.shape == shape:
            mask |= obj.instance_mask
    return mask


def build_shared_observations(
    clip: VideoClip,
    *,
    object_provider: ObjectProvider,
    depth_provider: DepthIntrinsicsProvider,
    pose_provider: PoseProvider,
    track_provider: TrackProvider | None = None,
) -> ClipObservation:
    frames: list[FrameObservation] = []
    for image, index, timestamp in zip(clip.frames, clip.frame_indices, clip.timestamps):
        objects = list(object_provider.predict(image, index))
        depth, valid, depth_confidence, intrinsics, quality = depth_provider.predict(image, index)
        frames.append(
            FrameObservation(
                video_id=clip.video_id,
                clip_id=clip.clip_id,
                frame_index=index,
                timestamp=timestamp,
                image=image,
                objects=objects,
                metric_depth=depth,
                depth_valid_mask=valid,
                depth_confidence=depth_confidence,
                intrinsics=intrinsics,
                availability={"objects": True, "metric_depth": True, "intrinsics": True},
                confidence={"metric_depth": quality},
            )
        )
    for previous, current in zip(frames, frames[1:]):
        transform, confidence, status = pose_provider.estimate(
            previous.image,
            current.image,
            previous.frame_index,
            current.frame_index,
            previous.metric_depth,
            previous.depth_valid_mask,
            previous.intrinsics,
            current.intrinsics,
            _foreground(previous.objects, previous.image.shape[:2]),
            _foreground(current.objects, current.image.shape[:2]),
        )
        current.relative_pose_from_previous = transform
        current.availability["relative_pose"] = transform is not None
        current.confidence["relative_pose"] = confidence
        if transform is None:
            current.occlusion_states["_pose"] = status
    tracks = [] if track_provider is None else list(track_provider.track(clip, frames))
    return ClipObservation(video_id=clip.video_id, clip_id=clip.clip_id, frames=frames, tracks=tracks)
