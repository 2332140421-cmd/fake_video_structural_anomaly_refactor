"""Builders that convert extracted video frames into observation JSON objects."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import cv2

from .observations import ClipObservationJSON, FrameObservationJSON
from .providers import BaseObjectProvider

PathLike = Union[str, Path]


def build_frame_observation(
    frame_path: PathLike,
    frame_index: int,
    object_provider: BaseObjectProvider,
) -> FrameObservationJSON:
    """Build a FrameObservationJSON by reading image size and mock objects."""

    path = Path(frame_path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read frame image with OpenCV: {path}")

    height, width = image.shape[:2]
    objects = object_provider.predict(path, frame_index, width, height)
    return FrameObservationJSON(
        frame_index=frame_index,
        frame_id=path.stem,
        width=width,
        height=height,
        objects=objects,
        image_path=str(path),
    )


def build_clip_observation(
    video_id: str,
    clip_id: str,
    frames: Iterable[FrameObservationJSON],
    metadata: Optional[Dict[str, Any]] = None,
) -> ClipObservationJSON:
    """Group frame observations into a ClipObservationJSON."""

    frame_list = list(frames)
    return ClipObservationJSON(
        clip_id=clip_id,
        video_id=video_id,
        frame_indices=[frame.frame_index for frame in frame_list],
        frames=frame_list,
        metadata=dict(metadata or {}),
    )
