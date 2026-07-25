"""Canonical in-memory video decode and clip splitting."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .schemas import VideoClip, VideoMetadata


def read_video(
    video_path: str | Path,
    *,
    resize: tuple[int, int] | None = None,
    max_frames: int | None = None,
) -> tuple[VideoMetadata, list[np.ndarray]]:
    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0.0 else 1.0
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if resize is not None:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {path}.")
    height, width = frames[0].shape[:2]
    count = len(frames) if max_frames is not None else max(len(frames), reported_count)
    metadata = VideoMetadata(
        video_path=path.resolve(),
        video_id=path.stem,
        fps=fps,
        width=width,
        height=height,
        frame_count=count,
        duration_seconds=count / fps,
    )
    return metadata, frames


def split_clips(
    metadata: VideoMetadata,
    frames: Sequence[np.ndarray],
    *,
    clip_length: int,
    clip_stride: int,
    scene_boundaries: Sequence[int] = (),
) -> list[VideoClip]:
    if clip_length < 2 or clip_stride < 1:
        raise ValueError("clip_length must be >=2 and clip_stride must be positive.")
    boundaries = {0, len(frames), *(int(value) for value in scene_boundaries)}
    ordered = sorted(value for value in boundaries if 0 <= value <= len(frames))
    clips: list[VideoClip] = []
    for segment_start, segment_end in zip(ordered, ordered[1:]):
        starts = list(range(segment_start, max(segment_start + 1, segment_end - clip_length + 1), clip_stride))
        if not starts:
            starts = [segment_start]
        final_start = max(segment_start, segment_end - clip_length)
        if starts[-1] != final_start:
            starts.append(final_start)
        for start in dict.fromkeys(starts):
            end = min(start + clip_length, segment_end)
            indices = tuple(range(start, end))
            if len(indices) < 2:
                continue
            clips.append(
                VideoClip(
                    clip_id=f"{metadata.video_id}_clip_{len(clips):04d}",
                    video_id=metadata.video_id,
                    frame_indices=indices,
                    timestamps=tuple(index / metadata.fps for index in indices),
                    frames=tuple(np.asarray(frames[index]) for index in indices),
                )
            )
    return clips
