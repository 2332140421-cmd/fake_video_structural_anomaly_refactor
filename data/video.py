"""Canonical video decode, deterministic sampling, and clip splitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .schemas import VideoClip, VideoMetadata


SAMPLER_VERSION = "uniform_full_video_integer_v1"


class InsufficientUniqueFramesError(ValueError):
    """Raised when a video cannot supply the requested number of unique frames."""


@dataclass(frozen=True)
class UniformVideoSample:
    """A fixed-size, full-duration sample with source-frame provenance."""

    metadata: VideoMetadata
    frames: tuple[np.ndarray, ...]
    frame_indices: tuple[int, ...]
    timestamps: tuple[float, ...]
    sampler_version: str = SAMPLER_VERSION


def uniform_frame_indices(source_frame_count: int, requested_frame_count: int) -> tuple[int, ...]:
    """Return unique integer indices spanning the first through last source frame."""

    source = int(source_frame_count)
    requested = int(requested_frame_count)
    if requested < 2:
        raise ValueError("requested_frame_count must be at least 2.")
    if source < requested:
        raise InsufficientUniqueFramesError(
            f"INSUFFICIENT_UNIQUE_FRAMES: source={source}, requested={requested}."
        )
    # The integer formula is version-independent and includes both endpoints.
    indices = tuple(
        (position * (source - 1)) // (requested - 1)
        for position in range(requested)
    )
    if len(set(indices)) != requested or any(
        current >= following for current, following in zip(indices, indices[1:])
    ):
        raise RuntimeError("Uniform sampler failed to produce strictly increasing indices.")
    return indices


def _count_decodable_frames(path: Path) -> tuple[int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0.0 else 1.0
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    return count, fps


def read_uniform_video_sample(
    video_path: str | Path,
    *,
    requested_frame_count: int = 32,
    resize: tuple[int, int] | None = None,
) -> UniformVideoSample:
    """Decode only a deterministic uniform sample while preserving source indices."""

    path = Path(video_path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps if fps > 0.0 else 1.0
    reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if reported_count < requested_frame_count:
        reported_count, fps = _count_decodable_frames(path)
    indices = uniform_frame_indices(reported_count, requested_frame_count)
    wanted = set(indices)
    selected: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {path}.")
    decoded_count = 0
    while decoded_count <= indices[-1]:
        ok, frame = capture.read()
        if not ok:
            break
        if decoded_count in wanted:
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
            selected[decoded_count] = frame
        decoded_count += 1
    capture.release()
    if len(selected) != requested_frame_count:
        # A container can over-report frame count. Count once, then retry with the
        # verified decodable count instead of looping or duplicating frames.
        actual_count, fps = _count_decodable_frames(path)
        indices = uniform_frame_indices(actual_count, requested_frame_count)
        wanted = set(indices)
        selected = {}
        capture = cv2.VideoCapture(str(path))
        decoded_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if decoded_count in wanted:
                if resize is not None:
                    frame = cv2.resize(frame, resize, interpolation=cv2.INTER_AREA)
                selected[decoded_count] = frame
            decoded_count += 1
        capture.release()
        reported_count = actual_count
    if len(selected) != requested_frame_count:
        raise InsufficientUniqueFramesError(
            f"INSUFFICIENT_UNIQUE_FRAMES: decoded={len(selected)}, "
            f"requested={requested_frame_count}."
        )
    frames = tuple(selected[index] for index in indices)
    height, width = frames[0].shape[:2]
    metadata = VideoMetadata(
        video_path=path.resolve(),
        video_id=path.stem,
        fps=fps,
        width=width,
        height=height,
        frame_count=reported_count,
        duration_seconds=reported_count / fps,
    )
    return UniformVideoSample(
        metadata=metadata,
        frames=frames,
        frame_indices=indices,
        timestamps=tuple(index / fps for index in indices),
    )


def split_uniform_sample(
    sample: UniformVideoSample,
    *,
    clip_length: int,
    clip_count: int,
) -> list[VideoClip]:
    """Partition a uniform sample into fixed, non-overlapping temporal clips.

    Clip-local algorithms receive consecutive sampled positions.  The original
    source-frame indices remain on ``UniformVideoSample`` as provenance, so
    skipped source frames are not misclassified as missing observations.
    """

    if clip_length < 2 or clip_count < 1:
        raise ValueError("clip_length must be >=2 and clip_count must be positive.")
    expected = clip_length * clip_count
    if len(sample.frames) != expected:
        raise ValueError(
            f"Uniform sample contains {len(sample.frames)} frames; expected {expected}."
        )
    clips = []
    for clip_index in range(clip_count):
        start = clip_index * clip_length
        end = start + clip_length
        clips.append(
            VideoClip(
                clip_id=f"{sample.metadata.video_id}_clip_{clip_index:04d}",
                video_id=sample.metadata.video_id,
                frame_indices=tuple(range(start, end)),
                timestamps=sample.timestamps[start:end],
                frames=sample.frames[start:end],
            )
        )
    return clips


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


__all__ = [
    "InsufficientUniqueFramesError",
    "SAMPLER_VERSION",
    "UniformVideoSample",
    "read_uniform_video_sample",
    "read_video",
    "split_clips",
    "split_uniform_sample",
    "uniform_frame_indices",
]
