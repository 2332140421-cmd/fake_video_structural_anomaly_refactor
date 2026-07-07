"""Video frame extraction and clip-window utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ClipWindow:
    """A sliding-window clip over extracted frame paths."""

    clip_id: str
    frame_indices: List[int]
    frame_paths: List[Path]


def extract_frames(
    video_path: PathLike,
    output_dir: PathLike,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> Tuple[List[Path], List[int]]:
    """Extract frames from a local video with OpenCV.

    Args:
        video_path: Path to the input video.
        output_dir: Directory where extracted PNG frames are written.
        fps: Optional target sampling rate. If omitted, every decoded frame is
            saved. If larger than the native FPS, every frame is saved.
        max_frames: Optional cap on the number of saved frames.

    Returns:
        A tuple of saved frame paths and their original video frame indices.
    """

    input_path = Path(video_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {input_path}")
    if fps is not None and fps <= 0:
        raise ValueError(f"fps must be > 0 when provided, got {fps}.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be > 0 when provided, got {max_frames}.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video with OpenCV: {input_path}")

    native_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    sample_every = 1
    if fps is not None and native_fps > 0:
        sample_every = max(1, int(round(native_fps / fps)))

    frame_paths: List[Path] = []
    frame_indices: List[int] = []
    frame_number = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % sample_every == 0:
                frame_file = output_path / f"{input_path.stem}_frame_{frame_number:06d}.png"
                if not cv2.imwrite(str(frame_file), frame):
                    raise IOError(f"Failed to write extracted frame: {frame_file}")
                frame_paths.append(frame_file)
                frame_indices.append(frame_number)
                if max_frames is not None and len(frame_paths) >= max_frames:
                    break
            frame_number += 1
    finally:
        capture.release()

    return frame_paths, frame_indices


def build_clips(
    frame_paths: Sequence[PathLike], clip_len: int = 8, stride: int = 4
) -> List[ClipWindow]:
    """Split extracted frames into sliding-window clips.

    Args:
        frame_paths: Ordered frame paths.
        clip_len: Number of frames per clip window.
        stride: Sliding step between two adjacent clips.

    Returns:
        A list of ClipWindow objects. If the sequence is shorter than
        clip_len, one shorter clip is returned so small demos can still run.
    """

    if clip_len <= 0:
        raise ValueError(f"clip_len must be > 0, got {clip_len}.")
    if stride <= 0:
        raise ValueError(f"stride must be > 0, got {stride}.")

    paths = [Path(path) for path in frame_paths]
    if not paths:
        return []

    windows: List[ClipWindow] = []
    if len(paths) <= clip_len:
        windows.append(
            ClipWindow(
                clip_id="clip_000",
                frame_indices=list(range(len(paths))),
                frame_paths=paths,
            )
        )
        return windows

    clip_number = 0
    for start in range(0, len(paths) - clip_len + 1, stride):
        end = start + clip_len
        windows.append(
            ClipWindow(
                clip_id=f"clip_{clip_number:03d}",
                frame_indices=list(range(start, end)),
                frame_paths=paths[start:end],
            )
        )
        clip_number += 1
    return windows
