"""Video indexing, scene segmentation, and overlap-safe clip manifests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .ids import StableIdFactory, stable_id
from .writer import sha256_file


@dataclass(frozen=True)
class VideoProbe:
    """Decoded container metadata used in videos.parquet."""

    frame_count: int
    fps: float
    width: int
    height: int
    decode_status: str
    failure_reason: str = ""


def inspect_video(path: str | Path) -> VideoProbe:
    """Inspect one video and verify that at least its first frame decodes."""

    source = Path(path)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        return VideoProbe(0, math.nan, 0, 0, "failed", "video_open_failed")
    try:
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        return VideoProbe(frame_count, fps, width, height, "failed", "first_frame_decode_failed")
    return VideoProbe(frame_count, fps, width, height, "ok", "")


def split_scene_segments(
    frame_signatures: Sequence[float], *, cut_threshold: float
) -> list[tuple[int, int]]:
    """Split scalar frame signatures into inclusive scene segments."""

    if cut_threshold < 0.0:
        raise ValueError("cut_threshold must be non-negative")
    if not frame_signatures:
        return []
    starts = [0]
    for index in range(1, len(frame_signatures)):
        if abs(float(frame_signatures[index]) - float(frame_signatures[index - 1])) >= cut_threshold:
            starts.append(index)
    starts.append(len(frame_signatures))
    return [(starts[i], starts[i + 1] - 1) for i in range(len(starts) - 1)]


def decode_frame_signatures(video_path: str | Path) -> tuple[list[float], list[int]]:
    """Decode every frame and return deterministic low-cost scene signatures."""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    signatures: list[float] = []
    failures: list[int] = []
    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame is None:
                failures.append(frame_index)
                signatures.append(signatures[-1] if signatures else 0.0)
            else:
                small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
                signatures.append(float(np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))) / 255.0)
            frame_index += 1
    finally:
        capture.release()
    return signatures, failures


def _window_starts(start: int, end: int, window_size: int, stride: int) -> list[int]:
    length = end - start + 1
    if length <= window_size:
        return [start]
    starts = list(range(start, end - window_size + 2, stride))
    final = end - window_size + 1
    if starts[-1] != final:
        starts.append(final)
    return sorted(set(starts))


def build_clip_manifests(
    *,
    video_id: str,
    frame_count: int,
    scene_segments: Sequence[tuple[int, int]],
    id_factory: StableIdFactory,
    window_size: int,
    stride: int,
    left_context: int,
    right_context: int,
    minimum_clip_length: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build overlapping clips with one deterministic owner per source frame."""

    if min(window_size, stride, minimum_clip_length) <= 0:
        raise ValueError("window_size, stride, and minimum_clip_length must be positive")
    if left_context < 0 or right_context < 0:
        raise ValueError("context sizes must be non-negative")
    clips: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    candidate_by_frame: dict[int, list[dict[str, Any]]] = {}
    for scene_id, (scene_start, scene_end) in enumerate(scene_segments):
        scene_length = scene_end - scene_start + 1
        if scene_length < minimum_clip_length:
            continue
        for ordinal, core_start in enumerate(_window_starts(scene_start, scene_end, window_size, stride)):
            core_end = min(scene_end, core_start + window_size - 1)
            visible_start = max(scene_start, core_start - left_context)
            visible_end = min(scene_end, core_end + right_context)
            clip_id = id_factory.clip(video_id, scene_id, visible_start, visible_end)
            coordinate_system_id = id_factory.coordinate_system(clip_id)
            clip = {
                "clip_id": clip_id,
                "video_id": video_id,
                "scene_id": scene_id,
                "clip_ordinal": ordinal,
                "start_frame_index": visible_start,
                "end_frame_index": visible_end,
                "core_start_frame_index": core_start,
                "core_end_frame_index": core_end,
                "reference_frame_index": (core_start + core_end) // 2,
                "frame_count": visible_end - visible_start + 1,
                "coordinate_system_id": coordinate_system_id,
                "geometry_mode": "unavailable",
                "sequence_scale_status": "unknown",
                "depth_alignment_domain": "clip_local",
                "pose_graph_id": stable_id("pose_graph", clip_id, prefix="posegraph"),
                "scale_alignment_id": stable_id("scale_alignment", clip_id, prefix="scalealign"),
                "valid": True,
                "missing_reason": "",
            }
            clips.append(clip)
            center = (visible_start + visible_end) / 2.0
            for frame_index in range(visible_start, visible_end + 1):
                candidate_by_frame.setdefault(frame_index, []).append(
                    {"clip": clip, "distance": abs(frame_index - center)}
                )
    owners: dict[int, str] = {}
    for frame_index, candidates in candidate_by_frame.items():
        owner = min(candidates, key=lambda item: (item["distance"], item["clip"]["clip_id"]))
        owners[frame_index] = owner["clip"]["clip_id"]
    for frame_index, candidates in sorted(candidate_by_frame.items()):
        frame_id = id_factory.frame(video_id, frame_index)
        for item in sorted(candidates, key=lambda value: value["clip"]["clip_id"]):
            clip = item["clip"]
            is_owned = owners[frame_index] == clip["clip_id"]
            memberships.append(
                {
                    "frame_record_id": stable_id("frame_record", frame_id, clip["clip_id"], prefix="frec"),
                    "frame_id": frame_id,
                    "video_id": video_id,
                    "clip_id": clip["clip_id"],
                    "frame_index": frame_index,
                    "scene_id": clip["scene_id"],
                    "is_context_frame": not is_owned,
                    "is_owned_frame": is_owned,
                    "owner_clip_id": owners[frame_index],
                    "decode_status": "pending",
                    "failure_reason": "",
                    "decoded_frame_path": "",
                }
            )
    return clips, memberships


def video_manifest_row(
    *, source_root: Path, source_path: Path, dataset_id: str
) -> dict[str, Any]:
    """Create one content-addressed video manifest row without labels."""

    relative = source_path.resolve().relative_to(source_root.resolve()).as_posix()
    source_hash = sha256_file(source_path)
    factory = StableIdFactory(dataset_id)
    probe = inspect_video(source_path)
    return {
        "video_id": factory.video(relative, source_hash),
        "source_name": source_path.stem,
        "source_relative_path": relative,
        "source_sha256": source_hash,
        "file_size": source_path.stat().st_size,
        "frame_count": probe.frame_count,
        "fps": probe.fps,
        "width": probe.width,
        "height": probe.height,
        "decode_status": probe.decode_status,
        "failure_reason": probe.failure_reason,
    }
