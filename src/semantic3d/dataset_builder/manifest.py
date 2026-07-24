"""Video indexing, scene segmentation, and overlap-safe clip manifests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .formal_schema import FormalVideoSample, OPTIONAL_METADATA_FIELDS
from .ids import StableIdFactory, stable_id
from .writer import sha256_file

VIDEO_EXTENSIONS = frozenset(
    {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}
)


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


def scan_video_files(
    video_root: str | Path,
    *,
    recursive: bool = True,
    extensions: Iterable[str] = VIDEO_EXTENSIONS,
) -> list[Path]:
    """Return supported source videos in deterministic path order."""

    root = Path(video_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {root}")
    allowed = {
        value.lower() if value.startswith(".") else f".{value.lower()}"
        for value in extensions
    }
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        (
            path.resolve()
            for path in candidates
            if path.is_file() and path.suffix.lower() in allowed
        ),
        key=lambda path: path.as_posix(),
    )


def _resolved_source_path(source_path: str | Path, data_root: str | Path | None) -> Path:
    source = Path(source_path).expanduser()
    if source.is_absolute():
        return source.resolve()
    if data_root is None:
        raise ValueError("Relative source paths require an explicit data_root")
    return (Path(data_root).expanduser().resolve() / source).resolve()


def normalize_manifest_video_path(
    source_path: str | Path,
    *,
    data_root: str | Path | None,
    path_mode: str,
) -> tuple[Path, str]:
    """Resolve a source and return an absolute or explicit data-root-relative path."""

    source = _resolved_source_path(source_path, data_root)
    if path_mode == "absolute":
        return source, source.as_posix()
    if path_mode != "data_root_relative":
        raise ValueError("path_mode must be absolute or data_root_relative")
    if data_root is None:
        raise ValueError("data_root_relative mode requires data_root")
    root = Path(data_root).expanduser().resolve()
    try:
        relative = source.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Source path is outside data_root: {source}") from exc
    return source, relative


def _metadata_status(
    values: Mapping[str, Any],
    *,
    status_overrides: Mapping[str, str] | None = None,
) -> dict[str, str]:
    statuses = {
        name: ("missing" if values.get(name) is None else "provided")
        for name in OPTIONAL_METADATA_FIELDS
    }
    for name in ("duration", "fps", "frame_count", "width", "height"):
        if values.get(name) is not None:
            statuses[name] = "derived"
    statuses.update(dict(status_overrides or {}))
    return statuses


def build_formal_video_sample(
    source_path: str | Path,
    *,
    data_root: str | Path | None,
    source_dataset: str,
    split: str | None = None,
    label: int | None = None,
    source_id: str | None = None,
    source_lineage: Mapping[str, Any] | None = None,
    generator: str | None = None,
    source_domain: str | None = None,
    is_real: bool | None = None,
    temporal_annotation: Any | None = None,
    spatial_annotation: Any | None = None,
    official_split: bool = False,
    expected_sha256: str | None = None,
    path_mode: str = "absolute",
    metadata_status: Mapping[str, str] | None = None,
    probe: Callable[[str | Path], VideoProbe] = inspect_video,
) -> FormalVideoSample:
    """Build one canonical sample without assigning or changing dataset splits."""

    raw_source_path = str(source_path)
    source, manifest_path = normalize_manifest_video_path(
        source_path,
        data_root=data_root,
        path_mode=path_mode,
    )
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    digest = sha256_file(source)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise ValueError(f"Checksum mismatch for source video: {source}")

    inspected = probe(source)
    probe_ok = inspected.decode_status == "ok"
    fps = (
        float(inspected.fps)
        if probe_ok and math.isfinite(float(inspected.fps)) and inspected.fps > 0
        else None
    )
    frame_count = int(inspected.frame_count) if probe_ok and inspected.frame_count >= 0 else None
    width = int(inspected.width) if probe_ok and inspected.width > 0 else None
    height = int(inspected.height) if probe_ok and inspected.height > 0 else None
    duration = (
        float(frame_count) / fps
        if frame_count is not None and fps is not None
        else None
    )
    if label is not None and is_real is None:
        is_real = label == 0
    if label is None and is_real is not None:
        label = 0 if is_real else 1

    root = Path(data_root).expanduser().resolve() if data_root is not None else None
    if root is not None:
        try:
            identity_path = source.relative_to(root).as_posix()
        except ValueError:
            identity_path = source.as_posix()
    else:
        identity_path = source.as_posix()
    effective_source_id = source_id if source_id not in {"", None} else identity_path
    sample_id = stable_id(
        "formal_video_sample",
        source_dataset,
        identity_path,
        digest,
        prefix="sample",
    )
    values = {
        "label": label,
        "split": split,
        "source_id": effective_source_id,
        "source_lineage": source_lineage,
        "generator": generator,
        "source_domain": source_domain,
        "is_real": is_real,
        "duration": duration,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "temporal_annotation": temporal_annotation,
        "spatial_annotation": spatial_annotation,
    }
    statuses = _metadata_status(values, status_overrides=metadata_status)
    if source_id in {"", None} and (
        metadata_status is None or "source_id" not in metadata_status
    ):
        statuses["source_id"] = "derived"
    return FormalVideoSample(
        sample_id=sample_id,
        video_path=manifest_path,
        label=label,
        split=split,
        source_dataset=source_dataset,
        source_id=effective_source_id,
        source_lineage=source_lineage,
        generator=generator,
        source_domain=source_domain,
        is_real=is_real,
        duration=duration,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        file_size=source.stat().st_size,
        sha256=digest,
        temporal_annotation=temporal_annotation,
        spatial_annotation=spatial_annotation,
        metadata_status=statuses,
        source_path=raw_source_path,
        path_mode=path_mode,
        official_split=official_split,
    )


def build_formal_manifest_from_directory(
    video_root: str | Path,
    *,
    data_root: str | Path,
    source_dataset: str,
    split: str | None = None,
    recursive: bool = True,
    path_mode: str = "data_root_relative",
    probe: Callable[[str | Path], VideoProbe] = inspect_video,
) -> list[FormalVideoSample]:
    """Recursively index one directory without inferring labels or lineage."""

    return [
        build_formal_video_sample(
            path,
            data_root=data_root,
            source_dataset=source_dataset,
            split=split,
            path_mode=path_mode,
            probe=probe,
        )
        for path in scan_video_files(video_root, recursive=recursive)
    ]


def disambiguate_source_names(
    rows: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Keep legacy stems when unique and suffix only colliding source names."""

    output = [dict(row) for row in rows]
    counts: dict[str, int] = {}
    for row in output:
        name = str(row["source_name"])
        counts[name] = counts.get(name, 0) + 1
    for row in output:
        name = str(row["source_name"])
        if counts[name] > 1:
            row["source_name"] = f"{name}__{row['video_id']}"
    return output


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

    root = source_root.expanduser().resolve()
    source = source_path.expanduser().resolve()
    try:
        relative = source.relative_to(root).as_posix()
        path_kind = "source_root_relative"
    except ValueError:
        relative = source.as_posix()
        path_kind = "absolute"
    source_hash = sha256_file(source)
    factory = StableIdFactory(dataset_id)
    probe = inspect_video(source)
    return {
        "video_id": factory.video(relative, source_hash),
        "source_name": source.stem,
        "source_stem": source.stem,
        "source_relative_path": relative,
        "source_path": source.as_posix(),
        "source_original_path": str(source_path),
        "source_path_kind": path_kind,
        "source_sha256": source_hash,
        "file_size": source.stat().st_size,
        "frame_count": probe.frame_count,
        "fps": probe.fps,
        "width": probe.width,
        "height": probe.height,
        "decode_status": probe.decode_status,
        "failure_reason": probe.failure_reason,
    }
