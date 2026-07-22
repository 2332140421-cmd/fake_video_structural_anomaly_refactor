"""Read P4-C0 and P4-B.5 indexes for deterministic P4-C1 clip inventory."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from semantic3d.dataset_builder.writer import sha256_file

from .manifest_schema import SampleAvailability


@dataclass(frozen=True)
class P4C1DataInventory:
    """All table rows and indexes needed to build P4-C1 samples."""

    videos: tuple[Mapping[str, Any], ...]
    clips: tuple[Mapping[str, Any], ...]
    inventory_by_video: Mapping[str, Mapping[str, Any]]
    split_by_video: Mapping[str, Mapping[str, Any]]
    source_group_by_video: Mapping[str, Mapping[str, Any]]
    duplicate_rows: tuple[Mapping[str, Any], ...]
    structural_manifest: Mapping[str, Any]
    experiment_protocol: Mapping[str, Any]


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required P4-C1 input table is missing: {path}")
    return pq.read_table(path).to_pylist()


def load_p4c1_inventory(
    protocol_root: str | Path,
    structural_dataset_root: str | Path,
) -> P4C1DataInventory:
    """Load existing P4-C0/P4-B.5 metadata without recomputing split decisions."""

    protocol = Path(protocol_root)
    structural = Path(structural_dataset_root)
    video_inventory = _rows(protocol / "video_inventory.parquet")
    split_rows = _rows(protocol / "split_manifest.parquet")
    source_groups = _rows(protocol / "source_groups.parquet")
    videos = _rows(structural / "manifests/videos.parquet")
    clips = _rows(structural / "manifests/clips.parquet")
    experiment_protocol = json.loads(
        (protocol / "experiment_protocol.json").read_text(encoding="utf-8")
    )
    structural_manifest = json.loads(
        (structural / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    duplicate_path = protocol / "duplicate_audit.parquet"
    duplicate_rows = _rows(duplicate_path) if duplicate_path.is_file() else []
    return P4C1DataInventory(
        videos=tuple(videos),
        clips=tuple(clips),
        inventory_by_video={str(row["video_id"]): row for row in video_inventory},
        split_by_video={str(row["video_id"]): row for row in split_rows},
        source_group_by_video={str(row["video_id"]): row for row in source_groups},
        duplicate_rows=tuple(duplicate_rows),
        structural_manifest=structural_manifest,
        experiment_protocol=experiment_protocol,
    )


def _valid(row: Mapping[str, Any]) -> bool:
    return bool(row.get("valid", True))


def _frame_index_counts(
    rows: Iterable[Mapping[str, Any]],
) -> Mapping[str, Counter[int]]:
    output: dict[str, Counter[int]] = defaultdict(Counter)
    for row in rows:
        if _valid(row):
            output[str(row["video_id"])][int(row["frame_index"])] += 1
    return output


class AvailabilityIndex:
    """Pre-index structural rows and audit one clip without reading residuals."""

    def __init__(self, structural_dataset_root: str | Path, project_root: str | Path) -> None:
        self.root = Path(structural_dataset_root)
        self.project_root = Path(project_root)
        frames = _rows(self.root / "manifests/frames.parquet")
        objects = _rows(self.root / "observations/objects.parquet")
        depth = _rows(self.root / "observations/depth.parquet")
        camera = _rows(self.root / "observations/camera.parquet")
        tracks = _rows(self.root / "observations/point_tracks_2d.parquet")
        shared_frames = _rows(self.root / "observations/shared_3d_frames.parquet")
        shared_clips = _rows(self.root / "observations/shared_3d_clips.parquet")

        self.frames_by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in frames:
            self.frames_by_clip[str(row["clip_id"])].append(row)
        self.objects_by_frame = _frame_index_counts(objects)
        self.depth_by_frame = _frame_index_counts(depth)
        self.shared_by_frame = _frame_index_counts(shared_frames)
        self.camera_by_clip: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in camera:
            self.camera_by_clip[str(row["clip_id"])].append(row)
        self.tracks_by_clip: Counter[str] = Counter(
            str(row["clip_id"]) for row in tracks if _valid(row)
        )
        self.shared_clips_by_clip: Counter[str] = Counter(
            str(row["clip_id"]) for row in shared_clips if _valid(row)
        )

    @staticmethod
    def _count_range(counter: Counter[int], start: int, end: int) -> int:
        return sum(count for index, count in counter.items() if start <= index <= end)

    def audit_clip(
        self,
        clip: Mapping[str, Any],
        video: Mapping[str, Any],
    ) -> SampleAvailability:
        """Return modality counts and source-file integrity for one clip."""

        clip_id = str(clip["clip_id"])
        video_id = str(clip["video_id"])
        start = int(clip["start_frame_index"])
        end = int(clip["end_frame_index"])
        expected = end - start + 1
        source_path = self.project_root / str(video["source_relative_path"])
        exists = source_path.is_file()
        readable = exists and source_path.stat().st_size > 0
        hash_matches = bool(
            readable and sha256_file(source_path) == str(video["source_sha256"])
        )
        clip_frames = self.frames_by_clip[clip_id]
        indexed = len(clip_frames)
        decoded = sum(
            str(row.get("decode_status", "")) in {"ok", "decoded", "indexed_decodable"}
            and not str(row.get("failure_reason", ""))
            for row in clip_frames
        )
        camera_rows = self.camera_by_clip[clip_id]
        camera_count = len(camera_rows)
        pose_count = sum(
            _valid(row)
            and row.get("T_world_camera") not in {None, "", "null"}
            and bool(row.get("sequence_geometry_valid", False))
            for row in camera_rows
        )
        object_count = self._count_range(self.objects_by_frame[video_id], start, end)
        depth_count = self._count_range(self.depth_by_frame[video_id], start, end)
        shared_frame_count = self._count_range(self.shared_by_frame[video_id], start, end)
        semantic_count = shared_frame_count + self.shared_clips_by_clip[clip_id]
        track_count = int(self.tracks_by_clip[clip_id])
        missing = []
        for name, count in (
            ("objects", object_count),
            ("depth", depth_count),
            ("camera", camera_count),
            ("pose", pose_count),
            ("tracks", track_count),
            ("semantic3d", semantic_count),
        ):
            if count <= 0:
                missing.append(name)
        missing.append("camera_identity")
        return SampleAvailability(
            video_exists=exists,
            video_readable=readable,
            video_hash_matches=hash_matches,
            expected_frame_count=expected,
            indexed_frame_count=indexed,
            decoded_frame_count=decoded,
            valid_object_count=object_count,
            valid_depth_count=depth_count,
            camera_observation_count=camera_count,
            valid_pose_count=pose_count,
            valid_track_point_count=track_count,
            valid_semantic3d_count=semantic_count,
            camera_identity_available=False,
            missing_modalities=tuple(sorted(missing)),
            metadata={
                "camera_identity_reason": "camera_identity_unavailable",
                "semantic3d_count_includes": "valid_shared_3d_frames_and_clips",
                "residual_values_read": False,
            },
        )


def project_relative_path(path: Path, project_root: Path) -> str:
    """Return a portable project-relative path when possible."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)

