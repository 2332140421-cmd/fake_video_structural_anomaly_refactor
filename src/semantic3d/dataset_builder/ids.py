"""Stable identifiers derived only from canonical, reproducible inputs."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized[:32] or "item"


def stable_id(namespace: str, *parts: Any, prefix: str | None = None) -> str:
    """Return a deterministic ID that is independent of process ordering."""

    if not namespace.strip():
        raise ValueError("namespace must be non-empty")
    payload = _canonical({"namespace": namespace, "parts": parts})
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"{_slug(prefix or namespace)}_{digest}"


@dataclass(frozen=True)
class StableIdFactory:
    """Create all P4-B IDs under one immutable dataset identity."""

    dataset_id: str

    def video(self, relative_path: str, source_sha256: str) -> str:
        return stable_id("video", self.dataset_id, relative_path, source_sha256, prefix="vid")

    def clip(self, video_id: str, scene_id: int, start: int, end: int) -> str:
        return stable_id("clip", self.dataset_id, video_id, scene_id, start, end, prefix="clip")

    def frame(self, video_id: str, frame_index: int) -> str:
        return stable_id("frame", self.dataset_id, video_id, int(frame_index), prefix="frm")

    def object_track(self, video_id: str, source_track_id: str) -> str:
        return stable_id("object_track", self.dataset_id, video_id, source_track_id, prefix="objtrk")

    def point(self, object_track_id: str, source_point_id: str) -> str:
        return stable_id("point", self.dataset_id, object_track_id, source_point_id, prefix="pt")

    def edge(self, object_track_id: str, source_edge_id: str) -> str:
        return stable_id("edge", self.dataset_id, object_track_id, source_edge_id, prefix="edge")

    def evidence(self, branch: str, level: str, *source_ids: str) -> str:
        return stable_id("evidence", self.dataset_id, branch, level, source_ids, prefix="ev")

    def coordinate_system(self, clip_id: str) -> str:
        return stable_id("coordinate_system", self.dataset_id, clip_id, prefix="coord")
