"""Typed, deterministic records for the P4-C1 experiment manifest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

P4C1_MANIFEST_SCHEMA_VERSION = "p4c1_experiment_manifest_v1"
P4C1_SAMPLE_ID_VERSION = "clip_identity_sha256_v1"
ALLOWED_SPLITS = frozenset({"train", "validation", "test", "official_conflict"})


def stable_sample_id(
    dataset_name: str,
    source_video_id: str,
    clip_id: str,
    clip_start: int,
    clip_end: int,
) -> str:
    """Return a deterministic sample ID from immutable clip identity fields."""

    values = (dataset_name, source_video_id, clip_id)
    if any(not str(value).strip() for value in values):
        raise ValueError("dataset_name, source_video_id, and clip_id must be non-empty")
    if clip_start < 0 or clip_end < clip_start:
        raise ValueError("clip frame range is invalid")
    payload = json.dumps(
        [
            P4C1_SAMPLE_ID_VERSION,
            dataset_name.strip(),
            source_video_id.strip(),
            clip_id.strip(),
            int(clip_start),
            int(clip_end),
        ],
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f"sample_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class SampleAvailability:
    """Observed data availability for one clip without anomaly interpretation."""

    video_exists: bool
    video_readable: bool
    video_hash_matches: bool
    expected_frame_count: int
    indexed_frame_count: int
    decoded_frame_count: int
    valid_object_count: int
    valid_depth_count: int
    camera_observation_count: int
    valid_pose_count: int
    valid_track_point_count: int
    valid_semantic3d_count: int
    camera_identity_available: bool
    missing_modalities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentSampleRecord:
    """One frozen clip sample and references to all structural input tables."""

    sample_id: str
    manifest_schema_version: str
    dataset_name: str
    source_video_id: str
    source_video_name: str
    source_group_id: str
    source_group_review_required: bool
    source_sha256: str
    video_path: str
    clip_id: str
    clip_start: int
    clip_end: int
    core_clip_start: int
    core_clip_end: int
    num_frames: int
    split: str
    authenticity_label: int | None
    authenticity_label_name: str
    manipulation_type: str
    scene_id: int
    camera_id: str
    camera_identity_status: str
    coordinate_system_id: str
    frame_manifest_path: str
    object_observations_path: str
    depth_observations_path: str
    pose_observations_path: str
    track_observations_path: str
    semantic3d_observations_path: str
    valid_object_count: int
    valid_depth_count: int
    valid_pose_count: int
    valid_track_point_count: int
    valid_semantic3d_count: int
    usable: bool
    exclusion_reason: str
    protocol_sha256: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.split not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {self.split!r}")
        if self.authenticity_label not in {0, 1, None}:
            raise ValueError("authenticity_label must be 0, 1, or None")
        if self.clip_start < 0 or self.clip_end < self.clip_start:
            raise ValueError("clip frame range is invalid")
        if self.num_frames != self.clip_end - self.clip_start + 1:
            raise ValueError("num_frames must match the inclusive clip range")
        if self.usable and self.exclusion_reason:
            raise ValueError("usable samples cannot carry an exclusion_reason")
        if not self.usable and not self.exclusion_reason:
            raise ValueError("excluded samples must carry an exclusion_reason")

    def to_dict(self) -> dict[str, Any]:
        """Return a serialization-ready mapping with stable field order."""

        return asdict(self)


@dataclass(frozen=True)
class LeakageFinding:
    """One source, duplicate, or split-isolation finding."""

    finding_type: str
    severity: str
    source_video_id_a: str
    source_video_id_b: str
    source_group_id: str
    split_a: str
    split_b: str
    details: str
    review_required: bool


@dataclass(frozen=True)
class ManifestBuildResult:
    """In-memory P4-C1 build result before deterministic artifact writing."""

    records: tuple[ExperimentSampleRecord, ...]
    availability: tuple[SampleAvailability, ...]
    leakage_findings: tuple[LeakageFinding, ...]
    leakage_summary: Mapping[str, Any]
    protocol_sha256: str
    p4c0_config_sha256: str
    source_manifest_sha256: str
    config_sha256: str
    structural_dataset_id: str

