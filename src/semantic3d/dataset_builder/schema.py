"""Schema contracts for the label-isolated P4-B offline dataset."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "semantic3d_structural_enhancement_v1"
PIPELINE_VERSION = "p4b_v1"
P4B5_PIPELINE_VERSION = "p4b5_full_observation_v1"


class Applicability(str, Enum):
    """Controlled evidence applicability states."""

    APPLICABLE_VALID = "applicable_valid"
    APPLICABLE_INVALID = "applicable_invalid"
    NOT_APPLICABLE = "not_applicable"
    OBSERVATION_MISSING = "observation_missing"
    INVALID_GEOMETRY = "invalid_geometry"
    UNSUPPORTED_MODE = "unsupported_mode"


@dataclass(frozen=True)
class DatasetManifest:
    """Reproducibility metadata for one immutable dataset build."""

    dataset_id: str
    schema_version: str
    pipeline_version: str
    git_commit: str
    creation_time: str
    config_path: str
    config_sha256: str
    source_root: str
    source_video_count: int
    provider_metadata: Mapping[str, Any]
    weight_sha256_by_provider: Mapping[str, str]
    branch_registry_version: str
    coordinate_convention: str
    depth_convention: str
    label_isolation: bool
    random_seed: int
    environment: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.label_isolation:
            raise ValueError("P4-B dataset construction requires label_isolation=true")
        if self.source_video_count < 0:
            raise ValueError("source_video_count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceRecord:
    """Evidence row with explicit validity, applicability, and provenance."""

    evidence_id: str
    branch_name: str
    evidence_level: str
    video_id: str
    clip_id: str
    frame_id: str = ""
    object_track_id: str = ""
    point_id: str = ""
    edge_id: str = ""
    raw_value: float = math.nan
    intrinsic_normalized_value: float = math.nan
    statistically_normalized_value: float = math.nan
    normalization_fit_source: str = "none"
    valid: bool = False
    quality: float = 0.0
    applicability: Applicability = Applicability.OBSERVATION_MISSING
    missing_reason: str = "observation_missing"
    geometry_mode: str = "unavailable"
    sequence_scale_status: str = "unknown"
    coordinate_system_id: str = ""
    source_evidence_ids: tuple[str, ...] = ()
    localization_reference: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        applicability = Applicability(self.applicability)
        quality = float(self.quality)
        if not 0.0 <= quality <= 1.0 or not math.isfinite(quality):
            raise ValueError("Evidence quality must be finite and in [0, 1]")
        if self.valid:
            if applicability != Applicability.APPLICABLE_VALID:
                raise ValueError("Valid evidence must be applicable_valid")
            if not math.isfinite(float(self.raw_value)):
                raise ValueError("Valid evidence requires finite raw_value")
            if self.missing_reason:
                raise ValueError("Valid evidence cannot have missing_reason")
        else:
            if not math.isnan(float(self.raw_value)):
                raise ValueError("Invalid or missing evidence must keep raw_value=NaN")
            if not self.missing_reason:
                raise ValueError("Invalid evidence requires missing_reason")
        if not math.isnan(float(self.statistically_normalized_value)):
            raise ValueError("P4-B does not fit statistical normalization")
        if self.normalization_fit_source != "none":
            raise ValueError("P4-B normalization_fit_source must be 'none'")
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "source_evidence_ids", tuple(self.source_evidence_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["applicability"] = self.applicability.value
        return row


EVIDENCE_COLUMNS = tuple(EvidenceRecord.__dataclass_fields__)


TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "manifests/videos.parquet": ("video_id",),
    "manifests/clips.parquet": ("clip_id",),
    "manifests/frames.parquet": ("frame_record_id",),
    "manifests/stage_status.parquet": ("run_id", "stage_name"),
    "observations/objects.parquet": ("object_observation_id",),
    "observations/camera.parquet": ("camera_observation_id",),
    "observations/depth.parquet": ("depth_observation_id",),
    "observations/masks.parquet": ("mask_observation_id",),
    "observations/keypoints.parquet": ("keypoint_observation_id",),
    "observations/shared_3d_frames.parquet": ("shared_3d_frame_id",),
    "observations/shared_3d_clips.parquet": ("shared_3d_clip_id",),
    "evidence/point_evidence.parquet": ("evidence_id",),
    "evidence/edge_evidence.parquet": ("evidence_id",),
    "evidence/object_evidence.parquet": ("evidence_id",),
    "evidence/frame_evidence.parquet": ("evidence_id",),
    "evidence/clip_evidence.parquet": ("evidence_id",),
}


# P4-B.5 extends the immutable P4-B schema without making the new tables
# mandatory for older datasets.  Validation enables these keys only when the
# dataset manifest declares ``p4b5_full_observation_v1``.
P4B5_TABLE_PRIMARY_KEYS: dict[str, tuple[str, ...]] = {
    "observations/point_tracks_2d.parquet": ("point_track_2d_observation_id",),
    "observations/point_tracks_3d.parquet": ("point_track_3d_observation_id",),
    "observations/keypoints_3d.parquet": ("keypoint_3d_observation_id",),
    "observations/structure_graphs.parquet": ("structure_graph_id",),
    "observations/structure_transitions.parquet": ("structure_transition_id",),
    "observations/clip_track_handoffs.parquet": ("handoff_id",),
    "observations/mask_tracks.parquet": ("mask_track_observation_id",),
    "observations/dynamic_readiness.parquet": ("clip_id",),
    "reports/coverage_metrics.parquet": ("metric_name", "scope_type", "scope_id"),
}
