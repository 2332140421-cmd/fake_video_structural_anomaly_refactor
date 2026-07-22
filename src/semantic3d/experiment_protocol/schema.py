"""Typed records for the label-isolated P4-C0 experiment protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PROTOCOL_VERSION = "p4c0_experiment_protocol_v1"
SPLIT_ALGORITHM_VERSION = "group_aware_balanced_v1"

DATASET_ROLES = {
    "geometry_validation",
    "detection_training",
    "validation",
    "final_test",
    "cross_dataset_test",
    "occlusion_validation",
    "localization_validation",
}


@dataclass(frozen=True)
class DatasetCatalog:
    """One source dataset and its intended, explicitly scoped roles."""

    dataset_name: str
    dataset_version: str
    source_root: str
    license_or_usage_note: str
    official_split_available: bool
    video_count: int
    real_count: int
    fake_count: int
    manipulation_types: tuple[str, ...] = ()
    temporal_annotation_available: bool = False
    spatial_annotation_available: bool = False
    original_source_identity_available: bool = False
    compression_variants: tuple[str, ...] = ()
    expected_storage_bytes: int = 0
    roles: tuple[str, ...] = ("geometry_validation",)
    sample_scope: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.roles) - DATASET_ROLES
        if unknown:
            raise ValueError(f"Unsupported dataset role(s): {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        """Return a table-ready mapping."""

        return asdict(self)


@dataclass(frozen=True)
class LabelRecord:
    """Independent annotation record; never inferred from a source filename."""

    video_id: str
    binary_label: int | None
    label_name: str
    manipulation_type: str = "unknown"
    temporal_intervals: tuple[tuple[float, float], ...] = ()
    spatial_mask_path: str = ""
    spatial_bbox: tuple[float, ...] = ()
    object_annotation_path: str = ""
    source_group_id: str = ""
    annotation_quality: str = "unknown"
    annotation_source: str = "independent_manifest"
    declared_split: str = ""
    original_source_identity: str = ""


@dataclass(frozen=True)
class SourceGroupRecord:
    """Stable source identity grouping all known derivatives of one origin."""

    video_id: str
    source_group_id: str
    grouping_basis: str
    original_source_identity: str
    source_group_review_required: bool
    review_reason: str = ""


@dataclass(frozen=True)
class VideoInventoryRecord:
    """Metadata, annotation availability, and task eligibility for one video."""

    video_id: str
    dataset_name: str
    source_name: str
    source_path: str
    source_sha256: str
    file_size: int
    frame_count: int
    fps: float
    width: int
    height: int
    duration_seconds: float
    binary_label: int | None
    label_name: str
    manipulation_type: str
    source_group_id: str
    source_group_review_required: bool
    declared_split: str
    temporal_annotation_available: bool
    spatial_annotation_available: bool
    object_annotation_available: bool
    annotation_quality: str
    detection_training_eligible: bool
    video_classification_eligible: bool
    temporal_localization_eligible: bool
    spatial_localization_eligible: bool
    object_localization_eligible: bool
    occlusion_validation_eligible: bool
    geometry_validation_only: bool
    exclusion_reasons: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitAssignment:
    """One group-aware video split assignment and its provenance."""

    video_id: str
    source_group_id: str
    split: str
    split_source: str
    official_split_preserved: bool
    algorithm_version: str
    random_seed: int
    decision_inputs: tuple[str, ...]


@dataclass(frozen=True)
class BranchEligibilityRecord:
    """Applicability of one structural branch for a video or clip."""

    entity_type: str
    entity_id: str
    video_id: str
    tier: str
    branch_name: str
    applicable: bool
    eligibility_status: str
    expected_observation_available: bool
    geometry_requirement: str
    mask_requirement: str
    keypoint_requirement: str
    event_requirement: str
    exclusion_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProtocolValidationResult:
    """Machine-readable protocol validation result."""

    valid: bool
    error_count: int
    warning_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    checks: dict[str, bool]
    metrics: dict[str, Any]
