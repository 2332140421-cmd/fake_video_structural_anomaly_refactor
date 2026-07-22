"""Typed P4-C2 contracts for formal dataset onboarding readiness."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

P4C2_SCHEMA_VERSION = "p4c2_formal_data_readiness_v1"
ELIGIBILITY_STATES = frozenset(
    {"eligible", "ineligible", "unknown", "not_applicable", "provider_failed"}
)
VERIFICATION_STATES = frozenset({"verified", "unverified", "conflict"})
FORMAL_SPLITS = frozenset({"train", "validation", "test", ""})


@dataclass(frozen=True)
class FormalDatasetRecord:
    """One registered dataset with source, license, split, and storage facts."""

    dataset_name: str
    version: str
    official_source: str
    license: str
    citation: str
    download_method: str
    official_split: str
    annotation_types: tuple[str, ...]
    expected_size: Mapping[str, Any]
    checksum_policy: str
    local_root: str
    dataset_role: str
    eligible_for_training: bool
    eligible_for_model_selection: bool
    eligible_for_threshold_selection: bool
    eligible_for_final_evaluation: bool
    registry_status: str
    missing_requirements: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = (
            self.dataset_name,
            self.version,
            self.official_source,
            self.license,
            self.citation,
            self.download_method,
            self.official_split,
            self.checksum_policy,
            self.local_root,
            self.dataset_role,
            self.registry_status,
        )
        if any(not value.strip() for value in required):
            raise ValueError("Formal dataset registry fields must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serialization mapping."""

        return asdict(self)


@dataclass(frozen=True)
class SourceLineageRecord:
    """Identity and derivation evidence for one original or derived video."""

    dataset_name: str
    original_source_id: str
    source_group_id: str
    derived_video_id: str
    source_video_name: str
    video_path: str
    source_sha256: str
    authenticity_label: int | None
    manipulation_type: str
    derivation_status: str
    identity_evidence: str
    verification_status: str
    formal_split_eligible: bool
    ineligibility_reason: str
    dataset_role: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verification_status not in VERIFICATION_STATES:
            raise ValueError(f"Unsupported verification status: {self.verification_status!r}")
        if self.authenticity_label not in {0, 1, None}:
            raise ValueError("authenticity_label must be 0, 1, or None")
        if self.formal_split_eligible and (
            self.verification_status != "verified" or not self.original_source_id
        ):
            raise ValueError("Formal split eligibility requires verified original_source_id")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serialization mapping."""

        return asdict(self)


@dataclass(frozen=True)
class FormalSplitRecord:
    """Formal split decision for one video, separate from smoke-only splits."""

    dataset_name: str
    derived_video_id: str
    original_source_id: str
    source_group_id: str
    formal_split: str
    split_source: str
    eligible: bool
    blocked_reason: str
    official_split_preserved: bool
    algorithm_version: str
    random_seed: int

    def __post_init__(self) -> None:
        if self.formal_split not in FORMAL_SPLITS:
            raise ValueError(f"Unsupported formal split: {self.formal_split!r}")
        if self.eligible != bool(self.formal_split):
            raise ValueError("eligible must match the presence of a formal split")
        if self.eligible and self.blocked_reason:
            raise ValueError("Eligible formal splits cannot carry blocked_reason")
        if not self.eligible and not self.blocked_reason:
            raise ValueError("Blocked formal split rows require blocked_reason")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serialization mapping."""

        return asdict(self)


@dataclass(frozen=True)
class TaskEligibilityRecord:
    """Task-specific eligibility and missingness state for one video."""

    dataset_name: str
    derived_video_id: str
    task_name: str
    eligibility: str
    eligible_for_declared_role: bool
    eligible_for_formal_experiment: bool
    evaluation_scope: str
    ineligibility_reason: str
    provider_name: str
    available_observation_count: int
    expected_observation_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.eligibility not in ELIGIBILITY_STATES:
            raise ValueError(f"Unsupported eligibility state: {self.eligibility!r}")
        if self.available_observation_count < 0 or self.expected_observation_count < 0:
            raise ValueError("Observation counts must be non-negative")
        if self.eligible_for_formal_experiment and self.eligibility != "eligible":
            raise ValueError("Formal task eligibility requires eligibility='eligible'")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic serialization mapping."""

        return asdict(self)


@dataclass(frozen=True)
class P4C2BuildResult:
    """All deterministic P4-C2 records before artifact writing."""

    datasets: tuple[FormalDatasetRecord, ...]
    lineage: tuple[SourceLineageRecord, ...]
    formal_splits: tuple[FormalSplitRecord, ...]
    task_eligibility: tuple[TaskEligibilityRecord, ...]
    data_inventory: tuple[Mapping[str, Any], ...]
    leakage_audit: Mapping[str, Any]
    storage_plan: Mapping[str, Any]
    missingness_plan: Mapping[str, Any]
    readiness: Mapping[str, Any]
    protocol_sha256: str
    p4c1_manifest_sha256: str
    config_sha256: str

