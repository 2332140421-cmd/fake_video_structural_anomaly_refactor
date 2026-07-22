"""P4-C0/P4-C1/P4-C2 metadata protocols without fitting or classification."""

from .branch_eligibility import build_branch_eligibility
from .coverage_planning import build_coverage_targets
from .duplicate_audit import VisualEmbeddingProvider, audit_duplicates
from .inventory import build_dataset_catalog, build_video_inventory, load_isolated_labels
from .leakage_audit import audit_leakage, audit_manifest_leakage
from .manifest_builder import (
    build_p4c1_manifest,
    manifest_sha256,
    validate_manifest_artifacts,
)
from .manifest_schema import (
    ExperimentSampleRecord,
    LeakageFinding,
    ManifestBuildResult,
    SampleAvailability,
    stable_sample_id,
)
from .formal_schema import (
    FormalDatasetRecord,
    FormalSplitRecord,
    P4C2BuildResult,
    SourceLineageRecord,
    TaskEligibilityRecord,
)
from .p4c2_builder import build_p4c2_readiness, readiness_manifest_sha256
from .p4c2_builder import evaluate_formal_build_readiness
from .p4c2_validation import validate_p4c2_artifacts
from .evidence_eligibility_policy import EvidenceUse, can_contribute_evidence, validate_evidence_use
from .source_lineage import audit_formal_split_leakage, plan_verified_formal_splits
from .schema import (
    BranchEligibilityRecord,
    DatasetCatalog,
    LabelRecord,
    SourceGroupRecord,
    SplitAssignment,
    VideoInventoryRecord,
)
from .source_grouping import assign_source_groups, stable_source_group_id
from .split_planner import plan_group_aware_split
from .storage_planning import estimate_storage
from .validation import validate_protocol

__all__ = [
    "BranchEligibilityRecord",
    "DatasetCatalog",
    "ExperimentSampleRecord",
    "EvidenceUse",
    "FormalDatasetRecord",
    "FormalSplitRecord",
    "LabelRecord",
    "LeakageFinding",
    "ManifestBuildResult",
    "P4C2BuildResult",
    "SampleAvailability",
    "SourceGroupRecord",
    "SourceLineageRecord",
    "SplitAssignment",
    "TaskEligibilityRecord",
    "VideoInventoryRecord",
    "VisualEmbeddingProvider",
    "assign_source_groups",
    "audit_duplicates",
    "audit_leakage",
    "audit_manifest_leakage",
    "audit_formal_split_leakage",
    "build_branch_eligibility",
    "build_coverage_targets",
    "build_dataset_catalog",
    "build_video_inventory",
    "build_p4c1_manifest",
    "build_p4c2_readiness",
    "can_contribute_evidence",
    "evaluate_formal_build_readiness",
    "estimate_storage",
    "load_isolated_labels",
    "plan_group_aware_split",
    "plan_verified_formal_splits",
    "manifest_sha256",
    "readiness_manifest_sha256",
    "stable_source_group_id",
    "stable_sample_id",
    "validate_manifest_artifacts",
    "validate_evidence_use",
    "validate_p4c2_artifacts",
    "validate_protocol",
]
