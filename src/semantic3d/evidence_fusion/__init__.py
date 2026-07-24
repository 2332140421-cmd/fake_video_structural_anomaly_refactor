"""P4-C3B-M6 deterministic missing-aware evidence fusion."""

from .audits import (
    branch_availability_rows,
    coverage_by_group,
    missingness_only_features,
    provider_failure_balance,
)
from .contracts import (
    EvidenceBranchGroup,
    UnifiedEvidence,
    provider_status_is_failure,
)
from .fusion import (
    BranchContribution,
    FusionResult,
    branch_dropout_audit,
    fuse_unified_evidence,
)
from .localization import (
    SpatialEvidenceProducts,
    map_unified_evidence_spatially,
    rank_object_and_track_evidence,
)
from .routing import BranchRouteDecision, route_evidence_branches
from .temporal import (
    FrameFusionEvidence,
    TemporalEvidenceSequence,
    build_frame_fusions,
    build_temporal_evidence_sequences,
    merge_temporal_intervals,
)

__all__ = [
    "BranchContribution",
    "BranchRouteDecision",
    "EvidenceBranchGroup",
    "FrameFusionEvidence",
    "FusionResult",
    "SpatialEvidenceProducts",
    "TemporalEvidenceSequence",
    "UnifiedEvidence",
    "branch_availability_rows",
    "branch_dropout_audit",
    "build_frame_fusions",
    "build_temporal_evidence_sequences",
    "coverage_by_group",
    "fuse_unified_evidence",
    "map_unified_evidence_spatially",
    "merge_temporal_intervals",
    "missingness_only_features",
    "provider_failure_balance",
    "provider_status_is_failure",
    "rank_object_and_track_evidence",
    "route_evidence_branches",
]
