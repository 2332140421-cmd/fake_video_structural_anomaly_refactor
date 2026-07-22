"""Quality-aware P4-A aggregation without classification semantics."""

from .aggregation import aggregate_evidence_v2
from .applicability import (
    AggregationEvidence,
    EvidenceApplicability,
    EvidenceRecord,
    applicability_from_missing_reason,
    from_residual_evidence,
)
from .contracts import (
    ClipEvidenceAggregate,
    EdgeEvidenceAggregate,
    FrameEvidenceAggregate,
    ObjectEvidenceAggregate,
    PointEvidenceAggregate,
)
from .evidence_registry import (
    DEFAULT_EVIDENCE_REGISTRY,
    EvidenceBranchSpec,
    EvidenceFormality,
    EvidenceLevel,
    get_evidence_registry,
    register_evidence_branch,
)
from .hierarchy import (
    MultilevelAggregationResult,
    aggregate_clip_evidence,
    aggregate_edge_evidence,
    aggregate_frame_evidence,
    aggregate_multilevel_evidence,
    aggregate_object_evidence,
    aggregate_point_evidence,
)
from .temporal_localization import (
    TemporalInterval,
    causal_moving_median,
    localize_temporal_intervals,
)

__all__ = [
    "AggregationEvidence", "EvidenceRecord", "EvidenceApplicability",
    "EvidenceBranchSpec", "EvidenceFormality", "EvidenceLevel",
    "DEFAULT_EVIDENCE_REGISTRY", "get_evidence_registry",
    "register_evidence_branch", "applicability_from_missing_reason",
    "from_residual_evidence", "PointEvidenceAggregate",
    "EdgeEvidenceAggregate", "ObjectEvidenceAggregate",
    "FrameEvidenceAggregate", "ClipEvidenceAggregate",
    "MultilevelAggregationResult", "aggregate_evidence_v2",
    "aggregate_point_evidence", "aggregate_edge_evidence",
    "aggregate_object_evidence", "aggregate_frame_evidence",
    "aggregate_clip_evidence", "aggregate_multilevel_evidence",
    "TemporalInterval", "causal_moving_median",
    "localize_temporal_intervals",
]
