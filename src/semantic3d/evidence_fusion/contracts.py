"""Unified, missing-aware evidence contracts for P4-C3B-M6."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..aggregation_v2.applicability import (
    AggregationEvidence,
    EvidenceApplicability,
)


class EvidenceBranchGroup(str, Enum):
    """Method-level evidence groups used by deterministic M6 fusion."""

    STATIC_METRIC_GEOMETRY = "static_metric_geometry"
    STATIC_RELATIVE_GEOMETRY = "static_relative_geometry"
    BOUNDARY_STRUCTURE = "boundary_structure"
    INTERNAL_STRUCTURE = "internal_structure"
    TEMPORAL_SCALE = "temporal_scale"
    D1_LOCAL_MOTION = "D1_local_motion"
    D2_POSE_REPROJECTION = "D2_pose_reprojection"
    D3_STRUCTURAL_RELATION = "D3_structural_relation"
    OCCLUSION_REAPPEARANCE = "occlusion_reappearance"


PROVIDER_FAILURE_STATUSES = frozenset(
    {
        "provider_failed",
        "dependency_missing",
        "weights_missing",
        "execution_failed",
    }
)


def provider_status_is_failure(status: str) -> bool:
    """Return whether a provider status represents execution failure."""

    normalized = str(status).strip().lower()
    return normalized in PROVIDER_FAILURE_STATUSES or normalized.endswith(
        "_provider_failed"
    )


@dataclass(frozen=True)
class UnifiedEvidence:
    """One residual with complete applicability, quality, and localization context.

    ``confidence`` and ``uncertainty`` are engineering quality diagnostics, not
    calibrated probabilities. Invalid evidence always carries ``NaN`` and can
    never enter the deterministic risk calculation.
    """

    evidence_id: str
    residual_value: float
    applicable: bool
    valid: bool
    confidence: float
    uncertainty: float
    provider_status: str
    failure_reason: str
    branch_name: str
    branch_group: EvidenceBranchGroup | str
    object_id: str = ""
    track_id: str = ""
    frame_id: str = ""
    video_id: str = ""
    clip_id: str = ""
    frame_index: int | None = None
    spatial_reference: Mapping[str, Any] = field(default_factory=dict)
    temporal_reference: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        branch_group = EvidenceBranchGroup(self.branch_group)
        value = float(self.residual_value)
        confidence = float(self.confidence)
        uncertainty = float(self.uncertainty)
        if not self.evidence_id or not self.branch_name or not self.provider_status:
            raise ValueError(
                "UnifiedEvidence requires evidence_id, branch_name, and provider_status."
            )
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Evidence confidence must be finite and in [0, 1].")
        if not (math.isnan(uncertainty) or (math.isfinite(uncertainty) and uncertainty >= 0)):
            raise ValueError("Evidence uncertainty must be NaN or non-negative.")
        if self.valid:
            if not self.applicable:
                raise ValueError("Valid evidence must be applicable.")
            if not math.isfinite(value) or value < 0.0 or self.failure_reason:
                raise ValueError(
                    "Valid evidence requires a finite non-negative residual and no reason."
                )
            if provider_status_is_failure(self.provider_status):
                raise ValueError("Provider failure cannot produce valid evidence.")
        else:
            if not math.isnan(value) or not self.failure_reason:
                raise ValueError("Invalid evidence requires residual_value=NaN and a reason.")
        object.__setattr__(self, "branch_group", branch_group)
        object.__setattr__(self, "residual_value", value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(
            self, "frame_index", None if self.frame_index is None else int(self.frame_index)
        )
        object.__setattr__(self, "spatial_reference", dict(self.spatial_reference))
        object.__setattr__(self, "temporal_reference", dict(self.temporal_reference))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def observed(
        cls,
        *,
        evidence_id: str,
        residual_value: float,
        confidence: float,
        uncertainty: float,
        provider_status: str,
        branch_name: str,
        branch_group: EvidenceBranchGroup | str,
        **identity: Any,
    ) -> "UnifiedEvidence":
        """Construct valid evidence; a normal residual of zero remains valid."""

        return cls(
            evidence_id=evidence_id,
            residual_value=residual_value,
            applicable=True,
            valid=True,
            confidence=confidence,
            uncertainty=uncertainty,
            provider_status=provider_status,
            failure_reason="",
            branch_name=branch_name,
            branch_group=branch_group,
            **identity,
        )

    @classmethod
    def unavailable(
        cls,
        *,
        evidence_id: str,
        applicable: bool,
        provider_status: str,
        failure_reason: str,
        branch_name: str,
        branch_group: EvidenceBranchGroup | str,
        confidence: float = 0.0,
        uncertainty: float = float("nan"),
        **identity: Any,
    ) -> "UnifiedEvidence":
        """Construct unavailable evidence without substituting a zero."""

        return cls(
            evidence_id=evidence_id,
            residual_value=float("nan"),
            applicable=applicable,
            valid=False,
            confidence=confidence,
            uncertainty=uncertainty,
            provider_status=provider_status,
            failure_reason=failure_reason,
            branch_name=branch_name,
            branch_group=branch_group,
            **identity,
        )

    def effective_confidence(self) -> float:
        """Return deterministic quality after uncertainty attenuation."""

        if not self.valid:
            return 0.0
        penalty = 1.0 + self.uncertainty if math.isfinite(self.uncertainty) else 1.0
        return self.confidence / penalty

    def to_aggregation_evidence(self) -> AggregationEvidence:
        """Adapt into the existing P4-A aggregation contract without losing status."""

        metadata = {
            **dict(self.provenance),
            "video_id": self.video_id,
            "clip_id": self.clip_id,
            "frame_id": self.frame_id,
            "object_id": self.object_id,
            "spatial_reference": dict(self.spatial_reference),
            "temporal_reference": dict(self.temporal_reference),
            "provider_status": self.provider_status,
            "uncertainty": self.uncertainty,
            "branch_group": self.branch_group.value,
            "adapted_from_unified_evidence": True,
        }
        point_or_edge_id = str(
            self.spatial_reference.get("point_id")
            or self.spatial_reference.get("edge_id")
            or ""
        )
        if self.valid:
            return AggregationEvidence.observed(
                self.residual_value,
                quality=self.effective_confidence(),
                branch_name=self.branch_group.value,
                source_id=self.evidence_id,
                frame_index=self.frame_index,
                object_track_id=self.track_id,
                point_or_edge_id=point_or_edge_id,
                metadata=metadata,
            )
        normalized = self.failure_reason.lower()
        if not self.applicable:
            status = EvidenceApplicability.NOT_APPLICABLE
        elif "unsupported" in normalized:
            status = EvidenceApplicability.UNSUPPORTED_MODE
        elif "invalid_geometry" in normalized:
            status = EvidenceApplicability.INVALID_GEOMETRY
        else:
            status = EvidenceApplicability.OBSERVATION_MISSING
        return AggregationEvidence.unavailable(
            applicability=status,
            reason=self.failure_reason,
            branch_name=self.branch_group.value,
            source_id=self.evidence_id,
            frame_index=self.frame_index,
            object_track_id=self.track_id,
            point_or_edge_id=point_or_edge_id,
            metadata=metadata,
        )

