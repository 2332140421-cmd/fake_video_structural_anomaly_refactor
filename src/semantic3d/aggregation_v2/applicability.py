"""Applicability-aware evidence records for P4 aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..validity import ResidualEvidence


class EvidenceApplicability(str, Enum):
    """Why one branch value may or may not enter anomaly aggregation."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"
    OBSERVATION_MISSING = "observation_missing"
    INVALID_GEOMETRY = "invalid_geometry"
    UNSUPPORTED_MODE = "unsupported_mode"


@dataclass(frozen=True)
class AggregationEvidence:
    """One fully localized, applicability-aware residual observation.

    ``quality`` is an engineering evidence-quality score, not a probability.
    Only finite, valid, applicable records may enter anomaly aggregation.
    """

    value: float
    valid: bool
    quality: float
    applicability: EvidenceApplicability | str
    missing_reason: str
    branch_name: str
    source_id: str
    frame_index: int | None = None
    object_track_id: str = ""
    point_or_edge_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = float(self.value)
        quality = float(self.quality)
        applicability = EvidenceApplicability(self.applicability)
        if not self.branch_name or not self.source_id:
            raise ValueError("AggregationEvidence requires branch_name and source_id.")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("AggregationEvidence quality must be in [0, 1].")
        if self.valid:
            if applicability != EvidenceApplicability.APPLICABLE:
                raise ValueError("Valid evidence must be applicable.")
            if not math.isfinite(value) or self.missing_reason:
                raise ValueError("Valid evidence requires a finite value and no missing reason.")
        else:
            if applicability == EvidenceApplicability.APPLICABLE:
                raise ValueError("Invalid evidence requires a non-applicable status.")
            if not math.isnan(value) or not self.missing_reason:
                raise ValueError("Invalid evidence requires NaN and a missing reason.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "applicability", applicability)
        object.__setattr__(self, "frame_index", None if self.frame_index is None else int(self.frame_index))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def observed(
        cls,
        value: float,
        *,
        quality: float,
        branch_name: str,
        source_id: str,
        frame_index: int | None = None,
        object_track_id: str = "",
        point_or_edge_id: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> "AggregationEvidence":
        """Create valid applicable evidence; a normal value of zero is valid."""

        return cls(
            value=float(value), valid=True, quality=float(quality),
            applicability=EvidenceApplicability.APPLICABLE, missing_reason="",
            branch_name=branch_name, source_id=source_id,
            frame_index=frame_index, object_track_id=object_track_id,
            point_or_edge_id=point_or_edge_id, metadata=dict(metadata or {}),
        )

    @classmethod
    def unavailable(
        cls,
        *,
        applicability: EvidenceApplicability | str,
        reason: str,
        branch_name: str,
        source_id: str,
        frame_index: int | None = None,
        object_track_id: str = "",
        point_or_edge_id: str = "",
        quality: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AggregationEvidence":
        """Create unavailable evidence without substituting a zero residual."""

        status = EvidenceApplicability(applicability)
        if status == EvidenceApplicability.APPLICABLE:
            raise ValueError("Unavailable evidence cannot use applicability='applicable'.")
        default_quality = 1.0 if status == EvidenceApplicability.NOT_APPLICABLE else 0.0
        return cls(
            value=float("nan"), valid=False,
            quality=default_quality if quality is None else float(quality),
            applicability=status, missing_reason=str(reason),
            branch_name=branch_name, source_id=source_id,
            frame_index=frame_index, object_track_id=object_track_id,
            point_or_edge_id=point_or_edge_id, metadata=dict(metadata or {}),
        )


def applicability_from_missing_reason(reason: str) -> EvidenceApplicability:
    """Map legacy missing-reason strings to explicit applicability states."""

    normalized = str(reason).strip().lower()
    if normalized == "not_applicable" or normalized.startswith("no_observable_"):
        return EvidenceApplicability.NOT_APPLICABLE
    if normalized == "invalid_geometry" or "invalid_geometry" in normalized:
        return EvidenceApplicability.INVALID_GEOMETRY
    if normalized == "unsupported_mode" or "unsupported" in normalized:
        return EvidenceApplicability.UNSUPPORTED_MODE
    return EvidenceApplicability.OBSERVATION_MISSING


def from_residual_evidence(
    evidence: ResidualEvidence,
    *,
    branch_name: str | None = None,
    source_id: str | None = None,
    frame_index: int | None = None,
    object_track_id: str = "",
    point_or_edge_id: str = "",
) -> AggregationEvidence:
    """Adapt the canonical P0 residual contract without losing NaN semantics."""

    branch = branch_name or str(evidence.metadata.get("branch_name", evidence.name))
    source = source_id or (evidence.source_ids[0] if evidence.source_ids else evidence.name)
    metadata = {**dict(evidence.metadata), "adapted_from_residual_evidence": True}
    if evidence.valid:
        return AggregationEvidence.observed(
            evidence.value, quality=evidence.quality, branch_name=branch,
            source_id=source, frame_index=frame_index,
            object_track_id=object_track_id, point_or_edge_id=point_or_edge_id,
            metadata=metadata,
        )
    return AggregationEvidence.unavailable(
        applicability=applicability_from_missing_reason(evidence.missing_reason),
        reason=evidence.missing_reason, branch_name=branch, source_id=source,
        frame_index=frame_index, object_track_id=object_track_id,
        point_or_edge_id=point_or_edge_id, metadata=metadata,
    )


# Concise public alias for callers that prefer a generic evidence name.
EvidenceRecord = AggregationEvidence
