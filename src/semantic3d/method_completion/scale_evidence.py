"""Unified evidence contracts for the P4-C3A-MD2 scale-geometry routes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class ScaleBranchName(str, Enum):
    """Stable names for the three scale-geometry branches."""

    METRIC_SINGLE_OBJECT = "metric_single_object_scale"
    TEMPORAL_SAME_OBJECT = "temporal_same_object_scale"
    RELATIVE_PAIR = "relative_pair_scale_depth"
    NONE = "no_scale_evidence"


class ScaleEvidenceRole(str, Enum):
    """Permitted roles in downstream evidence aggregation."""

    PRIMARY = "primary"
    TEMPORAL_SUPPORT = "temporal_support"
    FALLBACK = "fallback"
    AUDIT_CROSSCHECK = "audit_crosscheck"


class ProviderStatus(str, Enum):
    """Provider state kept separate from physical applicability."""

    OK = "ok"
    PROVIDER_FAILED = "provider_failed"
    NOT_EXECUTED = "not_executed"
    INTERFACE_ONLY = "interface_only"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class ScaleGeometryEvidence:
    """One validity-aware scale residual with routing and provenance metadata.

    A valid normal residual may be zero. Missing or inapplicable evidence must
    carry ``NaN`` and a stable ``failure_reason``. Audit cross-check evidence is
    deliberately distinguishable from evidence used by the primary route.
    """

    video_id: str
    clip_id: str
    frame_id: str
    object_id: str
    track_id: str
    branch_name: ScaleBranchName | str
    branch_priority: int
    evidence_role: ScaleEvidenceRole | str
    residual_name: str
    residual_value: float
    valid: bool
    applicable: bool
    confidence: float
    uncertainty: float
    failure_reason: str
    provider_status: ProviderStatus | str
    depth_type: str
    depth_unit: str
    depth_definition: str
    coordinate_system: str
    localization_reference: str
    provenance: Mapping[str, Any] = field(default_factory=dict)
    config_sha256: str = ""
    software_commit: str = ""

    def __post_init__(self) -> None:
        branch = ScaleBranchName(self.branch_name)
        role = ScaleEvidenceRole(self.evidence_role)
        provider = ProviderStatus(self.provider_status)
        residual = float(self.residual_value)
        confidence = float(self.confidence)
        uncertainty = float(self.uncertainty)
        if self.branch_priority < 0:
            raise ValueError("branch_priority must be non-negative.")
        if not 0.0 <= confidence <= 1.0 or not math.isfinite(confidence):
            raise ValueError("confidence must be finite and in [0, 1].")
        if not math.isnan(uncertainty) and (uncertainty < 0.0 or not math.isfinite(uncertainty)):
            raise ValueError("uncertainty must be non-negative, finite, or NaN.")
        if self.valid:
            if not self.applicable:
                raise ValueError("Valid scale evidence must be applicable.")
            if not math.isfinite(residual):
                raise ValueError("Valid scale evidence requires a finite residual.")
            if self.failure_reason:
                raise ValueError("Valid scale evidence cannot have failure_reason.")
            if provider != ProviderStatus.OK:
                raise ValueError("Valid scale evidence requires provider_status='ok'.")
        else:
            if not math.isnan(residual):
                raise ValueError("Invalid scale evidence must use residual_value=NaN.")
            if not self.failure_reason:
                raise ValueError("Invalid scale evidence requires failure_reason.")
        if provider == ProviderStatus.PROVIDER_FAILED and self.valid:
            raise ValueError("Provider failure cannot become valid anomaly evidence.")
        object.__setattr__(self, "branch_name", branch)
        object.__setattr__(self, "evidence_role", role)
        object.__setattr__(self, "provider_status", provider)
        object.__setattr__(self, "residual_value", residual)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "uncertainty", uncertainty)
        object.__setattr__(self, "provenance", dict(self.provenance))

    @classmethod
    def observed(
        cls,
        *,
        residual_value: float,
        confidence: float,
        uncertainty: float = 0.0,
        **kwargs: Any,
    ) -> "ScaleGeometryEvidence":
        """Create valid observed scale evidence."""

        return cls(
            residual_value=float(residual_value),
            valid=True,
            applicable=True,
            confidence=float(confidence),
            uncertainty=float(uncertainty),
            failure_reason="",
            provider_status=ProviderStatus.OK,
            **kwargs,
        )

    @classmethod
    def missing(
        cls,
        *,
        failure_reason: str,
        applicable: bool = False,
        provider_status: ProviderStatus | str = ProviderStatus.BLOCKED,
        **kwargs: Any,
    ) -> "ScaleGeometryEvidence":
        """Create missing evidence without substituting a normal zero."""

        return cls(
            residual_value=float("nan"),
            valid=False,
            applicable=bool(applicable),
            confidence=0.0,
            uncertainty=float("nan"),
            failure_reason=failure_reason,
            provider_status=provider_status,
            **kwargs,
        )

    @property
    def eligible_for_primary_aggregation(self) -> bool:
        """Return whether this evidence may enter the main scale aggregate."""

        return bool(
            self.valid
            and self.evidence_role != ScaleEvidenceRole.AUDIT_CROSSCHECK
            and self.provider_status == ProviderStatus.OK
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly deterministic mapping."""

        result = asdict(self)
        result["branch_name"] = self.branch_name.value
        result["evidence_role"] = self.evidence_role.value
        result["provider_status"] = self.provider_status.value
        return result


@dataclass(frozen=True)
class ScaleRouteDecision:
    """Router output preserving executed, skipped, and selected branches."""

    selected_primary_branch: str
    available_branches: tuple[str, ...]
    executed_branches: tuple[str, ...]
    skipped_branches: Mapping[str, str]
    fallback_used: bool
    audit_crosscheck_used: bool
    routing_reason: str
    evidence_confidence: float
    evidences: tuple[ScaleGeometryEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.evidence_confidence) <= 1.0:
            raise ValueError("evidence_confidence must be in [0, 1].")
        object.__setattr__(self, "available_branches", tuple(self.available_branches))
        object.__setattr__(self, "executed_branches", tuple(self.executed_branches))
        object.__setattr__(self, "skipped_branches", dict(self.skipped_branches))
        object.__setattr__(self, "evidences", tuple(self.evidences))

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable route decision."""

        return {
            "selected_primary_branch": self.selected_primary_branch,
            "available_branches": list(self.available_branches),
            "executed_branches": list(self.executed_branches),
            "skipped_branches": dict(self.skipped_branches),
            "fallback_used": self.fallback_used,
            "audit_crosscheck_used": self.audit_crosscheck_used,
            "routing_reason": self.routing_reason,
            "evidence_confidence": self.evidence_confidence,
            "evidences": [item.to_dict() for item in self.evidences],
        }


def valid_scale_evidences(
    evidences: Sequence[ScaleGeometryEvidence], *, include_audit: bool = False
) -> tuple[ScaleGeometryEvidence, ...]:
    """Filter valid evidence without interpreting provider failure as anomaly."""

    return tuple(
        item
        for item in evidences
        if item.valid
        and item.provider_status == ProviderStatus.OK
        and (include_audit or item.evidence_role != ScaleEvidenceRole.AUDIT_CROSSCHECK)
    )
