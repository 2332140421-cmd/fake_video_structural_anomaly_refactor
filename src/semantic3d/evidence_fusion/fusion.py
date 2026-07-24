"""Deterministic missing-aware fusion with separate risk and completeness."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ..aggregation_v2 import aggregate_evidence_v2
from .contracts import (
    EvidenceBranchGroup,
    UnifiedEvidence,
    provider_status_is_failure,
)


@dataclass(frozen=True)
class BranchContribution:
    """One active branch's bounded risk and quality contribution."""

    branch_group: EvidenceBranchGroup
    residual_aggregate: float
    bounded_risk: float
    confidence: float
    configured_weight: float
    quality_adjusted_weight: float
    normalized_active_weight: float
    quality_adjusted_bounded_risk: float
    contribution_to_weighted_mean: float
    valid_evidence_count: int
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class FusionResult:
    """Risk and evidence completeness kept as separate outputs."""

    risk_score: float
    evidence_confidence: float
    active_branch_count: int
    available_weight_ratio: float
    missing_reason_summary: Mapping[str, int]
    branch_contributions: tuple[BranchContribution, ...]
    valid: bool
    failure_reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.valid:
            if not math.isfinite(self.risk_score) or self.failure_reason:
                raise ValueError("Valid fusion requires finite risk and no reason.")
        elif not math.isnan(self.risk_score) or not self.failure_reason:
            raise ValueError("Invalid fusion requires NaN risk and a reason.")
        for name in ("evidence_confidence", "available_weight_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")
        object.__setattr__(
            self, "missing_reason_summary", dict(self.missing_reason_summary)
        )
        object.__setattr__(self, "branch_contributions", tuple(self.branch_contributions))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _bounded_residual(value: float) -> float:
    """Map non-negative residuals monotonically to [0, 1) without fitting."""

    return 1.0 - math.exp(-max(0.0, float(value)))


def fuse_unified_evidence(
    evidences: Sequence[UnifiedEvidence],
    *,
    branch_weights: Mapping[EvidenceBranchGroup | str, float] | None = None,
    top_k: int = 3,
    excluded_groups: Iterable[EvidenceBranchGroup | str] = (),
) -> FusionResult:
    """Fuse valid residuals while missingness affects confidence, never risk."""

    excluded = {EvidenceBranchGroup(group) for group in excluded_groups}
    weights = {group: 1.0 for group in EvidenceBranchGroup}
    for key, value in dict(branch_weights or {}).items():
        group = EvidenceBranchGroup(key)
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise ValueError("Branch weights must be finite and non-negative.")
        weights[group] = numeric
    items = tuple(item for item in evidences if item.branch_group not in excluded)
    missing = Counter(item.failure_reason for item in items if not item.valid)
    grouped: dict[EvidenceBranchGroup, list[UnifiedEvidence]] = {}
    for item in items:
        grouped.setdefault(item.branch_group, []).append(item)
    applicable_groups = {
        group
        for group, rows in grouped.items()
        if any(row.applicable for row in rows)
    }
    denominator = sum(weights[group] for group in applicable_groups)
    branch_rows: list[tuple[EvidenceBranchGroup, float, float, float, tuple[str, ...]]] = []
    for group in sorted(grouped, key=lambda item: item.value):
        valid = [
            row
            for row in grouped[group]
            if row.valid
            and row.applicable
            and not provider_status_is_failure(row.provider_status)
        ]
        if not valid:
            continue
        aggregate = aggregate_evidence_v2(
            [row.to_aggregation_evidence() for row in valid],
            level="clip",
            method="hybrid_median_top_k",
            top_k=top_k,
            identity={"video_id": "", "clip_id": group.value},
        )
        if not aggregate.valid:
            continue
        confidence = float(
            np.mean([row.effective_confidence() for row in valid])
        )
        branch_rows.append(
            (
                group,
                aggregate.value,
                _bounded_residual(aggregate.value),
                confidence,
                tuple(row.evidence_id for row in valid),
            )
        )
    available_weight = sum(weights[group] for group, *_ in branch_rows)
    available_ratio = available_weight / denominator if denominator > 0 else 0.0
    quality_weights = {
        group: weights[group] * confidence
        for group, _, _, confidence, _ in branch_rows
    }
    active_quality_weight = sum(quality_weights.values())
    if not branch_rows or active_quality_weight <= 0:
        return FusionResult(
            risk_score=float("nan"),
            evidence_confidence=0.0,
            active_branch_count=0,
            available_weight_ratio=available_ratio,
            missing_reason_summary=dict(sorted(missing.items())),
            branch_contributions=(),
            valid=False,
            failure_reason="no_valid_fusion_evidence",
            metadata={
                "missingness_used_in_risk": False,
                "formal_threshold_selected": False,
                "classification_output": False,
            },
        )
    normalized = {
        group: quality_weights[group] / active_quality_weight
        for group, *_ in branch_rows
    }
    weighted_mean = sum(
        normalized[group] * bounded
        for group, _, bounded, _, _ in branch_rows
    )
    maximum = max(
        bounded * confidence
        for _, _, bounded, confidence, _ in branch_rows
    )
    risk = 0.5 * weighted_mean + 0.5 * maximum
    mean_quality = (
        sum(
            weights[group] * confidence
            for group, _, _, confidence, _ in branch_rows
        )
        / available_weight
    )
    confidence = float(np.clip(mean_quality * available_ratio, 0.0, 1.0))
    contributions = tuple(
        BranchContribution(
            branch_group=group,
            residual_aggregate=residual,
            bounded_risk=bounded,
            confidence=quality,
            configured_weight=weights[group],
            quality_adjusted_weight=quality_weights[group],
            normalized_active_weight=normalized[group],
            quality_adjusted_bounded_risk=bounded * quality,
            contribution_to_weighted_mean=normalized[group] * bounded,
            valid_evidence_count=len(source_ids),
            source_ids=source_ids,
        )
        for group, residual, bounded, quality, source_ids in branch_rows
    )
    return FusionResult(
        risk_score=risk,
        evidence_confidence=confidence,
        active_branch_count=len(branch_rows),
        available_weight_ratio=available_ratio,
        missing_reason_summary=dict(sorted(missing.items())),
        branch_contributions=contributions,
        valid=True,
        failure_reason="",
        metadata={
            "risk_formula": (
                "0.5 * quality_weighted_active_branch_mean + "
                "0.5 * max_quality_adjusted_active_branch"
            ),
            "residual_transform": "1-exp(-max(residual,0))",
            "missingness_used_in_risk": False,
            "missingness_affects_evidence_confidence_only": True,
            "formal_threshold_selected": False,
            "classification_output": False,
        },
    )


def branch_dropout_audit(
    evidences: Sequence[UnifiedEvidence],
    *,
    branch_weights: Mapping[EvidenceBranchGroup | str, float] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Recompute deterministic risk after excluding each branch; no fitting occurs."""

    output = []
    for group in EvidenceBranchGroup:
        result = fuse_unified_evidence(
            evidences,
            branch_weights=branch_weights,
            excluded_groups=(group,),
        )
        output.append(
            {
                "dropped_branch": group.value,
                "risk_score": result.risk_score,
                "evidence_confidence": result.evidence_confidence,
                "active_branch_count": result.active_branch_count,
                "available_weight_ratio": result.available_weight_ratio,
                "valid": result.valid,
            }
        )
    return tuple(output)
