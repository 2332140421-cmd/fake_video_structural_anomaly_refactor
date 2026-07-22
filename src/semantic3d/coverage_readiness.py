"""Observation coverage readiness for later P4 aggregation experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class BranchCoverageStatus(str, Enum):
    """Distinguish absent events from failed branch observations."""

    AVAILABLE = "available"
    NOT_APPLICABLE = "not_applicable"
    OBSERVATION_MISSING = "observation_missing"


@dataclass(frozen=True)
class CoverageReadiness:
    """Coverage-only readiness result; this is not an anomaly score."""

    formal_mask_valid_ratio: float
    mask_association_success_rate: float
    stable_mask_track_ratio: float
    person_structure_track_count: int
    ordinary_structure_track_count: int
    structure_residual_count: int
    partial_occlusion_event_count: int
    full_occlusion_event_count: int
    reappearance_event_count: int
    depth_order_evidence_count: int
    boundary_occlusion_evidence_count: int
    branch_coverage: Mapping[str, BranchCoverageStatus | str]
    ready_for_partial_p4: bool
    ready_for_full_p4: bool
    missing_reasons: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ratios = (
            float(self.formal_mask_valid_ratio),
            float(self.mask_association_success_rate),
            float(self.stable_mask_track_ratio),
        )
        for value in ratios:
            if not math.isnan(value) and not 0.0 <= value <= 1.0:
                raise ValueError("Coverage ratios must be NaN or in [0, 1].")
        counts = (
            self.person_structure_track_count,
            self.ordinary_structure_track_count,
            self.structure_residual_count,
            self.partial_occlusion_event_count,
            self.full_occlusion_event_count,
            self.reappearance_event_count,
            self.depth_order_evidence_count,
            self.boundary_occlusion_evidence_count,
        )
        if any(int(value) < 0 for value in counts):
            raise ValueError("Coverage evidence counts must be non-negative.")
        coverage = {
            str(name): BranchCoverageStatus(status)
            for name, status in self.branch_coverage.items()
        }
        object.__setattr__(self, "formal_mask_valid_ratio", ratios[0])
        object.__setattr__(self, "mask_association_success_rate", ratios[1])
        object.__setattr__(self, "stable_mask_track_ratio", ratios[2])
        object.__setattr__(self, "branch_coverage", coverage)
        object.__setattr__(self, "missing_reasons", tuple(dict.fromkeys(self.missing_reasons)))
        object.__setattr__(self, "metadata", dict(self.metadata))


def evaluate_coverage_readiness(
    *,
    formal_mask_valid_ratio: float,
    mask_association_success_rate: float,
    stable_mask_track_ratio: float,
    person_structure_track_count: int,
    ordinary_structure_track_count: int,
    structure_residual_count: int,
    partial_occlusion_event_count: int,
    full_occlusion_event_count: int,
    reappearance_event_count: int,
    depth_order_evidence_count: int,
    boundary_occlusion_evidence_count: int,
    mask_observation_missing: bool = False,
    structure_observation_missing: bool = False,
    missing_reasons: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> CoverageReadiness:
    """Evaluate reproducible branch readiness without inferring normality.

    A branch with usable observations but no event is ``not_applicable``.
    A branch whose required observation failed is ``observation_missing``.
    """

    stable_masks = (
        not mask_observation_missing
        and math.isfinite(float(formal_mask_valid_ratio))
        and formal_mask_valid_ratio > 0.0
        and math.isfinite(float(mask_association_success_rate))
        and mask_association_success_rate > 0.0
    )
    structure_available = (
        not structure_observation_missing
        and person_structure_track_count + ordinary_structure_track_count > 0
        and structure_residual_count > 0
    )
    any_occlusion_event = (
        partial_occlusion_event_count
        + full_occlusion_event_count
        + reappearance_event_count
    ) > 0
    if mask_observation_missing:
        mask_status = BranchCoverageStatus.OBSERVATION_MISSING
        occlusion_status = BranchCoverageStatus.OBSERVATION_MISSING
    else:
        mask_status = (
            BranchCoverageStatus.AVAILABLE
            if stable_masks else BranchCoverageStatus.OBSERVATION_MISSING
        )
        occlusion_status = (
            BranchCoverageStatus.AVAILABLE
            if any_occlusion_event else BranchCoverageStatus.NOT_APPLICABLE
        )
    if structure_observation_missing:
        structure_status = BranchCoverageStatus.OBSERVATION_MISSING
    else:
        structure_status = (
            BranchCoverageStatus.AVAILABLE
            if structure_available else BranchCoverageStatus.NOT_APPLICABLE
        )
    depth_order_status = (
        BranchCoverageStatus.OBSERVATION_MISSING
        if mask_observation_missing
        else BranchCoverageStatus.AVAILABLE
        if depth_order_evidence_count > 0
        else BranchCoverageStatus.NOT_APPLICABLE
    )
    boundary_status = (
        BranchCoverageStatus.OBSERVATION_MISSING
        if mask_observation_missing
        else BranchCoverageStatus.AVAILABLE
        if boundary_occlusion_evidence_count > 0
        else BranchCoverageStatus.NOT_APPLICABLE
    )
    branch_coverage = {
        "formal_mask": mask_status,
        "structure_temporal": structure_status,
        "occlusion_event": occlusion_status,
        "depth_order": depth_order_status,
        "boundary_occlusion": boundary_status,
    }
    partial = structure_available or occlusion_status == BranchCoverageStatus.AVAILABLE
    full = (
        structure_available
        and occlusion_status == BranchCoverageStatus.AVAILABLE
        and depth_order_status == BranchCoverageStatus.AVAILABLE
        and boundary_status == BranchCoverageStatus.AVAILABLE
    )
    return CoverageReadiness(
        formal_mask_valid_ratio=formal_mask_valid_ratio,
        mask_association_success_rate=mask_association_success_rate,
        stable_mask_track_ratio=stable_mask_track_ratio,
        person_structure_track_count=person_structure_track_count,
        ordinary_structure_track_count=ordinary_structure_track_count,
        structure_residual_count=structure_residual_count,
        partial_occlusion_event_count=partial_occlusion_event_count,
        full_occlusion_event_count=full_occlusion_event_count,
        reappearance_event_count=reappearance_event_count,
        depth_order_evidence_count=depth_order_evidence_count,
        boundary_occlusion_evidence_count=boundary_occlusion_evidence_count,
        branch_coverage=branch_coverage,
        ready_for_partial_p4=partial,
        ready_for_full_p4=full,
        missing_reasons=tuple(missing_reasons),
        metadata={"readiness_is_anomaly_score": False, **dict(metadata or {})},
    )
