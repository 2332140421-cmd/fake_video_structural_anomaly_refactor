"""Strict observation gate for formal partial/full occlusion events."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..coverage_readiness import BranchCoverageStatus


@dataclass(frozen=True)
class OcclusionEventInputs:
    """Required observation predicates for one candidate visibility event."""

    event_type: str
    formal_instance_mask: bool
    history_prediction: bool
    visible_area_change: bool
    candidate_occluder: bool
    depth_order: bool
    scene_cut: bool
    tracking_quality: float

    def __post_init__(self) -> None:
        if self.event_type not in {"partial_occlusion", "full_occlusion"}:
            raise ValueError("event_type must be partial_occlusion or full_occlusion.")
        if not math.isfinite(float(self.tracking_quality)) or not 0.0 <= float(self.tracking_quality) <= 1.0:
            raise ValueError("tracking_quality must be in [0, 1].")


@dataclass(frozen=True)
class OcclusionEventValidation:
    """Formal event gate result without an anomaly score."""

    event_type: str
    status: BranchCoverageStatus | str
    valid: bool
    quality: float
    passed_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    missing_reason: str = ""

    def __post_init__(self) -> None:
        status = BranchCoverageStatus(self.status)
        quality = float(self.quality)
        if not 0.0 <= quality <= 1.0:
            raise ValueError("Event quality must be in [0, 1].")
        if self.valid and (status != BranchCoverageStatus.AVAILABLE or self.failed_checks or self.missing_reason):
            raise ValueError("Valid event requires all checks to pass.")
        if not self.valid and (status == BranchCoverageStatus.AVAILABLE or not self.missing_reason):
            raise ValueError("Invalid event requires a non-available status and reason.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "quality", quality)


def validate_occlusion_event(
    inputs: OcclusionEventInputs,
    *,
    minimum_tracking_quality: float = 0.5,
) -> OcclusionEventValidation:
    """Require every physical observation before promoting a formal event."""

    checks = {
        "formal_instance_mask": inputs.formal_instance_mask,
        "history_prediction": inputs.history_prediction,
        "visible_area_change": inputs.visible_area_change,
        "candidate_occluder": inputs.candidate_occluder,
        "depth_order": inputs.depth_order,
        "no_scene_cut": not inputs.scene_cut,
        "tracking_quality": inputs.tracking_quality >= minimum_tracking_quality,
    }
    passed = tuple(name for name, value in checks.items() if value)
    failed = tuple(name for name, value in checks.items() if not value)
    observation_failures = {
        "formal_instance_mask",
        "history_prediction",
        "tracking_quality",
    }.intersection(failed)
    if observation_failures:
        status = BranchCoverageStatus.OBSERVATION_MISSING
        reason = "insufficient_" + "_and_".join(sorted(observation_failures))
    elif failed:
        status = BranchCoverageStatus.NOT_APPLICABLE
        reason = "occlusion_event_conditions_not_met"
    else:
        status = BranchCoverageStatus.AVAILABLE
        reason = ""
    return OcclusionEventValidation(
        event_type=inputs.event_type,
        status=status,
        valid=not failed,
        quality=inputs.tracking_quality if not failed else 0.0,
        passed_checks=passed,
        failed_checks=failed,
        missing_reason=reason,
    )
