"""Motion- and pose-aware branch routing without authenticity inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..method_completion.observability import ClipObservability
from .contracts import EvidenceBranchGroup


@dataclass(frozen=True)
class BranchRouteDecision:
    """One branch's eligibility under measured clip observability."""

    branch_group: EvidenceBranchGroup
    applicable: bool
    input_expected: bool
    status: str
    reason: str


_STATIC_GROUPS = frozenset(
    {
        EvidenceBranchGroup.STATIC_METRIC_GEOMETRY,
        EvidenceBranchGroup.STATIC_RELATIVE_GEOMETRY,
        EvidenceBranchGroup.BOUNDARY_STRUCTURE,
        EvidenceBranchGroup.INTERNAL_STRUCTURE,
        EvidenceBranchGroup.TEMPORAL_SCALE,
    }
)


def route_evidence_branches(
    observability: ClipObservability | str,
    *,
    pose_available: bool,
    event_observed: bool = False,
) -> Mapping[EvidenceBranchGroup, BranchRouteDecision]:
    """Return deterministic branch applicability independent of residual values."""

    mode = ClipObservability(observability)
    dynamic = mode in {
        ClipObservability.OBJECT_MOTION,
        ClipObservability.CAMERA_MOTION,
        ClipObservability.MIXED_MOTION,
    }
    output: dict[EvidenceBranchGroup, BranchRouteDecision] = {}
    for group in EvidenceBranchGroup:
        applicable = True
        expected = True
        status = "enabled"
        reason = ""
        if mode in {ClipObservability.STATIC, ClipObservability.LOW_MOTION}:
            if group not in _STATIC_GROUPS:
                applicable = False
                expected = False
                status = "not_applicable"
                reason = "dynamic_branch_not_applicable_to_static_or_low_motion"
        elif mode == ClipObservability.MOTION_UNRELIABLE:
            if group not in _STATIC_GROUPS:
                status = "blocked_by_input"
                reason = "motion_observability_unreliable"
        elif dynamic and not pose_available and group in {
            EvidenceBranchGroup.D2_POSE_REPROJECTION,
            EvidenceBranchGroup.D3_STRUCTURAL_RELATION,
        }:
            status = "blocked_by_input"
            reason = "pose_unavailable_for_pose_compensated_branch"
        if group == EvidenceBranchGroup.OCCLUSION_REAPPEARANCE and not event_observed:
            applicable = False
            expected = False
            status = "not_applicable"
            reason = "not_applicable_no_event"
        output[group] = BranchRouteDecision(
            branch_group=group,
            applicable=applicable,
            input_expected=expected,
            status=status,
            reason=reason,
        )
    return output

