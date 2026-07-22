from __future__ import annotations

from semantic3d.coverage_readiness import (
    BranchCoverageStatus,
    evaluate_coverage_readiness,
)
from semantic3d.occlusion import OcclusionEventInputs, validate_occlusion_event


def test_no_event_is_not_applicable_not_zero_residual() -> None:
    result = evaluate_coverage_readiness(
        formal_mask_valid_ratio=0.9,
        mask_association_success_rate=0.9,
        stable_mask_track_ratio=0.8,
        person_structure_track_count=1,
        ordinary_structure_track_count=0,
        structure_residual_count=5,
        partial_occlusion_event_count=0,
        full_occlusion_event_count=0,
        reappearance_event_count=0,
        depth_order_evidence_count=0,
        boundary_occlusion_evidence_count=0,
    )
    assert result.branch_coverage["occlusion_event"] == BranchCoverageStatus.NOT_APPLICABLE
    assert result.ready_for_partial_p4 and not result.ready_for_full_p4


def test_failed_mask_observation_is_observation_missing() -> None:
    result = evaluate_coverage_readiness(
        formal_mask_valid_ratio=float("nan"),
        mask_association_success_rate=float("nan"),
        stable_mask_track_ratio=float("nan"),
        person_structure_track_count=0,
        ordinary_structure_track_count=0,
        structure_residual_count=0,
        partial_occlusion_event_count=0,
        full_occlusion_event_count=0,
        reappearance_event_count=0,
        depth_order_evidence_count=0,
        boundary_occlusion_evidence_count=0,
        mask_observation_missing=True,
        missing_reasons=("instance_segmentation_weights_missing",),
    )
    assert result.branch_coverage["occlusion_event"] == BranchCoverageStatus.OBSERVATION_MISSING
    assert not result.ready_for_partial_p4 and not result.ready_for_full_p4


def test_formal_occlusion_requires_all_observation_checks() -> None:
    valid = validate_occlusion_event(OcclusionEventInputs(
        event_type="partial_occlusion", formal_instance_mask=True,
        history_prediction=True, visible_area_change=True,
        candidate_occluder=True, depth_order=True, scene_cut=False,
        tracking_quality=0.8,
    ))
    assert valid.valid and valid.status == BranchCoverageStatus.AVAILABLE

    no_event = validate_occlusion_event(OcclusionEventInputs(
        event_type="partial_occlusion", formal_instance_mask=True,
        history_prediction=True, visible_area_change=False,
        candidate_occluder=False, depth_order=False, scene_cut=False,
        tracking_quality=0.8,
    ))
    assert not no_event.valid and no_event.status == BranchCoverageStatus.NOT_APPLICABLE

    missing = validate_occlusion_event(OcclusionEventInputs(
        event_type="full_occlusion", formal_instance_mask=False,
        history_prediction=False, visible_area_change=True,
        candidate_occluder=True, depth_order=True, scene_cut=False,
        tracking_quality=0.0,
    ))
    assert not missing.valid and missing.status == BranchCoverageStatus.OBSERVATION_MISSING
