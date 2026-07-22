from __future__ import annotations

import math

import numpy as np

from semantic3d.occlusion import (
    OcclusionRelation,
    VisibilityExplanation,
    VisibilityState,
    build_occlusion_graph,
    compute_boundary_occlusion_residual,
    compute_occlusion_depth_order_residual,
    compute_visibility_explanation_residual,
    evaluate_reappearance,
    infer_visibility_state,
)
from synthetic_occlusion import normal_occlusion_scene, observed, rectangle, support


def _normal_relation_scene():
    scene = normal_occlusion_scene()
    graph = build_occlusion_graph(
        video_id="synthetic", frame_index=2,
        predicted_supports={"foreground": scene.foreground_support, "background": scene.background_support},
        observed_masks={"foreground": scene.foreground_visible, "background": scene.background_visible},
        object_depths={"foreground": 3.0, "background": 7.0},
    )
    return scene, graph, graph.relations[0]


def test_normal_occlusion_depth_order_is_low_and_reversal_is_high() -> None:
    _, _, relation = _normal_relation_scene()
    assert relation.valid
    normal = compute_occlusion_depth_order_residual(relation)
    reversed_order = compute_occlusion_depth_order_residual(relation, observed_foreground_depth=8.0, observed_background_depth=3.0)
    assert normal.evidence.valid and normal.evidence.value == 0.0
    assert reversed_order.evidence.valid and reversed_order.evidence.value > 0.5
    assert normal.evidence.metadata["center_depth_low_quality"] is True


def test_partial_and_full_occlusion_states_are_distinct() -> None:
    scene = normal_occlusion_scene()
    partial = infer_visibility_state(scene.background_support, scene.background_visible, nearer_object_masks={"foreground": scene.foreground_visible.visible_mask})
    assert partial.current_state == VisibilityState.PARTIALLY_OCCLUDED
    full_foreground = observed("foreground", 2, scene.background_support.support_mask)
    full = infer_visibility_state(scene.background_support, None, nearer_object_masks={"foreground": full_foreground.visible_mask})
    assert full.current_state == VisibilityState.FULLY_OCCLUDED
    assert full.valid


def test_out_of_frame_and_detector_missing_are_not_anomalies() -> None:
    tiny = support("o", 2, rectangle(0, 10, 2, 20), in_frame_ratio=0.1)
    out = infer_visibility_state(tiny, None)
    assert out.current_state == VisibilityState.OUT_OF_FRAME
    assert not compute_visibility_explanation_residual(out).residual_evidence.valid
    prediction = support("o", 2, rectangle(10, 10, 20, 20))
    miss = infer_visibility_state(prediction, None, detector_confidence=0.2)
    residual = compute_visibility_explanation_residual(miss)
    assert miss.current_state == VisibilityState.DETECTOR_MISSING
    assert not residual.residual_evidence.valid


def test_unexplained_disappearance_and_appearance_are_high() -> None:
    prediction = support("o", 2, rectangle(10, 10, 20, 20))
    disappear = infer_visibility_state(prediction, None, detector_confidence=1.0, detection_confirmed_absent=True)
    disappearance = compute_visibility_explanation_residual(disappear)
    assert disappearance.explanation == VisibilityExplanation.UNEXPLAINED_DISAPPEARANCE
    assert disappearance.residual_evidence.value == 1.0
    missing_prediction = type(prediction).missing(video_id="synthetic", object_track_id="o", target_frame_index=2, image_shape=(64, 64), geometry_mode="static_camera_3d", reason="no_history")
    appearance_state = infer_visibility_state(
        missing_prediction,
        observed("o", 2, rectangle(10, 10, 20, 20)),
        appearance_without_history_is_event=True,
    )
    appearance = compute_visibility_explanation_residual(appearance_state)
    assert appearance.explanation == VisibilityExplanation.UNEXPLAINED_APPEARANCE
    assert appearance.residual_evidence.value == 1.0


def test_scene_cut_blocks_state_and_reappearance() -> None:
    prediction = support("o", 2, rectangle(10, 10, 20, 20))
    state = infer_visibility_state(prediction, None, scene_cut=True)
    assert state.current_state == VisibilityState.SCENE_CUT and not state.valid
    result = evaluate_reappearance(previous_object_track_id="o", candidate_object_track_id="o", frame_index=2, predicted_reappearance_region=(10, 10, 20, 20), semantic_label_match=True, appearance_similarity=1.0, structure_similarity=1.0, relative_depth_consistency=1.0, motion_direction_consistency=1.0, reid_source="synthetic", scene_cut=True)
    assert not result.valid and math.isnan(result.evidence.value)


def test_normal_reappearance_low_and_wrong_reid_rejected() -> None:
    normal = evaluate_reappearance(previous_object_track_id="o", candidate_object_track_id="o", frame_index=4, predicted_reappearance_region=(10, 10, 20, 20), semantic_label_match=True, appearance_similarity=0.95, structure_similarity=0.95, relative_depth_consistency=0.95, motion_direction_consistency=0.95, reid_source="multi_cue")
    wrong = evaluate_reappearance(previous_object_track_id="o", candidate_object_track_id="other", frame_index=4, predicted_reappearance_region=(10, 10, 20, 20), semantic_label_match=False, appearance_similarity=0.1, structure_similarity=0.1, relative_depth_consistency=0.2, motion_direction_consistency=0.1, reid_source="multi_cue")
    assert normal.valid and normal.evidence.value < 0.1
    assert not wrong.valid and wrong.missing_reason == "semantic_identity_mismatch"


def test_boundary_residual_uses_independent_high_quality_masks() -> None:
    scene, _, relation = _normal_relation_scene()
    result = compute_boundary_occlusion_residual(relation, predicted_foreground=scene.foreground_support, predicted_background=scene.background_support, observed_foreground=scene.foreground_visible, observed_background=scene.background_visible)
    assert result.valid and result.residual_evidence.valid
    assert result.metadata["current_observed_boundary_used_for_prediction"] is False


def test_bbox_overlap_alone_does_not_form_occlusion_relation() -> None:
    first = support("a", 2, rectangle(10, 10, 30, 30), legacy=True)
    second = support("b", 2, rectangle(20, 20, 40, 40), legacy=True)
    graph = build_occlusion_graph(video_id="v", frame_index=2, predicted_supports={"a": first, "b": second}, observed_masks={"a": observed("a", 2, first.support_mask, legacy=True), "b": observed("b", 2, second.support_mask, legacy=True)}, object_depths={"a": 2.0, "b": 5.0})
    assert graph.relations and not graph.relations[0].valid
    assert graph.relations[0].missing_reason == "bbox_overlap_only_not_occlusion"


def test_missing_mask_and_unavailable_support_remain_nan() -> None:
    missing = support("o", 2, rectangle(10, 10, 20, 20))
    missing = type(missing).missing(video_id="v", object_track_id="o", target_frame_index=2, image_shape=(64, 64), geometry_mode="unavailable", reason="dynamic_geometry_unavailable")
    state = infer_visibility_state(missing, None)
    evidence = compute_visibility_explanation_residual(state)
    assert not state.valid
    assert not evidence.residual_evidence.valid and math.isnan(evidence.residual_evidence.value)
    assert math.isnan(state.predicted_support_area)
    assert math.isnan(state.visible_ratio)
    assert not evidence.diagnostic_evidence.valid
    assert math.isnan(evidence.diagnostic_evidence.value)


def test_unavailable_mode_rejects_even_when_current_mask_exists() -> None:
    missing = type(support("o", 2, rectangle(10, 10, 20, 20))).missing(
        video_id="v", object_track_id="o", target_frame_index=2,
        image_shape=(64, 64), geometry_mode="unavailable",
        reason="dynamic_geometry_unavailable",
    )
    state = infer_visibility_state(missing, observed("o", 2, rectangle(10, 10, 20, 20)))
    result = compute_visibility_explanation_residual(state)
    assert state.current_state == VisibilityState.UNCERTAIN
    assert not state.valid and not result.residual_evidence.valid
    assert math.isnan(result.residual_evidence.value)


def test_sequence_start_is_not_unexplained_appearance_by_default() -> None:
    missing = type(support("o", 2, rectangle(10, 10, 20, 20))).missing(
        video_id="v", object_track_id="o", target_frame_index=2,
        image_shape=(64, 64), geometry_mode="static_camera_3d",
        reason="insufficient_mask_history",
    )
    state = infer_visibility_state(missing, observed("o", 2, rectangle(10, 10, 20, 20)))
    assert state.current_state == VisibilityState.UNCERTAIN
    assert not state.valid and state.missing_reason == "insufficient_history_for_appearance_explanation"
