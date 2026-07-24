"""Tests for P4-C3B-M5 D3 relations, event typing, and NaN semantics."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import yaml

from semantic3d.d3 import (
    D3GraphNode,
    D3NodeType,
    D3StructureResidualExecutor,
    D3TransitionContext,
    OcclusionEventInputsV2,
    OcclusionEventType,
    build_d3_frame_graph,
    classify_occlusion_event,
    compute_reappearance_residual,
)
from semantic3d.d3 import smoke as smoke_module
from semantic3d.method_completion.d3_relations import D3RelationType
from semantic3d.pose_d2 import PoseProviderStatus


def _node(
    node_id: str,
    node_type: D3NodeType,
    frame_index: int,
    xyz: tuple[float, float, float],
    *,
    track_id: str = "a",
    reliable: bool = True,
    orientation: tuple[float, float, float] | None = None,
) -> D3GraphNode:
    return D3GraphNode(
        node_id=node_id,
        node_type=node_type,
        frame_index=frame_index,
        object_id=f"obj_{track_id}",
        track_id=track_id,
        semantic_label="synthetic",
        xyz_m=xyz,
        coordinate_frame="clip_local_aligned",
        source_observation_id=f"source:{frame_index}:{node_id}",
        confidence=0.9,
        identity_reliable=reliable,
        visibility="visible",
        valid=True,
        orientation_vector=orientation,
        localization_reference={"node_id": node_id},
        provenance={"synthetic_truth": True},
    )


def _graph(
    frame_index: int,
    *,
    object_b: tuple[float, float, float] = (1.0, 0.0, 4.0),
    object_b_orientation: tuple[float, float, float] = (1.0, 0.0, 0.0),
    deform: bool = False,
    reliable_points: bool = True,
):
    nodes = (
        _node(
            "object:a",
            D3NodeType.OBJECT_NODE,
            frame_index,
            (0.0, 0.0, 3.0),
            orientation=(1.0, 0.0, 0.0),
        ),
        _node(
            "object:b",
            D3NodeType.OBJECT_NODE,
            frame_index,
            object_b,
            track_id="b",
            orientation=object_b_orientation,
        ),
        _node(
            "boundary:a:0",
            D3NodeType.BOUNDARY_NODE,
            frame_index,
            (-0.3, 0.0, 3.0),
        ),
        _node(
            "boundary:a:1",
            D3NodeType.BOUNDARY_NODE,
            frame_index,
            (0.3, 0.0, 3.0),
        ),
        _node(
            "p1",
            D3NodeType.GEOMETRIC_TRACK_NODE,
            frame_index,
            (-0.2, 0.0, 3.0),
            reliable=reliable_points,
        ),
        _node(
            "p2",
            D3NodeType.GEOMETRIC_TRACK_NODE,
            frame_index,
            ((0.5 if deform else 0.2), 0.0, 3.0),
            reliable=reliable_points,
        ),
        _node(
            "p3",
            D3NodeType.GEOMETRIC_TRACK_NODE,
            frame_index,
            (0.0, 0.3, 3.0),
            reliable=reliable_points,
        ),
    )
    return build_d3_frame_graph(
        graph_id=f"graph:{frame_index}:{object_b}:{deform}",
        video_id="synthetic",
        clip_id="clip",
        frame_index=frame_index,
        nodes=nodes,
        pose_source="synthetic_known_pose",
        fixed_structure_edges=(("e12", "p1", "p2"), ("e13", "p1", "p3")),
        containment_relations={("a", "b"): (0.1, 0.2)},
        support_contact_relations={("a", "b"): (0.02, 0.9)},
    )


def _context(*, valid: bool = True) -> D3TransitionContext:
    return D3TransitionContext(
        video_id="synthetic",
        clip_id="clip",
        frame_t=0,
        frame_t1=1,
        pose_status=(
            PoseProviderStatus.ESTIMATED_VALID
            if valid
            else PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE
        ),
        pose_confidence=0.9 if valid else 0.0,
        pose_valid=valid,
        correspondence_identity_reliable=valid,
        source_coordinate_frame="clip_local_aligned",
        target_coordinate_frame="clip_local_aligned",
        valid=valid,
        failure_reason="" if valid else "blocked_by_pose_or_correspondence",
    )


def _event_input(**overrides: object) -> OcclusionEventInputsV2:
    values = {
        "video_id": "synthetic",
        "clip_id": "events",
        "frame_index": 3,
        "object_track_id": "track_a",
        "previous_event_type": OcclusionEventType.UNKNOWN,
        "formal_visible_mask_available": True,
        "history_prediction_available": True,
        "observed_object_available": False,
        "candidate_object_available": False,
        "identity_consistent": True,
        "mask_overlap_ratio": 0.0,
        "depth_order_supported": False,
        "visible_ratio": float("nan"),
        "predicted_in_frame_ratio": 1.0,
        "trajectory_prediction_quality": 0.9,
        "d2_reprojection_supported": True,
        "detector_attempted": True,
        "detector_reliable": True,
        "detection_confirmed_absent": False,
        "tracker_failed": False,
        "persistent_absence_frames": 0,
        "possible_occluder_ids": (),
    }
    values.update(overrides)
    return OcclusionEventInputsV2(**values)


def _valid_residuals(previous, current):
    return [
        item
        for item in D3StructureResidualExecutor().compare_graphs(
            previous, current, _context()
        )
        if item.valid
    ]


def test_d3_graph_contains_required_node_and_relation_types() -> None:
    graph = _graph(0)
    node_types = {node.node_type for node in graph.nodes}
    relation_types = {relation.relation_type for relation in graph.relations}
    assert {
        D3NodeType.OBJECT_NODE,
        D3NodeType.BOUNDARY_NODE,
        D3NodeType.GEOMETRIC_TRACK_NODE,
    } <= node_types
    assert {
        D3RelationType.OBJECT_RELATIVE_DISTANCE,
        D3RelationType.DEPTH_ORDER,
        D3RelationType.OBJECT_BOUNDARY_RELATION,
        D3RelationType.STRUCTURE_EDGE_LENGTH,
        D3RelationType.LOCAL_RIGIDITY,
        D3RelationType.RELATIVE_ORIENTATION,
        D3RelationType.CONTAINMENT_OR_OVERLAP,
        D3RelationType.SUPPORT_OR_CONTACT,
    } <= relation_types
    assert graph.coordinate_frame == "clip_local_aligned"
    assert not graph.metadata["world_frame_claimed"]


def test_rigid_graph_has_zero_core_residuals_and_localization() -> None:
    residuals = _valid_residuals(_graph(0), _graph(1))
    required = {
        "R_relative_distance",
        "R_depth_order",
        "R_edge_length",
        "R_local_rigidity",
        "R_relative_orientation",
    }
    assert required <= {item.residual_name for item in residuals}
    assert all(item.value <= 1e-12 for item in residuals)
    assert all(item.source_nodes and item.localization_reference for item in residuals)


def test_d3_core_residuals_respond_to_controlled_changes() -> None:
    reference = _graph(0)
    cases = (
        (_graph(1, object_b=(2.0, 0.0, 4.0)), "R_relative_distance"),
        (_graph(1, object_b=(1.0, 0.0, 2.0)), "R_depth_order"),
        (_graph(1, deform=True), "R_edge_length"),
        (_graph(1, deform=True), "R_local_rigidity"),
        (
            _graph(1, object_b_orientation=(0.0, 1.0, 0.0)),
            "R_relative_orientation",
        ),
    )
    for current, residual_name in cases:
        matching = [
            item.value
            for item in _valid_residuals(reference, current)
            if item.residual_name == residual_name
        ]
        assert matching and max(matching) > 0.0


def test_pose_or_correspondence_gate_blocks_with_nan() -> None:
    residuals = D3StructureResidualExecutor().compare_graphs(
        _graph(0), _graph(1), _context(valid=False)
    )
    assert residuals
    assert all(not item.valid for item in residuals)
    assert all(math.isnan(item.value) for item in residuals)
    assert {
        item.failure_reason for item in residuals
    } == {"blocked_by_pose_or_correspondence"}


def test_unreliable_structure_point_identity_blocks_edge_evidence() -> None:
    residuals = D3StructureResidualExecutor().compare_graphs(
        _graph(0, reliable_points=False),
        _graph(1, reliable_points=False),
        _context(),
    )
    structure = [
        item
        for item in residuals
        if item.residual_name in {"R_edge_length", "R_local_rigidity"}
    ]
    assert structure
    assert all(not item.valid and math.isnan(item.value) for item in structure)


def test_occlusion_classifier_distinguishes_partial_full_and_out_of_frame() -> None:
    partial = classify_occlusion_event(
        _event_input(
            observed_object_available=True,
            candidate_object_available=True,
            visible_ratio=0.4,
            mask_overlap_ratio=0.6,
            depth_order_supported=True,
            possible_occluder_ids=("track_b",),
        )
    )
    full = classify_occlusion_event(
        _event_input(
            mask_overlap_ratio=0.9,
            depth_order_supported=True,
            possible_occluder_ids=("track_b",),
        )
    )
    outside = classify_occlusion_event(
        _event_input(predicted_in_frame_ratio=0.05)
    )
    assert partial.event_type == OcclusionEventType.PARTIAL_OCCLUSION
    assert full.event_type == OcclusionEventType.FULL_OCCLUSION
    assert outside.event_type == OcclusionEventType.OUT_OF_FRAME
    assert partial.valid and full.valid and outside.valid


def test_event_classifier_distinguishes_detector_track_and_disappearance() -> None:
    detector = classify_occlusion_event(
        _event_input(detector_reliable=False)
    )
    track = classify_occlusion_event(_event_input(tracker_failed=True))
    disappearance = classify_occlusion_event(
        _event_input(
            detection_confirmed_absent=True,
            persistent_absence_frames=3,
        )
    )
    assert detector.event_type == OcclusionEventType.DETECTOR_MISS
    assert track.event_type == OcclusionEventType.TRACK_FAILURE
    assert disappearance.event_type == OcclusionEventType.TRUE_DISAPPEARANCE


def test_reappearance_and_id_switch_are_distinct() -> None:
    reappearance = classify_occlusion_event(
        _event_input(
            previous_event_type=OcclusionEventType.FULL_OCCLUSION,
            observed_object_available=True,
            candidate_object_available=True,
            identity_consistent=True,
        )
    )
    switched = classify_occlusion_event(
        _event_input(
            previous_event_type=OcclusionEventType.FULL_OCCLUSION,
            observed_object_available=True,
            candidate_object_available=True,
            identity_consistent=False,
        )
    )
    assert reappearance.event_type == OcclusionEventType.REAPPEARANCE
    assert switched.event_type == OcclusionEventType.TRACK_FAILURE

    residual = compute_reappearance_residual(
        event=reappearance,
        previous_track_id="track_a",
        current_track_id="track_a",
        identity_consistent=True,
        predicted_position_error_normalized=0.02,
        previous_depth_m=3.0,
        current_depth_m=3.1,
        previous_physical_scale_m=1.5,
        current_physical_scale_m=1.45,
        structure_change=0.03,
        motion_trend_change=0.04,
        confidence=0.8,
    )
    assert residual.valid and residual.combined_residual > 0.0


def test_no_event_is_not_applicable_and_never_zero_residual() -> None:
    no_event = classify_occlusion_event(
        _event_input(
            observed_object_available=True,
            candidate_object_available=True,
            visible_ratio=1.0,
        )
    )
    assert not no_event.valid
    assert no_event.status == "not_applicable"
    assert no_event.failure_reason == "not_applicable_no_event"
    residual = compute_reappearance_residual(
        event=no_event,
        previous_track_id="track_a",
        current_track_id="track_a",
        identity_consistent=True,
        predicted_position_error_normalized=0.0,
        previous_depth_m=3.0,
        current_depth_m=3.0,
        previous_physical_scale_m=1.5,
        current_physical_scale_m=1.5,
        structure_change=0.0,
        motion_trend_change=0.0,
        confidence=1.0,
    )
    assert not residual.valid
    assert math.isnan(residual.combined_residual)


def test_provider_failure_is_masked_not_anomaly() -> None:
    event = classify_occlusion_event(_event_input(provider_failed=True))
    assert not event.valid
    assert event.status == "provider_failed"
    assert event.failure_reason == "event_provider_failed"


def test_smoke_writes_required_artifacts_without_model_inference(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "outputs"
    config = {
        "schema_version": "test",
        "stage": "P4-C3B-M5",
        "inputs": {
            "metric_scene3d_root": "unused_scene",
            "pose_d2_root": "unused_pose",
        },
        "output_dir": str(output.relative_to(tmp_path)),
        "persisted_video_clips": [],
        "graph": {
            "maximum_boundary_nodes_per_object": 4,
            "maximum_internal_nodes_per_object": 4,
        },
        "events": {
            "partial_visible_ratio": 0.8,
            "full_overlap_ratio": 0.75,
            "out_of_frame_ratio": 0.2,
            "minimum_track_quality": 0.5,
            "true_disappearance_confirmation_frames": 2,
        },
        "constraints": {
            "authenticity_labels_used": False,
            "training_executed": False,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        smoke_module,
        "_load_persisted_graphs",
        lambda *_: ({}, {}, [], set()),
    )
    validation = smoke_module.run_d3_occlusion_smoke(
        tmp_path, config_path.relative_to(tmp_path)
    )
    required = {
        "d3_graph_manifest.csv",
        "d3_relation_residuals.csv",
        "occlusion_event_manifest.csv",
        "reappearance_event_manifest.csv",
        "event_classification_audit.json",
        "d3_eligibility_funnel.json",
        "synthetic_event_validation.json",
        "blocked_features.json",
        "validation_report.json",
        "D3_OCCLUSION_REPORT.md",
    }
    assert required <= {path.name for path in output.iterdir()}
    assert validation["d3_relation_residuals_synthetic_verified"]
    assert not validation["method_effectiveness_established"]
    with (output / "d3_relation_residuals.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    invalid = [row for row in rows if row["valid"] == "False"]
    assert invalid
    assert all(row["value"].lower() == "nan" for row in invalid)
