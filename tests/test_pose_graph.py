from __future__ import annotations

import math

import numpy as np

from semantic3d.sequence_geometry import (
    PoseEstimateCandidate,
    PoseModelType,
    TranslationScaleStatus,
    build_pose_graph,
)


def _edge(source: int, target: int, tx: float = 0.0, quality: float = 1.0):
    transform = np.eye(4)
    transform[0, 3] = tx
    return PoseEstimateCandidate(
        source,
        target,
        PoseModelType.DEPTH_PNP if tx else PoseModelType.STATIC_IDENTITY,
        transform,
        True,
        True,
        TranslationScaleStatus.DEPTH_RELATIVE if tx else TranslationScaleStatus.ZERO_STATIC,
        50,
        45,
        0.9,
        2.0 if tx else 0.0,
        0.2 if tx else 0.0,
        quality,
        True,
        selected_reference_frame=source,
        evidence_source="synthetic_background",
    )


def test_skip_frame_edges_connect_after_adjacent_failure() -> None:
    missing = PoseEstimateCandidate.missing(0, 1, "adjacent_low_baseline")
    graph = build_pose_graph(
        [0, 1, 2],
        [missing, _edge(0, 2, 0.2), _edge(2, 1, -0.1)],
        reference_frame=0,
    )
    assert graph.valid
    assert graph.connected_frame_ratio == 1.0
    assert graph.T_world_from_camera_by_frame[1] is not None
    assert graph.selected_reference_frame[1] == 2
    assert graph.pose_chain_length == 2


def test_single_failed_edge_does_not_create_identity_or_break_other_component_edges() -> None:
    graph = build_pose_graph(
        [0, 1, 2, 3],
        [_edge(0, 1), PoseEstimateCandidate.missing(1, 2, "failed"), _edge(0, 2), _edge(2, 3)],
    )
    assert graph.connected_frame_ratio == 1.0
    assert graph.T_world_from_camera_by_frame[2] is not None
    assert graph.metadata["missing_edges_filled_with_identity"] is False


def test_disconnected_frame_remains_missing() -> None:
    graph = build_pose_graph([0, 1, 2], [_edge(0, 1)])
    assert not graph.valid
    assert graph.disconnected_frames == (2,)
    assert graph.T_world_from_camera_by_frame[2] is None
    assert graph.T_camera_from_world_by_frame[2] is None
    assert graph.connected_frame_ratio == 2 / 3
    assert not math.isnan(graph.pose_graph_quality)

