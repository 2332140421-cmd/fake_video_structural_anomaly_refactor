from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from semantic3d.sequence_geometry import (
    DepthAlignmentMode,
    DepthAlignmentModelSelection,
    DepthAlignmentObservation,
    align_depth_sequence_to_reference,
    apply_sequence_depth_alignment,
    select_depth_alignment_model,
)


def _valid_edge(source: int, target: int, scale: float, shift: float, quality: float = 0.95):
    return DepthAlignmentObservation(
        source_frame=source,
        target_frame=target,
        alignment_mode=DepthAlignmentMode.AFFINE_DEPTH,
        scale=scale,
        shift=shift,
        support_count=100,
        inlier_ratio=0.95,
        fitting_error=0.01,
        quality=quality,
        valid=True,
        holdout_error=0.02,
        holdout_count=20,
        physical_valid=True,
        metadata={"normalized_holdout_error": 0.01},
    )


def _selection(edge: DepthAlignmentObservation) -> DepthAlignmentModelSelection:
    return DepthAlignmentModelSelection(
        edge.source_frame,
        edge.target_frame,
        (edge,),
        edge,
        80,
        20,
        False,
        True,
    )


def test_inverse_depth_candidate_uses_raw_model_output_and_holdout() -> None:
    raw_source = np.linspace(0.2, 1.2, 120)
    raw_target = 1.4 * raw_source + 0.08
    geometry_source = 1.0 / raw_source
    geometry_target = 1.0 / raw_target
    result = select_depth_alignment_model(
        geometry_source,
        geometry_target,
        source_frame=0,
        target_frame=1,
        source_raw_values=raw_source,
        target_raw_values=raw_target,
    )
    assert result.valid
    assert result.selected.alignment_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
    assert result.selected.metadata["raw_model_output_used"] is True
    assert result.selected.metadata["visualization_depth_used"] is False
    assert result.fitting_count > 0 and result.holdout_count > 0
    assert result.selected.scale == pytest.approx(1.4, rel=1e-8)
    assert result.selected.shift == pytest.approx(0.08, abs=1e-10)


def test_wrong_simple_model_is_rejected_by_holdout() -> None:
    source = np.linspace(1.0, 8.0, 100)
    target = 1.25 * source + 2.0
    result = select_depth_alignment_model(
        source,
        target,
        source_frame=0,
        target_frame=1,
    )
    by_mode = {candidate.alignment_mode: candidate for candidate in result.candidates}
    assert not by_mode[DepthAlignmentMode.SCALE_ONLY].valid
    assert by_mode[DepthAlignmentMode.SCALE_ONLY].missing_reason == "depth_alignment_holdout_rejected"
    assert result.selected.alignment_mode == DepthAlignmentMode.AFFINE_DEPTH


def test_global_graph_alignment_connects_through_skip_edge_and_stabilizes_depth() -> None:
    edge_01 = _valid_edge(0, 1, 2.0, 1.0)
    edge_02 = _valid_edge(0, 2, 0.5, 0.2)
    result = align_depth_sequence_to_reference(
        [0, 1, 2],
        [_selection(edge_01), _selection(edge_02)],
        reference_frame=0,
    )
    assert result.valid
    assert result.connected_frame_ratio == 1.0
    assert result.per_frame[1].scale == pytest.approx(0.5)
    assert result.per_frame[1].shift == pytest.approx(-0.5)
    assert result.per_frame[2].scale == pytest.approx(2.0)
    assert result.per_frame[2].shift == pytest.approx(-0.4)
    truth = np.linspace(2.0, 10.0, 20)
    frame_1 = 2.0 * truth + 1.0
    frame_2 = 0.5 * truth + 0.2
    aligned_1 = apply_sequence_depth_alignment(frame_1, result.per_frame[1])
    aligned_2 = apply_sequence_depth_alignment(frame_2, result.per_frame[2])
    np.testing.assert_allclose(aligned_1, truth, atol=1e-10)
    np.testing.assert_allclose(aligned_2, truth, atol=1e-10)
    before = np.mean(np.abs(frame_1 - truth)) + np.mean(np.abs(frame_2 - truth))
    after = np.mean(np.abs(aligned_1 - truth)) + np.mean(np.abs(aligned_2 - truth))
    assert after < before * 1e-9


def test_missing_depth_edge_keeps_disconnected_frame_nan() -> None:
    edge = _valid_edge(0, 1, 1.0, 0.0)
    result = align_depth_sequence_to_reference([0, 1, 2], [_selection(edge)])
    assert not result.valid
    assert not result.per_frame[2].valid
    assert np.isnan(result.per_frame[2].scale)
    assert result.per_frame[2].missing_reason == "depth_alignment_graph_disconnected"


def test_global_model_selection_prioritizes_reference_connectivity() -> None:
    connected_depth = _valid_edge(0, 1, 1.1, 0.1, quality=0.7)
    connected_depth = replace(
        connected_depth,
        alignment_mode=DepthAlignmentMode.AFFINE_DEPTH,
    )
    disconnected_inverse = replace(
        _valid_edge(1, 2, 1.0, 0.0, quality=0.99),
        alignment_mode=DepthAlignmentMode.AFFINE_INVERSE_DEPTH,
    )
    selections = [
        DepthAlignmentModelSelection(
            0, 1, (connected_depth,), connected_depth, 80, 20, False, True
        ),
        DepthAlignmentModelSelection(
            1,
            2,
            (disconnected_inverse,),
            disconnected_inverse,
            80,
            20,
            True,
            True,
        ),
    ]
    result = align_depth_sequence_to_reference([0, 1, 2], selections)
    assert result.alignment_mode == DepthAlignmentMode.AFFINE_DEPTH
    assert result.connected_frame_ratio == 2 / 3
    assert result.metadata["global_model_selection_rule"].startswith("reference_connectivity")


def test_redundant_global_refinement_is_better_than_noisy_chain_accumulation() -> None:
    edge_01 = _valid_edge(0, 1, 2.0, 0.0, quality=0.99)
    noisy_12 = _valid_edge(1, 2, 0.35, 0.0, quality=0.30)
    direct_02 = _valid_edge(0, 2, 0.5, 0.0, quality=0.99)
    result = align_depth_sequence_to_reference(
        [0, 1, 2],
        [_selection(edge_01), _selection(noisy_12), _selection(direct_02)],
    )
    true_scale_2 = 2.0
    noisy_chain_scale_2 = (1.0 / 2.0) / 0.35
    global_error = abs(result.per_frame[2].scale - true_scale_2)
    chain_error = abs(noisy_chain_scale_2 - true_scale_2)
    assert global_error < chain_error
    assert result.global_consistency_error < abs(np.log(0.35 / 0.25))
