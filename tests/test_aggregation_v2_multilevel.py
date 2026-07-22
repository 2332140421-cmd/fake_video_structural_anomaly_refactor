from __future__ import annotations

import math

import pytest

from semantic3d.aggregation_v2 import (
    AggregationEvidence,
    EvidenceApplicability,
    EvidenceBranchSpec,
    aggregate_evidence_v2,
    aggregate_multilevel_evidence,
    get_evidence_registry,
    localize_temporal_intervals,
    register_evidence_branch,
)


def observed(value: float, source: str, *, quality: float = 1.0, frame: int = 0, obj: str = "o", identity: str = "p", branch: str = "direction_consistency") -> AggregationEvidence:
    return AggregationEvidence.observed(
        value, quality=quality, branch_name=branch, source_id=source,
        frame_index=frame, object_track_id=obj, point_or_edge_id=identity,
        metadata={"video_id": "v"},
    )


def unavailable(status: EvidenceApplicability, source: str) -> AggregationEvidence:
    return AggregationEvidence.unavailable(
        applicability=status, reason=status.value, branch_name="boundary_occlusion",
        source_id=source, frame_index=0, object_track_id="o", point_or_edge_id="p",
        metadata={"video_id": "v"},
    )


def test_applicability_states_affect_coverage_without_zero_fill() -> None:
    normal = observed(0.0, "normal")
    not_applicable = unavailable(EvidenceApplicability.NOT_APPLICABLE, "na")
    missing = unavailable(EvidenceApplicability.OBSERVATION_MISSING, "missing")
    with_na = aggregate_evidence_v2([normal, not_applicable], level="point")
    with_missing = aggregate_evidence_v2([normal, missing], level="point")
    all_missing = aggregate_evidence_v2([missing], level="point")
    assert with_na.value == 0.0 and with_na.coverage == 1.0
    assert with_na.not_applicable_count == 1
    assert with_missing.coverage == pytest.approx(0.5)
    assert with_missing.observation_missing_count == 1
    assert not all_missing.valid and math.isnan(all_missing.value)
    assert all_missing.missing_reason == "observation_missing"


def test_invalid_geometry_and_unsupported_mode_are_counted_separately() -> None:
    result = aggregate_evidence_v2([
        unavailable(EvidenceApplicability.INVALID_GEOMETRY, "geometry"),
        unavailable(EvidenceApplicability.UNSUPPORTED_MODE, "mode"),
    ], level="clip")
    assert result.invalid_geometry_count == 1
    assert result.unsupported_mode_count == 1
    assert math.isnan(result.value)


def test_quality_gating_suppresses_low_quality_extreme_but_topk_keeps_local_signal() -> None:
    gated = aggregate_evidence_v2(
        [observed(1.0, "good"), observed(100.0, "bad", quality=0.01)],
        level="object", method="quality_weighted_top_k", top_k=1,
        quality_floor=0.1,
    )
    assert gated.value == pytest.approx(1.0)
    local = aggregate_evidence_v2(
        [observed(0.0, f"n{i}") for i in range(20)] + [observed(8.0, "local_high")],
        level="object", method="quality_weighted_top_k", top_k=3,
    )
    assert local.value > 2.0
    assert local.top_contributors[0]["source_id"] == "local_high"


def test_point_edge_object_frame_clip_traceability_and_object_localization() -> None:
    point_rows = [observed(5.0 if frame == 2 else 0.1, f"point:{frame}", frame=frame) for frame in range(4)]
    edge_rows = [observed(4.0, "edge:2", frame=2, identity="p0:p1", branch="structure_temporal")]
    result = aggregate_multilevel_evidence(
        point_evidences=point_rows, edge_evidences=edge_rows,
        video_id="v", clip_id="c", temporal_threshold=1.0,
        moving_median_window=1, minimum_duration=1,
        object_metadata={("v", 2, "o"): {
            "semantic_label": "cup", "localization_bbox": (1, 2, 8, 9),
            "localization_mask_reference": "mask.npy",
        }},
    )
    assert len(result.point_aggregates) == 4
    assert len(result.edge_aggregates) == 1
    target = next(item for item in result.object_aggregates if item.frame_index == 2)
    assert target.semantic_label == "cup"
    assert target.localization_mask_reference == "mask.npy"
    assert "p" in target.top_anomalous_point_ids
    assert "p0:p1" in target.top_anomalous_edge_ids
    assert max(result.frame_aggregates, key=lambda item: item.value).frame_index == 2
    assert result.clip_aggregate.valid
    assert result.clip_aggregate.metadata["classification_output"] is False


def test_two_objects_localize_only_anomalous_object() -> None:
    rows = [
        observed(0.1, "a", obj="a", identity="pa"),
        observed(7.0, "b", obj="b", identity="pb"),
    ]
    result = aggregate_multilevel_evidence(
        point_evidences=rows, edge_evidences=(), video_id="v", clip_id="c",
    )
    assert result.frame_aggregates[0].top_object_ids[0] == "b"


def test_temporal_localization_merges_one_missing_frame() -> None:
    smoothed, intervals = localize_temporal_intervals(
        range(6), [0.0, 2.0, 2.0, float("nan"), 2.0, 0.0],
        threshold=1.0, moving_median_window=1, max_gap=1, minimum_duration=3,
    )
    assert math.isnan(smoothed[3])
    assert len(intervals) == 1
    assert (intervals[0].start_frame, intervals[0].end_frame) == (1, 4)
    assert intervals[0].missing_frame_count == 1


def test_registry_does_not_auto_register_diagnostics() -> None:
    registry = get_evidence_registry(formal_only=True)
    assert "semantic_size_3d" in registry and "dynamic_reprojection" in registry
    diagnostic = EvidenceBranchSpec(
        "debug_only", ("static_camera_3d",), "point", False, 0.0,
        "none", "point", "diagnostic",
    )
    with pytest.raises(ValueError, match="Diagnostic"):
        register_evidence_branch(registry, diagnostic)
