from __future__ import annotations

import math

import pytest

from semantic3d.aggregation_v2 import aggregate_evidence_v2
from semantic3d.validity import ResidualEvidence


def _observed(value: float, quality: float, source: str) -> ResidualEvidence:
    return ResidualEvidence.observed(
        "r_structure", value, quality=quality, source_ids=(source,),
        metadata={"branch_name": "structure_temporal"},
    )


def test_missing_branch_stays_nan_and_states_are_distinct() -> None:
    missing = aggregate_evidence_v2(
        [ResidualEvidence.missing("r_occ", "observation_missing")], level="clip",
        identity={"video_id": "v", "clip_id": "c"},
    )
    not_applicable = aggregate_evidence_v2(
        [ResidualEvidence.missing("r_occ", "not_applicable")], level="clip",
        identity={"video_id": "v", "clip_id": "c"},
    )
    assert not missing.valid and math.isnan(missing.value)
    assert missing.missing_reason == "observation_missing"
    assert not_applicable.missing_reason == "not_applicable"


def test_quality_weight_topk_median_and_trimmed_mean() -> None:
    items = [_observed(0.0, 1.0, "p0"), _observed(1.0, 1.0, "p1"), _observed(10.0, 0.1, "p2")]
    top = aggregate_evidence_v2(items, level="object", method="topk_mean", top_k=1, identity={"object_track_id": "o"})
    median = aggregate_evidence_v2(items, level="object", method="median", identity={"object_track_id": "o"})
    trimmed = aggregate_evidence_v2(items, level="object", method="trimmed_mean", trim_fraction=0.34, identity={"object_track_id": "o"})
    assert top.value == pytest.approx(10.0)
    assert median.value == pytest.approx(1.0)
    assert trimmed.value == pytest.approx(1.0)
    assert set(top.contributing_source_ids) == {"p0", "p1", "p2"}


def test_low_quality_evidence_has_lower_weight_and_high_point_is_traceable() -> None:
    result = aggregate_evidence_v2(
        [_observed(1.0, 1.0, "normal"), _observed(9.0, 0.1, "high_point")],
        level="point", method="topk_mean", top_k=2,
        identity={"point_id": "p", "object_track_id": "o"},
    )
    assert 1.0 < result.value < 5.0
    assert "high_point" in result.contributing_source_ids
    assert result.metadata["classification_output"] is False
