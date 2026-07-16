from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

from pathlib import Path

import pytest

from semantic3d.observations import ObjectObservationJSON
from semantic3d.physical_prior_gate import evaluate_physical_prior_gate
from semantic3d.projected_measurement import (
    compute_projected_measurement,
    load_projected_measurement_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES = load_projected_measurement_rules(
    PROJECT_ROOT / "configs/projected_measurement_rules.yaml"
)
GATE = {
    "confidence_threshold": 0.3,
    "border_margin_px": 2.0,
    "border_margin_ratio": 0.005,
    "min_bbox_dimension_px": 8.0,
    "min_area_ratio": 0.0001,
    "max_area_ratio": 0.5,
    "aspect_ratio_range": [0.45, 1.6],
    "require_track_stability": False,
}


def _cup(**overrides: object) -> ObjectObservationJSON:
    data = {
        "object_id": "cup",
        "label": "cup",
        "mask_area": 1600.0,
        "frame_area": 1_000_000.0,
        "depth": 2.0,
        "confidence": 0.9,
        "bbox": [200.0, 200.0, 240.0, 250.0],
    }
    data.update(overrides)
    return ObjectObservationJSON(**data)  # type: ignore[arg-type]


def _evaluate(obj: ObjectObservationJSON):
    measurement = compute_projected_measurement(
        obj, 1000, 1000, "bbox_height_norm", RULES
    )
    return evaluate_physical_prior_gate(obj, 1000, 1000, measurement, GATE, RULES)


def test_complete_confident_cup_passes_gate() -> None:
    result = _evaluate(_cup())
    assert result.gate_passed
    assert result.gate_score == pytest.approx(1.0)
    assert not result.failed_gate_reasons


def test_low_confidence_fails_gate_score_is_heuristic() -> None:
    result = _evaluate(_cup(confidence=0.1))
    assert not result.gate_passed
    assert "detection_confidence" in result.failed_gate_reasons
    assert 0.0 <= result.gate_score < 1.0


def test_boundary_contact_and_bad_aspect_fail_gate() -> None:
    result = _evaluate(_cup(bbox=[0.0, 200.0, 200.0, 220.0], mask_area=4000.0))
    assert not result.gate_passed
    assert "not_truncated_at_frame_boundary" in result.failed_gate_reasons
    assert "aspect_ratio_range" in result.failed_gate_reasons


def test_invalid_depth_fails_gate() -> None:
    result = _evaluate(_cup(depth=0.0))
    assert not result.gate_passed
    assert "valid_depth" in result.failed_gate_reasons

