from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import math
from pathlib import Path

import pytest

from semantic3d.observations import ObjectObservationJSON
from semantic3d.projected_measurement import (
    compute_projected_measurement,
    load_projected_measurement_rules,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES = load_projected_measurement_rules(
    PROJECT_ROOT / "configs/projected_measurement_rules.yaml"
)


def _object(**overrides: object) -> ObjectObservationJSON:
    data = {
        "object_id": "obj",
        "label": "cup",
        "mask_area": 2400.0,
        "frame_area": 20_000.0,
        "depth": 2.0,
        "bbox": [20.0, 10.0, 60.0, 70.0],
    }
    data.update(overrides)
    return ObjectObservationJSON(**data)  # type: ignore[arg-type]


def test_bbox_height_norm() -> None:
    result = compute_projected_measurement(_object(), 200, 100, "bbox_height_norm", RULES)
    assert result.valid
    assert result.value == pytest.approx(0.6)
    assert result.compatibility_group == "vertical_extent"


def test_bbox_width_norm() -> None:
    result = compute_projected_measurement(_object(), 200, 100, "bbox_width_norm", RULES)
    assert result.valid
    assert result.value == pytest.approx(0.2)
    assert result.compatibility_group == "horizontal_extent"


def test_equivalent_diameter_norm_uses_area() -> None:
    result = compute_projected_measurement(
        _object(), 200, 100, "equivalent_diameter_norm", RULES
    )
    expected = math.sqrt(4.0 * 2400.0 / math.pi) / math.sqrt(20_000.0)
    assert result.valid
    assert result.value == pytest.approx(expected)
    assert result.measurement_quality == "bbox_area_approximation"
    assert result.compatibility_group == "radial_extent"


@pytest.mark.parametrize(
    ("bbox", "reason"),
    [
        ([-1.0, 10.0, 60.0, 70.0], "bbox_out_of_bounds"),
        ([20.0, 10.0, 20.0, 70.0], "non_positive_bbox_extent"),
        ([20.0, 10.0, 24.0, 14.0], "bbox_too_small"),
    ],
)
def test_invalid_bbox_is_skipped_without_zero(bbox: list[float], reason: str) -> None:
    result = compute_projected_measurement(
        _object(bbox=bbox), 200, 100, "bbox_height_norm", RULES
    )
    assert not result.valid
    assert math.isnan(result.value)
    assert result.invalid_reason == reason

