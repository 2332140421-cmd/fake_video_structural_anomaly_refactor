from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import math
from pathlib import Path

import pytest

from semantic3d.dimension_aligned_scale_depth import (
    DimensionAlignedPriorEntry,
    DimensionAlignedPriorResolver,
    compute_dimension_aligned_rsd,
    load_dimension_aligned_prior_resolver,
)
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.projected_measurement import load_projected_measurement_rules


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RULES = load_projected_measurement_rules(
    PROJECT_ROOT / "configs/projected_measurement_rules.yaml"
)
RESOLVER = load_dimension_aligned_prior_resolver(
    PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
)


def _object(
    object_id: str,
    label: str,
    bbox: list[float],
    depth: float,
    confidence: float = 0.9,
) -> ObjectObservationJSON:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        mask_area=width * height,
        frame_area=1_000_000.0,
        depth=depth,
        confidence=confidence,
        bbox=bbox,
    )


def _frame(objects: list[ObjectObservationJSON]) -> FrameObservationJSON:
    return FrameObservationJSON(
        frame_index=0,
        frame_id="frame_0",
        width=1000,
        height=1000,
        objects=objects,
    )


def test_person_and_upright_cup_are_conditional_physical() -> None:
    person = _object("p", "person", [100, 100, 300, 600], depth=2.0)
    cup = _object("c", "cup", [700, 400, 740, 450], depth=1.0)
    frame = _frame([person, cup])

    row = compute_dimension_aligned_rsd(frame, person, cup, RESOLVER, RULES)

    assert row["valid"] is True
    assert row["evidence_tier"] == "conditional_physical"
    assert row["compatibility_group_a"] == row["compatibility_group_b"] == "vertical_extent"
    assert math.isfinite(float(row["rsd_log"]))
    assert "通用观测门控" in str(row["explanation_text"])


def test_failed_cup_gate_returns_nan() -> None:
    person = _object("p", "person", [100, 100, 300, 600], depth=2.0)
    cup = _object("c", "cup", [700, 400, 740, 450], depth=1.0, confidence=0.1)
    row = compute_dimension_aligned_rsd(_frame([person, cup]), person, cup, RESOLVER, RULES)

    assert row["valid"] is False
    assert row["skip_reason"] == "observation_gate_failed_b"
    assert math.isnan(float(row["rsd_log"]))


def _available_entry(label: str, measurement: str, group: str) -> DimensionAlignedPriorEntry:
    return DimensionAlignedPriorEntry(
        label=label,
        characteristic_dimension="test_dimension",
        dimension_definition="test-only aligned dimension",
        unit="m",
        min_size=1.0,
        max_size=2.0,
        projected_measurement=measurement,
        compatibility_group=group,
        reliability_status="strict_high",
        sources=({"source_id": "fixture"},),
        source_count=1,
        pose_sensitivity="low",
        observation_gate={
            "confidence_threshold": 0.3,
            "border_margin_px": 2,
            "min_bbox_dimension_px": 8,
            "min_area_ratio": 0.0001,
            "max_area_ratio": 0.9,
            "aspect_ratio_range": [0.1, 10.0],
        },
        reliability_reason="test fixture",
        prior_version="test_v2",
    )


def test_different_compatibility_groups_return_nan() -> None:
    resolver = DimensionAlignedPriorResolver(
        {
            "vertical": _available_entry("vertical", "bbox_height_norm", "vertical_extent"),
            "horizontal": _available_entry("horizontal", "bbox_width_norm", "horizontal_extent"),
        },
        metadata={"prior_version": "test_v2"},
    )
    a = _object("a", "vertical", [100, 100, 200, 500], 2.0)
    b = _object("b", "horizontal", [400, 300, 700, 500], 2.0)
    row = compute_dimension_aligned_rsd(_frame([a, b]), a, b, resolver, RULES)

    assert row["valid"] is False
    assert row["skip_reason"] == "incompatible_projected_dimensions"
    assert math.isnan(float(row["rsd_log"]))
    assert "维度不兼容" in str(row["explanation_text"])


def test_pose_sensitive_book_is_skipped() -> None:
    book = _object("b", "book", [100, 100, 300, 250], 2.0)
    cup = _object("c", "cup", [600, 300, 640, 350], 1.0)
    row = compute_dimension_aligned_rsd(_frame([book, cup]), book, cup, RESOLVER, RULES)

    assert row["valid"] is False
    assert row["skip_reason"] == "pose_sensitive_prior_a"
    assert math.isnan(float(row["rsd_log"]))
    assert "姿态" in str(row["explanation_text"])


def test_non_inverted_depth_mode_is_rejected() -> None:
    person = _object("p", "person", [100, 100, 300, 600], 2.0)
    cup = _object("c", "cup", [700, 400, 740, 450], 1.0)
    row = compute_dimension_aligned_rsd(
        _frame([person, cup]), person, cup, RESOLVER, RULES, depth_mode="real_depth_no_invert"
    )
    assert row["skip_reason"] == "invalid_depth_mode"
    assert math.isnan(float(row["rsd_log"]))

