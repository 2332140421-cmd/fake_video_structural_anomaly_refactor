from __future__ import annotations

import math

import pytest

from semantic3d.method_completion import (
    DimensionScalePrior,
    MultiIntervalScalePriorRegistry,
    ObjectPhysicalScalePrior,
    SizeInterval,
    log_distance_to_interval_union,
)


def test_disjoint_interval_union_and_alias_resolution():
    intervals = (SizeInterval(1.2, 2.2), SizeInterval(2.8, 4.5))
    prior = ObjectPhysicalScalePrior(
        "vehicle", {"height_m": DimensionScalePrior("height_m", intervals)}
    )
    registry = MultiIntervalScalePriorRegistry({"vehicle": prior}, {"bus": "vehicle"})
    assert registry.resolve("vehicle").resolution == "exact"
    assert registry.resolve("BUS").resolution == "alias"
    assert registry.resolve("unknown").prior is None
    assert log_distance_to_interval_union(1.5, intervals) == 0.0
    assert log_distance_to_interval_union(3.0, intervals) == 0.0


def test_gap_uses_nearest_interval_and_outside_is_monotonic():
    intervals = (SizeInterval(1.0, 2.0), SizeInterval(4.0, 5.0))
    expected = min(math.log(3.0) - math.log(2.0), math.log(4.0) - math.log(3.0))
    assert log_distance_to_interval_union(3.0, intervals) == pytest.approx(expected)
    assert log_distance_to_interval_union(6.0, intervals) < log_distance_to_interval_union(8.0, intervals)


def test_prior_rejects_invalid_units_and_intervals():
    with pytest.raises(ValueError):
        SizeInterval(2.0, 1.0)
    with pytest.raises(ValueError):
        DimensionScalePrior("height_m", (SizeInterval(1.0, 2.0),), unit="relative")
