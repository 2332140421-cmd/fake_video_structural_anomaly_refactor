from __future__ import annotations

import math

import pytest

from semantic3d.method_completion import (
    ProviderStatus,
    ScaleHistoryObservation,
    TemporalReferenceMethod,
    TemporalSameObjectScaleBranch,
    TemporalScaleMode,
)


def item(
    frame: int,
    size: float,
    *,
    video: str = "v",
    clip: str = "c",
    track: str = "t",
    mode: str = "metric",
    provider: str = "metric_provider",
    definition: str = "z_depth",
    **kwargs,
):
    return ScaleHistoryObservation(
        video, clip, f"f{frame}", frame, track, f"o{frame}", "height_m", size,
        "meter" if mode == "metric" else "relative_local_unit", mode,
        provider, definition, "K1", "aligned", **kwargs,
    )


def test_constant_scale_is_zero_and_jumps_are_monotonic():
    branch = TemporalSameObjectScaleBranch(
        reference_method=TemporalReferenceMethod.PREVIOUS_VALID
    )
    history = [item(0, 2.0)]
    assert branch.evaluate(item(1, 2.0), history).residual_value == pytest.approx(0.0)
    low = branch.evaluate(item(1, 2.5), history).residual_value
    high = branch.evaluate(item(1, 4.0), history).residual_value
    assert 0 < low < high


def test_rolling_median_outlier_does_not_permanently_pollute_reference():
    branch = TemporalSameObjectScaleBranch(
        reference_method=TemporalReferenceMethod.ROLLING_MEDIAN,
        min_valid_history=2,
        reference_window=5,
    )
    history = [item(0, 2.0), item(1, 2.0), item(2, 20.0), item(3, 2.0)]
    result = branch.evaluate(item(4, 2.0), history)
    assert result.valid
    assert result.residual_value == pytest.approx(0.0)
    assert result.provenance["reference_size"] == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"track": "other"}, "track_id_switch_or_mismatch"),
        ({"video": "other"}, "cross_video_history_forbidden"),
        ({"clip": "other"}, "cross_clip_history_forbidden"),
        ({"provider": "other"}, "depth_provider_changed"),
        ({"definition": "ray_distance"}, "depth_definition_changed"),
    ],
)
def test_history_domain_switches_are_blocked(changed, reason):
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    history = [item(0, 2.0, **changed)]
    result = branch.evaluate(item(1, 2.0), history)
    assert not result.valid and math.isnan(result.residual_value)
    assert result.failure_reason == reason


def test_occlusion_and_out_of_frame_are_missing_not_zero():
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    history = [item(0, 2.0)]
    for current in (
        item(1, 2.0, occlusion_status="fully_occluded"),
        item(1, 2.0, out_of_frame=True),
    ):
        result = branch.evaluate(current, history)
        assert not result.valid and math.isnan(result.residual_value)


def test_relative_local_mode_never_reports_meters_and_requires_alignment():
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    history = [item(0, 2.0, mode="relative_local")]
    result = branch.evaluate(item(1, 2.1, mode="relative_local"), history)
    assert result.valid
    assert result.depth_unit == "relative_local_unit"
    assert result.provenance["size_unit"] == "relative_local_unit"
    unaligned = ScaleHistoryObservation(
        "v", "c", "f2", 2, "t", "o2", "height_m", 2.1,
        "relative_local_unit", TemporalScaleMode.RELATIVE_LOCAL, "metric_provider",
        "z_depth", "K1", "unaligned",
    )
    blocked = branch.evaluate(unaligned, history)
    assert blocked.failure_reason == "relative_depth_scale_not_locally_aligned"


def test_provider_failure_never_becomes_temporal_anomaly():
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    current = item(1, 10.0, provider_status=ProviderStatus.PROVIDER_FAILED)
    result = branch.evaluate(current, [item(0, 2.0)])
    assert not result.valid and math.isnan(result.residual_value)
    assert result.provider_status == ProviderStatus.PROVIDER_FAILED
