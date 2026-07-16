from __future__ import annotations

import math

import numpy as np
import pytest

from semantic3d.sequence_geometry import (
    DepthAlignmentMode,
    SequenceScaleStatus,
    SyntheticSequenceGeometryProvider,
    apply_depth_alignment,
    estimate_depth_alignment,
)

from synthetic_sequence_geometry import depth_drift_pair, make_world_consistent_sequence


def test_scale_only_recovers_known_scale() -> None:
    source, target = depth_drift_pair(scale=2.5)
    result = estimate_depth_alignment(
        source,
        target,
        source_frame=0,
        target_frame=1,
        mode="scale_only",
    )
    assert result.valid
    assert result.scale == pytest.approx(2.5, abs=1e-10)
    assert result.shift == 0.0
    assert result.fitting_error < 1e-10


def test_affine_inverse_depth_recovers_scale_and_shift() -> None:
    source, target = depth_drift_pair(
        scale=1.7, shift=0.03, inverse_domain=True
    )
    result = estimate_depth_alignment(
        source,
        target,
        source_frame=0,
        target_frame=1,
        mode=DepthAlignmentMode.AFFINE_INVERSE_DEPTH,
    )
    assert result.valid
    assert result.scale == pytest.approx(1.7, rel=1e-8)
    assert result.shift == pytest.approx(0.03, abs=1e-10)
    np.testing.assert_allclose(apply_depth_alignment(source, result), target, rtol=1e-8)


def test_affine_depth_recovers_scale_shift_with_outliers() -> None:
    source, target = depth_drift_pair(scale=1.3, shift=0.4)
    target = target.copy()
    target[::17] += 20.0
    result = estimate_depth_alignment(
        source,
        target,
        source_frame=0,
        target_frame=1,
        mode=DepthAlignmentMode.AFFINE_DEPTH,
    )
    assert result.valid
    assert result.scale == pytest.approx(1.3, rel=1e-6)
    assert result.shift == pytest.approx(0.4, abs=1e-6)
    assert result.inlier_ratio < 1.0


def test_alignment_failure_uses_nan_not_identity() -> None:
    result = estimate_depth_alignment(
        [1.0, 2.0],
        [2.0, 4.0],
        source_frame=0,
        target_frame=1,
        mode="scale_only",
        minimum_support=12,
    )
    assert not result.valid
    assert math.isnan(result.scale)
    assert math.isnan(result.shift)
    assert math.isnan(result.fitting_error)


def test_valid_alignment_promotes_relative_per_frame_to_aligned_sequence() -> None:
    frames, poses = make_world_consistent_sequence(
        [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], metric=False
    )
    source, target = depth_drift_pair(scale=1.4)
    alignment = estimate_depth_alignment(
        source,
        target,
        source_frame=0,
        target_frame=1,
        mode="scale_only",
    )
    clip = SyntheticSequenceGeometryProvider(
        poses,
        depth_alignments=(alignment,),
        pose_scale_compatible_with_depth=True,
    ).predict_clip(frames, [0, 1])
    assert clip.sequence_scale_status == SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE
    assert clip.scale_allows_dynamic_3d
    assert clip.allows_dynamic_3d
