from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import numpy as np
import pytest

from semantic3d.residual_fusion import (
    ResidualValues,
    ResidualWeights,
    fuse_residuals,
    normalize_residuals,
    video_risk_score,
)


def test_weighted_fusion_result_is_correct() -> None:
    values = ResidualValues(
        flow=1.0,
        track=2.0,
        depth_cons=3.0,
        occ=4.0,
        corr=5.0,
        scale_depth=6.0,
    )
    weights = ResidualWeights(
        flow=0.1,
        track=0.2,
        depth_cons=0.3,
        occ=0.4,
        corr=0.5,
        scale_depth=0.6,
    )

    score = fuse_residuals(values, weights)

    assert score == pytest.approx(9.1)


def test_missing_residual_raises_clear_error() -> None:
    values = ResidualValues(
        flow=1.0,
        track=2.0,
        depth_cons=None,  # type: ignore[arg-type]
        occ=4.0,
        corr=5.0,
        scale_depth=6.0,
    )
    weights = ResidualWeights(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="Missing residual 'depth_cons'"):
        fuse_residuals(values, weights)


def test_depth_cons_and_scale_depth_are_independent_fields() -> None:
    values = ResidualValues(
        flow=0.0,
        track=0.0,
        depth_cons=10.0,
        occ=0.0,
        corr=0.0,
        scale_depth=1.0,
    )
    weights = ResidualWeights(
        flow=0.0,
        track=0.0,
        depth_cons=0.5,
        occ=0.0,
        corr=0.0,
        scale_depth=2.0,
    )

    score = fuse_residuals(values, weights)

    assert values.depth_cons != values.scale_depth
    assert score == pytest.approx(7.0)


def test_normalize_residuals_minmax_for_multiple_segments() -> None:
    values = [
        ResidualValues(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        ResidualValues(10.0, 1.0, 4.0, 9.0, 8.0, 15.0),
    ]

    normalized = normalize_residuals(values, method="minmax")

    assert isinstance(normalized, list)
    assert normalized[0].flow == pytest.approx(0.0)
    assert normalized[1].flow == pytest.approx(1.0)
    assert normalized[0].track == pytest.approx(0.0)
    assert normalized[1].track == pytest.approx(0.0)
    assert normalized[1].scale_depth == pytest.approx(1.0)


def test_multi_segment_fusion_and_video_risk_score() -> None:
    values = [
        ResidualValues(0.1, 0.2, 0.3, 0.1, 0.0, 0.4),
        ResidualValues(0.2, 0.3, 0.4, 0.2, 0.1, 1.0),
        ResidualValues(0.0, 0.1, 0.2, 0.0, 0.2, 0.2),
    ]
    weights = ResidualWeights(
        flow=1.0,
        track=1.0,
        depth_cons=1.0,
        occ=1.0,
        corr=1.0,
        scale_depth=2.0,
    )

    scores = fuse_residuals(values, weights)
    video_score, details = video_risk_score(
        scores, w_mean=0.5, w_max=0.25, w_topk=0.25, topk=2
    )

    expected_scores = np.asarray([1.5, 3.2, 0.9])
    expected_video = (
        0.5 * expected_scores.mean()
        + 0.25 * expected_scores.max()
        + 0.25 * np.asarray([1.5, 3.2]).mean()
    )

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)
    assert np.allclose(scores, expected_scores)
    assert details["topk_mean"] == pytest.approx(2.35)
    assert video_score == pytest.approx(expected_video)


def test_video_risk_score_rejects_empty_scores() -> None:
    with pytest.raises(ValueError, match="non-empty 1D"):
        video_risk_score([])
