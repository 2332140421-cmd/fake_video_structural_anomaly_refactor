import math

import numpy as np

from data.schemas import ClipObservation, FrameObservation, ResidualEvidence
from models.fusion import fuse_clip_residuals, fuse_video_results


def _clip(clip_id, start):
    frames = [
        FrameObservation(
            "video", clip_id, start + offset, float(start + offset),
            np.zeros((8, 8, 3), dtype=np.uint8),
        )
        for offset in range(2)
    ]
    return ClipObservation("video", clip_id, frames)


def test_missing_aware_fusion_coverage_confidence_and_contribution():
    observed = ResidualEvidence.observed("motion", "track", 1.0, confidence=0.8)
    missing = ResidualEvidence.unavailable("relation", "object_pair", "not_observed")
    result = fuse_clip_residuals(_clip("clip_0", 0), [observed, missing])
    assert result.risk_score > 0
    assert result.coverage == 0.5
    assert result.confidence == 0.4
    assert result.dominant_residual == "motion"
    assert result.risk_score != 0.0
    assert sum(result.contributions.values()) == result.risk_score


def test_all_missing_does_not_become_pseudo_normal():
    result = fuse_clip_residuals(
        _clip("clip_0", 0),
        [ResidualEvidence.unavailable("motion", "track", "blocked")],
    )
    assert math.isnan(result.risk_score)
    assert result.coverage == 0.0
    assert result.confidence == 0.0


def test_full_timeline_and_adjacent_suspicious_clip_merge():
    first = fuse_clip_residuals(
        _clip("clip_0", 0),
        [ResidualEvidence.observed("motion", "track", 2.0)],
    )
    second = fuse_clip_residuals(
        _clip("clip_1", 2),
        [ResidualEvidence.observed("relation", "object_pair", 2.0)],
    )
    video = fuse_video_results(
        video_id="video",
        video_path="<memory>",
        clips=[first, second],
        suspicious_threshold=0.5,
    )
    assert len(video.timeline) == 2
    assert len(video.suspicious_clips) == 1
    assert video.suspicious_clips[0]["clip_ids"] == ["clip_0", "clip_1"]
