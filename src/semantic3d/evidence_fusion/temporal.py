"""Threshold-free temporal evidence sequences with optional interval interfaces."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ..aggregation_v2.temporal_localization import (
    TemporalInterval,
    causal_moving_median,
    localize_temporal_intervals,
)
from .contracts import UnifiedEvidence
from .fusion import FusionResult, fuse_unified_evidence


@dataclass(frozen=True)
class FrameFusionEvidence:
    """One frame's fused evidence without an anomaly decision."""

    video_id: str
    clip_id: str
    frame_index: int
    frame_id: str
    risk_score: float
    evidence_confidence: float
    active_branch_count: int
    available_weight_ratio: float
    valid: bool
    failure_reason: str


@dataclass(frozen=True)
class TemporalEvidenceSequence:
    """Raw and smoothed frame evidence plus optional diagnostic intervals."""

    video_id: str
    clip_id: str
    frames: tuple[FrameFusionEvidence, ...]
    smoothed_risk: tuple[float, ...]
    intervals: tuple[Mapping[str, Any], ...]
    formal_threshold_selected: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)


def build_frame_fusions(
    evidences: Sequence[UnifiedEvidence],
    *,
    branch_weights: Mapping[str, float] | None = None,
) -> tuple[FrameFusionEvidence, ...]:
    """Aggregate evidence independently for each global frame."""

    grouped: dict[tuple[str, str, int, str], list[UnifiedEvidence]] = {}
    for item in evidences:
        if item.frame_index is None:
            continue
        key = (item.video_id, item.clip_id, item.frame_index, item.frame_id)
        grouped.setdefault(key, []).append(item)
    output = []
    for (video_id, clip_id, frame_index, frame_id), rows in sorted(grouped.items()):
        result = fuse_unified_evidence(rows, branch_weights=branch_weights)
        output.append(
            FrameFusionEvidence(
                video_id=video_id,
                clip_id=clip_id,
                frame_index=frame_index,
                frame_id=frame_id or f"frame_{frame_index:06d}",
                risk_score=result.risk_score,
                evidence_confidence=result.evidence_confidence,
                active_branch_count=result.active_branch_count,
                available_weight_ratio=result.available_weight_ratio,
                valid=result.valid,
                failure_reason=result.failure_reason,
            )
        )
    return tuple(output)


def merge_temporal_intervals(
    intervals: Sequence[TemporalInterval],
    *,
    max_gap: int,
) -> tuple[TemporalInterval, ...]:
    """Merge already generated intervals without changing their threshold semantics."""

    if max_gap < 0:
        raise ValueError("max_gap must be non-negative.")
    ordered = sorted(intervals, key=lambda item: (item.start_frame, item.end_frame))
    if not ordered:
        return ()
    merged: list[TemporalInterval] = [ordered[0]]
    for item in ordered[1:]:
        previous = merged[-1]
        if item.start_frame - previous.end_frame - 1 > max_gap:
            merged.append(item)
            continue
        valid_count = previous.valid_frame_count + item.valid_frame_count
        weight = max(1, valid_count)
        score = (
            previous.score * previous.valid_frame_count
            + item.score * item.valid_frame_count
        ) / weight
        quality = (
            previous.quality * previous.valid_frame_count
            + item.quality * item.valid_frame_count
        ) / weight
        merged[-1] = TemporalInterval(
            start_frame=previous.start_frame,
            end_frame=max(previous.end_frame, item.end_frame),
            score=score,
            quality=quality,
            valid_frame_count=valid_count,
            missing_frame_count=(
                previous.missing_frame_count
                + item.missing_frame_count
                + max(0, item.start_frame - previous.end_frame - 1)
            ),
            source_frame_indices=tuple(
                dict.fromkeys(
                    (*previous.source_frame_indices, *item.source_frame_indices)
                )
            ),
            metadata={
                "merged_without_threshold_refit": True,
                "max_gap": max_gap,
            },
        )
    return tuple(merged)


def build_temporal_evidence_sequences(
    frame_evidence: Sequence[FrameFusionEvidence],
    *,
    smoothing_window: int = 3,
    diagnostic_threshold: float | None = None,
    max_gap: int = 1,
    minimum_duration: int = 2,
) -> tuple[TemporalEvidenceSequence, ...]:
    """Build sequences; no intervals are emitted without an explicit threshold."""

    grouped: dict[tuple[str, str], list[FrameFusionEvidence]] = {}
    for item in frame_evidence:
        grouped.setdefault((item.video_id, item.clip_id), []).append(item)
    output = []
    for (video_id, clip_id), rows in sorted(grouped.items()):
        ordered = tuple(sorted(rows, key=lambda item: item.frame_index))
        scores = [item.risk_score for item in ordered]
        qualities = [item.evidence_confidence for item in ordered]
        smoothed = causal_moving_median(scores, smoothing_window)
        intervals: tuple[TemporalInterval, ...] = ()
        if diagnostic_threshold is not None:
            smoothed, intervals = localize_temporal_intervals(
                [item.frame_index for item in ordered],
                scores,
                threshold=diagnostic_threshold,
                moving_median_window=smoothing_window,
                max_gap=max_gap,
                minimum_duration=minimum_duration,
                qualities=qualities,
            )
        output.append(
            TemporalEvidenceSequence(
                video_id=video_id,
                clip_id=clip_id,
                frames=ordered,
                smoothed_risk=tuple(float(value) for value in smoothed),
                intervals=tuple(asdict(item) for item in intervals),
                formal_threshold_selected=False,
                metadata={
                    "smoothing": "causal_moving_median",
                    "smoothing_window": smoothing_window,
                    "diagnostic_threshold": diagnostic_threshold,
                    "interval_count": len(intervals),
                    "missing_frame_scores_remain_nan": any(
                        not math.isfinite(value) for value in scores
                    ),
                    "classification_output": False,
                },
            )
        )
    return tuple(output)

