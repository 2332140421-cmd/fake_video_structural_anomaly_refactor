"""Label-independent temporal localization over frame anomaly evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TemporalInterval:
    start_frame: int
    end_frame: int
    score: float
    quality: float
    valid_frame_count: int
    missing_frame_count: int
    source_frame_indices: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def causal_moving_median(scores: Sequence[float], window_size: int = 3) -> np.ndarray:
    """Smooth causally while preserving all-missing positions as NaN."""

    if window_size < 1:
        raise ValueError("window_size must be positive.")
    values = np.asarray(scores, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    for index in range(len(values)):
        window = values[max(0, index - window_size + 1):index + 1]
        finite = window[np.isfinite(window)]
        if finite.size:
            output[index] = float(np.median(finite))
    return output


def localize_temporal_intervals(
    frame_indices: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    moving_median_window: int = 3,
    max_gap: int = 1,
    minimum_duration: int = 2,
    qualities: Sequence[float] | None = None,
) -> tuple[np.ndarray, tuple[TemporalInterval, ...]]:
    """Group high-evidence frames, merge small gaps, and filter duration."""

    if len(frame_indices) != len(scores):
        raise ValueError("frame_indices and scores must have equal length.")
    if qualities is not None and len(qualities) != len(scores):
        raise ValueError("qualities and scores must have equal length.")
    if max_gap < 0 or minimum_duration < 1 or not math.isfinite(float(threshold)):
        raise ValueError("Invalid temporal localization configuration.")
    frames = tuple(int(item) for item in frame_indices)
    smoothed = causal_moving_median(scores, moving_median_window)
    high = [position for position, value in enumerate(smoothed) if math.isfinite(float(value)) and value >= threshold]
    if not high:
        return smoothed, ()
    groups: list[list[int]] = [[high[0]]]
    for position in high[1:]:
        previous = groups[-1][-1]
        frame_gap = frames[position] - frames[previous] - 1
        if frame_gap <= max_gap:
            groups[-1].append(position)
        else:
            groups.append([position])
    quality_values = np.ones(len(scores), dtype=float) if qualities is None else np.asarray(qualities, dtype=float)
    intervals = []
    for group in groups:
        start, end = frames[group[0]], frames[group[-1]]
        if end - start + 1 < minimum_duration:
            continue
        group_scores = np.asarray([smoothed[index] for index in group], dtype=float)
        group_qualities = np.asarray([quality_values[index] for index in group], dtype=float)
        missing = max(0, end - start + 1 - len(group))
        intervals.append(TemporalInterval(
            start_frame=start, end_frame=end,
            score=float(np.mean(group_scores)),
            quality=float(np.mean(group_qualities)),
            valid_frame_count=len(group), missing_frame_count=missing,
            source_frame_indices=tuple(frames[index] for index in group),
            metadata={
                "threshold": float(threshold),
                "moving_median_window": moving_median_window,
                "max_gap": max_gap,
                "minimum_duration": minimum_duration,
                "threshold_source": "explicit_configuration_or_synthetic_validation",
            },
        ))
    return smoothed, tuple(intervals)
