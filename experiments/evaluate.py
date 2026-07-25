"""Compact engineering metrics; no paper claims without formal data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .train import binary_metrics
from utils.io import write_json


def evaluate_video_scores(
    labels: Sequence[int],
    logits: Sequence[float],
    *,
    output_path: str | Path | None = None,
) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {
        **binary_metrics(labels, logits),
        "scope": "engineering_smoke_not_paper_performance",
    }
    if output_path is not None:
        write_json(output_path, metrics)
    return metrics


def interval_iou(
    predicted: Sequence[tuple[int, int]],
    target: Sequence[tuple[int, int]],
) -> float:
    predicted_frames = {frame for start, end in predicted for frame in range(start, end + 1)}
    target_frames = {frame for start, end in target for frame in range(start, end + 1)}
    union = predicted_frames | target_frames
    return len(predicted_frames & target_frames) / len(union) if union else 1.0


__all__ = ["evaluate_video_scores", "interval_iou"]
