"""Paper-core timeline, trajectory, and residual-support visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from data.schemas import ClipObservation, VideoResult


def save_timeline(result: VideoResult, path: str | Path) -> Path:
    target = Path(path)
    scores = [row["risk_score"] for row in result.timeline]
    times = [row["start_time"] for row in result.timeline]
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.plot(times, scores, marker="o")
    axis.set(xlabel="time (s)", ylabel="risk", title="Full-video structural anomaly timeline", ylim=(0, 1))
    figure.tight_layout()
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target


def save_track_deviations(
    observations: Sequence[ClipObservation], path: str | Path
) -> Path:
    target = Path(path)
    figure, axis = plt.subplots(figsize=(6, 5))
    rendered = False
    for clip in observations:
        for track in clip.tracks:
            actual = track.actual_xy
            axis.plot(actual[:, 0], actual[:, 1], "o-", label=f"{track.track_id} actual")
            if track.predicted_xy is not None:
                predicted = track.predicted_xy
                axis.plot(predicted[:, 0], predicted[:, 1], "x--", label=f"{track.track_id} predicted")
                for actual_point, predicted_point in zip(actual, predicted):
                    axis.plot(
                        [actual_point[0], predicted_point[0]],
                        [actual_point[1], predicted_point[1]],
                        color="crimson",
                        alpha=0.5,
                    )
            rendered = True
    if not rendered:
        axis.text(0.5, 0.5, "No valid point tracks", ha="center", va="center")
    axis.invert_yaxis()
    axis.set(xlabel="x (px)", ylabel="y (px)", title="Actual, predicted, and deviating tracks")
    if rendered:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target


def save_structural_heatmap(
    result: VideoResult,
    observations: Sequence[ClipObservation],
    path: str | Path,
    *,
    sigma: float = 5.0,
) -> Path:
    target = Path(path)
    shape = observations[0].frames[0].image.shape[:2]
    accumulated = np.zeros(shape, dtype=np.float32)
    support_count = np.zeros(shape, dtype=np.float32)
    for clip_result in result.clip_results:
        for support in clip_result.spatial_heatmaps.values():
            valid = np.isfinite(support)
            accumulated[valid] += support[valid]
            support_count[valid] += 1.0
    base = np.divide(
        accumulated,
        support_count,
        out=np.zeros_like(accumulated),
        where=support_count > 0,
    )
    if np.any(support_count):
        kernel = max(3, int(round(6 * sigma)) | 1)
        base = cv2.GaussianBlur(base, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
        maximum = float(base.max())
        if maximum > 0:
            base /= maximum
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.imshow(cv2.cvtColor(observations[0].frames[0].image, cv2.COLOR_BGR2RGB))
    axis.imshow(base, cmap="inferno", alpha=np.where(base > 0, 0.65, 0.0), vmin=0, vmax=1)
    axis.set_title("3D structural residual support")
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(target, dpi=140)
    plt.close(figure)
    return target


__all__ = ["save_structural_heatmap", "save_timeline", "save_track_deviations"]
