"""Single missing-aware hierarchy for clip and full-video fusion."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from data.schemas import ClipObservation, ClipResult, ResidualEvidence, VideoResult


def _weighted(rows: Sequence[ResidualEvidence]) -> tuple[float, float, float, dict[str, float]]:
    valid = [row for row in rows if row.valid_mask]
    coverage = len(valid) / max(len(rows), 1)
    if not valid:
        return float("nan"), coverage, 0.0, {}
    weights = np.asarray([max(row.confidence, 1e-8) for row in valid])
    values = np.asarray([row.normalized_value for row in valid])
    score = float(np.average(values, weights=weights))
    confidence = float(np.mean([row.confidence for row in valid]) * coverage)
    contribution = {
        f"{row.name}:{index}": float(value * weight / weights.sum())
        for index, (row, value, weight) in enumerate(zip(valid, values, weights))
    }
    return score, coverage, confidence, contribution


def _hierarchical(
    rows: Sequence[ResidualEvidence],
) -> tuple[float, float, float, dict[str, float]]:
    valid = [(index, row) for index, row in enumerate(rows) if row.valid_mask]
    coverage = len(valid) / max(len(rows), 1)
    if not valid:
        return float("nan"), coverage, 0.0, {}

    def entity_key(row: ResidualEvidence) -> tuple[object, ...]:
        spatial = row.spatial_support
        temporal = row.temporal_support
        frame = temporal.get("frame_index", spatial.get("frame_index", -1))
        identity = (
            spatial.get("object_id")
            or spatial.get("track_id")
            or temporal.get("track_id")
            or spatial.get("track_ids")
            or spatial.get("xy")
            or row.name
        )
        return int(frame), row.level, str(identity)

    entities: dict[tuple[object, ...], list[tuple[int, ResidualEvidence]]] = defaultdict(list)
    for item in valid:
        entities[entity_key(item[1])].append(item)
    entity_rows: list[tuple[int, float, float, dict[int, float]]] = []
    for key, members in entities.items():
        weights = np.asarray([max(row.confidence, 1e-8) for _, row in members])
        normalized_weights = weights / weights.sum()
        score = float(
            sum(row.normalized_value * weight for (_, row), weight in zip(members, normalized_weights))
        )
        entity_rows.append(
            (
                int(key[0]),
                score,
                float(np.mean([row.confidence for _, row in members])),
                {index: float(weight) for (index, _), weight in zip(members, normalized_weights)},
            )
        )
    frames: dict[int, list[tuple[float, float, dict[int, float]]]] = defaultdict(list)
    for frame, score, confidence, member_weights in entity_rows:
        frames[frame].append((score, confidence, member_weights))
    frame_rows: list[tuple[float, float, dict[int, float]]] = []
    for members in frames.values():
        weights = np.asarray([max(confidence, 1e-8) for _, confidence, _ in members])
        normalized_weights = weights / weights.sum()
        contribution: dict[int, float] = {}
        for (_, _, base), entity_weight in zip(members, normalized_weights):
            for index, value in base.items():
                contribution[index] = value * float(entity_weight)
        frame_rows.append(
            (
                float(sum(score * weight for (score, _, _), weight in zip(members, normalized_weights))),
                float(np.mean([confidence for _, confidence, _ in members])),
                contribution,
            )
        )
    frame_weights = np.asarray([max(confidence, 1e-8) for _, confidence, _ in frame_rows])
    frame_weights /= frame_weights.sum()
    contribution = {}
    for (_, _, base), frame_weight in zip(frame_rows, frame_weights):
        for index, value in base.items():
            row = rows[index]
            contribution[f"{row.name}:{index}"] = (
                row.normalized_value * value * float(frame_weight)
            )
    return (
        float(sum(score * weight for (score, _, _), weight in zip(frame_rows, frame_weights))),
        coverage,
        float(np.mean([confidence for _, confidence, _ in frame_rows]) * coverage),
        contribution,
    )


def _spatial_maps(rows: Sequence[ResidualEvidence], image_shape: tuple[int, int]) -> dict[int, np.ndarray]:
    maps: dict[int, np.ndarray] = {}
    for row in rows:
        if not row.valid_mask:
            continue
        frame = row.temporal_support.get("frame_index", row.spatial_support.get("frame_index"))
        if frame is None:
            continue
        target = maps.setdefault(int(frame), np.full(image_shape, np.nan, dtype=np.float32))
        support = row.spatial_support
        kind = support.get("kind")
        if kind == "object_mask" and support.get("mask") is not None:
            mask = np.asarray(support["mask"], dtype=bool)
            if mask.shape == image_shape:
                current = target[mask]
                target[mask] = np.where(
                    np.isfinite(current),
                    np.maximum(current, row.normalized_value),
                    row.normalized_value,
                )
        elif kind in {"point", "track"} and support.get("xy") is not None:
            x, y = (int(round(value)) for value in support["xy"])
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                target[y, x] = max(float(target[y, x]) if np.isfinite(target[y, x]) else 0.0, row.normalized_value)
    return maps


def fuse_clip_residuals(clip: ClipObservation, residuals: Sequence[ResidualEvidence]) -> ClipResult:
    score, coverage, confidence, contribution = _hierarchical(residuals)
    by_object: dict[str, list[ResidualEvidence]] = defaultdict(list)
    by_track: dict[str, list[ResidualEvidence]] = defaultdict(list)
    for row in residuals:
        object_id = str(row.spatial_support.get("object_id", ""))
        track_id = str(row.spatial_support.get("track_id", row.temporal_support.get("track_id", "")))
        if object_id:
            by_object[object_id].append(row)
        if track_id:
            by_track[track_id].append(row)
    object_scores = {key: _weighted(rows)[0] for key, rows in by_object.items()}
    track_scores = {key: _weighted(rows)[0] for key, rows in by_track.items()}
    dominant = max(contribution, key=contribution.get).split(":", 1)[0] if contribution else ""
    first, last = clip.frames[0], clip.frames[-1]
    return ClipResult(
        video_id=clip.video_id,
        clip_id=clip.clip_id,
        start_frame=first.frame_index,
        end_frame=last.frame_index,
        start_time=first.timestamp,
        end_time=last.timestamp,
        risk_score=score,
        coverage=coverage,
        confidence=confidence,
        residuals=list(residuals),
        object_scores=object_scores,
        track_scores=track_scores,
        contributions=contribution,
        dominant_residual=dominant,
        spatial_heatmaps=_spatial_maps(residuals, first.image.shape[:2]),
    )


def fuse_video_results(
    *,
    video_id: str,
    video_path: str,
    clips: Sequence[ClipResult],
    suspicious_threshold: float,
    merge_gap_frames: int = 1,
) -> VideoResult:
    valid_clips = [clip for clip in clips if math.isfinite(clip.risk_score)]
    video_score = (
        float(
            np.average(
                [clip.risk_score for clip in valid_clips],
                weights=[max(clip.confidence, 1e-8) for clip in valid_clips],
            )
        )
        if valid_clips
        else float("nan")
    )
    timeline = [
        {
            "clip_id": clip.clip_id,
            "start_frame": clip.start_frame,
            "end_frame": clip.end_frame,
            "start_time": clip.start_time,
            "end_time": clip.end_time,
            "risk_score": clip.risk_score,
            "coverage": clip.coverage,
            "confidence": clip.confidence,
            "dominant_residual": clip.dominant_residual,
        }
        for clip in clips
    ]
    selected = [clip for clip in clips if math.isfinite(clip.risk_score) and clip.risk_score >= suspicious_threshold]
    suspicious: list[dict[str, object]] = []
    for clip in selected:
        if suspicious and clip.start_frame - int(suspicious[-1]["end_frame"]) - 1 <= merge_gap_frames:
            suspicious[-1]["end_frame"] = max(int(suspicious[-1]["end_frame"]), clip.end_frame)
            suspicious[-1]["end_time"] = max(float(suspicious[-1]["end_time"]), clip.end_time)
            suspicious[-1]["risk_score"] = max(float(suspicious[-1]["risk_score"]), clip.risk_score)
            suspicious[-1]["clip_ids"].append(clip.clip_id)
        else:
            suspicious.append(
                {
                    "start_frame": clip.start_frame,
                    "end_frame": clip.end_frame,
                    "start_time": clip.start_time,
                    "end_time": clip.end_time,
                    "risk_score": clip.risk_score,
                    "clip_ids": [clip.clip_id],
                    "threshold": suspicious_threshold,
                    "threshold_role": "engineering_configuration_not_calibrated_probability",
                }
            )
    def maxima(attribute: str) -> dict[str, float]:
        output: dict[str, float] = {}
        for clip in clips:
            for key, value in getattr(clip, attribute).items():
                if math.isfinite(value):
                    output[key] = max(output.get(key, -math.inf), value)
        return output
    return VideoResult(
        video_id=video_id,
        video_path=video_path,
        risk_score=video_score,
        clip_results=list(clips),
        timeline=timeline,
        suspicious_clips=suspicious,
        object_scores=maxima("object_scores"),
        track_scores=maxima("track_scores"),
        metadata={
            "risk_is_calibrated_probability": False,
            "missing_evidence_is_zero": False,
            "historical_csv_read": False,
        },
    )
