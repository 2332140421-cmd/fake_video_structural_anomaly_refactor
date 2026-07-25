"""Persist the compact, interpretable outputs of one pipeline run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from data.schemas import ClipObservation, ResidualEvidence, VideoResult
from utils.io import ensure_output_dir, write_csv, write_json
from utils.visualization import (
    save_structural_heatmap,
    save_timeline,
    save_track_deviations,
)


def _residual_record(row: ResidualEvidence) -> dict[str, Any]:
    spatial = {
        key: value
        for key, value in row.spatial_support.items()
        if key != "mask"
    }
    if row.spatial_support.get("mask") is not None:
        spatial["mask_available"] = True
    return {
        "name": row.name,
        "level": row.level,
        "raw_value": row.raw_value,
        "normalized_value": row.normalized_value,
        "availability": row.availability,
        "confidence": row.confidence,
        "valid_mask": row.valid_mask,
        "reason": row.reason,
        "spatial_support": spatial,
        "temporal_support": row.temporal_support,
        "metadata": row.metadata,
    }


def _result_payload(result: VideoResult) -> dict[str, Any]:
    return {
        "video_id": result.video_id,
        "video_path": result.video_path,
        "risk_score": result.risk_score,
        "timeline": result.timeline,
        "suspicious_clips": result.suspicious_clips,
        "object_scores": result.object_scores,
        "track_scores": result.track_scores,
        "clips": [
            {
                "clip_id": clip.clip_id,
                "start_frame": clip.start_frame,
                "end_frame": clip.end_frame,
                "risk_score": clip.risk_score,
                "coverage": clip.coverage,
                "confidence": clip.confidence,
                "dominant_residual": clip.dominant_residual,
                "contributions": clip.contributions,
                "residuals": [_residual_record(row) for row in clip.residuals],
            }
            for clip in result.clip_results
        ],
        "metadata": result.metadata,
    }


def save_analysis_outputs(
    result: VideoResult,
    observations: Sequence[ClipObservation],
    output_dir: str | Path,
    *,
    heatmap_sigma: float = 5.0,
) -> dict[str, Path]:
    output = ensure_output_dir(output_dir)
    paths = {
        "result": write_json(output / "result.json", _result_payload(result)),
        "timeline": write_csv(
            output / "timeline.csv",
            result.timeline,
            [
                "clip_id", "start_frame", "end_frame", "start_time", "end_time",
                "risk_score", "coverage", "confidence", "dominant_residual",
            ],
        ),
        "clip_scores": write_csv(
            output / "clip_scores.csv",
            (
                {
                    "clip_id": clip.clip_id,
                    "risk_score": clip.risk_score,
                    "coverage": clip.coverage,
                    "confidence": clip.confidence,
                    "dominant_residual": clip.dominant_residual,
                }
                for clip in result.clip_results
            ),
            ["clip_id", "risk_score", "coverage", "confidence", "dominant_residual"],
        ),
        "object_scores": write_csv(
            output / "object_scores.csv",
            ({"object_id": key, "risk_score": value} for key, value in result.object_scores.items()),
            ["object_id", "risk_score"],
        ),
        "track_scores": write_csv(
            output / "track_scores.csv",
            ({"track_id": key, "risk_score": value} for key, value in result.track_scores.items()),
            ["track_id", "risk_score"],
        ),
        "suspicious_clips": write_json(output / "suspicious_clips.json", result.suspicious_clips),
        "timeline_plot": save_timeline(result, output / "timeline.png"),
        "abnormal_tracks": save_track_deviations(observations, output / "abnormal_tracks.png"),
        "structural_heatmap": save_structural_heatmap(
            result,
            observations,
            output / "structural_residual_heatmap.png",
            sigma=heatmap_sigma,
        ),
    }
    return paths


__all__ = ["save_analysis_outputs"]
