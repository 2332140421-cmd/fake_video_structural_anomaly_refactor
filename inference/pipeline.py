"""One in-memory video-to-structural-anomaly pipeline."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data.observations import build_shared_observations
from data.schemas import ClipObservation, VideoResult
from data.video import read_video, split_clips
from models.fusion import fuse_clip_residuals, fuse_video_results
from models.geometry import predict_target_positions
from models.motion_residuals import compute_motion_residuals
from models.object_semantic import compute_object_semantic_residuals
from models.providers import DepthIntrinsicsProvider, ObjectProvider, PoseProvider, TrackProvider
from models.relation_residuals import compute_relation_residuals
from models.reprojection_residuals import compute_reprojection_residuals


class ForgeryAnalysisPipeline:
    """Run all paper residual branches over one shared observation object."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        object_provider: ObjectProvider,
        depth_provider: DepthIntrinsicsProvider,
        pose_provider: PoseProvider,
        track_provider: TrackProvider | None = None,
    ) -> None:
        self.config = dict(config)
        self.object_provider = object_provider
        self.depth_provider = depth_provider
        self.pose_provider = pose_provider
        self.track_provider = track_provider
        self.last_observations: list[ClipObservation] = []

    def analyze_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
        max_clips: int | None = None,
    ) -> VideoResult:
        started = time.perf_counter()
        torch = None
        try:
            import torch as torch_module

            torch = torch_module
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except (ImportError, RuntimeError):
            torch = None
        video = self.config["video"]
        resize = video.get("resize")
        resize_tuple = None if resize is None else tuple(int(value) for value in resize)
        metadata, frames = read_video(
            video_path,
            resize=resize_tuple,
            max_frames=max_frames,
        )
        clips = split_clips(
            metadata,
            frames,
            clip_length=int(video["clip_length"]),
            clip_stride=int(video["clip_stride"]),
        )
        if max_clips is not None:
            if max_clips < 1:
                raise ValueError("max_clips must be positive when supplied.")
            clips = clips[:max_clips]
        observations = [
            build_shared_observations(
                clip,
                object_provider=self.object_provider,
                depth_provider=self.depth_provider,
                pose_provider=self.pose_provider,
                track_provider=self.track_provider,
            )
            for clip in clips
        ]
        result = self.analyze_observations(
            observations,
            video_path=str(metadata.video_path),
        )
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        else:
            peak_memory = 0.0
        result.metadata.update(
            {
                "decoded_frames": len(frames),
                "runtime_seconds": time.perf_counter() - started,
                "peak_gpu_memory_mb": float(peak_memory),
                "failure_reason": "",
            }
        )
        return result

    @staticmethod
    def _attach_predictions(clip: ClipObservation) -> None:
        frames = {frame.frame_index: frame for frame in clip.frames}
        for track in clip.tracks:
            if track.points_3d is None:
                continue
            predicted = np.full_like(track.actual_xy, np.nan, dtype=float)
            predicted[0] = track.actual_xy[0]
            for position in range(1, len(track.frame_indices)):
                target = frames.get(track.frame_indices[position])
                if (
                    target is None
                    or target.relative_pose_from_previous is None
                    or target.intrinsics is None
                    or not bool(track.valid_mask[position - 1])
                ):
                    continue
                projected = predict_target_positions(
                    track.points_3d[position - 1 : position],
                    target.relative_pose_from_previous,
                    target.intrinsics,
                )
                if len(projected):
                    predicted[position] = projected[0]
            track.predicted_xy = predicted

    def _clip_result(self, clip: ClipObservation):
        self._attach_predictions(clip)
        semantic_config = self.config["object_semantic"]
        residuals = compute_object_semantic_residuals(
            clip,
            prior_path=semantic_config["prior_path"],
            min_depth_coverage=float(semantic_config.get("min_depth_coverage", 0.5)),
            max_occlusion_ratio=float(semantic_config.get("max_occlusion_ratio", 0.5)),
            min_mask_quality=float(semantic_config.get("min_mask_quality", 0.3)),
        )
        residuals.extend(compute_motion_residuals(clip))
        residuals.extend(compute_reprojection_residuals(clip))
        residuals.extend(compute_relation_residuals(clip))
        return fuse_clip_residuals(clip, residuals)

    def analyze_observations(
        self,
        observations: Sequence[ClipObservation],
        *,
        video_path: str = "<in-memory>",
    ) -> VideoResult:
        if not observations:
            raise ValueError("At least one clip observation is required.")
        self.last_observations = list(observations)
        clip_results = [self._clip_result(clip) for clip in observations]
        fusion = self.config["fusion"]
        result = fuse_video_results(
            video_id=observations[0].video_id,
            video_path=video_path,
            clips=clip_results,
            suspicious_threshold=float(fusion["suspicious_clip_threshold"]),
            merge_gap_frames=int(fusion.get("merge_gap_frames", 1)),
        )
        frames = [frame for clip in observations for frame in clip.frames]
        residuals = [row for clip in result.clip_results for row in clip.residuals]

        def available(names: set[str]) -> int:
            return sum(row.valid_mask and row.name in names for row in residuals)

        objects = [obj for frame in frames for obj in frame.objects]
        semantic_prior = available({"semantic_metric_prior"})
        semantic_temporal = available({"semantic_metric_temporal"})
        d1 = available(
            {
                "dynamic_reprojection",
                "track_3d_continuity",
                "direction_consistency",
                "relative_velocity",
            }
        )
        d2 = available(
            {"point_reprojection", "depth_reprojection", "boundary_reprojection"}
        )
        d3 = available({"relation", "occlusion", "reappearance"})
        result.metadata.update(
            {
                "analyzed_frames": len({frame.frame_index for frame in frames}),
                "clip_count": len(observations),
                "object_detection_frames": sum(
                    frame.availability.get("objects", False) for frame in frames
                ),
                "instance_mask_frames": sum(
                    any(obj.instance_mask is not None for obj in frame.objects)
                    for frame in frames
                ),
                "metric_depth_frames": sum(
                    frame.metric_depth is not None for frame in frames
                ),
                "intrinsics_frames": sum(frame.intrinsics is not None for frame in frames),
                "pose_pairs_total": sum(max(len(clip.frames) - 1, 0) for clip in observations),
                "pose_pairs_valid": sum(
                    frame.relative_pose_from_previous is not None
                    for clip in observations
                    for frame in clip.frames[1:]
                ),
                "object_tracks": len({obj.track_id for obj in objects}),
                "point_tracks": len(
                    {track.track_id for clip in observations for track in clip.tracks}
                ),
                "semantic_prior_available": bool(semantic_prior),
                "semantic_temporal_available": bool(semantic_temporal),
                "d1_available": bool(d1),
                "d2_available": bool(d2),
                "d3_available": bool(d3),
                "branch_valid_evidence_counts": {
                    "object_semantic_prior": semantic_prior,
                    "object_semantic_temporal": semantic_temporal,
                    "d1": d1,
                    "d2": d2,
                    "d3": d3,
                },
                "overall_coverage": float(
                    np.mean([clip.coverage for clip in result.clip_results])
                ),
                "objects_total": len(objects),
                "objects_with_metric_depth": sum(
                    obj.instance_mask is not None
                    and frame.metric_depth is not None
                    and bool(
                        np.any(
                            obj.instance_mask
                            & (
                                np.isfinite(frame.metric_depth)
                                if frame.depth_valid_mask is None
                                else frame.depth_valid_mask
                            )
                        )
                    )
                    for frame in frames
                    for obj in frame.objects
                ),
                "objects_with_scale_prior": sum(
                    row.name == "semantic_metric_prior"
                    and row.reason != "missing_category_metric_prior"
                    for row in residuals
                ),
                "objects_with_observable_dimension": semantic_prior,
                "objects_with_semantic_prior_residual": semantic_prior,
                "tracks_with_semantic_temporal_residual": len(
                    {
                        str(row.spatial_support.get("track_id"))
                        for row in residuals
                        if row.name == "semantic_metric_temporal" and row.valid_mask
                    }
                ),
                "authenticity_label_used": False,
                "m6_to_a2_bridge_called": False,
                "real_analysis_reads_historical_csv": False,
            }
        )
        if not math.isfinite(result.risk_score):
            result.metadata["risk_score_status"] = "unavailable"
        return result


__all__ = ["ForgeryAnalysisPipeline"]
