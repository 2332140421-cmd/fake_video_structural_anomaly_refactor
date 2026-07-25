"""One in-memory video-to-structural-anomaly pipeline."""

from __future__ import annotations

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

    def analyze_video(self, video_path: str | Path) -> VideoResult:
        video = self.config["video"]
        resize = video.get("resize")
        resize_tuple = None if resize is None else tuple(int(value) for value in resize)
        metadata, frames = read_video(video_path, resize=resize_tuple)
        clips = split_clips(
            metadata,
            frames,
            clip_length=int(video["clip_length"]),
            clip_stride=int(video["clip_stride"]),
        )
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
        return self.analyze_observations(
            observations,
            video_path=str(metadata.video_path),
        )

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
        return fuse_video_results(
            video_id=observations[0].video_id,
            video_path=video_path,
            clips=clip_results,
            suspicious_threshold=float(fusion["suspicious_clip_threshold"]),
            merge_gap_frames=int(fusion.get("merge_gap_frames", 1)),
        )


__all__ = ["ForgeryAnalysisPipeline"]
