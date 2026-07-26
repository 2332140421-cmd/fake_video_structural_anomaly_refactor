"""One in-memory video-to-structural-anomaly pipeline."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from data.observations import build_shared_observations
from data.schemas import ClipObservation, VideoResult
from data.video import (
    SAMPLER_VERSION,
    read_uniform_video_sample,
    read_video,
    split_clips,
    split_uniform_sample,
)
from models.fusion import fuse_clip_residuals, fuse_video_results
from models.geometry import predict_target_positions
from models.motion_residuals import compute_motion_residuals
from models.object_semantic import (
    compute_object_semantic_residuals,
    load_canonical_axis_registry,
    load_scale_prior_registry,
)
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

    def reset_video_state(self) -> None:
        """Clear video-local adapter and output state while retaining model weights."""

        self.last_observations = []
        pair_metadata = getattr(self.pose_provider, "pair_metadata", None)
        if isinstance(pair_metadata, dict):
            pair_metadata.clear()
        for provider in (
            self.object_provider,
            self.depth_provider,
            self.pose_provider,
            self.track_provider,
        ):
            reset = getattr(provider, "reset_video_state", None)
            if callable(reset):
                reset()

    def analyze_video(
        self,
        video_path: str | Path,
        *,
        max_frames: int | None = None,
        max_clips: int | None = None,
        sampling_mode: str | None = None,
        sampled_frames: int = 32,
        clip_length: int | None = None,
        clip_count: int | None = None,
    ) -> VideoResult:
        started = time.perf_counter()
        self.reset_video_state()
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
        if sampling_mode is None:
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
            selected_frame_indices = tuple(range(len(frames)))
            selected_timestamps = tuple(index / metadata.fps for index in range(len(frames)))
            effective_sampling_mode = "sequential_decode"
            sampler_version = "legacy_sequential_v1"
        elif sampling_mode == "uniform_full_video":
            if max_frames is not None or max_clips is not None:
                raise ValueError(
                    "max_frames/max_clips cannot be combined with uniform_full_video."
                )
            effective_clip_length = int(
                video["clip_length"] if clip_length is None else clip_length
            )
            effective_clip_count = int(
                sampled_frames // effective_clip_length
                if clip_count is None
                else clip_count
            )
            if effective_clip_length * effective_clip_count != int(sampled_frames):
                raise ValueError("sampled_frames must equal clip_length * clip_count.")
            sample = read_uniform_video_sample(
                video_path,
                requested_frame_count=int(sampled_frames),
                resize=resize_tuple,
            )
            metadata = sample.metadata
            frames = list(sample.frames)
            clips = split_uniform_sample(
                sample,
                clip_length=effective_clip_length,
                clip_count=effective_clip_count,
            )
            selected_frame_indices = sample.frame_indices
            selected_timestamps = sample.timestamps
            effective_sampling_mode = sampling_mode
            sampler_version = SAMPLER_VERSION
        else:
            raise ValueError(f"Unsupported sampling_mode: {sampling_mode!r}.")
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
                "sampling_mode": effective_sampling_mode,
                "sampler_version": sampler_version,
                "source_frame_count": int(metadata.frame_count),
                "source_duration_seconds": float(metadata.duration_seconds),
                "source_fps": float(metadata.fps),
                "selected_frame_count": len(selected_frame_indices),
                "selected_frame_indices": list(selected_frame_indices),
                "selected_timestamps": list(selected_timestamps),
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
            canonical_axis_path=semantic_config["canonical_axis_path"],
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
        prior_registry = load_scale_prior_registry(
            self.config["object_semantic"]["prior_path"]
        )
        canonical_axis_registry = load_canonical_axis_registry(
            self.config["object_semantic"]["canonical_axis_path"]
        )
        min_mask_quality = float(
            self.config["object_semantic"].get("min_mask_quality", 0.3)
        )
        max_occlusion_ratio = float(
            self.config["object_semantic"].get("max_occlusion_ratio", 0.5)
        )

        def has_valid_metric_depth(frame, obj) -> bool:
            if obj.instance_mask is None or frame.metric_depth is None:
                return False
            valid = (
                np.isfinite(frame.metric_depth)
                if frame.depth_valid_mask is None
                else frame.depth_valid_mask
            )
            return bool(np.any(obj.instance_mask & valid))

        valid_objects = [
            (frame, obj)
            for frame in frames
            for obj in frame.objects
            if has_valid_metric_depth(frame, obj)
        ]
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
        semantic_rows = [row for row in residuals if row.name == "semantic_metric_prior"]
        semantic_dimensions = [
            tuple(row.metadata.get("dimension_observability", ()))
            for row in semantic_rows
        ]
        observable_dimensions = {
            dimension: sum(
                any(
                    record.get("dimension") == dimension
                    and bool(record.get("observable", False))
                    for record in records
                )
                for records in semantic_dimensions
            )
            for dimension in ("height", "width", "length")
        }
        objects_with_dimension_axis = sum(
            any(bool(record.get("axis_source")) for record in records)
            for records in semantic_dimensions
        )
        objects_with_viewpoint_evidence = sum(
            any(bool(record.get("viewpoint_evidence", False)) for record in records)
            for records in semantic_dimensions
        )
        objects_with_any_observable_dimension = sum(
            any(bool(record.get("observable", False)) for record in records)
            for records in semantic_dimensions
        )
        objects_with_supported_prior = sum(
            bool(row.metadata.get("scale_prior_entry_id")) for row in semantic_rows
        )
        objects_with_supported_prior_and_observable_dimension = sum(
            bool(row.metadata.get("scale_prior_entry_id"))
            and any(
                record.get("dimension") == row.metadata.get("dimension")
                and bool(record.get("observable", False))
                for record in row.metadata.get("dimension_observability", ())
            )
            for row in semantic_rows
        )
        objects_with_canonical_axis = sum(
            any(
                record.get("dimension") in {"height", "width", "length"}
                and bool(record.get("observable", False))
                and bool(record.get("canonical_mapping_rule"))
                for record in row.metadata.get("dimension_observability", ())
            )
            for row in semantic_rows
        )
        canonical_failure_reasons: dict[str, int] = {}
        for row in semantic_rows:
            if row.valid_mask or not row.metadata.get("scale_prior_entry_id"):
                continue
            canonical_failure_reasons[row.reason] = (
                canonical_failure_reasons.get(row.reason, 0) + 1
            )
        unavailable_reason_map = {
            "missing_category_metric_prior": "NO_SCALE_PRIOR",
            "category_too_broad_without_subtype": (
                "CATEGORY_TOO_BROAD_WITHOUT_SUBTYPE"
            ),
            "instance_mask_unavailable": "NO_INSTANCE_MASK",
            "severe_object_truncation": "TRUNCATED",
            "dimension_truncated": "TRUNCATED",
            "severe_object_occlusion": "OCCLUDED",
            "unstable_track_identity": "TRACK_UNSTABLE",
            "dimension_not_observable_from_current_view": "VIEWPOINT_UNAVAILABLE",
            "viewpoint_unavailable": "VIEWPOINT_UNAVAILABLE",
            "no_reliable_dimension_axis": "NO_RELIABLE_DIMENSION_AXIS",
            "dimension_not_observable": "DIMENSION_NOT_OBSERVABLE",
            "metric_object_surface_unavailable": "INVALID_METRIC_DEPTH",
            "insufficient_valid_metric_depth_ratio": "INSUFFICIENT_DEPTH_COVERAGE",
        }
        unavailable_reasons = {
            name: 0
            for name in (
                "NO_SCALE_PRIOR",
                "CATEGORY_TOO_BROAD_WITHOUT_SUBTYPE",
                "NO_INSTANCE_MASK",
                "INVALID_METRIC_DEPTH",
                "INSUFFICIENT_DEPTH_COVERAGE",
                "TRUNCATED",
                "OCCLUDED",
                "NO_RELIABLE_DIMENSION_AXIS",
                "VIEWPOINT_UNAVAILABLE",
                "DIMENSION_NOT_OBSERVABLE",
                "TRACK_TOO_SHORT",
                "TRACK_UNSTABLE",
                "OTHER",
            )
        }
        for row in semantic_rows:
            if not row.valid_mask:
                unavailable_reasons[unavailable_reason_map.get(row.reason, "OTHER")] += 1
        unavailable_reasons["TRACK_TOO_SHORT"] += sum(
            row.name == "semantic_metric_temporal"
            and not row.valid_mask
            and row.reason == "insufficient_same_track_metric_history"
            for row in residuals
        )

        valid_frames_by_track: dict[tuple[str, str], set[int]] = {}
        for frame, obj in valid_objects:
            valid_frames_by_track.setdefault(
                (frame.clip_id, obj.track_id), set()
            ).add(frame.frame_index)
        tracks_with_multiple_valid_frames = sum(
            len(indices) > 1 for indices in valid_frames_by_track.values()
        )
        tracks_with_semantic_temporal_residual = len(
            {
                (
                    str(row.spatial_support.get("clip_id")),
                    str(row.spatial_support.get("track_id")),
                )
                for row in residuals
                if row.name == "semantic_metric_temporal" and row.valid_mask
            }
        )
        visibility_states = {
            name: 0
            for name in (
                "VISIBLE",
                "PARTIALLY_OCCLUDED",
                "FULLY_OCCLUDED",
                "REAPPEARED",
                "TRUNCATED",
                "MISSING_UNEXPLAINED",
                "UNAVAILABLE",
            )
        }
        visibility_mapping = {
            "fully_visible": "VISIBLE",
            "partially_occluded": "PARTIALLY_OCCLUDED",
            "fully_occluded": "FULLY_OCCLUDED",
            "reappeared": "REAPPEARED",
            "out_of_frame": "TRUNCATED",
            "detector_missing": "MISSING_UNEXPLAINED",
        }
        for frame in frames:
            for observation in frame.visibility_observations.values():
                state = getattr(observation.current_state, "value", observation.current_state)
                visibility_states[visibility_mapping.get(str(state), "UNAVAILABLE")] += 1

        branch_names = {
            "semantic_prior": {"semantic_metric_prior"},
            "semantic_temporal": {"semantic_metric_temporal"},
            "d1": {
                "dynamic_reprojection",
                "track_3d_continuity",
                "direction_consistency",
                "relative_velocity",
            },
            "d2": {
                "point_reprojection",
                "depth_reprojection",
                "boundary_reprojection",
            },
            "d3": {"relation", "occlusion", "reappearance"},
        }
        branch_counts = {
            branch: {
                "total": sum(row.name in names for row in residuals),
                "available": sum(row.name in names and row.valid_mask for row in residuals),
            }
            for branch, names in branch_names.items()
        }
        point_tracks = [track for clip in observations for track in clip.tracks]
        point_support_counts = {
            name: sum(track.metadata.get("support_type") == name for track in point_tracks)
            for name in ("OBJECT", "BOUNDARY", "BACKGROUND", "KEYPOINT")
        }
        point_support_counts["OTHER"] = len(point_tracks) - sum(
            point_support_counts.values()
        )
        point_track_index_alignment_ok = all(
            track.actual_xy.shape == (len(track.frame_indices), 2)
            and track.valid_mask.shape == (len(track.frame_indices),)
            and (
                track.predicted_xy is None
                or track.predicted_xy.shape == track.actual_xy.shape
            )
            for track in point_tracks
        )
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
                    has_valid_metric_depth(frame, obj)
                    for frame in frames for obj in frame.objects
                ),
                "objects_with_scale_prior": sum(
                    bool(row.metadata.get("scale_prior_entry_id"))
                    for row in semantic_rows
                ),
                "objects_with_observable_dimension": (
                    objects_with_any_observable_dimension
                ),
                "objects_with_semantic_prior_residual": semantic_prior,
                "tracks_with_semantic_temporal_residual": (
                    tracks_with_semantic_temporal_residual
                ),
                "authenticity_label_used": False,
                "scale_prior_schema_version": prior_registry.schema_version,
                "scale_prior_sha256": prior_registry.prior_sha256,
                "scale_prior_source_table_sha256": (
                    prior_registry.source_table_sha256
                ),
                "canonical_axis_schema_version": (
                    canonical_axis_registry.schema_version
                ),
                "canonical_threshold_config_sha256": (
                    canonical_axis_registry.config_sha256
                ),
                "bottle_source_runtime_dimension_match": (
                    canonical_axis_registry.bottle_source_runtime_dimension_match
                ),
                "scale_prior_entry_id": sorted(
                    {
                        str(row.metadata["scale_prior_entry_id"])
                        for row in semantic_rows
                        if row.metadata.get("scale_prior_entry_id")
                    }
                ),
                "scale_prior_confidence": {
                    str(row.metadata["scale_prior_entry_id"]): str(
                        row.metadata["scale_prior_confidence"]
                    )
                    for row in semantic_rows
                    if row.metadata.get("scale_prior_entry_id")
                },
                "m6_to_a2_bridge_called": False,
                "real_analysis_reads_historical_csv": False,
                "branch_evidence_counts": branch_counts,
                "object_semantic_funnel": {
                    "objects_total": len(objects),
                    "objects_with_instance_mask": sum(
                        obj.instance_mask is not None and bool(np.any(obj.instance_mask))
                        for obj in objects
                    ),
                    "objects_with_valid_metric_depth": len(valid_objects),
                    "objects_with_scale_prior": objects_with_supported_prior,
                    "objects_with_supported_prior_and_observable_dimension": (
                        objects_with_supported_prior_and_observable_dimension
                    ),
                    "objects_not_severely_truncated": sum(not obj.truncated for obj in objects),
                    "objects_not_severely_occluded": sum(
                        obj.occlusion_ratio <= max_occlusion_ratio for obj in objects
                    ),
                    "objects_with_dimension_axis": objects_with_dimension_axis,
                    "objects_with_canonical_axis": objects_with_canonical_axis,
                    "objects_with_viewpoint_evidence": objects_with_viewpoint_evidence,
                    "objects_with_viewpoint_estimate": sum(
                        obj.viewpoint != "unknown" for obj in objects
                    ),
                    "observable_height": observable_dimensions["height"],
                    "observable_width": observable_dimensions["width"],
                    "observable_length": observable_dimensions["length"],
                    "objects_with_observable_height": observable_dimensions["height"],
                    "objects_with_observable_width": observable_dimensions["width"],
                    "objects_with_observable_length": observable_dimensions["length"],
                    "objects_with_any_observable_dimension": (
                        objects_with_any_observable_dimension
                    ),
                    "objects_with_semantic_prior_residual": semantic_prior,
                    "tracks_with_multiple_valid_frames": tracks_with_multiple_valid_frames,
                    "tracks_with_semantic_temporal_residual": (
                        tracks_with_semantic_temporal_residual
                    ),
                    "prior_support_rate": (
                        objects_with_supported_prior / len(objects)
                        if objects
                        else float("nan")
                    ),
                    "prior_execution_rate": (
                        semantic_prior
                        / objects_with_supported_prior_and_observable_dimension
                        if objects_with_supported_prior_and_observable_dimension
                        else float("nan")
                    ),
                },
                "canonical_axis_summary": {
                    "objects_with_canonical_axis": objects_with_canonical_axis,
                    "person_total": sum(obj.category == "person" for obj in objects),
                    "person_height_available": sum(
                        row.valid_mask
                        and row.metadata.get("class_name") == "person"
                        and row.metadata.get("canonical_dimension") == "height"
                        for row in semantic_rows
                    ),
                    "bottle_total": sum(obj.category == "bottle" for obj in objects),
                    "bottle_height_available": sum(
                        row.valid_mask
                        and row.metadata.get("class_name") == "bottle"
                        and row.metadata.get("canonical_dimension") == "height"
                        for row in semantic_rows
                    ),
                    "car_total": sum(obj.category == "car" for obj in objects),
                    "car_length_available": sum(
                        row.valid_mask
                        and row.metadata.get("class_name") == "car"
                        and row.metadata.get("canonical_dimension") == "length"
                        for row in semantic_rows
                    ),
                    "canonical_mapping_failure_reasons": (
                        canonical_failure_reasons
                    ),
                },
                "object_semantic_unavailable_reasons": unavailable_reasons,
                "visibility_state_counts": visibility_states,
                "d3_event_summary": {
                    "observable_relation_events": available({"relation"}),
                    "observable_occlusion_events": available({"occlusion"}),
                    "observable_reappearance_events": available({"reappearance"}),
                    "d3_residual_available": bool(d3),
                },
                "point_track_diagnostics": {
                    "support_counts": point_support_counts,
                    "track_ids_unique": len({track.track_id for track in point_tracks})
                    == len(point_tracks),
                    "index_alignment_ok": point_track_index_alignment_ok,
                    "invalid_point_samples": sum(
                        int(np.count_nonzero(~track.valid_mask)) for track in point_tracks
                    ),
                },
            }
        )
        if not math.isfinite(result.risk_score):
            result.metadata["risk_score_status"] = "unavailable"
        return result


__all__ = ["ForgeryAnalysisPipeline"]
