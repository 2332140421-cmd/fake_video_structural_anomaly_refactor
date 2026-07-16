#!/usr/bin/env python3
"""Run P3-0.5 sequence pose/depth stabilization on real short clips.

This script reports geometry quality only.  It does not compute forged-video
residuals, object velocity anomalies, occlusion anomalies, or classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.depth_provider import BaseDepthProvider, RealDepthProvider  # noqa: E402
from semantic3d.geometry.camera import CameraObservation  # noqa: E402
from semantic3d.observations import FrameObservationJSON  # noqa: E402
from semantic3d.providers import BaseObjectProvider  # noqa: E402
from semantic3d.real_object_provider import RealObjectProvider  # noqa: E402
from semantic3d.reconstruction import Shared3DFrameBuilder  # noqa: E402
from semantic3d.sequence_geometry import (  # noqa: E402
    CameraMotionRegime,
    DepthAlignmentMode,
    HistogramFeatureSceneCutDetector,
    SequenceScaleStatus,
    apply_sequence_depth_alignment,
    build_foreground_mask,
    stabilize_sequence_geometry,
)


DEFAULT_SUITE = (
    ("static_camera", "real_3.mp4", 0, 8),
    ("slowly_moving_camera", "real_2.mp4", 0, 8),
    ("clearly_moving_camera", "real_1.mp4", 0, 8),
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _decode_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    output_dir: Path,
) -> list[tuple[int, Path, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames: list[tuple[int, Path, np.ndarray]] = []
    try:
        for offset in range(num_frames):
            success, image = capture.read()
            if not success or image is None:
                break
            index = start_frame + offset
            frame_path = frame_dir / f"frame_{index:06d}.png"
            if not cv2.imwrite(str(frame_path), image):
                raise RuntimeError(f"Could not save decoded frame: {frame_path}")
            frames.append((index, frame_path, image))
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("Sequence stabilization requires at least two decoded frames.")
    return frames


def _approximate_camera(width: int, height: int) -> CameraObservation:
    focal = 1.2 * max(width, height)
    K = np.asarray(
        [[focal, 0.0, (width - 1) / 2], [0.0, focal, (height - 1) / 2], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return CameraObservation.from_parameters(
        K=K,
        image_width=width,
        image_height=height,
        intrinsics_source="approximate",
        quality=0.5,
        metadata={"metric_calibration": False, "P3_0_5_smoke_only": True},
    )


def _resized_raw(depth) -> Optional[np.ndarray]:
    if depth.raw_model_output is None or depth.depth_map is None:
        return None
    raw = np.asarray(depth.raw_model_output, dtype=float)
    if raw.shape != depth.depth_map.shape:
        raw = cv2.resize(
            raw.astype(np.float32),
            (depth.depth_map.shape[1], depth.depth_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(float)
    return raw


def _background_depth_stability(depths, masks, global_alignment) -> tuple[float, float]:
    before: list[float] = []
    after: list[float] = []
    inverse_domain = global_alignment.alignment_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
    for frame_index in global_alignment.frame_indices:
        depth = depths[frame_index]
        geometry = depth.require_geometry_depth()
        background = ~masks[frame_index]
        canonical_valid = np.asarray(depth.valid_mask, dtype=bool) & background
        if np.any(canonical_valid):
            before.append(float(np.median(geometry[canonical_valid])))
        frame_alignment = global_alignment.per_frame[frame_index]
        if not frame_alignment.valid:
            continue
        source = _resized_raw(depth) if inverse_domain else geometry
        if source is None:
            continue
        aligned = apply_sequence_depth_alignment(
            source,
            frame_alignment,
            values_are_inverse_domain=inverse_domain,
        )
        valid = np.isfinite(aligned) & (aligned > 0.0) & background
        if np.any(valid):
            after.append(float(np.median(aligned[valid])))

    def coefficient(values: Sequence[float]) -> float:
        if len(values) < 2:
            return float("nan")
        mean = float(np.mean(values))
        return float(np.std(values) / max(abs(mean), 1e-12))

    return coefficient(before), coefficient(after)


def _save_shared_geometry_cache(
    *,
    clip_id: str,
    video_path: Path,
    extracted: Sequence[tuple[int, Path, np.ndarray]],
    camera: CameraObservation,
    depths: Mapping[int, Any],
    foreground_masks: Mapping[int, np.ndarray],
    result: Any,
    scene_cut_flags: Mapping[int, bool],
    geometry_quality: Mapping[str, Any],
    output_dir: Path,
) -> Path:
    """Persist canonical P3-0.5 geometry for downstream reuse without inference."""

    cache_dir = output_dir / "shared_geometry_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    inverse_domain = (
        result.depth_alignment.alignment_mode == DepthAlignmentMode.AFFINE_INVERSE_DEPTH
    )
    depth_records: dict[str, dict[str, Any]] = {}
    for frame_index, _, _ in extracted:
        observation = depths[frame_index]
        per_frame_geometry = observation.require_geometry_depth()
        valid = np.asarray(observation.valid_mask, dtype=bool)
        unaligned_path = cache_dir / f"geometry_depth_per_frame_{frame_index:06d}.npy"
        aligned_path = cache_dir / f"geometry_depth_aligned_{frame_index:06d}.npy"
        valid_path = cache_dir / f"geometry_depth_valid_{frame_index:06d}.npy"
        foreground_path = cache_dir / f"foreground_mask_{frame_index:06d}.npy"
        np.save(unaligned_path, per_frame_geometry.astype(np.float32))
        np.save(valid_path, valid)
        np.save(foreground_path, np.asarray(foreground_masks[frame_index], dtype=bool))
        frame_alignment = result.depth_alignment.per_frame[frame_index]
        source = _resized_raw(observation) if inverse_domain else per_frame_geometry
        if frame_alignment.valid and source is not None:
            aligned = apply_sequence_depth_alignment(
                source,
                frame_alignment,
                values_are_inverse_domain=inverse_domain,
            )
        else:
            aligned = np.full(per_frame_geometry.shape, np.nan, dtype=float)
        np.save(aligned_path, aligned.astype(np.float32))
        depth_records[str(frame_index)] = {
            "per_frame_geometry_depth_path": str(unaligned_path.resolve()),
            "aligned_geometry_depth_path": str(aligned_path.resolve()),
            "valid_mask_path": str(valid_path.resolve()),
            "foreground_mask_path": str(foreground_path.resolve()),
            "provider_name": observation.provider_name,
            "input_representation": observation.depth_representation.value,
            "input_scale_status": observation.scale_status.value,
            "aligned_scale_status": result.sequence_scale_status.value,
            "quality": observation.quality,
            "visualization_depth_saved": False,
        }
    selected_by_target = {
        str(edge.target_frame_index): {
            "source_frame_index": edge.source_frame_index,
            "target_frame_index": edge.target_frame_index,
            "pose_model_type": edge.pose_model_type.value,
            "translation_scale_status": edge.translation_scale_status.value,
            "support_count": edge.support_count,
            "inlier_ratio": edge.inlier_ratio,
            "reprojection_error": edge.reprojection_error,
            "quality": edge.quality,
            "full_se3": edge.full_se3,
            "evidence_source": edge.evidence_source,
        }
        for edge in result.pose_graph.selected_edges
    }
    manifest = {
        "cache_version": "p3_0_5_shared_geometry_v1",
        "video_id": video_path.stem,
        "clip_id": clip_id,
        "video_path": str(video_path.resolve()),
        "frame_indices": [index for index, _, _ in extracted],
        "frame_paths": {
            str(index): str(path.resolve()) for index, path, _ in extracted
        },
        "image_width": camera.image_width,
        "image_height": camera.image_height,
        "K": camera.K.tolist() if camera.K is not None else None,
        "intrinsics_source": camera.intrinsics_source,
        "camera_quality": camera.quality,
        "reference_frame_index": result.reference_frame,
        "T_world_from_camera_by_frame": {
            str(index): (
                None if transform is None else transform.tolist()
            )
            for index, transform in result.pose_graph.T_world_from_camera_by_frame.items()
        },
        "T_camera_from_world_by_frame": {
            str(index): (
                None if transform is None else transform.tolist()
            )
            for index, transform in result.pose_graph.T_camera_from_world_by_frame.items()
        },
        "selected_pose_edges_by_target": selected_by_target,
        "sequence_scale_status": result.sequence_scale_status.value,
        "depth_alignment_mode": result.depth_alignment.alignment_mode.value,
        "depth_alignment_domain": result.depth_alignment.alignment_domain,
        "depth_records": depth_records,
        "scene_cut_flags": {str(key): bool(value) for key, value in scene_cut_flags.items()},
        "sequence_geometry_quality": result.quality,
        "geometry_quality": _json_safe(geometry_quality),
        "provenance": {
            "source": "run_sequence_geometry_stabilization.py",
            "depth_reestimated_by_downstream": False,
            "intrinsics_reestimated_by_downstream": False,
            "pose_reestimated_by_downstream": False,
            "visualization_depth_used": False,
            "semantic_scale_prior_used": False,
        },
    }
    manifest_path = cache_dir / "shared_3d_clip_manifest.json"
    manifest_path.write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8"
    )
    return manifest_path


def _save_diagnostics(result, output_dir: Path) -> None:
    pairs = result.pose_pairs
    adjacent = [pair for pair in pairs if pair.tracks.target_frame_index - pair.tracks.source_frame_index == 1]
    targets = [pair.tracks.target_frame_index for pair in adjacent]
    median_flow = [pair.tracks.median_flow for pair in adjacent]
    p90_flow = [pair.tracks.p90_flow for pair in adjacent]
    regimes = [pair.motion_regime.regime.value for pair in adjacent]
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))
    if targets:
        axes[0, 0].plot(targets, median_flow, marker="o", label="median background flow")
        axes[0, 0].plot(targets, p90_flow, marker="s", label="p90 background flow")
        axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Foreground-filtered background motion")
    axes[0, 0].set_xlabel("target frame")
    axes[0, 0].set_ylabel("pixels")
    axes[0, 0].grid(alpha=0.25)

    regime_names = [regime.value for regime in CameraMotionRegime]
    counts = Counter(regimes)
    axes[0, 1].barh(regime_names, [counts[name] for name in regime_names], color="tab:blue")
    axes[0, 1].set_title("Adjacent-frame camera motion regimes")
    axes[0, 1].set_xlabel("frame-pair count")

    frame_indices = list(result.frame_indices)
    pose_connected = [
        result.pose_graph.T_world_from_camera_by_frame[index] is not None
        for index in frame_indices
    ]
    depth_connected = [result.depth_alignment.per_frame[index].valid for index in frame_indices]
    dynamic_valid = [result.frame_validity[index].dynamic_3d_valid for index in frame_indices]
    x = np.arange(len(frame_indices))
    axes[1, 0].plot(x, pose_connected, marker="o", label="pose graph")
    axes[1, 0].plot(x, depth_connected, marker="s", label="depth graph")
    axes[1, 0].plot(x, dynamic_valid, marker="^", label="joint dynamic-3D gate")
    axes[1, 0].set_xticks(x, frame_indices, rotation=45)
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].set_title("Per-frame geometry validity")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.25)

    candidate_errors = [
        candidate.holdout_error
        for selection in result.depth_selections
        for candidate in selection.candidates
        if candidate.valid and math.isfinite(candidate.holdout_error)
    ]
    if candidate_errors:
        axes[1, 1].hist(candidate_errors, bins=min(12, len(candidate_errors)), color="tab:orange")
    else:
        axes[1, 1].text(0.5, 0.5, "No valid holdout depth alignment", ha="center", va="center", transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Depth alignment holdout errors")
    axes[1, 1].set_xlabel("domain error")
    axes[1, 1].set_ylabel("candidate count")
    figure.suptitle(
        f"P3-0.5 Geometry Stabilization, quality={result.quality:.3f}, "
        f"dynamic_3d_valid={result.dynamic_3d_valid}"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output_dir / "diagnostics.png", dpi=180)
    plt.close(figure)


def run_sequence_geometry_stabilization(
    *,
    clip_id: str,
    video_path: Path,
    start_frame: int,
    num_frames: int,
    output_dir: Path,
    object_provider: Optional[BaseObjectProvider] = None,
    depth_provider: Optional[BaseDepthProvider] = None,
    model_path: Path = PROJECT_ROOT / "checkpoints/yolov8n.pt",
    depth_model: str = "depth-anything/Depth-Anything-V2-Small",
    confidence_threshold: float = 0.3,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run one real short-clip stabilization and save all P3-0.5 diagnostics."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = _decode_frames(video_path, start_frame, num_frames, output_dir)
    height, width = extracted[0][2].shape[:2]
    camera = _approximate_camera(width, height)
    detector = object_provider or RealObjectProvider(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        default_depth=1.0,
        device=device,
        skip_unknown_scale_prior=False,
    )
    depth_estimator = depth_provider or RealDepthProvider(
        model_name=depth_model,
        device=device,
        invert_depth=True,
    )
    images: dict[int, np.ndarray] = {}
    depths = {}
    foreground_masks: dict[int, np.ndarray] = {}
    foreground_diagnostics: dict[int, dict[str, Any]] = {}
    for frame_index, frame_path, image in extracted:
        images[frame_index] = image
        objects = detector.predict(frame_path, frame_index, width, height)
        depth = depth_estimator.predict_observation(frame_path, frame_index=frame_index)
        depth.require_geometry_depth()
        depths[frame_index] = depth
        frame_2d = FrameObservationJSON(
            frame_index=frame_index,
            frame_id=f"{video_path.stem}_{frame_index:06d}",
            width=width,
            height=height,
            image_path=str(frame_path),
            objects=list(objects),
        )
        shared = Shared3DFrameBuilder().build(
            video_id=video_path.stem,
            frame=frame_2d,
            depth=depth,
            camera=camera,
        )
        foreground = build_foreground_mask(shared)
        foreground_masks[frame_index] = foreground.mask
        foreground_diagnostics[frame_index] = {
            "excluded_foreground_ratio": foreground.excluded_foreground_ratio,
            "quality": foreground.quality,
            "bbox_fallback_count": foreground.bbox_fallback_count,
            "mask_object_count": foreground.mask_object_count,
            "fallback_decision": dict(foreground.metadata),
        }
    frame_indices = tuple(images)
    scene_detector = HistogramFeatureSceneCutDetector()
    cut_flags = {frame_indices[0]: False}
    cut_details = []
    for source, target in zip(frame_indices, frame_indices[1:]):
        decision = scene_detector.detect(
            images[source],
            images[target],
            source_frame_index=source,
            target_frame_index=target,
        )
        cut_flags[target] = decision.is_cut
        cut_details.append(decision.__dict__)
    result = stabilize_sequence_geometry(
        images,
        depths,
        camera.K,
        frame_indices=frame_indices,
        foreground_masks=foreground_masks,
        scene_cut_flags=cut_flags,
        temporal_strides=(1, 2, 4),
    )

    motion_rows = []
    candidate_rows = []
    track_rows = []
    for pair_index, pair in enumerate(result.pose_pairs):
        motion_rows.append(pair.motion_regime.to_dict())
        for candidate in pair.candidates:
            payload = candidate.to_dict()
            payload["T_target_from_source"] = json.dumps(payload["T_target_from_source"])
            payload["metadata"] = json.dumps(_json_safe(payload["metadata"]))
            payload["selected"] = candidate is pair.selected
            candidate_rows.append(payload)
        for row in pair.tracks.track_rows:
            track_rows.append(
                {
                    **dict(row),
                    "pair_index": pair_index,
                    "candidate_count_before_mask": pair.tracks.candidate_count_before_mask,
                    "background_point_count_after_mask": pair.tracks.background_point_count_after_mask,
                    "match_count": pair.tracks.match_count,
                    "spatial_coverage_ratio": pair.tracks.spatial_coverage_ratio,
                    "quadrant_support": json.dumps(pair.tracks.quadrant_support),
                    "feature_concentrated": pair.tracks.feature_concentrated,
                    "failure_stage": pair.tracks.failure_stage,
                }
            )
    _write_csv(
        output_dir / "motion_regimes.csv",
        motion_rows,
        (
            "source_frame_index", "target_frame_index", "regime",
            "background_candidate_count", "background_point_count",
            "background_match_count", "background_inlier_count",
            "background_inlier_ratio", "median_background_flow",
            "p90_background_flow", "median_parallax", "homography_inlier_ratio",
            "essential_matrix_inlier_ratio", "model_reprojection_error",
            "image_difference", "foreground_excluded_ratio",
            "spatial_coverage_ratio", "quadrant_support", "feature_concentrated",
            "evidence_source", "valid", "missing_reason", "metadata",
        ),
    )
    candidate_fields = (
        "source_frame_index", "target_frame_index", "pose_model_type",
        "T_target_from_source", "rotation_valid", "translation_valid",
        "translation_scale_status", "support_count", "inlier_count",
        "inlier_ratio", "median_parallax", "reprojection_error", "quality",
        "valid", "missing_reason", "selected_reference_frame", "evidence_source",
        "metadata", "full_se3", "pose_scale_compatible_with_depth", "selected",
    )
    _write_csv(output_dir / "pose_candidates.csv", candidate_rows, candidate_fields)
    (output_dir / "pose_graph.json").write_text(
        json.dumps(_json_safe(result.pose_graph.to_dict()), indent=2) + "\n",
        encoding="utf-8",
    )
    camera_rows = []
    for index in frame_indices:
        transform = result.pose_graph.T_world_from_camera_by_frame[index]
        camera_rows.append(
            {
                "frame_index": index,
                "camera_center_x": None if transform is None else transform[0, 3],
                "camera_center_y": None if transform is None else transform[1, 3],
                "camera_center_z": None if transform is None else transform[2, 3],
                "connected_component_id": result.pose_graph.connected_component_id[index],
                "selected_reference_frame": result.pose_graph.selected_reference_frame[index],
                "valid": transform is not None,
                "dynamic_3d_valid": result.frame_validity[index].dynamic_3d_valid,
                "missing_reason": result.frame_validity[index].missing_reason,
            }
        )
    _write_csv(
        output_dir / "camera_trajectory.csv",
        camera_rows,
        (
            "frame_index", "camera_center_x", "camera_center_y", "camera_center_z",
            "connected_component_id", "selected_reference_frame", "valid",
            "dynamic_3d_valid", "missing_reason",
        ),
    )
    depth_candidate_rows = []
    for selection in result.depth_selections:
        for candidate in selection.candidates:
            depth_candidate_rows.append(
                {
                    "source_frame": candidate.source_frame,
                    "target_frame": candidate.target_frame,
                    "alignment_mode": candidate.alignment_mode.value,
                    "scale": candidate.scale,
                    "shift": candidate.shift,
                    "support_count": candidate.support_count,
                    "fitting_count": candidate.metadata.get("fitting_count", 0),
                    "holdout_count": candidate.holdout_count,
                    "inlier_ratio": candidate.inlier_ratio,
                    "fitting_error": candidate.fitting_error,
                    "holdout_error": candidate.holdout_error,
                    "normalized_holdout_error": candidate.metadata.get("normalized_holdout_error", math.nan),
                    "physical_valid": candidate.physical_valid,
                    "quality": candidate.quality,
                    "valid": candidate.valid,
                    "selected": candidate is selection.selected,
                    "raw_model_output_used": candidate.metadata.get("raw_model_output_used", False),
                    "visualization_depth_used": candidate.metadata.get("visualization_depth_used", False),
                    "missing_reason": candidate.missing_reason,
                }
            )
    depth_fields = (
        "source_frame", "target_frame", "alignment_mode", "scale", "shift",
        "support_count", "fitting_count", "holdout_count", "inlier_ratio",
        "fitting_error", "holdout_error", "normalized_holdout_error",
        "physical_valid", "quality", "valid", "selected",
        "raw_model_output_used", "visualization_depth_used", "missing_reason",
    )
    _write_csv(output_dir / "depth_alignment_candidates.csv", depth_candidate_rows, depth_fields)
    depth_global_rows = []
    for index, item in result.depth_alignment.per_frame.items():
        depth_global_rows.append(
            {
                "reference_frame": result.depth_alignment.reference_frame,
                "frame_index": index,
                "method": result.depth_alignment.method.value,
                "alignment_mode": result.depth_alignment.alignment_mode.value,
                "alignment_domain": item.alignment_domain,
                "scale": item.scale,
                "shift": item.shift,
                "supporting_edges": json.dumps(item.supporting_edges),
                "quality": item.quality,
                "valid": item.valid,
                "missing_reason": item.missing_reason,
                "global_consistency_error": result.depth_alignment.global_consistency_error,
                "scale_drift_before": result.depth_alignment.scale_drift_before,
                "scale_drift_after": result.depth_alignment.scale_drift_after,
            }
        )
    _write_csv(
        output_dir / "depth_alignment_global.csv",
        depth_global_rows,
        (
            "reference_frame", "frame_index", "method", "alignment_mode",
            "alignment_domain", "scale", "shift", "supporting_edges", "quality",
            "valid", "missing_reason", "global_consistency_error",
            "scale_drift_before", "scale_drift_after",
        ),
    )
    track_fields = (
        "pair_index", "track_id", "source_frame_index", "target_frame_index",
        "source_x", "source_y", "target_x", "target_y", "forward_backward_valid",
        "candidate_count_before_mask", "background_point_count_after_mask",
        "match_count", "spatial_coverage_ratio", "quadrant_support",
        "feature_concentrated", "failure_stage",
    )
    _write_csv(output_dir / "background_tracks.csv", track_rows, track_fields)
    stability_before, stability_after = _background_depth_stability(
        depths, foreground_masks, result.depth_alignment
    )
    adjacent_pairs = [
        pair
        for pair in result.pose_pairs
        if pair.tracks.target_frame_index - pair.tracks.source_frame_index == 1
    ]
    regime_counts = Counter(pair.motion_regime.regime.value for pair in adjacent_pairs)
    valid_identity_count = sum(
        pair.selected.valid and pair.selected.pose_model_type.value == "static_identity"
        for pair in result.pose_pairs
    )
    rotation_only_count = sum(
        edge.pose_model_type.value == "rotation_homography"
        for edge in result.pose_graph.selected_edges
    )
    full_se3_count = sum(edge.full_se3 for edge in result.pose_graph.selected_edges)
    background_motion_before = float(
        np.mean(
            [
                pair.tracks.median_flow
                for pair in adjacent_pairs
                if math.isfinite(pair.tracks.median_flow)
            ]
        )
    )
    background_reprojection_after = float(
        np.mean(
            [
                pair.selected.reprojection_error
                for pair in adjacent_pairs
                if pair.selected.valid
                and math.isfinite(pair.selected.reprojection_error)
            ]
        )
    )
    geometry_quality = {
        "clip_id": clip_id,
        "video_path": str(video_path),
        "frame_indices": list(frame_indices),
        "K_source": "approximate",
        "depth_provider": next(iter(depths.values())).provider_name,
        "depth_representation": next(iter(depths.values())).depth_representation.value,
        "input_depth_scale_status": next(iter(depths.values())).scale_status.value,
        "motion_regime_counts_adjacent": dict(regime_counts),
        "valid_identity_pose_count_all_candidates": valid_identity_count,
        "rotation_only_selected_edge_count": rotation_only_count,
        "full_se3_selected_edge_count": full_se3_count,
        "pose_graph_connected_frame_ratio": result.pose_graph.connected_frame_ratio,
        "pose_graph_valid_edge_ratio": result.pose_graph.valid_edge_ratio,
        "pose_chain_length": result.pose_graph.pose_chain_length,
        "disconnected_frames": list(result.pose_graph.disconnected_frames),
        "pose_graph_quality": result.pose_graph.pose_graph_quality,
        "background_motion_before_compensation_px": background_motion_before,
        "background_reprojection_after_compensation_px": background_reprojection_after,
        "depth_alignment_mode": result.depth_alignment.alignment_mode.value,
        "depth_alignment_domain": result.depth_alignment.alignment_domain,
        "depth_alignment_connected_frame_ratio": result.depth_alignment.connected_frame_ratio,
        "depth_alignment_global_consistency_error": result.depth_alignment.global_consistency_error,
        "depth_alignment_mean_fitting_error": float(np.mean([item.fitting_error for item in result.depth_alignment.supporting_edges])) if result.depth_alignment.supporting_edges else math.nan,
        "depth_alignment_mean_holdout_error": float(np.mean([item.holdout_error for item in result.depth_alignment.supporting_edges])) if result.depth_alignment.supporting_edges else math.nan,
        "sequence_scale_status": result.sequence_scale_status.value,
        "background_depth_stability_before": stability_before,
        "background_depth_stability_after": stability_after,
        "dynamic_3d_valid_frame_ratio": float(np.mean([item.dynamic_3d_valid for item in result.frame_validity.values()])),
        "dynamic_3d_valid": result.dynamic_3d_valid,
        "sequence_geometry_quality": result.quality,
        "scene_cut_flags": cut_flags,
        "scene_cut_details": cut_details,
        "foreground_diagnostics": foreground_diagnostics,
        "semantic_scale_prior_used": False,
        "scale_prior_config_read": False,
        "visualization_depth_used_for_alignment": False,
        "formal_dynamic_anomaly_residuals_computed": False,
        "metric_camera_trajectory": False,
        "geometry_failure_is_forgery": False,
    }
    cache_manifest = _save_shared_geometry_cache(
        clip_id=clip_id,
        video_path=video_path,
        extracted=extracted,
        camera=camera,
        depths=depths,
        foreground_masks=foreground_masks,
        result=result,
        scene_cut_flags=cut_flags,
        geometry_quality=geometry_quality,
        output_dir=output_dir,
    )
    geometry_quality["shared_geometry_cache_manifest"] = str(cache_manifest)
    (output_dir / "geometry_quality.json").write_text(
        json.dumps(_json_safe(geometry_quality), indent=2) + "\n", encoding="utf-8"
    )
    _save_diagnostics(result, output_dir)
    print(
        f"{clip_id}: pose_connected={result.pose_graph.connected_frame_ratio:.3f}, "
        f"depth_connected={result.depth_alignment.connected_frame_ratio:.3f}, "
        f"scale_status={result.sequence_scale_status.value}, "
        f"dynamic_3d_valid={result.dynamic_3d_valid}"
    )
    return geometry_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_path", type=Path)
    parser.add_argument("--clip_id")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sequence_geometry_stabilization",
    )
    parser.add_argument("--model_path", type=Path, default=PROJECT_ROOT / "checkpoints/yolov8n.pt")
    parser.add_argument("--depth_model", default="depth-anything/Depth-Anything-V2-Small")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video_path is not None:
        clip_id = args.clip_id or args.video_path.stem
        specs = ((clip_id, args.video_path, args.start_frame, args.num_frames),)
    else:
        base = PROJECT_ROOT / "data" / "tests_videos" / "tests_real_videos"
        specs = tuple(
            (clip_id, base / filename, start, count)
            for clip_id, filename, start, count in DEFAULT_SUITE
        )
    detector = RealObjectProvider(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold,
        default_depth=1.0,
        device=args.device,
        skip_unknown_scale_prior=False,
    )
    depth_estimator = RealDepthProvider(
        model_name=args.depth_model,
        device=args.device,
        invert_depth=True,
    )
    summaries = []
    for clip_id, video_path, start, count in specs:
        summaries.append(
            run_sequence_geometry_stabilization(
                clip_id=clip_id,
                video_path=video_path,
                start_frame=start,
                num_frames=count,
                output_dir=args.output_root / clip_id,
                model_path=args.model_path,
                depth_model=args.depth_model,
                confidence_threshold=args.confidence_threshold,
                device=args.device,
                object_provider=detector,
                depth_provider=depth_estimator,
            )
        )
    (args.output_root / "suite_summary.json").write_text(
        json.dumps(_json_safe(summaries), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
