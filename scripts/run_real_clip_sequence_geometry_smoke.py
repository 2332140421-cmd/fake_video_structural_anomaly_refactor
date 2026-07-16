#!/usr/bin/env python3
"""Run a real short-clip P3-0 sequence-geometry smoke test.

This script estimates geometry quality only. It does not compute dynamic
forgery residuals, metric camera trajectory, 3D velocity, or classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from enum import Enum
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
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON  # noqa: E402
from semantic3d.providers import BaseObjectProvider  # noqa: E402
from semantic3d.real_object_provider import RealObjectProvider  # noqa: E402
from semantic3d.reconstruction import Shared3DFrameBuilder  # noqa: E402
from semantic3d.sequence_geometry import (  # noqa: E402
    DepthAlignmentMode,
    HistogramFeatureSceneCutDetector,
    LegacyDepthPoseSequenceAdapter,
    RelativePoseObservation,
    SequenceGeometryQuality,
    build_foreground_mask,
    estimate_depth_alignment_from_correspondences,
)
from semantic3d.sequence_geometry.provider import (  # noqa: E402
    BackgroundPoseEstimate,
    estimate_background_relative_pose,
)
from semantic3d.shared_3d_observation import Shared3DFrameObservation  # noqa: E402


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _extract_clip_frames(
    video_path: Path,
    start_frame: int,
    num_frames: int,
    output_dir: Path,
) -> list[tuple[int, Path, np.ndarray]]:
    if start_frame < 0 or not 1 <= num_frames <= 64:
        raise ValueError("start_frame must be >= 0 and num_frames must be in [1, 64].")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    extracted: list[tuple[int, Path, np.ndarray]] = []
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        for offset in range(num_frames):
            success, image = capture.read()
            if not success or image is None:
                break
            frame_index = start_frame + offset
            path = frame_dir / f"frame_{frame_index:06d}.png"
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"Could not save decoded frame: {path}")
            extracted.append((frame_index, path, image))
    finally:
        capture.release()
    if not extracted:
        raise ValueError("No frames were decoded from the requested clip.")
    return extracted


def _approximate_camera(width: int, height: int, focal_factor: float) -> CameraObservation:
    focal = float(focal_factor * max(width, height))
    K = np.asarray(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return CameraObservation.from_parameters(
        K=K,
        image_width=width,
        image_height=height,
        intrinsics_source="approximate",
        quality=0.5,
        metadata={
            "focal_length_factor": focal_factor,
            "metric_calibration": False,
            "P3_0_smoke_only": True,
        },
    )


def _load_external_poses(path: Optional[Path]) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    if path is None:
        return {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("poses", payload)
    poses = {int(index): np.asarray(matrix, dtype=float) for index, matrix in raw.items() if str(index).lstrip("-").isdigit()}
    return poses, {
        "pose_source": payload.get("pose_source", "external_pose_json"),
        "pose_scale_compatible_with_depth": bool(payload.get("pose_scale_compatible_with_depth", False)),
    }


def _pose_record_from_estimate(
    estimate: BackgroundPoseEstimate,
    T_world_from_previous: Optional[np.ndarray],
) -> tuple[RelativePoseObservation, Optional[np.ndarray]]:
    if not estimate.valid or estimate.T_current_from_previous is None:
        return (
            RelativePoseObservation.missing(
                estimate.target_frame_index,
                estimate.missing_reason,
                source_frame_index=estimate.source_frame_index,
                pose_source="opencv_background_visual_odometry",
                background_support_count=estimate.support_count,
                background_inlier_ratio=estimate.inlier_ratio,
                metadata=estimate.metadata,
            ),
            None,
        )
    if T_world_from_previous is None:
        return (
            RelativePoseObservation.missing(
                estimate.target_frame_index,
                "pose_chain_broken",
                source_frame_index=estimate.source_frame_index,
                pose_source="opencv_background_visual_odometry",
                background_support_count=estimate.support_count,
                background_inlier_ratio=estimate.inlier_ratio,
                metadata=estimate.metadata,
            ),
            None,
        )
    current_twc = T_world_from_previous @ np.linalg.inv(estimate.T_current_from_previous)
    pose_source = (
        "opencv_lk_static_identity"
        if estimate.is_static_identity
        else "opencv_lk_essential_recover_pose"
    )
    return (
        RelativePoseObservation.from_transforms(
            source_frame_index=estimate.source_frame_index,
            target_frame_index=estimate.target_frame_index,
            T_world_from_camera=current_twc,
            relative_pose_from_previous=estimate.T_current_from_previous,
            pose_source=pose_source,
            pose_quality=estimate.quality,
            background_support_count=estimate.support_count,
            background_inlier_ratio=estimate.inlier_ratio,
            reprojection_error=estimate.reprojection_error,
            metadata=estimate.metadata,
        ),
        current_twc,
    )


def _clip_payload(clip, quality: SequenceGeometryQuality) -> dict[str, Any]:
    return {
        "schema": "Shared3DClipObservation/P3-0",
        "claim": "shared sequence-level 3D geometry foundation, not dynamic anomaly detection",
        "video_id": clip.video_id,
        "clip_id": clip.clip_id,
        "frame_indices": list(clip.frame_indices),
        "reference_frame_index": clip.reference_frame_index,
        "provider_name": clip.provider_name,
        "sequence_scale_status": clip.sequence_scale_status.value,
        "scale_allows_dynamic_3d": clip.scale_allows_dynamic_3d,
        "allows_dynamic_3d": clip.allows_dynamic_3d,
        "valid": clip.valid,
        "quality": clip.quality,
        "missing_reason": clip.missing_reason,
        "scene_cut_flags": _json_safe(clip.scene_cut_flags),
        "background_track_ids": list(clip.background_track_ids),
        "foreground_object_ids": list(clip.foreground_object_ids),
        "poses": [
            {
                "source_frame_index": pose.source_frame_index,
                "target_frame_index": pose.target_frame_index,
                "T_world_from_camera": _json_safe(pose.T_world_from_camera),
                "T_camera_from_world": _json_safe(pose.T_camera_from_world),
                "relative_pose_from_previous": _json_safe(pose.relative_pose_from_previous),
                "camera_center_world": _json_safe(pose.camera_center_world),
                "pose_source": pose.pose_source,
                "pose_quality": pose.pose_quality,
                "background_support_count": pose.background_support_count,
                "background_inlier_ratio": pose.background_inlier_ratio,
                "reprojection_error": _json_safe(pose.reprojection_error),
                "valid": pose.valid,
                "missing_reason": pose.missing_reason,
                "metadata": _json_safe(pose.metadata),
            }
            for pose in clip.relative_poses
        ],
        "depth_alignments": [
            {
                "source_frame": item.source_frame,
                "target_frame": item.target_frame,
                "alignment_mode": item.alignment_mode.value,
                "scale": _json_safe(item.scale),
                "shift": _json_safe(item.shift),
                "support_count": item.support_count,
                "inlier_ratio": item.inlier_ratio,
                "fitting_error": _json_safe(item.fitting_error),
                "quality": item.quality,
                "valid": item.valid,
                "missing_reason": item.missing_reason,
                "metadata": _json_safe(item.metadata),
            }
            for item in clip.depth_alignment_observations
        ],
        "frames": [
            {
                "frame_index": frame.frame_index,
                "valid": frame.valid,
                "quality": frame.quality,
                "missing_reason": frame.missing_reason,
                "object_count": len(frame.objects),
                "valid_object_count": sum(obj.valid for obj in frame.objects),
                "camera_intrinsics_source": frame.camera.intrinsics_source,
                "camera_pose_source": frame.camera.pose_source,
                "camera_pose_valid": frame.camera.pose_valid,
                "depth_provider": frame.depth.provider_name,
                "depth_representation": frame.depth.depth_representation.value,
                "depth_scale_status": frame.depth.scale_status.value,
            }
            for frame in clip.frames
        ],
        "sequence_geometry_quality": _json_safe(quality.to_dict()),
        "metadata": _json_safe(clip.metadata),
    }


def _save_plots(clip, quality: SequenceGeometryQuality, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))
    valid_centers = [
        (pose.target_frame_index, pose.camera_center_world)
        for pose in clip.relative_poses
        if (
            pose.source_frame_index is not None
            and pose.valid
            and pose.camera_center_world is not None
        )
    ]
    if valid_centers:
        indices = [item[0] for item in valid_centers]
        centers = np.stack([item[1] for item in valid_centers])
        axis.plot(centers[:, 0], centers[:, 2], marker="o", linewidth=1.5)
        for frame_index, center in zip(indices, centers, strict=True):
            axis.annotate(str(frame_index), (center[0], center[2]), fontsize=8)
    else:
        axis.text(
            0.5,
            0.5,
            "Reference gauge only; no valid inter-frame camera trajectory",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("world X (relative pose unit)")
    axis.set_ylabel("world Z (relative pose unit)")
    axis.set_title("Camera Trajectory Diagnostic (Not Metric Unless Externally Calibrated)")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "camera_trajectory.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    valid_transition_poses = [
        pose
        for pose in clip.relative_poses
        if pose.source_frame_index is not None and pose.valid
    ]
    pose_targets = [pose.target_frame_index for pose in valid_transition_poses]
    pose_errors = [pose.reprojection_error for pose in valid_transition_poses]
    if pose_targets:
        axes[0].plot(pose_targets, pose_errors, marker="o", label="background reprojection/epipolar error")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(
            0.5,
            0.5,
            "No valid inter-frame pose evidence",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )
    axes[0].set_ylabel("error (px)")
    axes[0].grid(alpha=0.25)
    alignment_targets = [item.target_frame for item in clip.depth_alignment_observations]
    alignment_errors = [item.fitting_error if item.valid else np.nan for item in clip.depth_alignment_observations]
    axes[1].plot(alignment_targets, alignment_errors, marker="s", color="tab:orange", label="depth alignment fitting error")
    axes[1].set_xlabel("target frame index")
    axes[1].set_ylabel("fit error")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    figure.suptitle(f"Sequence Geometry Diagnostics, quality={quality.quality:.3f}")
    figure.tight_layout()
    figure.savefig(output_dir / "reprojection_diagnostics.png", dpi=180)
    plt.close(figure)


def run_real_clip_sequence_geometry_smoke(
    *,
    video_path: Path,
    start_frame: int,
    num_frames: int,
    output_dir: Path,
    object_provider: Optional[BaseObjectProvider] = None,
    depth_provider: Optional[BaseDepthProvider] = None,
    model_path: Path = PROJECT_ROOT / "checkpoints" / "yolov8n.pt",
    depth_model: str = "depth-anything/Depth-Anything-V2-Small",
    confidence_threshold: float = 0.3,
    device: str = "cpu",
    focal_length_factor: float = 1.2,
    depth_alignment_mode: DepthAlignmentMode | str = DepthAlignmentMode.AFFINE_DEPTH,
    external_pose_json: Optional[Path] = None,
) -> tuple[Any, SequenceGeometryQuality]:
    """Build sequence geometry from real frames and save P3-0 diagnostics."""

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted = _extract_clip_frames(video_path, start_frame, num_frames, output_dir)
    height, width = extracted[0][2].shape[:2]
    camera = _approximate_camera(width, height, focal_length_factor)
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
    shared_frames: list[Shared3DFrameObservation] = []
    images: dict[int, np.ndarray] = {}
    for frame_index, frame_path, image in extracted:
        images[frame_index] = image
        objects = detector.predict(frame_path, frame_index, width, height)
        depth = depth_estimator.predict_observation(frame_path, frame_index=frame_index)
        depth.require_geometry_depth()
        frame = FrameObservationJSON(
            frame_index=frame_index,
            frame_id=f"{video_path.stem}_frame_{frame_index:06d}",
            width=width,
            height=height,
            objects=list(objects),
            image_path=str(frame_path),
            depth_metadata={
                "provider_name": depth.provider_name,
                "canonical_geometry_depth": True,
                "legacy_normalized_depth_used": False,
            },
        )
        shared_frames.append(
            Shared3DFrameBuilder().build(
                video_id=video_path.stem,
                frame=frame,
                depth=depth,
                camera=camera,
            )
        )

    foreground = {frame.frame_index: build_foreground_mask(frame) for frame in shared_frames}
    scene_detector = HistogramFeatureSceneCutDetector()
    scene_cut_flags: dict[int, bool] = {shared_frames[0].frame_index: False}
    cut_decisions: list[dict[str, Any]] = []
    pose_estimates: list[BackgroundPoseEstimate] = []
    track_rows: list[dict[str, Any]] = []
    alignments = []
    external_poses, external_metadata = _load_external_poses(external_pose_json)
    pose_map: dict[int, np.ndarray] = {}
    pose_records: dict[int, RelativePoseObservation] = {}
    first_index = shared_frames[0].frame_index
    if external_poses and first_index not in external_poses:
        raise ValueError(
            "External pose JSON does not contain the requested reference frame "
            f"{first_index}; refusing to fabricate an identity pose."
        )
    reference_pose = external_poses.get(first_index, np.eye(4))
    reference_source = external_metadata.get("pose_source", "opencv_sequence_reference_gauge")
    pose_map[first_index] = reference_pose
    pose_records[first_index] = RelativePoseObservation.from_transforms(
        source_frame_index=None,
        target_frame_index=first_index,
        T_world_from_camera=reference_pose,
        relative_pose_from_previous=np.eye(4),
        pose_source=reference_source,
        pose_quality=0.5 if not external_poses else 1.0,
        background_support_count=1,
        background_inlier_ratio=1.0,
        reprojection_error=0.0,
        metadata={
            "reference_coordinate_gauge": True,
            "background_support_ratio": 1.0,
            "metric_trajectory": False,
        },
    )

    retained_count = len(shared_frames)
    for position in range(1, len(shared_frames)):
        previous = shared_frames[position - 1]
        current = shared_frames[position]
        decision = scene_detector.detect(
            images[previous.frame_index],
            images[current.frame_index],
            source_frame_index=previous.frame_index,
            target_frame_index=current.frame_index,
        )
        scene_cut_flags[current.frame_index] = decision.is_cut
        cut_decisions.append(_json_safe(decision.__dict__))
        if decision.is_cut:
            retained_count = position
            break
        estimate = estimate_background_relative_pose(
            images[previous.frame_index],
            images[current.frame_index],
            camera.K,
            source_frame_index=previous.frame_index,
            target_frame_index=current.frame_index,
            source_foreground_mask=foreground[previous.frame_index].mask,
            target_foreground_mask=foreground[current.frame_index].mask,
        )
        pose_estimates.append(estimate)
        track_rows.extend(dict(row) for row in estimate.track_rows)
        if external_poses:
            current_twc = external_poses.get(current.frame_index)
            previous_twc = external_poses.get(previous.frame_index)
            if current_twc is not None and previous_twc is not None:
                relative = np.linalg.inv(current_twc) @ previous_twc
                pose_record = RelativePoseObservation.from_transforms(
                    source_frame_index=previous.frame_index,
                    target_frame_index=current.frame_index,
                    T_world_from_camera=current_twc,
                    relative_pose_from_previous=relative,
                    pose_source=str(external_metadata.get("pose_source", "external_pose_json")),
                    pose_quality=estimate.quality if estimate.valid else 0.5,
                    background_support_count=max(1, estimate.support_count),
                    background_inlier_ratio=estimate.inlier_ratio if estimate.valid else 0.0,
                    reprojection_error=estimate.reprojection_error if estimate.valid else 0.0,
                    metadata={
                        **dict(estimate.metadata),
                        "external_pose": True,
                    },
                )
                pose_map[current.frame_index] = current_twc
                pose_records[current.frame_index] = pose_record
            else:
                pose_records[current.frame_index] = RelativePoseObservation.missing(
                    current.frame_index,
                    "external_pose_missing",
                    source_frame_index=previous.frame_index,
                    pose_source=str(external_metadata.get("pose_source", "external_pose_json")),
                )
        else:
            pose_record, current_twc = _pose_record_from_estimate(
                estimate, pose_map.get(previous.frame_index)
            )
            pose_records[current.frame_index] = pose_record
            if current_twc is not None:
                pose_map[current.frame_index] = current_twc

        inlier_rows = [row for row in estimate.track_rows if bool(row["inlier"])]
        source_points = np.asarray(
            [[row["source_x"], row["source_y"]] for row in inlier_rows], dtype=float
        ).reshape(-1, 2)
        target_points = np.asarray(
            [[row["target_x"], row["target_y"]] for row in inlier_rows], dtype=float
        ).reshape(-1, 2)
        alignment = estimate_depth_alignment_from_correspondences(
            previous.depth,
            current.depth,
            source_points,
            target_points,
            source_frame=previous.frame_index,
            target_frame=current.frame_index,
            mode=depth_alignment_mode,
            source_foreground_mask=foreground[previous.frame_index].mask,
            target_foreground_mask=foreground[current.frame_index].mask,
            metadata={
                "pose_valid": estimate.valid,
                "pose_source": pose_records[current.frame_index].pose_source,
                "fit_assumption": "stable_background_correspondence_small_interframe_motion",
                "not_optimized_for_anomaly_residual": True,
            },
        )
        alignments.append(alignment)

    retained_frames = shared_frames[:retained_count]
    retained_indices = [frame.frame_index for frame in retained_frames]
    retained_pose_records = {
        index: pose_records.get(
            index,
            RelativePoseObservation.missing(index, "pose_not_computed_due_to_cut"),
        )
        for index in retained_indices
    }
    all_estimated_identity = bool(pose_estimates) and all(
        estimate.valid and estimate.is_static_identity for estimate in pose_estimates
    )
    pose_scale_compatible = bool(
        external_metadata.get("pose_scale_compatible_with_depth", False)
        if external_poses
        else all_estimated_identity
    )
    adapter = LegacyDepthPoseSequenceAdapter(
        T_world_from_camera_by_frame=pose_map,
        relative_pose_observations=retained_pose_records,
        depth_alignments=alignments,
        scene_cut_flags=scene_cut_flags,
        background_track_ids=tuple(str(row["track_id"]) for row in track_rows),
        pose_source=(
            str(external_metadata.get("pose_source", "external_pose_json"))
            if external_poses
            else "opencv_background_visual_odometry"
        ),
        pose_scale_compatible_with_depth=pose_scale_compatible,
        video_id=video_path.stem,
        clip_id=f"{video_path.stem}_{start_frame:06d}_{retained_indices[-1]:06d}",
        metadata={
            "K_source": "approximate",
            "canonical_depth": True,
            "legacy_normalized_depth_used": False,
            "foreground_exclusion": True,
            "excluded_foreground_ratio_by_frame": {
                index: foreground[index].excluded_foreground_ratio for index in retained_indices
            },
            "foreground_mask_quality_by_frame": {
                index: foreground[index].quality for index in retained_indices
            },
            "scene_cut_decisions": cut_decisions,
            "monocular_pose_translation_scale": (
                "external" if external_poses else "identity_or_direction_only"
            ),
        },
    )
    clip = adapter.predict_clip(retained_frames, retained_indices)
    quality = SequenceGeometryQuality.from_clip(clip)
    payload = _clip_payload(clip, quality)
    (output_dir / "shared_3d_clip.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    camera_rows = [
        {
            "frame_index": pose.target_frame_index,
            "camera_center_x": None if pose.camera_center_world is None else pose.camera_center_world[0],
            "camera_center_y": None if pose.camera_center_world is None else pose.camera_center_world[1],
            "camera_center_z": None if pose.camera_center_world is None else pose.camera_center_world[2],
            "pose_source": pose.pose_source,
            "pose_quality": pose.pose_quality,
            "valid": pose.valid,
            "missing_reason": pose.missing_reason,
        }
        for pose in clip.relative_poses
    ]
    _write_csv(
        output_dir / "camera_trajectory.csv",
        camera_rows,
        ("frame_index", "camera_center_x", "camera_center_y", "camera_center_z", "pose_source", "pose_quality", "valid", "missing_reason"),
    )
    relative_rows = [
        {
            "source_frame_index": pose.source_frame_index,
            "target_frame_index": pose.target_frame_index,
            "pose_source": pose.pose_source,
            "pose_quality": pose.pose_quality,
            "background_support_count": pose.background_support_count,
            "background_inlier_ratio": pose.background_inlier_ratio,
            "reprojection_error": pose.reprojection_error,
            "is_identity": pose.is_identity_relative_pose,
            "valid": pose.valid,
            "missing_reason": pose.missing_reason,
            "relative_pose_from_previous": json.dumps(_json_safe(pose.relative_pose_from_previous)),
        }
        for pose in clip.relative_poses
    ]
    _write_csv(
        output_dir / "relative_poses.csv",
        relative_rows,
        ("source_frame_index", "target_frame_index", "pose_source", "pose_quality", "background_support_count", "background_inlier_ratio", "reprojection_error", "is_identity", "valid", "missing_reason", "relative_pose_from_previous"),
    )
    alignment_rows = [
        {
            "source_frame": item.source_frame,
            "target_frame": item.target_frame,
            "alignment_mode": item.alignment_mode.value,
            "scale": item.scale,
            "shift": item.shift,
            "support_count": item.support_count,
            "inlier_ratio": item.inlier_ratio,
            "fitting_error": item.fitting_error,
            "quality": item.quality,
            "valid": item.valid,
            "missing_reason": item.missing_reason,
        }
        for item in clip.depth_alignment_observations
    ]
    _write_csv(
        output_dir / "depth_alignment.csv",
        alignment_rows,
        ("source_frame", "target_frame", "alignment_mode", "scale", "shift", "support_count", "inlier_ratio", "fitting_error", "quality", "valid", "missing_reason"),
    )
    track_fields = (
        "track_id",
        "source_frame_index",
        "target_frame_index",
        "source_x",
        "source_y",
        "target_x",
        "target_y",
        "inlier",
        "reprojection_error",
    )
    _write_csv(output_dir / "background_tracks.csv", track_rows, track_fields)
    _write_csv(
        output_dir / "background_reprojection.csv",
        [row for row in track_rows if bool(row["inlier"])],
        track_fields,
    )
    quality_payload = {
        **_json_safe(quality.to_dict()),
        "pose_source": adapter.pose_source,
        "K_source": "approximate",
        "depth_alignment_mode": DepthAlignmentMode(depth_alignment_mode).value,
        "sequence_scale_status": clip.sequence_scale_status.value,
        "allows_dynamic_3d": clip.allows_dynamic_3d,
        "metric_camera_trajectory": False,
        "dynamic_anomaly_residuals_computed": False,
        "valid_frames": [frame.frame_index for frame in clip.frames if frame.valid],
        "invalid_frames": [
            {"frame_index": frame.frame_index, "reason": frame.missing_reason}
            for frame in clip.frames
            if not frame.valid
        ],
    }
    (output_dir / "sequence_geometry_quality.json").write_text(
        json.dumps(quality_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _save_plots(clip, quality, output_dir)
    return clip, quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_path",
        type=Path,
        default=PROJECT_ROOT / "data" / "tests_videos" / "tests_real_videos" / "real_1.mp4",
    )
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "real_clip_sequence_geometry_smoke",
    )
    parser.add_argument("--model_path", type=Path, default=PROJECT_ROOT / "checkpoints" / "yolov8n.pt")
    parser.add_argument("--depth_model", default="depth-anything/Depth-Anything-V2-Small")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--focal_length_factor", type=float, default=1.2)
    parser.add_argument(
        "--depth_alignment_mode",
        choices=[mode.value for mode in DepthAlignmentMode if mode != DepthAlignmentMode.UNSUPPORTED],
        default=DepthAlignmentMode.AFFINE_DEPTH.value,
    )
    parser.add_argument("--external_pose_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clip, quality = run_real_clip_sequence_geometry_smoke(
        video_path=args.video_path,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        output_dir=args.output_dir,
        model_path=args.model_path,
        depth_model=args.depth_model,
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        focal_length_factor=args.focal_length_factor,
        depth_alignment_mode=args.depth_alignment_mode,
        external_pose_json=args.external_pose_json,
    )
    print(f"pose_source={clip.metadata.get('pose_source', 'unknown')}")
    print("K_source=approximate")
    print(f"sequence_scale_status={clip.sequence_scale_status.value}")
    print(f"valid_pose_ratio={quality.valid_pose_ratio:.6f}")
    print(f"depth_alignment_valid_ratio={quality.depth_alignment_valid_ratio:.6f}")
    print(f"sequence_geometry_quality={quality.quality:.6f}")
    print(f"allows_dynamic_3d={clip.allows_dynamic_3d}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
