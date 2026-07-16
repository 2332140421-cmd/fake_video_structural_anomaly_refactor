#!/usr/bin/env python3
"""Run P3-0.6/P3-A readiness and independent-point smoke on cached geometry.

The script never estimates depth, intrinsics, or camera pose. It consumes the
P3-0.5 shared geometry cache and reports geometry QA, not forged-video labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
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

from semantic3d.dynamic_3d import (  # noqa: E402
    Dynamic3DReadinessThresholds,
    ExistingInterfaceAdapter,
    PointTrack2DObservation,
    assess_dynamic_3d_readiness,
    compute_dynamic_reprojection_residual,
    compute_track_3d_continuity_residuals,
    load_shared_geometry_cache,
    reconstruct_point_tracks_3d,
    summarize_point_track_coverage,
)
from semantic3d.geometry.backprojection import backproject_pixel  # noqa: E402


DEFAULT_CLIPS = ("static_camera", "slowly_moving_camera", "clearly_moving_camera")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_images(frame_paths: Mapping[int, Path]) -> dict[int, np.ndarray]:
    images = {}
    for index, path in sorted(frame_paths.items()):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read cached source frame: {path}")
        images[index] = image
    return images


def _klt_callback(
    images: Mapping[int, np.ndarray],
    object_track_id: str,
    initial_points: Optional[np.ndarray],
) -> list[Mapping[str, Any]]:
    """Track stable image observations; coordinates never come from 3D projection."""

    indices = tuple(sorted(images))
    first_gray = cv2.cvtColor(images[indices[0]], cv2.COLOR_BGR2GRAY)
    points = (
        np.asarray(initial_points, dtype=np.float32).reshape(-1, 1, 2)
        if initial_points is not None
        else cv2.goodFeaturesToTrack(
            first_gray,
            maxCorners=160,
            qualityLevel=0.015,
            minDistance=7.0,
        )
    )
    if points is None or not len(points):
        return []
    ids = [f"klt_{index:04d}" for index in range(len(points))]
    active = np.ones(len(points), dtype=bool)
    current = points.reshape(-1, 2).astype(np.float32)
    rows: list[Mapping[str, Any]] = []

    def append_row(point_index: int, frame_index: int, uv: Optional[np.ndarray], valid: bool, reason: str = "") -> None:
        rows.append(
            {
                "point_id": ids[point_index],
                "object_track_id": object_track_id,
                "frame_index": frame_index,
                "pixel_uv": None if uv is None else (float(uv[0]), float(uv[1])),
                "visibility": "visible" if valid else "unknown",
                "occlusion_status": "visible" if valid else "tracker_lost",
                "tracking_confidence": 1.0 if valid else 0.0,
                "source_tracker": "opencv_klt_forward_backward",
                "valid": valid,
                "missing_reason": reason,
                "metadata": {
                    "independent_observation": True,
                    "generated_from_projection": False,
                    "forward_backward_checked": frame_index != indices[0],
                },
            }
        )

    for point_index, uv in enumerate(current):
        append_row(point_index, indices[0], uv, True)
    previous_gray = first_gray
    for frame_index in indices[1:]:
        current_gray = cv2.cvtColor(images[frame_index], cv2.COLOR_BGR2GRAY)
        active_indices = np.flatnonzero(active)
        if not len(active_indices):
            for point_index in range(len(ids)):
                append_row(point_index, frame_index, None, False, "tracker_lost")
            previous_gray = current_gray
            continue
        source_points = current[active_indices].reshape(-1, 1, 2)
        target_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray, current_gray, source_points, None
        )
        if target_points is None or forward_status is None:
            target_points = np.full_like(source_points, np.nan)
            forward_status = np.zeros((len(active_indices), 1), dtype=np.uint8)
            backward_points = np.full_like(source_points, np.nan)
            backward_status = np.zeros_like(forward_status)
        else:
            backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                current_gray, previous_gray, target_points, None
            )
            if backward_points is None or backward_status is None:
                backward_points = np.full_like(source_points, np.nan)
                backward_status = np.zeros_like(forward_status)
        forward_xy = target_points.reshape(-1, 2)
        backward_xy = backward_points.reshape(-1, 2)
        fb_error = np.linalg.norm(backward_xy - source_points.reshape(-1, 2), axis=1)
        valid_local = forward_status.reshape(-1).astype(bool)
        valid_local &= backward_status.reshape(-1).astype(bool)
        valid_local &= np.isfinite(forward_xy).all(axis=1) & (fb_error <= 1.5)
        height, width = current_gray.shape
        valid_local &= (
            (forward_xy[:, 0] >= 0)
            & (forward_xy[:, 0] < width)
            & (forward_xy[:, 1] >= 0)
            & (forward_xy[:, 1] < height)
        )
        local_by_global = {global_index: local for local, global_index in enumerate(active_indices)}
        for point_index in range(len(ids)):
            local = local_by_global.get(point_index)
            if local is None or not valid_local[local]:
                active[point_index] = False
                append_row(point_index, frame_index, None, False, "tracker_lost")
                continue
            current[point_index] = forward_xy[local]
            append_row(point_index, frame_index, current[point_index], True)
        previous_gray = current_gray
    return rows


def _background_3d_stability(
    points_2d: Sequence[PointTrack2DObservation],
    cache: Any,
) -> tuple[float, float]:
    """Compare common-frame KLT point scatter before and after depth alignment."""

    K = cache.clip.frames[0].camera.K
    assert K is not None
    by_point_before: dict[str, list[np.ndarray]] = defaultdict(list)
    by_point_after: dict[str, list[np.ndarray]] = defaultdict(list)
    frames = {frame.frame_index: frame for frame in cache.clip.frames}
    for point in points_2d:
        if not point.valid or point.pixel_uv is None:
            continue
        frame = frames[point.frame_index]
        transform = cache.clip.T_world_from_camera_by_frame.get(point.frame_index)
        if transform is None:
            continue
        u, v = point.pixel_uv
        row, column = int(round(v)), int(round(u))
        raw = cache.per_frame_geometry_depth[point.frame_index]
        aligned = frame.depth.require_geometry_depth()
        if not (0 <= row < raw.shape[0] and 0 <= column < raw.shape[1]):
            continue
        if cache.foreground_masks[point.frame_index][row, column]:
            continue
        for depth_map, destination in ((raw, by_point_before), (aligned, by_point_after)):
            depth = float(depth_map[row, column])
            if not math.isfinite(depth) or depth <= 0.0:
                continue
            camera_point = backproject_pixel(u, v, depth, K, point_id=point.point_id)
            if not camera_point.valid:
                continue
            world = np.asarray(transform) @ np.concatenate([camera_point.as_array(), [1.0]])
            destination[point.point_id].append(world[:3] / world[3])

    def stability(groups: Mapping[str, Sequence[np.ndarray]]) -> float:
        values = []
        for samples in groups.values():
            if len(samples) < 3:
                continue
            array = np.asarray(samples, dtype=float)
            center = np.median(array, axis=0)
            scale = max(float(np.median(np.linalg.norm(array, axis=1))), 1e-8)
            values.append(float(np.median(np.linalg.norm(array - center, axis=1)) / scale))
        return float(np.median(values)) if values else float("nan")

    return stability(by_point_before), stability(by_point_after)


def _track_rows(points: Sequence[PointTrack2DObservation]) -> list[dict[str, Any]]:
    return [
        {
            "point_id": point.point_id,
            "object_track_id": point.object_track_id,
            "frame_index": point.frame_index,
            "u": None if point.pixel_uv is None else point.pixel_uv[0],
            "v": None if point.pixel_uv is None else point.pixel_uv[1],
            "visibility": point.visibility.value,
            "occlusion_status": point.occlusion_status,
            "tracking_confidence": point.tracking_confidence,
            "source_tracker": point.source_tracker,
            "valid": point.valid,
            "missing_reason": point.missing_reason,
        }
        for point in points
    ]


def _save_plots(
    output_dir: Path,
    images: Mapping[int, np.ndarray],
    points_2d: Sequence[PointTrack2DObservation],
    track_residuals: Sequence[Any],
    reprojection: Sequence[Any],
) -> None:
    figure, axis = plt.subplots(figsize=(10, 7))
    first = images[min(images)]
    axis.imshow(cv2.cvtColor(first, cv2.COLOR_BGR2RGB))
    by_point: dict[str, list[PointTrack2DObservation]] = defaultdict(list)
    for point in points_2d:
        if point.valid:
            by_point[point.point_id].append(point)
    for samples in list(by_point.values())[:50]:
        samples.sort(key=lambda item: item.frame_index)
        uv = np.asarray([item.pixel_uv for item in samples], dtype=float)
        axis.plot(uv[:, 0], uv[:, 1], linewidth=0.8, alpha=0.65)
    axis.set_title("Independent KLT point tracks, coordinates shown in image plane")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output_dir / "track_diagnostics.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    valid_tracks = [item for item in track_residuals if item.valid]
    axes[0].scatter(
        [item.current_frame_index for item in valid_tracks],
        [item.raw_residual for item in valid_tracks],
        s=12,
        alpha=0.7,
    )
    axes[0].set_title("3D continuity interface output")
    axes[0].set_xlabel("frame")
    axes[0].set_ylabel("raw second difference")
    valid_reprojection = [item for item in reprojection if item.valid]
    axes[1].scatter(
        [item.target_frame_index for item in valid_reprojection],
        [item.pixel_error for item in valid_reprojection],
        s=12,
        alpha=0.7,
    )
    axes[1].set_title("Camera geometry QA reprojection")
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("pixel error")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "reprojection_diagnostics.png", dpi=180)
    plt.close(figure)


def run_real_dynamic_3d_smoke(
    *,
    geometry_cache_manifest: Path,
    output_dir: Path,
    thresholds: Optional[Dynamic3DReadinessThresholds] = None,
) -> dict[str, Any]:
    """Consume one shared cache and save readiness/track diagnostics."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_shared_geometry_cache(geometry_cache_manifest)
    images = _load_images(cache.frame_paths)
    tracker = ExistingInterfaceAdapter(
        _klt_callback, provider_name="opencv_klt_forward_backward"
    )
    first_index = min(images)
    first_gray = cv2.cvtColor(images[first_index], cv2.COLOR_BGR2GRAY)
    background_feature_mask = (
        ~np.asarray(cache.foreground_masks[first_index], dtype=bool)
    ).astype(np.uint8) * 255
    initial_points = cv2.goodFeaturesToTrack(
        first_gray,
        maxCorners=160,
        qualityLevel=0.015,
        minDistance=7.0,
        mask=background_feature_mask,
    )
    points_2d = tracker.track(
        images,
        object_track_id="scene_geometry_qa_points",
        initial_points=initial_points,
    )
    coverage, mean_length, track_count = summarize_point_track_coverage(
        points_2d, len(cache.clip.frame_indices)
    )
    background_before, background_after = _background_3d_stability(points_2d, cache)
    quality = cache.geometry_quality
    transition_count = max(len(cache.clip.frame_indices) - 1, 1)
    motion_counts = dict(quality.get("motion_regime_counts_adjacent", {}))
    static_ratio = float(motion_counts.get("static_camera", 0) / transition_count)
    rotation_ratio = float(
        quality.get("rotation_only_selected_edge_count", 0) / transition_count
    )
    full_ratio = float(quality.get("full_se3_selected_edge_count", 0) / transition_count)
    readiness = assess_dynamic_3d_readiness(
        cache.clip,
        valid_shared_frame_ratio=float(
            np.mean([frame.valid for frame in cache.clip.frames])
        ),
        pose_graph_connected_ratio=float(quality["pose_graph_connected_frame_ratio"]),
        static_pose_ratio=static_ratio,
        rotation_only_ratio=rotation_ratio,
        full_se3_ratio=full_ratio,
        depth_alignment_valid_ratio=float(
            quality["depth_alignment_connected_frame_ratio"]
        ),
        independent_track_coverage=coverage,
        mean_track_length=mean_length,
        reprojection_error_before=float(
            quality["background_motion_before_compensation_px"]
        ),
        reprojection_error_after=float(
            quality["background_reprojection_after_compensation_px"]
        ),
        depth_stability_before=float(quality["background_depth_stability_before"]),
        depth_stability_after=float(quality["background_depth_stability_after"]),
        background_3d_stability_before=background_before,
        background_3d_stability_after=background_after,
        thresholds=thresholds,
        metadata={
            "geometry_cache_manifest": str(geometry_cache_manifest),
            "point_tracker": "opencv_klt_forward_backward",
        },
    )
    points_3d = reconstruct_point_tracks_3d(points_2d, cache.clip, readiness)
    track_residuals = compute_track_3d_continuity_residuals(
        points_3d, scene_cut_flags=cache.clip.scene_cut_flags
    )
    two_by_key = {(point.point_id, point.frame_index): point for point in points_2d}
    three_by_key = {(point.point_id, point.frame_index): point for point in points_3d}
    pose_by_frame = {pose.target_frame_index: pose for pose in cache.clip.relative_poses}
    reprojection = []
    for point_id in sorted({point.point_id for point in points_2d}):
        for previous_frame, current_frame in zip(
            cache.clip.frame_indices, cache.clip.frame_indices[1:]
        ):
            previous = three_by_key.get((point_id, previous_frame))
            current = two_by_key.get((point_id, current_frame))
            pose = pose_by_frame[current_frame]
            if previous is None or current is None or pose.relative_pose_from_previous is None:
                continue
            reprojection.append(
                compute_dynamic_reprojection_residual(
                    previous,
                    current,
                    K_current=cache.clip.frames[0].camera.K,
                    image_width=cache.clip.frames[0].image_width,
                    image_height=cache.clip.frames[0].image_height,
                    relative_pose_current_from_previous=pose.relative_pose_from_previous,
                    geometry_mode=readiness.mode,
                    is_background=True,
                )
            )

    readiness_path = output_dir / "dynamic_readiness.json"
    readiness_path.write_text(
        json.dumps(_json_safe(readiness.to_dict()), indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(
        output_dir / "point_tracks_2d.csv",
        _track_rows(points_2d),
        (
            "point_id", "object_track_id", "frame_index", "u", "v", "visibility",
            "occlusion_status", "tracking_confidence", "source_tracker", "valid",
            "missing_reason",
        ),
    )
    rows_3d = [
        {
            "point_id": point.point_id,
            "object_track_id": point.object_track_id,
            "frame_index": point.frame_index,
            "u": None if point.pixel_uv is None else point.pixel_uv[0],
            "v": None if point.pixel_uv is None else point.pixel_uv[1],
            "observed_depth": point.observed_depth,
            "camera_x": None if point.point_3d_camera is None else point.point_3d_camera[0],
            "camera_y": None if point.point_3d_camera is None else point.point_3d_camera[1],
            "camera_z": None if point.point_3d_camera is None else point.point_3d_camera[2],
            "world_x": None if point.point_3d_world is None else point.point_3d_world[0],
            "world_y": None if point.point_3d_world is None else point.point_3d_world[1],
            "world_z": None if point.point_3d_world is None else point.point_3d_world[2],
            "tracking_confidence": point.tracking_confidence,
            "depth_quality": point.depth_quality,
            "reconstruction_quality": point.reconstruction_quality,
            "source_tracker": point.source_tracker,
            "scale_status": point.scale_status.value,
            "geometry_mode": point.geometry_mode.value,
            "valid": point.valid,
            "missing_reason": point.missing_reason,
        }
        for point in points_3d
    ]
    _write_csv(
        output_dir / "point_tracks_3d.csv",
        rows_3d,
        tuple(rows_3d[0]) if rows_3d else (
            "point_id", "object_track_id", "frame_index", "valid", "missing_reason"
        ),
    )
    track_rows = [
        {
            "point_id": item.point_id,
            "object_track_id": item.object_track_id,
            "previous_previous_frame_index": item.previous_previous_frame_index,
            "previous_frame_index": item.previous_frame_index,
            "current_frame_index": item.current_frame_index,
            "coordinate_frame": item.coordinate_frame,
            "first_order_displacement": item.first_order_displacement,
            "raw_residual": item.raw_residual,
            "normalized_residual": item.normalized_residual,
            "raw_evidence_valid": item.raw_evidence.valid,
            "normalized_evidence_valid": item.normalized_evidence.valid,
            "valid": item.valid,
            "missing_reason": item.missing_reason,
        }
        for item in track_residuals
    ]
    _write_csv(
        output_dir / "track_residuals.csv",
        track_rows,
        tuple(track_rows[0]) if track_rows else (
            "point_id", "current_frame_index", "raw_residual", "valid", "missing_reason"
        ),
    )
    reprojection_rows = [
        {
            "point_id": item.point_id,
            "object_track_id": item.object_track_id,
            "source_frame_index": item.source_frame_index,
            "target_frame_index": item.target_frame_index,
            "geometry_mode": item.geometry_mode.value,
            "evidence_type": item.evidence_type.value,
            "predicted_u": None if item.predicted_uv is None else item.predicted_uv[0],
            "predicted_v": None if item.predicted_uv is None else item.predicted_uv[1],
            "observed_u": None if item.observed_uv is None else item.observed_uv[0],
            "observed_v": None if item.observed_uv is None else item.observed_uv[1],
            "pixel_error": item.pixel_error,
            "normalized_pixel_error": item.normalized_pixel_error,
            "diagnostic_valid": item.diagnostic_evidence.valid,
            "formal_residual_valid": item.residual_evidence.valid,
            "valid": item.valid,
            "missing_reason": item.missing_reason,
        }
        for item in reprojection
    ]
    _write_csv(
        output_dir / "reprojection_residuals.csv",
        reprojection_rows,
        tuple(reprojection_rows[0]) if reprojection_rows else (
            "point_id", "source_frame_index", "target_frame_index", "pixel_error",
            "valid", "missing_reason",
        ),
    )
    _save_plots(output_dir, images, points_2d, track_residuals, reprojection)
    valid_3d = sum(point.valid for point in points_3d)
    report = {
        "video_id": cache.clip.video_id,
        "clip_id": cache.clip.clip_id,
        "geometry_mode": readiness.mode.value,
        "dynamic_3d_ready": readiness.dynamic_3d_ready,
        "allows_world_3d": readiness.allows_world_3d,
        "allows_rotation_compensation_only": readiness.mode.value == "rotation_compensated",
        "independent_track_count": track_count,
        "mean_track_length": mean_length,
        "independent_track_coverage": coverage,
        "valid_3d_observation_count": valid_3d,
        "total_3d_observation_count": len(points_3d),
        "valid_3d_observation_ratio": valid_3d / len(points_3d) if points_3d else 0.0,
        "valid_track_residual_count": sum(item.valid for item in track_residuals),
        "valid_reprojection_qa_count": sum(item.valid for item in reprojection),
        "formal_dynamic_reprojection_evidence_count": sum(
            item.residual_evidence.valid for item in reprojection
        ),
        "track_missing_reasons": dict(
            Counter(item.missing_reason for item in track_residuals if not item.valid)
        ),
        "reprojection_missing_reasons": dict(
            Counter(item.missing_reason for item in reprojection if not item.valid)
        ),
        "background_3d_stability_before": background_before,
        "background_3d_stability_after": background_after,
        "qa_outputs": [
            "dynamic_readiness",
            "background_3d_stability",
            "point_tracks_2d",
            "camera_reprojection_diagnostics",
        ],
        "formal_residual_interfaces": [
            "Track3DContinuityResidual",
            "DynamicReprojectionResidual",
        ],
        "formal_real_anomaly_evidence": False,
        "formal_real_anomaly_evidence_reason": (
            "KLT points are scene geometry QA points without object association or "
            "an independently fitted foreground motion history model."
        ),
        "geometry_failure_is_forgery": False,
        "truth_label_or_threshold_computed": False,
        "shared_clip_reused": True,
        "depth_reestimated": False,
        "intrinsics_reestimated": False,
        "pose_reestimated": False,
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{cache.clip.clip_id}: mode={readiness.mode.value}, "
        f"ready={readiness.dynamic_3d_ready}, tracks={track_count}, "
        f"valid_3d={valid_3d}/{len(points_3d)}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry_root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "sequence_geometry_stabilization",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "real_dynamic_3d_smoke",
    )
    parser.add_argument(
        "--readiness_config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dynamic_3d_readiness.yaml",
    )
    parser.add_argument("--clip_id", action="append", dest="clip_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = Dynamic3DReadinessThresholds.from_yaml(args.readiness_config)
    reports = []
    for clip_id in tuple(args.clip_ids or DEFAULT_CLIPS):
        manifest = (
            args.geometry_root
            / clip_id
            / "shared_geometry_cache"
            / "shared_3d_clip_manifest.json"
        )
        if not manifest.exists():
            raise FileNotFoundError(
                f"Missing P3-0.5 shared geometry cache: {manifest}. Run "
                "scripts/run_sequence_geometry_stabilization.py first."
            )
        reports.append(
            run_real_dynamic_3d_smoke(
                geometry_cache_manifest=manifest,
                output_dir=args.output_root / clip_id,
                thresholds=thresholds,
            )
        )
    (args.output_root / "suite_report.json").write_text(
        json.dumps(_json_safe(reports), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
