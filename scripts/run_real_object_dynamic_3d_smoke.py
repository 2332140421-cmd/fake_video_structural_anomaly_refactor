#!/usr/bin/env python3
"""Run object-bound P3-A.5/P3-B smoke from immutable shared geometry caches.

This script never estimates depth, intrinsics, camera pose, or truth labels.
It binds independent KLT observations to existing associated object tracks and
reports dynamic structural diagnostics without producing a real/fake score.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.dynamic_3d import (  # noqa: E402
    Dynamic3DReadiness,
    DynamicGeometryMode,
    ObjectMedianTranslationModel,
    PointConstantVelocityModel,
    PointTrack2DObservation,
    aggregate_object_dynamic_evidence,
    assemble_object_point_tracks_3d,
    bind_point_tracks_to_objects,
    build_object_structure_graph,
    compute_direction_consistency_residuals,
    compute_dynamic_reprojection_residual,
    compute_relative_velocity_residuals,
    compute_structure_temporal_residuals,
    load_shared_geometry_cache,
    reconstruct_point_tracks_3d,
    select_stable_object_point_tracks,
)
from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.observations import ClipObservationJSON, FrameObservationJSON  # noqa: E402
from scripts.run_real_dynamic_3d_smoke import _klt_callback, _load_images  # noqa: E402


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields_: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields_)
        writer.writeheader()
        writer.writerows(rows)


def _read_readiness(path: Path) -> Dynamic3DReadiness:
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = {item.name for item in fields(Dynamic3DReadiness)}
    return Dynamic3DReadiness(**{name: payload[name] for name in names})


def _find_associated_observation(video_id: str, root: Path) -> Path:
    candidates = sorted((root / "videos" / video_id / "associated_observations").rglob("*.json"))
    if not candidates:
        candidates = sorted(root.rglob(f"{video_id}_associated_tracks.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No globally associated observation JSON found for {video_id} under {root}."
        )
    return candidates[0]


def _clip_frames(observation: ClipObservationJSON, frame_indices: Sequence[int]) -> tuple[FrameObservationJSON, ...]:
    wanted = set(int(value) for value in frame_indices)
    frames = tuple(sorted((frame for frame in observation.frames if frame.frame_index in wanted), key=lambda item: item.frame_index))
    if len({frame.frame_index for frame in frames}) != len(frames):
        raise ValueError("Associated observations contain duplicate global frame indices.")
    return frames


def _bbox_feature_points(image: np.ndarray, bbox: Sequence[float], max_points: int = 24) -> Optional[np.ndarray]:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    margin_x, margin_y = 0.12 * (x2 - x1), 0.12 * (y2 - y1)
    left, top = max(0, int(x1 + margin_x)), max(0, int(y1 + margin_y))
    right, bottom = min(width, int(x2 - margin_x)), min(height, int(y2 - margin_y))
    if right - left < 4 or bottom - top < 4:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[top:bottom, left:right] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.goodFeaturesToTrack(gray, maxCorners=max_points, qualityLevel=0.01, minDistance=5.0, mask=mask)


def _track_object_points(images: Mapping[int, np.ndarray], frames: Sequence[FrameObservationJSON]) -> tuple[PointTrack2DObservation, ...]:
    by_track: dict[str, list[tuple[FrameObservationJSON, Any]]] = defaultdict(list)
    for frame in frames:
        for obj in frame.objects:
            track_id = str(obj.track_id or obj.person_track_id or "")
            if track_id and obj.bbox is not None:
                by_track[track_id].append((frame, obj))
    output: list[PointTrack2DObservation] = []
    for track_id, observations in sorted(by_track.items()):
        observations.sort(key=lambda item: item[0].frame_index)
        first_frame, first_object = observations[0]
        if first_frame.frame_index not in images or first_object.bbox is None:
            continue
        subset = {index: image for index, image in images.items() if index >= first_frame.frame_index}
        initial = _bbox_feature_points(images[first_frame.frame_index], first_object.bbox)
        if initial is None or not len(initial):
            continue
        rows = _klt_callback(subset, track_id, initial)
        for row in rows:
            row = dict(row)
            row["point_id"] = f"{track_id}:{row['point_id']}"
            output.append(PointTrack2DObservation(**row))
    return tuple(output)


def _rotation_compensate_bearings(points: Sequence[Any], cache: Any) -> tuple[Any, ...]:
    output = []
    for point in points:
        if point.valid and point.geometry_mode == DynamicGeometryMode.ROTATION_COMPENSATED and point.point_3d_camera is not None:
            transform = cache.clip.T_world_from_camera_by_frame.get(point.frame_index)
            if transform is not None:
                ray = np.asarray(point.point_3d_camera, dtype=float)
                ray /= max(float(np.linalg.norm(ray)), 1e-12)
                compensated = np.asarray(transform, dtype=float)[:3, :3] @ ray
                output.append(replace(point, metadata={**dict(point.metadata), "rotation_compensated_bearing": tuple(compensated), "translation_used": False}))
                continue
        output.append(point)
    return tuple(output)


def _object_scales(points: Sequence[Any], mode: DynamicGeometryMode) -> dict[str, dict[int, Optional[float]]]:
    if mode not in {DynamicGeometryMode.STATIC_CAMERA_3D, DynamicGeometryMode.FULL_SE3_3D}:
        return {object_id: {} for object_id in {point.object_track_id for point in points}}
    grouped: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for point in points:
        if not point.valid:
            continue
        xyz = point.point_3d_world if mode == DynamicGeometryMode.FULL_SE3_3D else point.point_3d_camera
        if xyz is not None:
            grouped[point.object_track_id][point.frame_index].append(np.asarray(xyz, dtype=float))
    output: dict[str, dict[int, Optional[float]]] = {}
    for object_id, by_frame in grouped.items():
        reference = None
        for _, coordinates in sorted(by_frame.items()):
            if len(coordinates) < 2:
                continue
            distances = [float(np.linalg.norm(a - b)) for index, a in enumerate(coordinates) for b in coordinates[index + 1 :]]
            positive = [value for value in distances if math.isfinite(value) and value > 1e-8]
            if positive:
                reference = float(np.percentile(positive, 75))
                break
        output[object_id] = {frame: reference for frame in by_frame}
    return output


def _graph_neighbours(graphs: Sequence[Any]) -> dict[str, tuple[str, ...]]:
    result: dict[str, set[str]] = defaultdict(set)
    for graph in graphs:
        if not graph.valid:
            continue
        for edge in graph.edges:
            result[edge.point_id_a].add(edge.point_id_b)
            result[edge.point_id_b].add(edge.point_id_a)
    return {key: tuple(sorted(value)) for key, value in result.items()}


def _motion_predictions(points: Sequence[Any], K: np.ndarray) -> tuple[Any, ...]:
    by_object_point: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for point in points:
        by_object_point[point.object_track_id][point.point_id].append(point)
    output = []
    for object_id, histories in sorted(by_object_point.items()):
        object_model = ObjectMedianTranslationModel(histories)
        for point_id, samples in sorted(histories.items()):
            samples.sort(key=lambda item: item.frame_index)
            for current in samples[2:]:
                history = [item for item in samples if item.frame_index < current.frame_index]
                prediction = object_model.predict(history, target_frame_index=current.frame_index, K_current=K)
                if not prediction.valid:
                    prediction = PointConstantVelocityModel().predict(history, target_frame_index=current.frame_index, K_current=K)
                output.append(prediction)
    return tuple(output)


def _prediction_current_camera(prediction: Any, mode: DynamicGeometryMode, target_frame: int, cache: Any) -> Optional[np.ndarray]:
    if not prediction.valid or prediction.predicted_point_3d is None:
        return None
    value = np.asarray(prediction.predicted_point_3d, dtype=float)
    if mode == DynamicGeometryMode.FULL_SE3_3D:
        transform = cache.clip.T_camera_from_world_by_frame.get(target_frame)
        if transform is None:
            return None
        value = (np.asarray(transform) @ np.concatenate([value, [1.0]]))[:3]
    elif mode == DynamicGeometryMode.ROTATION_COMPENSATED:
        transform = cache.clip.T_camera_from_world_by_frame.get(target_frame)
        if transform is None:
            return None
        value = np.asarray(transform)[:3, :3] @ value
    return value


def _save_plot(output_dir: Path, images: Mapping[int, np.ndarray], points: Sequence[Any], direction: Sequence[Any], structure: Sequence[Any]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(images[min(images)], cv2.COLOR_BGR2RGB))
    by_point: dict[str, list[Any]] = defaultdict(list)
    for point in points:
        if point.valid:
            by_point[point.point_id].append(point)
    for samples in by_point.values():
        samples.sort(key=lambda item: item.frame_index)
        uv = np.asarray([item.pixel_uv for item in samples], dtype=float)
        axes[0].plot(uv[:, 0], uv[:, 1], linewidth=0.9, alpha=0.7)
    axes[0].set_title("Object-bound independent point tracks")
    axes[0].set_axis_off()
    valid_direction = [item for item in direction if item.own_history.valid]
    axes[1].scatter([item.current_frame_index for item in valid_direction], [item.own_history.value for item in valid_direction], s=12, label="direction")
    valid_structure = [item for item in structure if item.object_structure_residual.valid]
    axes[1].scatter([item.frame_index for item in valid_structure], [item.object_structure_residual.value for item in valid_structure], s=14, label="structure")
    axes[1].set_title("Point and fixed-edge diagnostics")
    axes[1].set_xlabel("global frame index")
    axes[1].set_ylabel("residual, no threshold")
    axes[1].grid(alpha=0.25)
    if valid_direction or valid_structure:
        axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_dir / "point_and_edge_diagnostics.png", dpi=180)
    plt.close(figure)


def run_real_object_dynamic_3d_smoke(
    *,
    geometry_cache_manifest: Path,
    readiness_path: Path,
    associated_observation_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one object-bound smoke without re-estimating shared geometry."""

    output_dir.mkdir(parents=True, exist_ok=True)
    cache = load_shared_geometry_cache(geometry_cache_manifest)
    readiness = _read_readiness(readiness_path)
    images = _load_images(cache.frame_paths)
    observation = load_clip_observation(associated_observation_path)
    frames = _clip_frames(observation, cache.clip.frame_indices)
    raw_points_2d = _track_object_points(images, frames)
    binding_result = bind_point_tracks_to_objects(
        raw_points_2d, frames, video_id=cache.clip.video_id, clip_id=cache.clip.clip_id
    )
    points_3d = reconstruct_point_tracks_3d(binding_result.points_2d, cache.clip, readiness)
    points_3d = _rotation_compensate_bearings(points_3d, cache)
    scales = _object_scales(points_3d, readiness.mode)
    object_tracks = assemble_object_point_tracks_3d(binding_result.bindings, binding_result.points_2d, points_3d, object_scale_by_track_and_frame=scales)
    stable_tracks, rejected_stable_points = select_stable_object_point_tracks(object_tracks)
    stable_point_ids = {item.binding.point_id for item in stable_tracks}
    formal_points_3d = tuple(point for point in points_3d if point.point_id in stable_point_ids)
    labels = {
        str(obj.track_id or obj.person_track_id): obj.label
        for frame in frames for obj in frame.objects if obj.track_id or obj.person_track_id
    }
    graphs = tuple(
        build_object_structure_graph(stable_tracks, object_track_id=object_id, semantic_label=label)
        for object_id, label in sorted(labels.items())
    )
    neighbours = _graph_neighbours(graphs)
    direction = compute_direction_consistency_residuals(formal_points_3d, neighbour_ids=neighbours)
    velocity = compute_relative_velocity_residuals(formal_points_3d, scales)
    structure = tuple(
        residual for graph in graphs
        for residual in compute_structure_temporal_residuals(graph, formal_points_3d, scales.get(graph.object_track_id, {}))
    )
    predictions = _motion_predictions(formal_points_3d, cache.clip.frames[0].camera.K)
    two_lookup = {(point.point_id, point.frame_index): point for point in binding_result.points_2d}
    three_lookup = {(point.point_id, point.frame_index): point for point in points_3d}
    pose_lookup = {pose.target_frame_index: pose for pose in cache.clip.relative_poses}
    reprojection = []
    for prediction in predictions:
        history_end = max(prediction.history_frames) if prediction.valid else None
        if history_end is None:
            continue
        previous = three_lookup.get((prediction.point_id, history_end))
        current = two_lookup.get((prediction.point_id, prediction.target_frame_index))
        pose = pose_lookup.get(prediction.target_frame_index)
        predicted_camera = _prediction_current_camera(prediction, readiness.mode, prediction.target_frame_index, cache)
        if previous is None or current is None or pose is None or pose.relative_pose_from_previous is None or predicted_camera is None:
            continue
        reprojection.append(compute_dynamic_reprojection_residual(
            previous, current, K_current=cache.clip.frames[0].camera.K,
            image_width=cache.clip.frames[0].image_width,
            image_height=cache.clip.frames[0].image_height,
            relative_pose_current_from_previous=pose.relative_pose_from_previous,
            geometry_mode=readiness.mode, is_background=False,
            predicted_foreground_point_current_camera=predicted_camera,
            has_history_motion_model=True,
            motion_model_type=prediction.model_type,
            history_frames=prediction.history_frames,
            support_point_ids=prediction.support_point_ids,
        ))

    binding_rows = [
        {**_json_safe(asdict(item)), "point_role": item.point_role.value, "frame_indices": ";".join(map(str, item.frame_indices))}
        for item in binding_result.bindings
    ]
    binding_fields = tuple(binding_rows[0]) if binding_rows else ("video_id", "clip_id", "object_track_id", "point_id", "valid", "missing_reason")
    _write_csv(output_dir / "object_point_bindings.csv", binding_rows, binding_fields)
    (output_dir / "object_structure_graphs.json").write_text(json.dumps(_json_safe([asdict(item) for item in graphs]), indent=2) + "\n", encoding="utf-8")
    prediction_rows = [_json_safe(asdict(item)) for item in predictions]
    _write_csv(output_dir / "object_motion_predictions.csv", prediction_rows, tuple(prediction_rows[0]) if prediction_rows else ("point_id", "object_track_id", "target_frame_index", "valid", "missing_reason"))
    direction_rows = [{
        "point_id": item.point_id, "object_track_id": item.object_track_id,
        "current_frame_index": item.current_frame_index, "geometry_mode": item.geometry_mode.value,
        "direction_domain": item.direction_domain, "own_history": item.own_history.value,
        "own_history_valid": item.own_history.valid, "object_median": item.object_median.value,
        "object_median_valid": item.object_median.valid, "local_neighbour": item.local_neighbour.value,
        "local_neighbour_valid": item.local_neighbour.valid, "valid": item.valid,
        "missing_reason": item.missing_reason,
    } for item in direction]
    _write_csv(output_dir / "direction_residuals.csv", direction_rows, tuple(direction_rows[0]) if direction_rows else ("point_id", "current_frame_index", "valid", "missing_reason"))
    velocity_rows = [{
        "point_id": item.point_id, "object_track_id": item.object_track_id,
        "previous_frame_index": item.previous_frame_index, "current_frame_index": item.current_frame_index,
        "raw_displacement": item.raw_displacement, "normalized_relative_speed": item.normalized_relative_speed,
        "speed_unit": item.speed_unit, "speed_change": item.speed_change_residual.value,
        "speed_change_valid": item.speed_change_residual.valid,
        "point_vs_object_median_speed": item.object_median_speed_residual.value,
        "point_vs_object_median_speed_valid": item.object_median_speed_residual.valid,
        "valid": item.valid, "missing_reason": item.missing_reason,
    } for item in velocity]
    _write_csv(output_dir / "relative_velocity.csv", velocity_rows, tuple(velocity_rows[0]) if velocity_rows else ("point_id", "current_frame_index", "valid", "missing_reason"))
    structure_rows = [{
        "object_track_id": item.object_track_id, "frame_index": item.frame_index,
        "object_structure_residual": item.object_structure_residual.value,
        "valid_edge_ratio": item.valid_edge_ratio, "anomalous_edge_ids": ";".join(item.anomalous_edge_ids),
        "anomalous_point_ids": ";".join(item.anomalous_point_ids), "valid": item.valid,
        "missing_reason": item.missing_reason,
    } for item in structure]
    _write_csv(output_dir / "structure_residuals.csv", structure_rows, tuple(structure_rows[0]) if structure_rows else ("object_track_id", "frame_index", "valid", "missing_reason"))
    reprojection_rows = [{
        "point_id": item.point_id, "object_track_id": item.object_track_id,
        "source_frame_index": item.source_frame_index, "target_frame_index": item.target_frame_index,
        "geometry_mode": item.geometry_mode.value, "evidence_type": item.evidence_type.value,
        "pixel_error": item.pixel_error, "normalized_pixel_error": item.normalized_pixel_error,
        "motion_model_type": item.metadata.get("motion_model_type", ""),
        "formal_residual": item.residual_evidence.value,
        "formal_residual_valid": item.residual_evidence.valid, "valid": item.valid,
        "missing_reason": item.missing_reason,
    } for item in reprojection]
    _write_csv(output_dir / "dynamic_reprojection_residuals.csv", reprojection_rows, tuple(reprojection_rows[0]) if reprojection_rows else ("point_id", "target_frame_index", "formal_residual_valid", "missing_reason"))

    summaries = []
    for object_id in sorted(labels):
        point_evidence = [(f"{item.point_id}:{item.current_frame_index}", item.own_history) for item in direction if item.object_track_id == object_id]
        edge_evidence = [(f"{row.point_id_a}:{row.point_id_b}:{item.frame_index}", row.normalized_edge_length_change) for item in structure if item.object_track_id == object_id for row in item.edge_residuals]
        aggregate = aggregate_object_dynamic_evidence(object_id, point_evidence, edge_evidence)
        summaries.append({
            "object_track_id": object_id, "semantic_label": labels[object_id],
            "median": aggregate.median.value, "trimmed_mean": aggregate.trimmed_mean.value,
            "topk_mean": aggregate.topk_mean.value, "valid_point_ratio": aggregate.valid_point_ratio,
            "valid_edge_ratio": aggregate.valid_edge_ratio,
            "top_anomalous_points": ";".join(aggregate.top_anomalous_points),
            "top_anomalous_edges": ";".join(aggregate.top_anomalous_edges),
            "quality": aggregate.quality, "valid": aggregate.valid,
            "missing_reason": aggregate.missing_reason,
        })
    _write_csv(output_dir / "object_dynamic_summary.csv", summaries, tuple(summaries[0]) if summaries else ("object_track_id", "valid", "missing_reason"))
    _save_plot(output_dir, images, binding_result.points_2d, direction, structure)

    valid_bindings = [item for item in binding_result.bindings if item.valid and item.object_track_id != "background"]
    track_lengths = [len(item.frame_indices) for item in valid_bindings]
    object_history = Counter()
    for item in valid_bindings:
        object_history[item.object_track_id] = max(object_history[item.object_track_id], len(item.frame_indices))
    report = {
        "video_id": cache.clip.video_id,
        "clip_id": cache.clip.clip_id,
        "geometry_mode": readiness.mode.value,
        "dynamic_3d_ready": readiness.dynamic_3d_ready,
        "bound_object_point_count": len(valid_bindings),
        "stable_object_point_count": len(stable_tracks),
        "stable_point_rejection_reasons": dict(Counter(rejected_stable_points.values())),
        "mask_binding_count": binding_result.statistics["mask_bindings"],
        "semantic_keypoint_binding_count": binding_result.statistics["semantic_keypoint_bindings"],
        "shrunk_bbox_binding_count": binding_result.statistics["shrunk_bbox_bindings"],
        "bbox_fallback_binding_count": binding_result.statistics["bbox_fallback_bindings"],
        "mask_binding_ratio": binding_result.statistics["mask_bindings"] / len(valid_bindings) if valid_bindings else 0.0,
        "shrunk_bbox_binding_ratio": binding_result.statistics["shrunk_bbox_bindings"] / len(valid_bindings) if valid_bindings else 0.0,
        "bbox_fallback_binding_ratio": binding_result.statistics["bbox_fallback_bindings"] / len(valid_bindings) if valid_bindings else 0.0,
        "background_point_count": binding_result.statistics["background_bindings"],
        "assignment_switch_count": binding_result.statistics["assignment_switch_count"],
        "assignment_lost_count": binding_result.statistics["assignment_lost_count"],
        "mean_bound_point_track_length": float(np.mean(track_lengths)) if track_lengths else 0.0,
        "objects_with_at_least_3_frame_history": sum(length >= 3 for length in object_history.values()),
        "objects_with_valid_structure_graph": [graph.object_track_id for graph in graphs if graph.valid],
        "formal_direction_evidence_count": sum(item.own_history.valid for item in direction),
        "formal_relative_velocity_change_evidence_count": sum(item.speed_change_residual.valid for item in velocity),
        "formal_structure_evidence_count": sum(item.object_structure_residual.valid for item in structure),
        "formal_dynamic_reprojection_evidence_count": sum(item.residual_evidence.valid for item in reprojection),
        "diagnostic_reprojection_count": sum(item.diagnostic_evidence.valid for item in reprojection),
        "mode_limitations": {
            "static_camera_3d": "camera-gauge shared-scale 3D only; not metric world speed",
            "rotation_compensated": "bearing direction, angular motion, and rotation-only reprojection only",
            "full_se3_3d": "world trajectory path reserved when scale-compatible SE3 is available",
            "unavailable": "all formal dynamic evidence is missing/NaN",
        }[readiness.mode.value],
        "truth_label_or_threshold_computed": False,
        "shared_clip_reused": True,
        "depth_reestimated": False,
        "intrinsics_reestimated": False,
        "pose_reestimated": False,
        "current_frame_used_for_prediction": False,
    }
    (output_dir / "smoke_report.json").write_text(json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8")
    print(f"{cache.clip.clip_id}: mode={readiness.mode.value}, bound_points={len(valid_bindings)}, formal_reprojection={report['formal_dynamic_reprojection_evidence_count']}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_root", type=Path, default=PROJECT_ROOT / "outputs/sequence_geometry_stabilization")
    parser.add_argument("--readiness_root", type=Path, default=PROJECT_ROOT / "outputs/real_dynamic_3d_smoke")
    parser.add_argument("--observation_root", type=Path, default=PROJECT_ROOT / "outputs/evaluation/pilot_6video")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs/real_object_dynamic_3d_smoke")
    parser.add_argument("--clip_id", action="append", dest="clip_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for clip_id in tuple(args.clip_ids or DEFAULT_CLIPS):
        manifest = args.geometry_root / clip_id / "shared_geometry_cache/shared_3d_clip_manifest.json"
        cache = load_shared_geometry_cache(manifest)
        reports.append(run_real_object_dynamic_3d_smoke(
            geometry_cache_manifest=manifest,
            readiness_path=args.readiness_root / clip_id / "dynamic_readiness.json",
            associated_observation_path=_find_associated_observation(cache.clip.video_id, args.observation_root),
            output_dir=args.output_root / clip_id,
        ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "suite_report.json").write_text(json.dumps(_json_safe(reports), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
