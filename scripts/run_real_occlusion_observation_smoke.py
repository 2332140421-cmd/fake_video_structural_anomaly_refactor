#!/usr/bin/env python3
"""Run P3-C mask/visibility/occlusion diagnostics from shared geometry caches.

The smoke test never estimates depth, intrinsics, pose, or masks. Existing
instance-mask artifacts are consumed when present; bbox fallback is explicitly
low-quality legacy diagnostic evidence and cannot produce formal residuals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_DIR):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from semantic3d.dynamic_3d import (  # noqa: E402
    Dynamic3DReadiness,
    DynamicGeometryMode,
    load_shared_geometry_cache,
)
from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.observations import ClipObservationJSON, FrameObservationJSON  # noqa: E402
from semantic3d.occlusion import (  # noqa: E402
    BaseInstanceMaskProvider,
    ExistingDetectionMaskAdapter,
    MaskTracker,
    ObjectVisibilityObservation,
    PredictedObjectSupport,
    VisibilityState,
    build_occlusion_graph,
    compute_boundary_occlusion_residual,
    compute_occlusion_depth_order_residual,
    compute_visibility_explanation_residual,
    evaluate_reappearance,
    infer_visibility_state,
    predict_object_support,
)
from semantic3d.occlusion.scene_cut_statistics import compute_scene_cut_statistics  # noqa: E402


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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
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
        raise FileNotFoundError(f"No associated observation for {video_id} under {root}.")
    return candidates[0]


def _clip_frames(observation: ClipObservationJSON, frame_indices: Sequence[int]) -> tuple[FrameObservationJSON, ...]:
    wanted = {int(value) for value in frame_indices}
    frames = tuple(sorted(
        (frame for frame in observation.frames if frame.frame_index in wanted),
        key=lambda item: item.frame_index,
    ))
    if len(frames) != len({frame.frame_index for frame in frames}):
        raise ValueError("Associated observations contain duplicate global frame indices.")
    return frames


def _save_mask(mask: Optional[np.ndarray], path: Path) -> Optional[str]:
    if mask is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=bool), allow_pickle=False)
    return str(path)


def _mask_depth(depth_map: np.ndarray, valid_map: np.ndarray, mask: Optional[np.ndarray]) -> float:
    if mask is None or mask.shape != depth_map.shape:
        return float("nan")
    valid = np.asarray(mask, dtype=bool) & np.asarray(valid_map, dtype=bool)
    values = np.asarray(depth_map, dtype=float)[valid]
    values = values[np.isfinite(values) & (values > 0.0)]
    return float(np.median(values)) if values.size else float("nan")


def _prediction_for_track(
    *,
    video_id: str,
    track_id: str,
    frame_index: int,
    image_shape: tuple[int, int],
    mode: DynamicGeometryMode,
    history: Sequence[Any],
    scene_cut: bool,
) -> PredictedObjectSupport:
    if scene_cut:
        return PredictedObjectSupport.missing(
            video_id=video_id, object_track_id=track_id,
            target_frame_index=frame_index, image_shape=image_shape,
            geometry_mode=mode, reason="scene_cut_breaks_mask_history",
        )
    if mode == DynamicGeometryMode.UNAVAILABLE:
        return PredictedObjectSupport.missing(
            video_id=video_id, object_track_id=track_id,
            target_frame_index=frame_index, image_shape=image_shape,
            geometry_mode=mode, reason="dynamic_geometry_unavailable",
        )
    if len([item for item in history if item.valid]) < 2:
        return PredictedObjectSupport.missing(
            video_id=video_id, object_track_id=track_id,
            target_frame_index=frame_index, image_shape=image_shape,
            geometry_mode=mode, reason="insufficient_mask_history",
        )
    return predict_object_support(history, target_frame_index=frame_index, geometry_mode=mode)


def _save_diagnostics(
    output_path: Path,
    frame_paths: Mapping[int, Path],
    masks: Sequence[Any],
    tracked: Sequence[Any],
    visibility: Sequence[ObjectVisibilityObservation],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    if frame_paths:
        first_index = min(frame_paths)
        image = cv2.imread(str(frame_paths[first_index]))
        if image is not None:
            axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for item in masks:
            if item.frame_index != first_index or not item.valid or item.visible_mask is None:
                continue
            contours, _ = cv2.findContours(item.visible_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                axes[0].plot(contour[:, 0, 0], contour[:, 0, 1], linewidth=1.2)
            if item.mask_bbox is not None:
                axes[0].text(item.mask_bbox[0], max(0.0, item.mask_bbox[1] - 3), item.object_track_id, fontsize=7)
    axes[0].set_title("Available masks (bbox fallback marked diagnostic)")
    axes[0].set_axis_off()

    valid_tracks = [item for item in tracked if item.valid]
    axes[1].scatter(
        [item.frame_index for item in valid_tracks],
        [item.mask_iou for item in valid_tracks],
        s=16, alpha=0.75,
    )
    axes[1].set_title("History prediction vs current mask")
    axes[1].set_xlabel("global frame index")
    axes[1].set_ylabel("mask IoU (diagnostic for bbox fallback)")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.25)

    counts = Counter(item.current_state.value for item in visibility)
    axes[2].bar(list(counts), list(counts.values()), color="#4c78a8")
    axes[2].set_title("Visibility states")
    axes[2].set_ylabel("count")
    axes[2].tick_params(axis="x", rotation=45, labelsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_real_occlusion_observation_smoke(
    *,
    geometry_cache_manifest: Path,
    readiness_path: Path,
    associated_observation_path: Path,
    output_dir: Path,
    mask_provider: Optional[BaseInstanceMaskProvider] = None,
) -> dict[str, Any]:
    """Run one cache-only mask/occlusion smoke and write auditable artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "mask_arrays"
    support_dir = output_dir / "predicted_support_arrays"
    cache = load_shared_geometry_cache(geometry_cache_manifest)
    readiness = _read_readiness(readiness_path)
    observation = load_clip_observation(associated_observation_path)
    frames = _clip_frames(observation, cache.clip.frame_indices)
    frame_3d = {frame.frame_index: frame for frame in cache.clip.frames}
    provider = mask_provider or ExistingDetectionMaskAdapter(allow_legacy_bbox_fallback=True)
    tracker = MaskTracker()

    histories: dict[str, list[Any]] = defaultdict(list)
    previous_states: dict[str, VisibilityState] = {}
    labels: dict[str, str] = {}
    all_masks: list[Any] = []
    all_predictions: list[Any] = []
    all_tracked: list[Any] = []
    all_visibility: list[ObjectVisibilityObservation] = []
    all_relations: list[Any] = []
    all_depth_residuals: list[Any] = []
    all_visibility_residuals: list[Any] = []
    all_boundary_residuals: list[Any] = []
    all_reappearances: list[Any] = []

    for position, frame in enumerate(frames):
        scene_cut = position == 0 or bool(cache.clip.scene_cut_flags.get(frame.frame_index, False))
        if scene_cut:
            histories.clear()
            previous_states.clear()
        current_masks = provider.predict(video_id=cache.clip.video_id, frame=frame)
        current_by_track: dict[str, Any] = {}
        object_by_track: dict[str, Any] = {}
        for obj, mask in zip(frame.objects, current_masks):
            track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
            if track_id in current_by_track:
                raise ValueError(f"Duplicate object track {track_id!r} in frame {frame.frame_index}.")
            current_by_track[track_id] = mask
            object_by_track[track_id] = obj
            labels[track_id] = obj.label
        all_masks.extend(current_masks)

        known_tracks = sorted(set(histories) | set(current_by_track))
        predictions: dict[str, PredictedObjectSupport] = {}
        tracked_by_track: dict[str, Any] = {}
        image_shape = (frame.height, frame.width)
        for track_id in known_tracks:
            prediction = _prediction_for_track(
                video_id=cache.clip.video_id, track_id=track_id,
                frame_index=frame.frame_index, image_shape=image_shape,
                mode=readiness.mode, history=histories.get(track_id, ()),
                scene_cut=scene_cut,
            )
            predictions[track_id] = prediction
            all_predictions.append(prediction)
            tracked = tracker.track(prediction, current_by_track.get(track_id))
            tracked_by_track[track_id] = tracked
            all_tracked.append(tracked)

        shared_frame = frame_3d[frame.frame_index]
        depth_map = shared_frame.depth.require_geometry_depth()
        valid_depth = shared_frame.depth.valid_mask
        object_depths: dict[str, float] = {}
        for track_id in known_tracks:
            current_mask = current_by_track.get(track_id)
            source_mask = (
                current_mask.visible_mask
                if current_mask is not None and current_mask.valid
                else predictions[track_id].support_mask
            )
            object_depths[track_id] = _mask_depth(depth_map, valid_depth, source_mask)

        graph = build_occlusion_graph(
            video_id=cache.clip.video_id,
            frame_index=frame.frame_index,
            predicted_supports=predictions,
            observed_masks=current_by_track,
            object_depths=object_depths,
        )
        all_relations.extend(graph.relations)
        for relation in graph.relations:
            depth_result = compute_occlusion_depth_order_residual(relation)
            all_depth_residuals.append(depth_result)
            foreground_prediction = predictions[relation.foreground_object_id]
            background_prediction = predictions[relation.background_object_id]
            boundary = compute_boundary_occlusion_residual(
                relation,
                predicted_foreground=foreground_prediction,
                predicted_background=background_prediction,
                observed_foreground=current_by_track.get(relation.foreground_object_id),
                observed_background=current_by_track.get(relation.background_object_id),
            )
            all_boundary_residuals.append(boundary)

        for track_id in known_tracks:
            own_depth = object_depths.get(track_id, float("nan"))
            nearer = {
                other_id: other.visible_mask
                for other_id, other in current_by_track.items()
                if other_id != track_id and other.valid and other.visible_mask is not None
                and math.isfinite(own_depth) and math.isfinite(object_depths.get(other_id, float("nan")))
                and object_depths[other_id] < own_depth
            }
            current_object = object_by_track.get(track_id)
            detector_confidence = float(current_object.confidence) if current_object is not None else 0.0
            state = infer_visibility_state(
                predictions[track_id], current_by_track.get(track_id),
                previous_state=previous_states.get(track_id, VisibilityState.UNCERTAIN),
                nearer_object_masks=nearer,
                detector_confidence=detector_confidence,
                scene_cut=scene_cut,
                detection_confirmed_absent=False,
            )
            all_visibility.append(state)
            previous_states[track_id] = state.current_state
            all_visibility_residuals.append(compute_visibility_explanation_residual(state))
            if state.current_state == VisibilityState.REAPPEARED and not scene_cut:
                all_reappearances.append(evaluate_reappearance(
                    previous_object_track_id=track_id,
                    candidate_object_track_id=track_id,
                    frame_index=frame.frame_index,
                    predicted_reappearance_region=None,
                    semantic_label_match=True,
                    appearance_similarity=0.0,
                    structure_similarity=0.0,
                    relative_depth_consistency=0.0,
                    motion_direction_consistency=0.0,
                    reid_source="unavailable_real_smoke",
                ))

        for track_id, mask in current_by_track.items():
            histories[track_id].append(mask)

    instance_rows = []
    for item in all_masks:
        path = mask_dir / f"frame_{item.frame_index:06d}_{item.object_track_id}.npy"
        instance_rows.append({
            "video_id": item.video_id,
            "frame_index": item.frame_index,
            "object_track_id": item.object_track_id,
            "semantic_label": item.semantic_label,
            "visible_mask_path": _save_mask(item.visible_mask, path),
            "amodal_mask_path": None,
            "mask_area": item.mask_area,
            "mask_bbox": item.mask_bbox,
            "confidence": item.confidence,
            "source_provider": item.source_provider,
            "is_visible_mask": item.is_visible_mask,
            "is_amodal_mask": item.is_amodal_mask,
            "valid": item.valid,
            "missing_reason": item.missing_reason,
            "metadata": item.metadata,
        })
    (output_dir / "instance_masks.json").write_text(
        json.dumps(_json_safe(instance_rows), indent=2) + "\n", encoding="utf-8"
    )

    prediction_rows = []
    for item in all_predictions:
        path = support_dir / f"frame_{item.target_frame_index:06d}_{item.object_track_id}.npy"
        prediction_rows.append({
            "video_id": item.video_id,
            "object_track_id": item.object_track_id,
            "target_frame_index": item.target_frame_index,
            "support_mask_path": _save_mask(item.support_mask, path),
            "predicted_area": item.predicted_area,
            "in_frame_ratio": item.in_frame_ratio,
            "history_frames": item.history_frames,
            "geometry_mode": item.geometry_mode.value,
            "prediction_method": item.prediction_method,
            "quality": item.quality,
            "valid": item.valid,
            "missing_reason": item.missing_reason,
            "metadata": item.metadata,
        })
    (output_dir / "predicted_support_masks.json").write_text(
        json.dumps(_json_safe(prediction_rows), indent=2) + "\n", encoding="utf-8"
    )

    tracked_rows = [{
        "video_id": item.video_id,
        "object_track_id": item.object_track_id,
        "frame_index": item.frame_index,
        "mask_iou": item.mask_iou,
        "boundary_distance": item.boundary_distance,
        "area_change_ratio": item.area_change_ratio,
        "assignment_consistency": item.assignment_consistency,
        "track_quality": item.track_quality,
        "propagation_source": item.propagation_source,
        "valid": item.valid,
        "missing_reason": item.missing_reason,
        "metadata": item.metadata,
    } for item in all_tracked]
    (output_dir / "tracked_masks.json").write_text(
        json.dumps(_json_safe(tracked_rows), indent=2) + "\n", encoding="utf-8"
    )

    visibility_rows = [{
        "object_track_id": item.object_track_id,
        "frame_index": item.frame_index,
        "previous_state": item.previous_state.value,
        "current_state": item.current_state.value,
        "predicted_support_area": item.predicted_support_area,
        "observed_visible_area": item.observed_visible_area,
        "visible_ratio": item.visible_ratio,
        "occluded_ratio": item.occluded_ratio,
        "in_frame_ratio": item.in_frame_ratio,
        "possible_occluder_ids": ";".join(item.possible_occluder_ids),
        "state_quality": item.state_quality,
        "valid": item.valid,
        "missing_reason": item.missing_reason,
    } for item in all_visibility]
    _write_csv(output_dir / "visibility_states.csv", visibility_rows, (
        "object_track_id", "frame_index", "previous_state", "current_state",
        "predicted_support_area", "observed_visible_area", "visible_ratio",
        "occluded_ratio", "in_frame_ratio", "possible_occluder_ids",
        "state_quality", "valid", "missing_reason",
    ))

    relation_rows = [{
        **{key: value for key, value in asdict(item).items() if key != "metadata"},
        "metadata": json.dumps(_json_safe(item.metadata), sort_keys=True),
    } for item in all_relations]
    _write_csv(output_dir / "occlusion_relations.csv", relation_rows, (
        "foreground_object_id", "background_object_id", "frame_index",
        "predicted_overlap_area", "visible_overlap_area", "foreground_depth",
        "background_depth", "depth_margin", "boundary_contact_length",
        "occlusion_confidence", "valid", "missing_reason", "metadata",
    ))

    depth_rows = [{
        "foreground_object_id": item.foreground_object_id,
        "background_object_id": item.background_object_id,
        "frame_index": item.frame_index,
        "foreground_depth": item.foreground_depth,
        "background_depth": item.background_depth,
        "depth_uncertainty": item.depth_uncertainty,
        "depth_source": item.depth_source,
        "residual": item.evidence.value,
        "quality": item.evidence.quality,
        "valid": item.evidence.valid,
        "missing_reason": item.evidence.missing_reason,
    } for item in all_depth_residuals]
    _write_csv(output_dir / "depth_order_residuals.csv", depth_rows, (
        "foreground_object_id", "background_object_id", "frame_index",
        "foreground_depth", "background_depth", "depth_uncertainty",
        "depth_source", "residual", "quality", "valid", "missing_reason",
    ))

    visibility_residual_rows = [{
        "object_track_id": item.object_track_id,
        "frame_index": item.frame_index,
        "explanation": item.explanation.value,
        "diagnostic_value": item.diagnostic_evidence.value,
        "residual": item.residual_evidence.value,
        "quality": item.residual_evidence.quality,
        "valid": item.residual_evidence.valid,
        "missing_reason": item.residual_evidence.missing_reason,
    } for item in all_visibility_residuals]
    _write_csv(output_dir / "visibility_residuals.csv", visibility_residual_rows, (
        "object_track_id", "frame_index", "explanation", "diagnostic_value",
        "residual", "quality", "valid", "missing_reason",
    ))

    boundary_rows = [{
        "foreground_object_id": item.foreground_object_id,
        "background_object_id": item.background_object_id,
        "frame_index": item.frame_index,
        "boundary_distance": item.boundary_distance,
        "boundary_motion_consistency": item.boundary_motion_consistency,
        "diagnostic_value": item.diagnostic_evidence.value,
        "residual": item.residual_evidence.value,
        "quality": item.residual_evidence.quality,
        "valid": item.residual_evidence.valid,
        "missing_reason": item.residual_evidence.missing_reason,
    } for item in all_boundary_residuals]
    _write_csv(output_dir / "boundary_residuals.csv", boundary_rows, (
        "foreground_object_id", "background_object_id", "frame_index",
        "boundary_distance", "boundary_motion_consistency", "diagnostic_value",
        "residual", "quality", "valid", "missing_reason",
    ))

    reappearance_rows = [{
        **{key: value for key, value in asdict(item.observation).items() if key != "metadata"},
        "residual": item.evidence.value,
        "residual_valid": item.evidence.valid,
        "residual_missing_reason": item.evidence.missing_reason,
    } for item in all_reappearances]
    reappearance_columns = (
        "previous_object_track_id", "candidate_object_track_id", "frame_index",
        "predicted_reappearance_region", "semantic_label_match",
        "appearance_similarity", "structure_similarity",
        "relative_depth_consistency", "motion_direction_consistency",
        "reid_source", "quality", "valid", "missing_reason",
        "residual", "residual_valid", "residual_missing_reason",
    )
    _write_csv(output_dir / "reappearance_observations.csv", reappearance_rows, reappearance_columns)
    _save_diagnostics(output_dir / "occlusion_diagnostics.png", cache.frame_paths, all_masks, all_tracked, all_visibility)

    formal_masks = [
        item for item in all_masks
        if item.valid and not item.is_legacy_bbox_fallback
        and bool(item.metadata.get("formal_mask_evidence", True))
    ]
    legacy_masks = [item for item in all_masks if item.valid and item.is_legacy_bbox_fallback]
    valid_tracked = [item for item in all_tracked if item.valid]
    scene_cut_statistics = compute_scene_cut_statistics(
        frame_indices=cache.clip.frame_indices,
        scene_cut_flags=cache.clip.scene_cut_flags,
        visibility_observations=all_visibility,
    )
    report = {
        "video_id": cache.clip.video_id,
        "clip_id": cache.clip.clip_id,
        "geometry_mode": readiness.mode.value,
        "dynamic_3d_ready": readiness.dynamic_3d_ready,
        "total_mask_observations": len(all_masks),
        "diagnostic_valid_mask_count": sum(item.valid for item in all_masks),
        "diagnostic_mask_valid_rate": (sum(item.valid for item in all_masks) / len(all_masks)) if all_masks else 0.0,
        "formal_valid_mask_count": len(formal_masks),
        "formal_mask_valid_rate": len(formal_masks) / len(all_masks) if all_masks else 0.0,
        "legacy_bbox_mask_count": len(legacy_masks),
        "legacy_bbox_mask_ratio": len(legacy_masks) / len(all_masks) if all_masks else 0.0,
        "valid_tracked_mask_count": len(valid_tracked),
        "mean_mask_iou": float(np.mean([item.mask_iou for item in valid_tracked])) if valid_tracked else None,
        "mean_boundary_distance_px": float(np.mean([item.boundary_distance for item in valid_tracked])) if valid_tracked else None,
        "visibility_state_counts": dict(Counter(item.current_state.value for item in all_visibility)),
        "legacy_visibility_state_counts": dict(Counter(item.current_state.value for item in all_visibility)),
        "scene_cut_statistics_version": scene_cut_statistics.statistics_version,
        "clip_scene_cut_count": scene_cut_statistics.clip_scene_cut_count,
        "clip_scene_cut_frame_indices": scene_cut_statistics.clip_scene_cut_frame_indices,
        "object_visibility_scene_cut_markers": scene_cut_statistics.object_visibility_scene_cut_markers,
        "track_initialization_markers": scene_cut_statistics.track_initialization_markers,
        "state_machine_boundary_markers": scene_cut_statistics.state_machine_boundary_markers,
        "legacy_scene_cut_state_count": scene_cut_statistics.legacy_scene_cut_state_count,
        "candidate_partial_occlusion_count": sum(item.current_state == VisibilityState.PARTIALLY_OCCLUDED for item in all_visibility),
        "candidate_full_occlusion_count": sum(item.current_state == VisibilityState.FULLY_OCCLUDED for item in all_visibility),
        "candidate_occlusion_relation_count": len(all_relations),
        "formal_occlusion_relation_count": sum(item.valid for item in all_relations),
        "formal_depth_order_evidence_count": sum(item.evidence.valid for item in all_depth_residuals),
        "formal_visibility_residual_count": sum(item.residual_evidence.valid for item in all_visibility_residuals),
        "formal_boundary_residual_count": sum(item.residual_evidence.valid for item in all_boundary_residuals),
        "reappearance_candidate_count": len(all_reappearances),
        "formal_reappearance_evidence_count": sum(item.evidence.valid for item in all_reappearances),
        "bbox_fallback_formal_evidence_count": 0,
        "shared_clip_reused": True,
        "shared_object_dynamic_contract_reused": True,
        "depth_reestimated": False,
        "intrinsics_reestimated": False,
        "pose_reestimated": False,
        "mask_model_downloaded_or_trained": False,
        "truth_labels_used": False,
        "current_frame_used_for_prediction": False,
        "real_fake_classification_performed": False,
        "threshold_tuned_on_six_videos": False,
        "limitations": [
            "Existing observations currently provide bbox fallback rather than true instance masks.",
            "BBox fallback is diagnostic-only and cannot establish formal occlusion evidence.",
            "Object-region center depth is low-quality for depth-order diagnostics.",
            "No cross-scene-cut re-identification is performed.",
        ],
    }
    (output_dir / "smoke_report.json").write_text(
        json.dumps(_json_safe(report), indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{cache.clip.clip_id}: mode={readiness.mode.value}, "
        f"formal_masks={len(formal_masks)}/{len(all_masks)}, "
        f"candidate_relations={len(all_relations)}, "
        f"formal_depth_order={report['formal_depth_order_evidence_count']}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_root", type=Path, default=PROJECT_ROOT / "outputs/sequence_geometry_stabilization")
    parser.add_argument("--readiness_root", type=Path, default=PROJECT_ROOT / "outputs/real_dynamic_3d_smoke")
    parser.add_argument("--observation_root", type=Path, default=PROJECT_ROOT / "outputs/evaluation/pilot_6video")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs/real_occlusion_observation_smoke")
    parser.add_argument("--clip_id", action="append", dest="clip_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for clip_id in tuple(args.clip_ids or DEFAULT_CLIPS):
        manifest = args.geometry_root / clip_id / "shared_geometry_cache/shared_3d_clip_manifest.json"
        cache = load_shared_geometry_cache(manifest)
        reports.append(run_real_occlusion_observation_smoke(
            geometry_cache_manifest=manifest,
            readiness_path=args.readiness_root / clip_id / "dynamic_readiness.json",
            associated_observation_path=_find_associated_observation(cache.clip.video_id, args.observation_root),
            output_dir=args.output_root / clip_id,
        ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "suite_report.json").write_text(
        json.dumps(_json_safe(reports), indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
