#!/usr/bin/env python3
"""Run P3-D.1 full-video observation coverage and P4 readiness diagnostics.

This script never consumes real/fake labels or anomaly residual magnitudes. It
does not download weights and does not turn filled bboxes into formal masks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

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

from semantic3d.coverage_readiness import evaluate_coverage_readiness  # noqa: E402
from semantic3d.object_association import ObjectAssociator  # noqa: E402
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON  # noqa: E402
from semantic3d.occlusion import (  # noqa: E402
    MaskTracker,
    RealInstanceMaskProvider,
    associate_instance_masks,
    mask_bbox,
    predict_object_support,
    track_formal_mask_internal_points,
)
from semantic3d.dynamic_3d import DynamicGeometryMode  # noqa: E402
from semantic3d.real_object_provider import normalize_label  # noqa: E402
from scripts.find_real_3d_evidence_clips import find_evidence_clips, write_candidates  # noqa: E402
from scripts.run_real_3d_evidence_coverage import run_real_3d_evidence_coverage  # noqa: E402


CSV_COLUMNS = {
    "mask_coverage.csv": (
        "video_id", "frame_index", "segmentation_instance_id", "object_track_id",
        "class_id", "class_name", "confidence", "visible_mask_path", "mask_area",
        "mask_bbox", "boundary_point_count", "source_provider", "weight_sha256",
        "is_visible_mask", "is_amodal_mask", "valid", "missing_reason",
    ),
    "mask_object_association.csv": (
        "video_id", "frame_index", "object_id", "object_track_id", "candidate_id",
        "association_quality", "association_source", "valid", "missing_reason",
        "candidate_details",
    ),
    "mask_tracking_quality.csv": (
        "video_id", "object_track_id", "frame_index", "independently_observed_mask",
        "history_predicted_mask", "tracked_mask", "observed_vs_predicted_iou",
        "normalized_boundary_distance", "area_change_ratio", "assignment_consistency",
        "track_switch_count", "track_quality", "valid", "missing_reason",
    ),
    "keypoint_coverage.csv": (
        "video_id", "frame_index", "object_track_id", "total_keypoints",
        "valid_keypoints", "valid_ratio", "provider_name", "status", "valid",
        "missing_reason", "clip_id",
    ),
    "structure_graph_coverage.csv": (
        "video_id", "clip_id", "object_track_id", "semantic_label", "graph_type",
        "point_source", "point_count", "edge_count", "valid", "quality",
        "structure_temporal_evidence_count", "missing_reason",
    ),
    "structure_residual_coverage.csv": (
        "video_id", "object_track_id", "graph_type", "valid_residual_count",
        "valid", "missing_reason",
    ),
    "visibility_event_coverage.csv": (
        "video_id", "object_track_id", "frame_index", "event_type", "valid",
        "quality", "missing_reason", "has_history_prediction", "scene_cut",
    ),
    "occlusion_evidence_coverage.csv": (
        "video_id", "clip_id", "evidence_type", "frame_index", "valid",
        "residual", "quality", "missing_reason",
    ),
    "reappearance_coverage.csv": (
        "video_id", "object_track_id", "frame_index", "valid", "quality",
        "missing_reason",
    ),
}


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def discover_test_videos(video_root: Path) -> tuple[Path, ...]:
    """Return all six-video pilot inputs without exposing labels to selection."""

    return tuple(sorted(video_root.rglob("*.mp4")))


def _video_metadata(path: Path) -> tuple[int, float, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    result = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        float(capture.get(cv2.CAP_PROP_FPS)),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    capture.release()
    return result


def _candidate_object(candidate: Any, frame_area: float) -> ObjectObservationJSON:
    bbox = mask_bbox(candidate.visible_mask)
    if bbox is None:
        raise ValueError("A non-empty segmentation candidate must have a bbox.")
    label = normalize_label(candidate.class_name)
    return ObjectObservationJSON(
        object_id=candidate.source_detection_id,
        label=label,
        canonical_label=label,
        mask_area=float(np.count_nonzero(candidate.visible_mask)),
        frame_area=frame_area,
        depth=1.0,
        confidence=candidate.confidence,
        bbox=list(bbox),
        provenance={
            "object_provider": "frozen_instance_segmenter_detection_head",
            "source_detection_id": candidate.source_detection_id,
            "weight_sha256": candidate.weight_sha256,
        },
        quality=candidate.confidence,
        metadata={"mask_area_source": "real_instance_mask", "geometry_depth_available": False},
    )


def _save_array(mask: np.ndarray | None, path: Path) -> str:
    if mask is None:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(mask, dtype=bool), allow_pickle=False)
    return str(path)


def _scene_cut(previous: np.ndarray | None, current: np.ndarray) -> bool:
    """Conservative image-only cut diagnostic independent of object evidence."""

    if previous is None:
        return False
    previous_small = cv2.resize(previous, (64, 64), interpolation=cv2.INTER_AREA)
    current_small = cv2.resize(current, (64, 64), interpolation=cv2.INTER_AREA)
    difference = float(np.mean(cv2.absdiff(previous_small, current_small))) / 255.0
    return difference > 0.55


def _scan_video_masks(
    video_path: Path,
    *,
    provider: RealInstanceMaskProvider,
    output_root: Path,
    max_frames: int | None,
) -> dict[str, Any]:
    """Run full-frame segmentation, association, and history-only mask checks."""

    video_id = video_path.stem
    total_frames, fps, width, height = _video_metadata(video_path)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    result: dict[str, Any] = {
        "video_id": video_id,
        "num_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "mask_rows": [],
        "association_rows": [],
        "tracking_rows": [],
        "visibility_rows": [],
        "reappearance_rows": [],
        "structure_rows": [],
        "frame_records": [],
        "statistics": Counter(),
        "scene_cut_flags": {},
    }
    if not provider.available:
        result["status"] = "observation_missing"
        result["missing_reason"] = provider.unavailable_reason or "instance_segmentation_provider_unavailable"
        return result

    frame_dir = output_root / "frames" / video_id
    capture = cv2.VideoCapture(str(video_path))
    frames: list[FrameObservationJSON] = []
    candidates_by_frame: dict[int, tuple[Any, ...]] = {}
    images: dict[int, np.ndarray] = {}
    previous_image = None
    frame_index = 0
    while frame_index < total_frames:
        success, image = capture.read()
        if not success:
            break
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(frame_path), image)
        empty_frame = FrameObservationJSON(
            frame_index=frame_index, frame_id=frame_path.stem,
            width=image.shape[1], height=image.shape[0], objects=[],
            image_path=str(frame_path),
        )
        candidates = provider.predict_candidates(empty_frame)
        candidates_by_frame[frame_index] = candidates
        objects = [_candidate_object(item, float(image.shape[0] * image.shape[1])) for item in candidates]
        frames.append(replace(empty_frame, objects=objects))
        images[frame_index] = image
        result["scene_cut_flags"][frame_index] = _scene_cut(previous_image, image)
        previous_image = image
        frame_index += 1
    capture.release()
    result["num_frames"] = len(frames)
    associated_frames = ObjectAssociator(max_frame_gap=2).associate(frames)

    mask_by_frame_track: dict[tuple[int, str], Any] = {}
    masks_by_track: dict[str, list[Any]] = defaultdict(list)
    diagnostics_by_frame: dict[int, tuple[Any, ...]] = {}
    assigned_candidate_ids: set[tuple[int, str]] = set()
    for frame in associated_frames:
        candidates = candidates_by_frame[frame.frame_index]
        association = associate_instance_masks(video_id=video_id, frame=frame, candidates=candidates)
        diagnostics_by_frame[frame.frame_index] = association.diagnostics
        result["statistics"]["total_object_observations"] += len(frame.objects)
        result["statistics"]["total_mask_candidates"] += len(candidates)
        result["statistics"]["matched_masks"] += len(association.assigned_candidate_ids)
        result["statistics"]["unmatched_objects"] += sum(not item.valid for item in association.diagnostics)
        result["statistics"]["unmatched_masks"] += len(association.rejected_candidate_ids)
        assigned_candidate_ids.update((frame.frame_index, item) for item in association.assigned_candidate_ids)
        candidate_lookup = {item.candidate_id: item for item in candidates}
        for obj, mask, diagnostic in zip(frame.objects, association.masks, association.diagnostics):
            track_id = str(obj.track_id or obj.object_id)
            mask_by_frame_track[(frame.frame_index, track_id)] = mask
            masks_by_track[track_id].append(mask)
            details = tuple(dict(item) for item in diagnostic.candidate_details)
            result["statistics"]["category_conflicts"] += sum(
                item.get("rejection_reason") == "semantic_category_conflict" for item in details
            )
            result["association_rows"].append({
                "video_id": video_id,
                "frame_index": frame.frame_index,
                "object_id": obj.object_id,
                "object_track_id": track_id,
                "candidate_id": diagnostic.candidate_id or "",
                "association_quality": diagnostic.association_quality if diagnostic.valid else float("nan"),
                "association_source": diagnostic.association_source,
                "valid": diagnostic.valid,
                "missing_reason": diagnostic.missing_reason,
                "candidate_details": json.dumps(_json_safe(details), sort_keys=True),
            })
            candidate = candidate_lookup.get(diagnostic.candidate_id or "")
            mask_path = ""
            if mask.valid and mask.visible_mask is not None:
                mask_path = _save_array(
                    mask.visible_mask,
                    output_root / "formal_masks" / video_id / f"frame_{frame.frame_index:06d}_{track_id}.npy",
                )
            result["mask_rows"].append({
                "video_id": video_id,
                "frame_index": frame.frame_index,
                "segmentation_instance_id": diagnostic.candidate_id or "",
                "object_track_id": track_id,
                "class_id": "" if candidate is None else candidate.class_id,
                "class_name": obj.label if candidate is None else candidate.class_name,
                "confidence": mask.confidence,
                "visible_mask_path": mask_path,
                "mask_area": mask.mask_area,
                "mask_bbox": json.dumps(mask.mask_bbox),
                "boundary_point_count": len(mask.boundary_points),
                "source_provider": mask.source_provider,
                "weight_sha256": "" if candidate is None else candidate.weight_sha256,
                "is_visible_mask": mask.is_visible_mask,
                "is_amodal_mask": mask.is_amodal_mask,
                "valid": mask.valid,
                "missing_reason": mask.missing_reason,
            })

    eligible_by_candidate: Counter[str] = Counter()
    for row in result["association_rows"]:
        for item in json.loads(row["candidate_details"]):
            if item.get("association_score", 0.0) >= 0.35:
                eligible_by_candidate[f"{row['frame_index']}:{item['candidate_id']}"] += 1
    result["statistics"]["one_to_many_conflicts"] = sum(value - 1 for value in eligible_by_candidate.values() if value > 1)
    result["statistics"]["track_switches"] = 0

    tracker = MaskTracker()
    histories: dict[str, list[Any]] = defaultdict(list)
    previous_state: dict[str, str] = defaultdict(lambda: "uncertain")
    active_history_age: dict[str, int] = {}
    stable_track_ids: set[str] = set()
    frame_map = {frame.frame_index: frame for frame in associated_frames}
    for frame_index in sorted(frame_map):
        scene_cut = bool(result["scene_cut_flags"].get(frame_index, False))
        if scene_cut:
            histories.clear()
            active_history_age.clear()
            previous_state.clear()
        frame = frame_map[frame_index]
        current = {
            str(obj.track_id or obj.object_id): mask_by_frame_track[(frame_index, str(obj.track_id or obj.object_id))]
            for obj in frame.objects
        }
        known_tracks = sorted(set(current) | {track for track, age in active_history_age.items() if age <= 2})
        stable_count = 0
        for track_id in known_tracks:
            history = histories.get(track_id, ())
            if len([item for item in history if item.valid]) >= 2:
                prediction = predict_object_support(
                    history,
                    target_frame_index=frame_index,
                    geometry_mode=DynamicGeometryMode.STATIC_CAMERA_3D,
                )
            else:
                from semantic3d.occlusion import PredictedObjectSupport
                prediction = PredictedObjectSupport.missing(
                    video_id=video_id, object_track_id=track_id,
                    target_frame_index=frame_index, image_shape=(height, width),
                    geometry_mode=DynamicGeometryMode.STATIC_CAMERA_3D,
                    reason="insufficient_mask_history",
                )
            observed = current.get(track_id)
            tracked = tracker.track(prediction, observed)
            independent_path = _save_array(
                None if observed is None else observed.visible_mask,
                output_root / "tracking_masks" / video_id / f"frame_{frame_index:06d}_{track_id}_observed.npy",
            )
            predicted_path = _save_array(
                prediction.support_mask,
                output_root / "tracking_masks" / video_id / f"frame_{frame_index:06d}_{track_id}_predicted.npy",
            )
            tracked_path = _save_array(
                tracked.observed_mask if tracked.valid else None,
                output_root / "tracking_masks" / video_id / f"frame_{frame_index:06d}_{track_id}_tracked.npy",
            )
            normalized_distance = (
                tracked.boundary_distance / max(math.hypot(height, width), 1e-8)
                if tracked.valid else float("nan")
            )
            result["tracking_rows"].append({
                "video_id": video_id,
                "object_track_id": track_id,
                "frame_index": frame_index,
                "independently_observed_mask": independent_path,
                "history_predicted_mask": predicted_path,
                "tracked_mask": tracked_path,
                "observed_vs_predicted_iou": tracked.mask_iou,
                "normalized_boundary_distance": normalized_distance,
                "area_change_ratio": tracked.area_change_ratio,
                "assignment_consistency": tracked.assignment_consistency,
                "track_switch_count": 0,
                "track_quality": tracked.track_quality,
                "valid": tracked.valid,
                "missing_reason": tracked.missing_reason,
            })
            if tracked.valid and tracked.track_quality >= 0.5:
                stable_track_ids.add(track_id)
                stable_count += 1
            if prediction.valid and observed is None:
                event_type, valid, reason = "detector_missing", False, "observation_missing"
            elif tracked.valid:
                event_type, valid, reason = "no_occlusion_event", True, ""
            else:
                event_type, valid, reason = "observation_missing", False, tracked.missing_reason
            result["visibility_rows"].append({
                "video_id": video_id, "object_track_id": track_id,
                "frame_index": frame_index, "event_type": event_type,
                "valid": valid, "quality": tracked.track_quality,
                "missing_reason": reason,
                "has_history_prediction": prediction.valid,
                "scene_cut": scene_cut,
            })
            previous_state[track_id] = event_type
        for track_id in list(active_history_age):
            active_history_age[track_id] += 1
        for track_id, observed in current.items():
            histories[track_id].append(observed)
            active_history_age[track_id] = 0
        frame_masks = [mask for (index, _), mask in mask_by_frame_track.items() if index == frame_index and mask.valid]
        result["frame_records"].append({
            "video_id": video_id,
            "frame_index": frame_index,
            "object_track_ids": ";".join(sorted(current)),
            "semantic_labels": ";".join(sorted(obj.label for obj in frame.objects)),
            "mask_valid_ratio": len(frame_masks) / len(frame.objects) if frame.objects else float("nan"),
            "formal_mask_object_count": len(frame_masks),
            "keypoint_valid_ratio": float("nan"),
            "ordinary_structure_graph_count": 0,
            "stable_mask_track_count": stable_count,
            "has_formal_mask_overlap": False,
            "valid_depth_order_count": 0,
            "visibility_state": "detector_missing" if any(
                row["frame_index"] == frame_index and row["event_type"] == "detector_missing"
                for row in result["visibility_rows"]
            ) else "no_occlusion_event",
            "has_history_prediction": any(
                row["frame_index"] == frame_index and row["has_history_prediction"]
                for row in result["visibility_rows"]
            ),
            "mean_tracking_quality": float(np.mean([
                row["track_quality"] for row in result["tracking_rows"]
                if row["frame_index"] == frame_index and row["valid"]
            ])) if stable_count else float("nan"),
            "depth_order_confidence": float("nan"),
            "scene_cut": scene_cut,
            "geometry_mode": "full_video_mask_observation_only",
            "observation_quality": float(np.mean([mask.confidence for mask in frame_masks])) if frame_masks else 0.0,
        })

    for track_id, observations in sorted(masks_by_track.items()):
        semantic_label = observations[0].semantic_label if observations else ""
        if semantic_label == "person":
            continue
        points = track_formal_mask_internal_points(
            images, observations, scene_cut_flags=result["scene_cut_flags"],
        )
        valid_by_id: dict[str, int] = Counter(item.point_id for item in points if item.valid)
        stable_points = sum(count >= 3 for count in valid_by_id.values())
        result["structure_rows"].append({
            "video_id": video_id, "clip_id": "full_video_scan",
            "object_track_id": track_id, "semantic_label": semantic_label,
            "graph_type": "mask_internal_points_2d_ready",
            "point_source": "formal_instance_mask_internal",
            "point_count": stable_points, "edge_count": 0,
            "valid": False, "quality": 0.0,
            "structure_temporal_evidence_count": 0,
            "missing_reason": "shared_3d_reconstruction_required",
        })

    result["statistics"]["stable_mask_track_count"] = len(stable_track_ids)
    result["status"] = "observation_available"
    result["missing_reason"] = ""
    return result


def _plot(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    videos = [str(row["video_id"]) for row in rows]
    x = np.arange(len(videos))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    metrics = (
        ("formal_mask_valid_ratio", "Formal mask ratio"),
        ("mask_association_success_rate", "Mask association rate"),
        ("stable_mask_track_ratio", "Stable mask track ratio"),
        ("structure_residual_count", "Formal structure residual count"),
    )
    for axis, (field, title) in zip(axes.flat, metrics):
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        valid = np.isfinite(values)
        axis.bar(x[valid], values[valid], color="#4c78a8")
        axis.scatter(x[~valid], np.zeros(np.count_nonzero(~valid)), marker="x", color="#e15759", label="observation missing")
        axis.set_title(title)
        axis.set_xticks(x, videos, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if np.any(~valid):
            axis.legend(fontsize=8)
    figure.suptitle("P3-D.1 Real Observation Coverage (No Real/Fake Evaluation)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run_real_3d_evidence_coverage_v2(
    *,
    video_root: Path,
    output_root: Path,
    mask_model_path: Path,
    pose_model_path: Path,
    device: str = "cpu",
    max_frames: int | None = None,
    run_shared_3d_smoke: bool = True,
) -> dict[str, Any]:
    """Run six-video P3-D.1 coverage with explicit missing observations."""

    output_root.mkdir(parents=True, exist_ok=True)
    provider = RealInstanceMaskProvider(model_path=mask_model_path, device=device)
    videos = discover_test_videos(video_root)
    model_metadata = asdict(provider.model_metadata)
    model_metadata.update({"automatic_download_attempted": False, "bbox_used_as_formal_mask": False})
    (output_root / "model_metadata.json").write_text(
        json.dumps(_json_safe(model_metadata), indent=2) + "\n", encoding="utf-8"
    )

    scans = [
        _scan_video_masks(
            path, provider=provider, output_root=output_root, max_frames=max_frames,
        )
        for path in videos
    ]

    smoke_summary: dict[str, Any] = {"per_video": []}
    smoke_root = output_root / "shared_3d_smoke"
    if run_shared_3d_smoke:
        geometry_root = PROJECT_ROOT / "outputs/sequence_geometry_stabilization"
        readiness_root = PROJECT_ROOT / "outputs/real_dynamic_3d_smoke"
        observation_root = PROJECT_ROOT / "outputs/evaluation/pilot_6video"
        required = (
            geometry_root / "static_camera/shared_geometry_cache/shared_3d_clip_manifest.json",
            readiness_root / "static_camera/dynamic_readiness.json",
        )
        if all(path.exists() for path in required):
            smoke_summary = run_real_3d_evidence_coverage(
                geometry_root=geometry_root,
                readiness_root=readiness_root,
                observation_root=observation_root,
                output_root=smoke_root,
                mask_model_path=mask_model_path,
                pose_model_path=pose_model_path,
                device=device,
            )

    rows_by_file: dict[str, list[dict[str, Any]]] = {name: [] for name in CSV_COLUMNS}
    for scan in scans:
        rows_by_file["mask_coverage.csv"].extend(scan["mask_rows"])
        rows_by_file["mask_object_association.csv"].extend(scan["association_rows"])
        rows_by_file["mask_tracking_quality.csv"].extend(scan["tracking_rows"])
        rows_by_file["visibility_event_coverage.csv"].extend(scan["visibility_rows"])
        rows_by_file["reappearance_coverage.csv"].extend(scan["reappearance_rows"])
        rows_by_file["structure_graph_coverage.csv"].extend(scan["structure_rows"])
    for filename in ("keypoint_coverage.csv", "structure_graph_coverage.csv", "occlusion_evidence_coverage.csv"):
        rows_by_file[filename].extend(_read_csv(smoke_root / filename))
    for row in _read_csv(smoke_root / "structure_graph_coverage.csv"):
        rows_by_file["structure_residual_coverage.csv"].append({
            "video_id": row["video_id"], "object_track_id": row["object_track_id"],
            "graph_type": row["graph_type"],
            "valid_residual_count": int(row["structure_temporal_evidence_count"]),
            "valid": int(row["structure_temporal_evidence_count"]) > 0,
            "missing_reason": "" if int(row["structure_temporal_evidence_count"]) > 0 else row["missing_reason"],
        })
    for filename, columns in CSV_COLUMNS.items():
        _write_csv(output_root / filename, rows_by_file[filename], columns)

    smoke_by_video = {row["video_id"]: row for row in smoke_summary.get("per_video", [])}
    per_video = []
    all_frame_records = []
    for scan in scans:
        video_id = scan["video_id"]
        smoke = smoke_by_video.get(video_id, {})
        statistics = scan["statistics"]
        object_count = int(statistics.get("total_object_observations", 0))
        candidate_count = int(statistics.get("total_mask_candidates", 0))
        matched = int(statistics.get("matched_masks", 0))
        valid_tracks = {row["object_track_id"] for row in scan["tracking_rows"] if row["valid"]}
        all_tracks = {row["object_track_id"] for row in scan["mask_rows"]}
        mask_ratio = matched / object_count if object_count else float("nan")
        association_rate = matched / object_count if object_count else float("nan")
        stable_ratio = len(valid_tracks) / len(all_tracks) if all_tracks else float("nan")
        person_count = int(smoke.get("person_structure_graph_count", 0))
        ordinary_count = int(smoke.get("ordinary_mask_structure_graph_count", 0))
        structure_count = int(smoke.get("formal_structure_temporal_evidence_count", 0))
        partial = int(smoke.get("partial_occlusion_candidates", 0))
        full = int(smoke.get("full_occlusion_candidates", 0))
        reappearance = int(smoke.get("reappearance_event_count", 0))
        occlusion_rows = [row for row in rows_by_file["occlusion_evidence_coverage.csv"] if row.get("video_id") == video_id and str(row.get("valid", "")).lower() == "true"]
        depth_order = sum(row.get("evidence_type") == "depth_order" for row in occlusion_rows)
        boundary = sum(row.get("evidence_type") == "boundary" for row in occlusion_rows)
        reasons = [scan["missing_reason"]] if scan["missing_reason"] else []
        readiness = evaluate_coverage_readiness(
            formal_mask_valid_ratio=mask_ratio,
            mask_association_success_rate=association_rate,
            stable_mask_track_ratio=stable_ratio,
            person_structure_track_count=person_count,
            ordinary_structure_track_count=ordinary_count,
            structure_residual_count=structure_count,
            partial_occlusion_event_count=partial,
            full_occlusion_event_count=full,
            reappearance_event_count=reappearance,
            depth_order_evidence_count=depth_order,
            boundary_occlusion_evidence_count=boundary,
            mask_observation_missing=not provider.available or object_count == 0,
            structure_observation_missing=False,
            missing_reasons=reasons,
            metadata={"video_id": video_id},
        )
        association_qualities = [row["association_quality"] for row in scan["association_rows"] if row["valid"]]
        row = {
            "video_id": video_id,
            "num_frames": scan["num_frames"],
            "total_object_observations": object_count,
            "total_mask_candidates": candidate_count,
            "matched_masks": matched,
            "unmatched_objects": int(statistics.get("unmatched_objects", 0)),
            "unmatched_masks": int(statistics.get("unmatched_masks", 0)),
            "category_conflicts": int(statistics.get("category_conflicts", 0)),
            "one_to_many_conflicts": int(statistics.get("one_to_many_conflicts", 0)),
            "track_switches": int(statistics.get("track_switches", 0)),
            "association_quality_mean": float(np.mean(association_qualities)) if association_qualities else float("nan"),
            **asdict(readiness),
            "bbox_fallback_ratio": 0.0 if object_count else float("nan"),
            "status": scan["status"],
            "primary_missing_reason": scan["missing_reason"],
        }
        row["branch_coverage"] = json.dumps(_json_safe(row["branch_coverage"]), sort_keys=True)
        row["missing_reasons"] = ";".join(row["missing_reasons"])
        row["metadata"] = json.dumps(_json_safe(row["metadata"]), sort_keys=True)
        per_video.append(row)
        all_frame_records.extend(scan["frame_records"])

    candidates = find_evidence_clips(all_frame_records, minimum_duration=2)
    existing_keys = {
        (row["video_id"], row["start_frame"], row["end_frame"], row["candidate_type"])
        for row in candidates
    }
    for row in _read_csv(smoke_root / "evidence_clip_candidates.csv"):
        key = (row["video_id"], row["start_frame"], row["end_frame"], row["candidate_type"])
        if key not in existing_keys:
            candidates.append(row)
            existing_keys.add(key)
    write_candidates(candidates, output_root / "evidence_clip_candidates.csv")
    per_video_columns = tuple(per_video[0]) if per_video else ("video_id", "status")
    _write_csv(output_root / "per_video_summary.csv", per_video, per_video_columns)
    final_model_metadata = asdict(provider.model_metadata)
    final_model_metadata.update({"automatic_download_attempted": False, "bbox_used_as_formal_mask": False})
    (output_root / "model_metadata.json").write_text(
        json.dumps(_json_safe(final_model_metadata), indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "coverage_version": "p3_d_1_real_evidence_coverage_v2",
        "video_count": len(videos),
        "full_video_scan": True,
        "total_video_frames": sum(scan["num_frames"] for scan in scans),
        "model_metadata": _json_safe(final_model_metadata),
        "total_formal_masks": sum(row["matched_masks"] for row in per_video),
        "total_stable_mask_tracks": sum(scan["statistics"].get("stable_mask_track_count", 0) for scan in scans),
        "total_person_structure_tracks": sum(row["person_structure_track_count"] for row in per_video),
        "total_ordinary_structure_tracks": sum(row["ordinary_structure_track_count"] for row in per_video),
        "total_structure_residuals": sum(row["structure_residual_count"] for row in per_video),
        "total_partial_occlusion_events": sum(row["partial_occlusion_event_count"] for row in per_video),
        "total_full_occlusion_events": sum(row["full_occlusion_event_count"] for row in per_video),
        "total_reappearance_events": sum(row["reappearance_event_count"] for row in per_video),
        "ready_for_partial_p4_videos": [row["video_id"] for row in per_video if row["ready_for_partial_p4"]],
        "ready_for_full_p4_videos": [row["video_id"] for row in per_video if row["ready_for_full_p4"]],
        "candidate_clip_count": len(candidates),
        "truth_labels_used_for_selection": False,
        "residual_magnitude_used_for_selection": False,
        "real_fake_performance_evaluation_performed": False,
        "per_video": _json_safe(per_video),
    }
    (output_root / "global_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8"
    )
    _plot(output_root / "coverage_diagnostics.png", per_video)
    print(json.dumps(_json_safe(summary), indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video_root", type=Path, default=PROJECT_ROOT / "data/tests_videos")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage_v2")
    parser.add_argument("--mask_model_path", type=Path, default=PROJECT_ROOT / "checkpoints/yolov8n-seg.pt")
    parser.add_argument("--pose_model_path", type=Path, default=PROJECT_ROOT / "checkpoints/yolov8n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_frames", type=int)
    parser.add_argument("--skip_shared_3d_smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_real_3d_evidence_coverage_v2(
        video_root=args.video_root,
        output_root=args.output_root,
        mask_model_path=args.mask_model_path,
        pose_model_path=args.pose_model_path,
        device=args.device,
        max_frames=args.max_frames,
        run_shared_3d_smoke=not args.skip_shared_3d_smoke,
    )


if __name__ == "__main__":
    main()
