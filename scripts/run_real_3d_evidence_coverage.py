#!/usr/bin/env python3
"""Measure P3-D real observation coverage without labels or anomaly selection."""

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
    assemble_object_point_tracks_3d,
    bind_person_keypoints_to_shared_3d,
    bind_point_tracks_to_objects,
    build_object_structure_graph,
    compute_structure_temporal_residuals,
    load_shared_geometry_cache,
    reconstruct_point_tracks_3d,
    select_stable_object_point_tracks,
)
from semantic3d.dynamic_3d.motion_model import trajectory_coordinate  # noqa: E402
from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.keypoint_provider import RealHumanKeypointProvider  # noqa: E402
from semantic3d.observations import ClipObservationJSON, FrameObservationJSON  # noqa: E402
from semantic3d.occlusion import (  # noqa: E402
    RealInstanceMaskProvider,
    associate_instance_masks,
)
from scripts.find_real_3d_evidence_clips import find_evidence_clips, write_candidates  # noqa: E402
from scripts.run_real_object_dynamic_3d_smoke import _track_object_points  # noqa: E402
from scripts.run_real_occlusion_observation_smoke import run_real_occlusion_observation_smoke  # noqa: E402


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


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


def _clip_frames(observation: ClipObservationJSON, indices: Sequence[int]) -> tuple[FrameObservationJSON, ...]:
    wanted = {int(value) for value in indices}
    frames = tuple(sorted((frame for frame in observation.frames if frame.frame_index in wanted), key=lambda item: item.frame_index))
    if len(frames) != len({frame.frame_index for frame in frames}):
        raise ValueError("Duplicate global frame indices in associated observations.")
    return frames


def _point_scales(points: Sequence[Any]) -> dict[str, dict[int, Optional[float]]]:
    grouped: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for point in points:
        coordinate = trajectory_coordinate(point)
        if coordinate is not None:
            grouped[point.object_track_id][point.frame_index].append(np.asarray(coordinate, dtype=float))
    output: dict[str, dict[int, Optional[float]]] = {}
    for track_id, by_frame in grouped.items():
        output[track_id] = {}
        for frame_index, coordinates in by_frame.items():
            distances = [
                float(np.linalg.norm(first - second))
                for index, first in enumerate(coordinates)
                for second in coordinates[index + 1 :]
            ]
            finite = [value for value in distances if math.isfinite(value) and value > 1e-8]
            output[track_id][frame_index] = float(np.percentile(finite, 75)) if finite else None
    return output


def _bool(value: str) -> bool:
    return value.strip().lower() == "true"


def _plot_coverage(output: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    videos = [str(row["video_id"]) for row in summaries]
    x = np.arange(len(videos))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].bar(x, [row["formal_mask_valid_ratio"] for row in summaries], label="formal mask")
    axes[0, 0].bar(x, [row["bbox_fallback_ratio"] for row in summaries], bottom=[row["formal_mask_valid_ratio"] for row in summaries], label="bbox fallback")
    axes[0, 0].set_title("Mask source coverage")
    axes[0, 0].legend()
    axes[0, 1].bar(x, [row["keypoint_valid_ratio"] for row in summaries], color="#59a14f")
    axes[0, 1].set_title("Human keypoint coverage")
    axes[1, 0].bar(x, [row["formal_structure_graph_count"] for row in summaries], color="#f28e2b")
    axes[1, 0].set_title("Formal fixed structure graphs")
    axes[1, 1].bar(x, [row["formal_occlusion_evidence_count"] for row in summaries], color="#e15759")
    axes[1, 1].set_title("Formal occlusion evidence")
    for axis in axes.flat:
        axis.set_xticks(x, videos, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("P3-D Real 3D Evidence Coverage (No Classification)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def _append_structure_temporal_rows(
    target: list[dict[str, Any]],
    *,
    video_id: str,
    clip_id: str,
    semantic_label: str,
    residuals: Sequence[Any],
) -> None:
    """Persist already-computed object and edge residuals for P4 tracing."""

    for residual in residuals:
        evidence = residual.object_structure_residual
        target.append({
            "video_id": video_id, "clip_id": clip_id,
            "object_track_id": residual.object_track_id,
            "semantic_label": semantic_label,
            "frame_index": residual.frame_index,
            "evidence_level": "object", "point_or_edge_id": "",
            "branch_name": "structure_temporal", "value": evidence.value,
            "valid": evidence.valid, "quality": evidence.quality,
            "missing_reason": evidence.missing_reason,
            "source_ids": ";".join(evidence.source_ids),
            "anomalous_point_ids": ";".join(residual.anomalous_point_ids),
            "anomalous_edge_ids": ";".join(residual.anomalous_edge_ids),
            "truth_labels_used": False,
        })
        for edge in residual.edge_residuals:
            edge_evidence = edge.normalized_edge_length_change
            target.append({
                "video_id": video_id, "clip_id": clip_id,
                "object_track_id": edge.object_track_id,
                "semantic_label": semantic_label,
                "frame_index": edge.frame_index,
                "evidence_level": "edge",
                "point_or_edge_id": f"{edge.point_id_a}:{edge.point_id_b}",
                "branch_name": "structure_temporal", "value": edge_evidence.value,
                "valid": edge_evidence.valid, "quality": edge_evidence.quality,
                "missing_reason": edge_evidence.missing_reason,
                "source_ids": ";".join(edge_evidence.source_ids),
                "anomalous_point_ids": f"{edge.point_id_a};{edge.point_id_b}",
                "anomalous_edge_ids": f"{edge.point_id_a}:{edge.point_id_b}",
                "truth_labels_used": False,
            })


def run_real_3d_evidence_coverage(
    *,
    geometry_root: Path,
    readiness_root: Path,
    observation_root: Path,
    output_root: Path,
    mask_model_path: Path,
    pose_model_path: Path,
    device: str = "cpu",
    clip_ids: Sequence[str] = DEFAULT_CLIPS,
) -> dict[str, Any]:
    """Run observation-only coverage over existing P3 shared clips."""

    output_root.mkdir(parents=True, exist_ok=True)
    mask_provider = RealInstanceMaskProvider(model_path=mask_model_path, device=device)
    pose_provider = RealHumanKeypointProvider(model_path=pose_model_path, device=device) if pose_model_path.exists() else None
    mask_rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    keypoint_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    structure_temporal_rows: list[dict[str, Any]] = []
    visibility_rows: list[dict[str, Any]] = []
    occlusion_rows: list[dict[str, Any]] = []
    frame_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for clip_id in clip_ids:
        manifest = geometry_root / clip_id / "shared_geometry_cache/shared_3d_clip_manifest.json"
        cache = load_shared_geometry_cache(manifest)
        readiness = _read_readiness(readiness_root / clip_id / "dynamic_readiness.json")
        observation_path = _find_associated_observation(cache.clip.video_id, observation_root)
        observation = load_clip_observation(observation_path)
        frames = _clip_frames(observation, cache.clip.frame_indices)
        mask_by_key: dict[tuple[int, str], Any] = {}
        updated_frames = []
        clip_mask_rows = []
        clip_association_rows = []
        for frame in frames:
            if mask_provider.available:
                candidates = mask_provider.predict_candidates(frame)
                association = associate_instance_masks(video_id=cache.clip.video_id, frame=frame, candidates=candidates)
                masks = association.masks
                diagnostics = association.diagnostics
            else:
                candidates = ()
                masks = mask_provider.predict(video_id=cache.clip.video_id, frame=frame)
                diagnostics = ()
            diagnostic_by_track = {item.object_track_id: item for item in diagnostics}
            replaced_objects = []
            for obj, mask in zip(frame.objects, masks):
                track_id = str(obj.track_id or obj.person_track_id or obj.object_id)
                mask_by_key[(frame.frame_index, track_id)] = mask
                formal = bool(mask.valid and not mask.is_legacy_bbox_fallback and mask.metadata.get("formal_mask_evidence", False))
                path = None
                if formal and mask.visible_mask is not None:
                    path = output_root / "formal_masks" / cache.clip.video_id / f"frame_{frame.frame_index:06d}_{track_id}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, mask.visible_mask, allow_pickle=False)
                row = {
                    "video_id": cache.clip.video_id,
                    "clip_id": clip_id,
                    "frame_index": frame.frame_index,
                    "object_id": obj.object_id,
                    "object_track_id": track_id,
                    "semantic_label": obj.label,
                    "mask_valid": mask.valid,
                    "formal_mask_valid": formal,
                    "visible_mask": mask.is_visible_mask,
                    "amodal_mask": mask.is_amodal_mask,
                    "bbox_fallback": mask.is_legacy_bbox_fallback,
                    "confidence": mask.confidence,
                    "source_provider": mask.source_provider,
                    "missing_reason": mask.missing_reason,
                    "association_quality": mask.metadata.get("association_quality", float("nan")),
                    "association_source": mask.metadata.get("association_source", "unassigned"),
                }
                mask_rows.append(row)
                clip_mask_rows.append(row)
                diagnostic = diagnostic_by_track.get(track_id)
                association_row = {
                    "video_id": cache.clip.video_id,
                    "clip_id": clip_id,
                    "frame_index": frame.frame_index,
                    "object_id": obj.object_id,
                    "object_track_id": track_id,
                    "candidate_id": None if diagnostic is None else diagnostic.candidate_id,
                    "association_quality": 0.0 if diagnostic is None else diagnostic.association_quality,
                    "association_source": "provider_unavailable" if diagnostic is None else diagnostic.association_source,
                    "valid": False if diagnostic is None else diagnostic.valid,
                    "missing_reason": mask.missing_reason if diagnostic is None else diagnostic.missing_reason,
                    "candidate_details": "[]" if diagnostic is None else json.dumps(_json_safe(diagnostic.candidate_details), sort_keys=True),
                }
                association_rows.append(association_row)
                clip_association_rows.append(association_row)
                replaced_objects.append(replace(
                    obj,
                    mask_path=None if path is None else str(path),
                    metadata={**dict(obj.metadata), "tracked_instance_mask": formal, "mask_source_provider": mask.source_provider},
                ))
            updated_frames.append(replace(frame, objects=replaced_objects))

        person_graphs, person_structure_count = [], 0
        person_keypoint_result = None
        if pose_provider is not None:
            person_keypoint_result = bind_person_keypoints_to_shared_3d(
                video_id=cache.clip.video_id,
                clip_id=clip_id,
                frames=updated_frames,
                provider=pose_provider,
                shared_clip=cache.clip,
                readiness=readiness,
            )
            keypoint_rows.extend({**asdict(item), "clip_id": clip_id} for item in person_keypoint_result.coverage)
            point_scales = _point_scales(person_keypoint_result.points_3d)
            person_tracks = assemble_object_point_tracks_3d(
                person_keypoint_result.bindings,
                person_keypoint_result.points_2d,
                person_keypoint_result.points_3d,
                object_scale_by_track_and_frame=point_scales,
            )
            person_ids = sorted({item.binding.object_track_id for item in person_tracks})
            for track_id in person_ids:
                graph = build_object_structure_graph(person_tracks, object_track_id=track_id, semantic_label="person")
                residuals = compute_structure_temporal_residuals(graph, person_keypoint_result.points_3d, point_scales.get(track_id, {}))
                _append_structure_temporal_rows(
                    structure_temporal_rows, video_id=cache.clip.video_id,
                    clip_id=clip_id, semantic_label="person", residuals=residuals,
                )
                evidence_count = sum(item.object_structure_residual.valid for item in residuals)
                person_structure_count += evidence_count
                person_graphs.append(graph)
                graph_rows.append({
                    "video_id": cache.clip.video_id, "clip_id": clip_id,
                    "object_track_id": track_id, "semantic_label": "person",
                    "graph_type": graph.graph_type, "point_source": "semantic_keypoints",
                    "point_count": len(graph.point_ids), "edge_count": len(graph.edges),
                    "valid": graph.valid, "quality": graph.quality,
                    "structure_temporal_evidence_count": evidence_count,
                    "missing_reason": graph.missing_reason,
                })

        formal_masks = [row for row in clip_mask_rows if row["formal_mask_valid"]]
        ordinary_graph_count = 0
        ordinary_structure_count = 0
        point_source_counts = Counter()
        if formal_masks and readiness.mode in {DynamicGeometryMode.STATIC_CAMERA_3D, DynamicGeometryMode.FULL_SE3_3D}:
            images = {index: __import__("cv2").imread(str(path)) for index, path in cache.frame_paths.items()}
            raw_points = _track_object_points(images, updated_frames)
            binding = bind_point_tracks_to_objects(raw_points, updated_frames, video_id=cache.clip.video_id, clip_id=clip_id)
            points_3d = reconstruct_point_tracks_3d(binding.points_2d, cache.clip, readiness)
            tracks = assemble_object_point_tracks_3d(binding.bindings, binding.points_2d, points_3d, object_scale_by_track_and_frame=_point_scales(points_3d))
            stable, _ = select_stable_object_point_tracks(tracks)
            point_source_counts.update(item.binding.assignment_source for item in stable)
            formal_tracks = [item for item in stable if item.binding.assignment_source in {"instance_mask", "tracked_instance_mask"}]
            ordinary = {
                str(obj.track_id or obj.person_track_id or obj.object_id): obj.label
                for frame in updated_frames for obj in frame.objects if obj.label != "person"
            }
            scales = _point_scales(points_3d)
            for track_id, label in sorted(ordinary.items()):
                graph = build_object_structure_graph(formal_tracks, object_track_id=track_id, semantic_label=label)
                residuals = compute_structure_temporal_residuals(graph, points_3d, scales.get(track_id, {}))
                _append_structure_temporal_rows(
                    structure_temporal_rows, video_id=cache.clip.video_id,
                    clip_id=clip_id, semantic_label=label, residuals=residuals,
                )
                evidence_count = sum(item.object_structure_residual.valid for item in residuals)
                ordinary_graph_count += int(graph.valid)
                ordinary_structure_count += evidence_count
                graph_rows.append({
                    "video_id": cache.clip.video_id, "clip_id": clip_id,
                    "object_track_id": track_id, "semantic_label": label,
                    "graph_type": graph.graph_type, "point_source": "real_mask_internal_points",
                    "point_count": len(graph.point_ids), "edge_count": len(graph.edges),
                    "valid": graph.valid, "quality": graph.quality,
                    "structure_temporal_evidence_count": evidence_count,
                    "missing_reason": graph.missing_reason,
                })
        else:
            ordinary = {
                str(obj.track_id or obj.person_track_id or obj.object_id): obj.label
                for frame in updated_frames for obj in frame.objects if obj.label != "person"
            }
            for track_id, label in sorted(ordinary.items()):
                graph_rows.append({
                    "video_id": cache.clip.video_id, "clip_id": clip_id,
                    "object_track_id": track_id, "semantic_label": label,
                    "graph_type": "unavailable", "point_source": "real_mask_internal_points",
                    "point_count": 0, "edge_count": 0, "valid": False,
                    "quality": 0.0, "structure_temporal_evidence_count": 0,
                    "missing_reason": "no_formal_instance_mask_points",
                })

        occlusion_dir = output_root / "occlusion" / clip_id
        occlusion_report = run_real_occlusion_observation_smoke(
            geometry_cache_manifest=manifest,
            readiness_path=readiness_root / clip_id / "dynamic_readiness.json",
            associated_observation_path=observation_path,
            output_dir=occlusion_dir,
            mask_provider=mask_provider,
        )
        clip_visibility = _read_csv(occlusion_dir / "visibility_states.csv")
        clip_depth = _read_csv(occlusion_dir / "depth_order_residuals.csv")
        clip_visibility_residuals = _read_csv(occlusion_dir / "visibility_residuals.csv")
        clip_boundary = _read_csv(occlusion_dir / "boundary_residuals.csv")
        for row in clip_visibility:
            visibility_rows.append({"video_id": cache.clip.video_id, "clip_id": clip_id, **row})
        for evidence_type, rows in (("depth_order", clip_depth), ("visibility", clip_visibility_residuals), ("boundary", clip_boundary)):
            for row in rows:
                occlusion_rows.append({
                    "video_id": cache.clip.video_id, "clip_id": clip_id,
                    "evidence_type": evidence_type,
                    "frame_index": row.get("frame_index", ""),
                    "valid": row.get("valid", "False"),
                    "residual": row.get("residual", "nan"),
                    "quality": row.get("quality", "0"),
                    "missing_reason": row.get("missing_reason", ""),
                })

        kp_clip_rows = [row for row in keypoint_rows if row["video_id"] == cache.clip.video_id and row["clip_id"] == clip_id]
        kp_ratio = float(np.mean([row["valid_ratio"] for row in kp_clip_rows])) if kp_clip_rows else float("nan")
        mask_ratio = len(formal_masks) / len(clip_mask_rows) if clip_mask_rows else float("nan")
        association_success = sum(row["valid"] for row in clip_association_rows) / len(clip_association_rows) if clip_association_rows else float("nan")
        visibility_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in clip_visibility:
            visibility_by_frame[int(row["frame_index"])].append(row)
        relations = _read_csv(occlusion_dir / "occlusion_relations.csv")
        relations_by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in relations:
            relations_by_frame[int(row["frame_index"])].append(row)
        keypoints_by_frame = {int(row["frame_index"]): float(row["valid_ratio"]) for row in kp_clip_rows}
        for frame in updated_frames:
            frame_masks = [row for row in clip_mask_rows if row["frame_index"] == frame.frame_index]
            states = visibility_by_frame.get(frame.frame_index, [])
            state_priority = next((name for name in ("fully_occluded", "partially_occluded", "reappeared", "out_of_frame") if any(row["current_state"] == name for row in states)), "no_occlusion_event")
            formal_relations = [row for row in relations_by_frame.get(frame.frame_index, []) if _bool(row["valid"])]
            frame_records.append({
                "video_id": cache.clip.video_id,
                "frame_index": frame.frame_index,
                "object_track_ids": ";".join(str(obj.track_id or obj.person_track_id or obj.object_id) for obj in frame.objects),
                "semantic_labels": ";".join(obj.label for obj in frame.objects),
                "mask_valid_ratio": sum(row["formal_mask_valid"] for row in frame_masks) / len(frame_masks) if frame_masks else 0.0,
                "formal_mask_object_count": sum(row["formal_mask_valid"] for row in frame_masks),
                "keypoint_valid_ratio": keypoints_by_frame.get(frame.frame_index, 0.0),
                "has_formal_mask_overlap": bool(formal_relations),
                "depth_order_confidence": max((float(row["occlusion_confidence"]) for row in formal_relations), default=0.0),
                "visibility_state": state_priority,
                "mean_tracking_quality": float(occlusion_report.get("mean_mask_iou") or 0.0),
                "scene_cut": bool(cache.clip.scene_cut_flags.get(frame.frame_index, False)),
                "geometry_mode": readiness.mode.value,
                "observation_quality": min(readiness.quality, max(mask_ratio if math.isfinite(mask_ratio) else 0.0, keypoints_by_frame.get(frame.frame_index, 0.0))),
            })

        formal_occlusion_count = (
            occlusion_report["formal_depth_order_evidence_count"]
            + occlusion_report["formal_visibility_residual_count"]
            + occlusion_report["formal_boundary_residual_count"]
            + occlusion_report["formal_reappearance_evidence_count"]
        )
        summaries.append({
            "video_id": cache.clip.video_id,
            "clip_id": clip_id,
            "geometry_mode": readiness.mode.value,
            "num_objects": len(clip_mask_rows),
            "formal_mask_count": len(formal_masks),
            "formal_mask_valid_ratio": mask_ratio,
            "bbox_fallback_count": sum(row["bbox_fallback"] for row in clip_mask_rows),
            "bbox_fallback_ratio": sum(row["bbox_fallback"] for row in clip_mask_rows) / len(clip_mask_rows) if clip_mask_rows else float("nan"),
            "mask_object_association_success_ratio": association_success,
            "keypoint_valid_ratio": kp_ratio,
            "person_structure_graph_count": sum(graph.valid for graph in person_graphs),
            "ordinary_mask_structure_graph_count": ordinary_graph_count,
            "formal_structure_graph_count": sum(graph.valid for graph in person_graphs) + ordinary_graph_count,
            "formal_structure_temporal_evidence_count": person_structure_count + ordinary_structure_count,
            "mask_internal_points": point_source_counts["instance_mask"] + point_source_counts["tracked_instance_mask"],
            "bbox_internal_points": point_source_counts["shrunk_bbox"] + point_source_counts["bbox_fallback"],
            "semantic_keypoints": sum(point.valid for point in person_keypoint_result.points_2d) if person_keypoint_result else 0,
            "boundary_points": 0,
            "partial_occlusion_candidates": occlusion_report["candidate_partial_occlusion_count"],
            "full_occlusion_candidates": occlusion_report["candidate_full_occlusion_count"],
            "formal_occlusion_evidence_count": formal_occlusion_count,
            "reappearance_event_count": occlusion_report["formal_reappearance_evidence_count"],
            "clip_scene_cut_count": occlusion_report["clip_scene_cut_count"],
            "object_visibility_scene_cut_markers": occlusion_report["object_visibility_scene_cut_markers"],
            "track_initialization_markers": occlusion_report["track_initialization_markers"],
            "state_machine_boundary_markers": occlusion_report["state_machine_boundary_markers"],
            "status": "observation_available" if len(formal_masks) or sum(graph.valid for graph in person_graphs) else "insufficient_real_3d_evidence",
            "primary_missing_reason": "" if len(formal_masks) or sum(graph.valid for graph in person_graphs) else (mask_provider.unavailable_reason or "no_formal_mask_or_structure_graph"),
        })

    candidates = find_evidence_clips(frame_records, minimum_duration=2)
    write_candidates(candidates, output_root / "evidence_clip_candidates.csv")
    (output_root / "frame_observation_availability.json").write_text(json.dumps(_json_safe(frame_records), indent=2) + "\n", encoding="utf-8")
    _write_csv(output_root / "mask_coverage.csv", mask_rows, tuple(mask_rows[0]) if mask_rows else ("video_id", "frame_index", "formal_mask_valid", "missing_reason"))
    _write_csv(output_root / "mask_object_association.csv", association_rows, tuple(association_rows[0]) if association_rows else ("video_id", "frame_index", "valid", "missing_reason"))
    _write_csv(output_root / "keypoint_coverage.csv", keypoint_rows, tuple(keypoint_rows[0]) if keypoint_rows else ("video_id", "frame_index", "valid", "missing_reason"))
    _write_csv(output_root / "structure_graph_coverage.csv", graph_rows, tuple(graph_rows[0]) if graph_rows else ("video_id", "object_track_id", "valid", "missing_reason"))
    _write_csv(
        output_root / "structure_temporal_evidence.csv", structure_temporal_rows,
        tuple(structure_temporal_rows[0]) if structure_temporal_rows else (
            "video_id", "clip_id", "object_track_id", "semantic_label",
            "frame_index", "evidence_level", "point_or_edge_id",
            "branch_name", "value", "valid", "quality", "missing_reason",
            "source_ids", "anomalous_point_ids", "anomalous_edge_ids",
            "truth_labels_used",
        ),
    )
    _write_csv(output_root / "visibility_event_coverage.csv", visibility_rows, tuple(visibility_rows[0]) if visibility_rows else ("video_id", "frame_index", "valid", "missing_reason"))
    _write_csv(output_root / "occlusion_evidence_coverage.csv", occlusion_rows, tuple(occlusion_rows[0]) if occlusion_rows else ("video_id", "evidence_type", "valid", "missing_reason"))
    _write_csv(output_root / "per_video_summary.csv", summaries, tuple(summaries[0]) if summaries else ("video_id", "status"))
    summary = {
        "coverage_version": "p3_d_real_evidence_coverage_v1",
        "video_count": len(summaries),
        "real_instance_mask_provider_available": mask_provider.available,
        "real_instance_mask_provider_missing_reason": mask_provider.unavailable_reason,
        "pose_provider_available": pose_provider is not None,
        "total_formal_masks": sum(row["formal_mask_count"] for row in summaries),
        "total_person_structure_graphs": sum(row["person_structure_graph_count"] for row in summaries),
        "total_ordinary_mask_structure_graphs": sum(row["ordinary_mask_structure_graph_count"] for row in summaries),
        "total_structure_temporal_evidence": sum(row["formal_structure_temporal_evidence_count"] for row in summaries),
        "total_partial_occlusion_candidates": sum(row["partial_occlusion_candidates"] for row in summaries),
        "total_full_occlusion_candidates": sum(row["full_occlusion_candidates"] for row in summaries),
        "total_formal_occlusion_evidence": sum(row["formal_occlusion_evidence_count"] for row in summaries),
        "total_reappearance_events": sum(row["reappearance_event_count"] for row in summaries),
        "candidate_clip_count": len(candidates),
        "truth_labels_used_for_selection": False,
        "residual_magnitude_used_for_selection": False,
        "classification_or_threshold_tuning_performed": False,
        "per_video": _json_safe(summaries),
    }
    (output_root / "global_summary.json").write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    _plot_coverage(output_root / "coverage_diagnostics.png", summaries)
    print(json.dumps(_json_safe(summary), indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_root", type=Path, default=PROJECT_ROOT / "outputs/sequence_geometry_stabilization")
    parser.add_argument("--readiness_root", type=Path, default=PROJECT_ROOT / "outputs/real_dynamic_3d_smoke")
    parser.add_argument("--observation_root", type=Path, default=PROJECT_ROOT / "outputs/evaluation/pilot_6video")
    parser.add_argument("--output_root", type=Path, default=PROJECT_ROOT / "outputs/real_3d_evidence_coverage")
    parser.add_argument("--mask_model_path", type=Path, default=PROJECT_ROOT / "checkpoints/yolov8n-seg.pt")
    parser.add_argument("--pose_model_path", type=Path, default=PROJECT_ROOT / "checkpoints/yolov8n-pose.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clip_id", action="append", dest="clip_ids")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_real_3d_evidence_coverage(
        geometry_root=args.geometry_root,
        readiness_root=args.readiness_root,
        observation_root=args.observation_root,
        output_root=args.output_root,
        mask_model_path=args.mask_model_path,
        pose_model_path=args.pose_model_path,
        device=args.device,
        clip_ids=tuple(args.clip_ids or DEFAULT_CLIPS),
    )


if __name__ == "__main__":
    main()
