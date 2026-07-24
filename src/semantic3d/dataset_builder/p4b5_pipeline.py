"""P4-B.5 full-observation extension of the immutable P4-B dataset builder.

The extension deliberately keeps frame-relative reconstruction separate from
clip-level sequence geometry.  It performs no label join, classifier training,
threshold fitting, or cross-clip 3D subtraction.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from semantic3d.keypoint_provider import COCO_PERSON_KEYPOINT_NAMES, RealHumanKeypointProvider
from semantic3d.occlusion.mask_observation import InstanceMaskObservation
from semantic3d.occlusion.mask_structure_points import track_formal_mask_internal_points

from .ids import stable_id
from .p4b5_contracts import (
    P4B5_PIPELINE_VERSION,
    ClipTrackHandoffObservation,
    CoverageMetric,
    adaptive_structure_point_target,
    build_fixed_structure_edges,
    classify_clip_geometry,
    synchronized_depth_order,
)
from .pipeline import (
    CAMERA_COLUMNS,
    CLIP_COLUMNS,
    DEPTH_COLUMNS,
    EVIDENCE_OUTPUT_COLUMNS,
    FRAME_COLUMNS,
    KEYPOINT_COLUMNS,
    SHARED_CLIP_COLUMNS,
    SHARED_FRAME_COLUMNS,
    Applicability,
    EvidenceRecord,
    StructuralEnhancementDatasetBuilder,
    _float,
    _json,
    _now,
)
from .reader import DatasetReader
from .writer import atomic_write_json, json_text, sha256_file, write_npz_array, write_parquet


P4B5_KEYPOINT_COLUMNS = KEYPOINT_COLUMNS + (
    "source_object_observation_id",
    "keypoints_2d",
    "matched_bbox",
    "source_version",
    "observed_independently",
)
P4B5_DEPTH_COLUMNS = DEPTH_COLUMNS + (
    "raw_array_name",
    "geometry_array_name",
    "visualization_array_name",
    "geometry_uses_visualization_depth",
    "provider_config_sha256",
)
P4B5_CAMERA_COLUMNS = CAMERA_COLUMNS + (
    "frame_geometry_valid",
    "sequence_geometry_valid",
    "sequence_missing_reason",
)
P4B5_SHARED_FRAME_COLUMNS = SHARED_FRAME_COLUMNS + (
    "frame_geometry_scope",
    "world_coordinates_available",
    "cross_frame_subtraction_allowed",
)
P4B5_SHARED_CLIP_COLUMNS = SHARED_CLIP_COLUMNS + (
    "frame_depth_coverage",
    "frame_shared_3d_coverage",
    "sequence_depth_aligned_coverage",
    "dynamic_3d_ready_coverage",
)
POINT_2D_COLUMNS = (
    "point_track_2d_observation_id", "video_id", "clip_id", "frame_id",
    "frame_index", "global_point_track_id", "clip_point_track_id",
    "global_object_track_id", "clip_object_track_id", "point_role",
    "semantic_keypoint_name", "pixel_uv", "tracking_confidence", "mask_support",
    "observed_independently", "is_context_frame", "is_owned_frame", "valid",
    "missing_reason", "source_provider", "metadata",
)
POINT_3D_COLUMNS = (
    "point_track_3d_observation_id", "video_id", "clip_id", "frame_id",
    "frame_index", "global_point_track_id", "clip_point_track_id",
    "global_object_track_id", "point_role", "semantic_keypoint_name", "pixel_uv", "observed_depth",
    "point_3d_camera", "point_3d_world", "ray_bearing", "trajectory_representation",
    "coordinate_system_id", "sequence_scale_status", "geometry_mode", "quality",
    "is_owned_frame", "valid", "missing_reason", "source_point_track_2d_id", "metadata",
)
KEYPOINT_3D_COLUMNS = (
    "keypoint_3d_observation_id", "video_id", "clip_id", "frame_id", "frame_index",
    "object_track_id", "keypoint_name", "pixel_uv", "point_3d_camera",
    "coordinate_system_id", "quality", "valid", "missing_reason",
    "source_keypoint_observation_id", "metadata",
)
MASK_TRACK_COLUMNS = (
    "mask_track_observation_id", "video_id", "clip_id", "object_track_id",
    "previous_frame_index", "current_frame_index", "independently_observed_mask",
    "history_predicted_mask", "tracked_mask", "observed_vs_predicted_iou",
    "normalized_boundary_distance", "area_change_ratio", "assignment_consistency",
    "track_switch_count", "valid", "missing_reason", "metadata",
)
HANDOFF_COLUMNS = (
    "handoff_id", "video_id", "source_clip_id", "target_clip_id",
    "global_object_track_id", "source_local_track_id", "target_local_track_id",
    "overlap_frame_ids", "mask_iou", "point_overlap_ratio", "appearance_similarity",
    "handoff_quality", "alignment_id", "allows_cross_clip_3d", "valid",
    "missing_reason", "metadata",
)
READINESS_COLUMNS = (
    "clip_id", "video_id", "geometry_mode", "sequence_scale_status", "valid",
    "dynamic_3d_ready", "quality", "missing_reason", "median_pixel_motion",
    "tracked_transition_ratio", "homography_inlier_ratio", "depth_aligned_ratio",
    "independent_track_coverage", "mean_track_length", "coordinate_system_id", "metadata",
)
STRUCTURE_GRAPH_COLUMNS = (
    "structure_graph_id", "video_id", "clip_id", "object_track_id", "semantic_label",
    "graph_type", "point_source", "anchor_frame_index", "point_ids", "edges",
    "coordinate_system_id", "fixed_topology", "formal_evidence", "valid",
    "missing_reason", "metadata",
)
STRUCTURE_TRANSITION_COLUMNS = (
    "structure_transition_id", "video_id", "clip_id", "object_track_id",
    "structure_graph_id", "previous_frame_index", "current_frame_index",
    "edge_count", "valid_edge_count", "raw_residual", "quality",
    "coordinate_system_id", "is_owned_frame", "valid", "missing_reason", "metadata",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _array_reference(path: Path) -> dict[str, Any]:
    with np.load(path) as data:
        shape = {name: list(np.asarray(data[name]).shape) for name in data.files}
        dtype = {name: str(np.asarray(data[name]).dtype) for name in data.files}
    return {"path": str(path), "shape": shape, "dtype": dtype, "sha256": sha256_file(path)}


def _safe_mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.count_nonzero(first | second))
    return int(np.count_nonzero(first & second)) / union if union else 0.0


def _bbox_mask(bbox: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    output = np.zeros(shape, dtype=bool)
    if x2 > x1 and y2 > y1:
        output[y1:y2, x1:x2] = True
    return output


class P4B5StructuralEnhancementDatasetBuilder(StructuralEnhancementDatasetBuilder):
    """Incremental P4-B.5 builder with full-frame and full-clip coverage."""

    def _previous_dataset_root(self) -> Path:
        value = self.config.get("dataset", {}).get(
            "previous_dataset_root",
            "outputs/structural_enhancement_dataset/p4b_six_video_smoke",
        )
        path = Path(str(value))
        return path if path.is_absolute() else self.project_root / path

    def _stage_01_video_index(self) -> list[Path]:
        artifacts = super()._stage_01_video_index()
        manifest_path = self.output_root / "dataset_manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload.update(
            {
                "pipeline_version": P4B5_PIPELINE_VERSION,
                "incremental_from": str(self._previous_dataset_root()),
                "coverage_semantics": [
                    "frame_depth_coverage",
                    "frame_shared_3d_coverage",
                    "sequence_depth_aligned_coverage",
                    "dynamic_3d_ready_coverage",
                    "formal_dynamic_evidence_coverage",
                ],
                "truth_labels_used": False,
                "classification_output": False,
                "strict_prior_hashes": {
                    name: sha256_file(self.project_root / "configs" / name)
                    for name in ("scale_priors_strict_v1.yaml", "scale_priors_strict_v2.yaml")
                },
            }
        )
        atomic_write_json(manifest_path, payload)
        return artifacts

    def _stage_02_frame_decode(self) -> list[Path]:
        """Reuse byte-identical old frames where possible, then decode gaps."""

        videos = self._source_map()
        owners = {
            (str(row["video_id"]), int(row["frame_index"])): row
            for row in self._owned_frames()
        }
        reused = 0
        previous = self._previous_dataset_root()
        for source_name, video in videos.items():
            for (video_id, frame_index), owner in owners.items():
                if video_id != video["video_id"]:
                    continue
                source = previous / "arrays/frames" / source_name / f"frame_{frame_index:06d}.jpg"
                target = self.output_root / str(owner["decoded_frame_path"])
                if not source.exists() or target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copy2(source, target)
                owner["decode_status"] = "ok"
                owner["failure_reason"] = ""
                reused += 1
        for source_name, video in videos.items():
            pending = {
                frame_index: owner
                for (video_id, frame_index), owner in owners.items()
                if video_id == video["video_id"]
                and not (self.output_root / str(owner["decoded_frame_path"])).exists()
            }
            if not pending:
                continue
            capture = cv2.VideoCapture(str(self._manifest_source_path(video)))
            try:
                frame_index = 0
                while capture.isOpened():
                    ok, image = capture.read()
                    if not ok:
                        break
                    owner = pending.get(frame_index)
                    if owner is not None:
                        target = self.output_root / str(owner["decoded_frame_path"])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(".tmp.jpg")
                        if image is not None and cv2.imwrite(str(temporary), image):
                            os.replace(temporary, target)
                            owner["decode_status"] = "ok"
                            owner["failure_reason"] = ""
                        else:
                            owner["decode_status"] = "failed"
                            owner["failure_reason"] = "cv2_imwrite_failed"
                    frame_index += 1
            finally:
                capture.release()
        summary = {
            "dataset_id": self.dataset_id,
            "unique_frames": len(owners),
            "decoded_frames": sum(
                (self.output_root / str(row["decoded_frame_path"])).exists()
                for row in owners.values()
            ),
            "reused_previous_frame_files": reused,
            "labels_used": False,
        }
        return [self.writer.json("reports/frame_decode_summary.json", summary)]

    def _stage_05_keypoints(self) -> list[Path]:
        """Run the frozen human-pose model for every detected person observation."""

        config = self.config.get("stages", {}).get("05_keypoints", {})
        weight = Path(str(self.config["providers"]["weights"]["human_keypoints"]))
        weight = weight if weight.is_absolute() else self.project_root / weight
        provider: Optional[RealHumanKeypointProvider] = None
        provider_error = ""
        try:
            provider = RealHumanKeypointProvider(
                model_path=weight,
                confidence_threshold=float(config.get("confidence_threshold", 0.25)),
                keypoint_confidence_threshold=float(
                    config.get("keypoint_confidence_threshold", 0.25)
                ),
                device=self.device,
            )
        except Exception as exc:  # each row remains explicit rather than aborting
            provider_error = f"keypoint_provider_unavailable:{type(exc).__name__}:{exc}"
        frame_by_id = {str(row["frame_id"]): row for row in self._owned_frames()}
        rows: list[dict[str, Any]] = []
        for obj in self._read("observations/objects.parquet"):
            if str(obj.get("canonical_label", "")).lower().replace(" ", "_") != "person":
                continue
            frame = frame_by_id.get(str(obj["frame_id"]))
            bbox = _json(obj.get("bbox"), obj.get("bbox"))
            valid = False
            reason = provider_error or "invalid_person_bbox"
            points: list[dict[str, Any]] = []
            matched_bbox = None
            provider_name = "ultralytics_yolov8_pose"
            if provider is not None and frame is not None and bbox and len(bbox) == 4:
                try:
                    prediction = provider.predict(
                        self.output_root / str(frame["decoded_frame_path"]), bbox, "person"
                    )
                    points = [point.to_dict() for point in prediction.keypoints]
                    matched_bbox = prediction.matched_bbox
                    provider_name = prediction.provider_name
                    valid = prediction.status == "ok" and any(point["valid"] for point in points)
                    reason = "" if valid else prediction.status
                except Exception as exc:
                    reason = f"keypoint_inference_failed:{type(exc).__name__}"
            valid_count = sum(bool(point.get("valid")) for point in points)
            rows.append(
                {
                    "keypoint_observation_id": stable_id(
                        "p4b5_keypoints", obj["object_observation_id"], prefix="kpobs"
                    ),
                    "video_id": obj["video_id"],
                    "frame_id": obj["frame_id"],
                    "frame_index": obj["frame_index"],
                    "object_track_id": obj["object_track_id"],
                    "total_keypoints": len(points),
                    "valid_keypoints": valid_count,
                    "valid_ratio": valid_count / len(points) if points else 0.0,
                    "provider_name": provider_name,
                    "valid": valid,
                    "missing_reason": "" if valid else reason,
                    "metadata": {
                        "full_video_inference": True,
                        "migrated_coverage_only": False,
                        "labels_used": False,
                        "weight_sha256": self.weight_hashes.get("human_keypoints", "missing"),
                    },
                    "source_object_observation_id": obj["object_observation_id"],
                    "keypoints_2d": points,
                    "matched_bbox": matched_bbox,
                    "source_version": P4B5_PIPELINE_VERSION,
                    "observed_independently": True,
                }
            )
        path = self.writer.parquet(
            "observations/keypoints.parquet", rows, P4B5_KEYPOINT_COLUMNS
        )
        return [path]

    def _old_depth_by_source_frame(self) -> dict[tuple[str, int], dict[str, Any]]:
        root = self._previous_dataset_root()
        reader = DatasetReader(root)
        videos = {row["video_id"]: row["source_name"] for row in reader.rows("manifests/videos.parquet")}
        return {
            (str(videos.get(row["video_id"], "")), int(row["frame_index"])): row
            for row in reader.rows("observations/depth.parquet")
            if row.get("valid")
        }

    def _stage_06_depth(self) -> list[Path]:
        """Run canonical relative depth for every decodable owned frame."""

        from semantic3d.depth_provider import RealDepthProvider

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        model_name = str(
            self.config.get("providers", {}).get(
                "depth_model", "depth-anything/Depth-Anything-V2-Small"
            )
        )
        source_by_id = {
            row["video_id"]: row for row in self._read("manifests/videos.parquet")
        }
        old_depth = self._old_depth_by_source_frame()
        provider: Optional[RealDepthProvider] = None
        provider_setup_error = ""
        rows: list[dict[str, Any]] = []
        artifacts: list[Path] = []
        for frame in self._owned_frames():
            video = source_by_id[frame["video_id"]]
            source_name = str(video["source_name"])
            frame_index = int(frame["frame_index"])
            target = self.output_root / "arrays/depth" / source_name / f"frame_{frame_index:06d}.npz"
            valid = False
            reason = "frame_image_unavailable"
            ref = {"path": "", "shape": {}, "dtype": {}, "sha256": ""}
            metadata: dict[str, Any] = {
                "metric_depth": False,
                "labels_used": False,
                "raw_model_output_semantics": "model_native_inverse_or_disparity_like",
                "geometry_depth_semantics": "canonical_relative_depth_larger_is_farther",
                "visualization_depth_saved": False,
                "visualization_depth_used_by_geometry": False,
                "conversion": "reciprocal_inverse_to_relative",
            }
            previous = old_depth.get((source_name, frame_index))
            if not target.exists() and previous:
                source = Path(str(previous.get("array_path", "")))
                if not source.is_absolute():
                    source = self._previous_dataset_root() / source
                if source.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                    metadata["reused_previous_p4b_depth"] = True
            if target.exists():
                try:
                    with np.load(target) as data:
                        required = {"depth_map", "valid_mask", "raw_model_output"}
                        if not required.issubset(data.files):
                            raise ValueError("missing raw/canonical depth arrays")
                        valid_mask = np.asarray(data["valid_mask"], dtype=bool)
                    ref = _array_reference(target)
                    valid = bool(np.any(valid_mask))
                    reason = "" if valid else "depth_has_no_valid_pixels"
                    metadata["resumed_from_existing_frame_artifact"] = True
                except Exception:
                    target.unlink(missing_ok=True)
            frame_path = self.output_root / str(frame["decoded_frame_path"])
            if not target.exists() and frame_path.exists():
                if provider is None and not provider_setup_error:
                    try:
                        provider = RealDepthProvider(
                            model_name=model_name,
                            device=self.device,
                            normalize=False,
                            invert_depth=True,
                        )
                    except Exception as exc:
                        provider_setup_error = (
                            f"depth_provider_unavailable:{type(exc).__name__}:{exc}"
                        )
                if provider is not None:
                    try:
                        observation = provider.predict_observation(
                            frame_path, frame_index=frame_index
                        )
                        geometry = observation.require_geometry_depth()
                        ref = write_npz_array(
                            target,
                            depth_map=np.asarray(geometry, dtype=np.float16),
                            relative_depth=np.asarray(geometry, dtype=np.float16),
                            valid_mask=np.asarray(observation.valid_mask, dtype=np.uint8),
                            raw_model_output=np.asarray(
                                observation.raw_model_output, dtype=np.float16
                            ),
                        )
                        valid = bool(observation.valid)
                        reason = observation.missing_reason
                        metadata.update(observation.metadata)
                    except Exception as exc:
                        reason = f"depth_inference_failed:{type(exc).__name__}:{exc}"
                else:
                    reason = provider_setup_error
            if target.exists() and not ref["path"]:
                ref = _array_reference(target)
                with np.load(target) as data:
                    valid = bool(np.any(np.asarray(data["valid_mask"], dtype=bool)))
                reason = "" if valid else "depth_has_no_valid_pixels"
            valid_ratio = 0.0
            if valid and target.exists():
                with np.load(target) as data:
                    valid_ratio = float(np.mean(np.asarray(data["valid_mask"], dtype=bool)))
                artifacts.append(target)
            rows.append(
                {
                    "depth_observation_id": stable_id(
                        "p4b5_depth", frame["frame_id"], prefix="depthobs"
                    ),
                    "video_id": frame["video_id"],
                    "frame_id": frame["frame_id"],
                    "frame_index": frame_index,
                    "array_path": ref["path"],
                    "array_shape": ref["shape"],
                    "array_dtype": ref["dtype"],
                    "array_sha256": ref["sha256"],
                    "depth_representation": "relative_depth" if valid else "unknown",
                    "scale_status": "relative_per_frame" if valid else "unknown",
                    "larger_value_means": "farther" if valid else "unknown",
                    "provider_name": "transformers:depth-anything/Depth-Anything-V2-Small-hf",
                    "valid_pixel_ratio": valid_ratio,
                    "valid": valid,
                    "missing_reason": "" if valid else reason,
                    "metadata": metadata,
                    "raw_array_name": "raw_model_output" if valid else "",
                    "geometry_array_name": "depth_map" if valid else "",
                    "visualization_array_name": "",
                    "geometry_uses_visualization_depth": False,
                    "provider_config_sha256": hashlib.sha256(
                        json_text(
                            {
                                "model_name": model_name,
                                "normalize": False,
                                "invert_depth": True,
                                "weight_hash": self.weight_hashes.get(
                                    "depth_model_cache", "missing"
                                ),
                            }
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        path = self.writer.parquet(
            "observations/depth.parquet", rows, P4B5_DEPTH_COLUMNS
        )
        return [path, *artifacts]

    def _mask_observation(
        self,
        mask_row: Mapping[str, Any],
        object_row: Mapping[str, Any],
    ) -> Optional[InstanceMaskObservation]:
        data = self._load_npz(str(mask_row.get("array_path", "")))
        if data is None or "visible_mask" not in data:
            return None
        mask = np.asarray(data["visible_mask"], dtype=bool)
        return InstanceMaskObservation.from_visible_mask(
            video_id=str(mask_row["video_id"]),
            frame_index=int(mask_row["frame_index"]),
            object_track_id=str(mask_row["object_track_id"]),
            semantic_label=str(object_row.get("semantic_label", "unknown")),
            mask=mask,
            confidence=float(mask_row.get("confidence") or 0.0),
            source_provider=str(mask_row.get("source_provider", "formal_instance_mask")),
            metadata={
                "formal_mask_evidence": True,
                "source_mask_observation_id": mask_row["mask_observation_id"],
                "legacy_bbox_fallback": False,
            },
        )

    def _stage_07_tracking(self) -> list[Path]:
        """Track formal-mask internal points independently inside every clip."""

        config = self.config.get("stages", {}).get("07_tracking", {})
        minimum = int(config.get("point_target_min", 4))
        maximum = int(config.get("point_target_max", 24))
        frames = self._read("manifests/frames.parquet")
        objects = self._read("observations/objects.parquet")
        object_by_key = {
            (str(row["frame_id"]), str(row["object_track_id"])): row for row in objects
        }
        masks = [row for row in self._read("observations/masks.parquet") if row.get("valid")]
        masks_by_key = {
            (str(row["frame_id"]), str(row["object_track_id"])): row for row in masks
        }
        keypoints = {
            (str(row["frame_id"]), str(row["object_track_id"])): row
            for row in self._read("observations/keypoints.parquet")
        }
        frames_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in frames:
            frames_by_clip[str(row["clip_id"])].append(row)
        point_rows: list[dict[str, Any]] = []
        mask_track_rows: list[dict[str, Any]] = []
        local_points: dict[tuple[str, str, int], list[tuple[str, float, float]]] = defaultdict(list)
        for clip_id, clip_frames in sorted(frames_by_clip.items()):
            clip_frames.sort(key=lambda row: int(row["frame_index"]))
            frame_info = {int(row["frame_index"]): row for row in clip_frames}
            images: dict[int, np.ndarray] = {}
            for row in clip_frames:
                image = cv2.imread(str(self.output_root / str(row["decoded_frame_path"])))
                if image is not None:
                    images[int(row["frame_index"])] = image
            tracks: dict[str, list[InstanceMaskObservation]] = defaultdict(list)
            mask_rows_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for frame in clip_frames:
                frame_id = str(frame["frame_id"])
                for (candidate_frame, track_id), mask_row in masks_by_key.items():
                    if candidate_frame != frame_id:
                        continue
                    obj = object_by_key.get((frame_id, track_id))
                    if obj is None:
                        continue
                    observation = self._mask_observation(mask_row, obj)
                    if observation is not None:
                        tracks[track_id].append(observation)
                        mask_rows_by_track[track_id].append(mask_row)
            for track_id, observations in sorted(tracks.items()):
                observations.sort(key=lambda item: item.frame_index)
                if len(observations) < 2:
                    continue
                point_target = adaptive_structure_point_target(
                    observations[0].mask_area, minimum=minimum, maximum=maximum
                )
                tracked = track_formal_mask_internal_points(
                    images,
                    observations,
                    max_points=point_target,
                    erosion_pixels=None,
                )
                for point in tracked:
                    frame = frame_info.get(point.frame_index)
                    if frame is None:
                        continue
                    clip_point_id = stable_id(
                        "clip_point", clip_id, point.point_id, prefix="cpt"
                    )
                    global_point_id = stable_id(
                        "unresolved_global_point", clip_id, point.point_id, prefix="gpt"
                    )
                    uv = list(point.pixel_uv) if point.pixel_uv is not None else None
                    row = {
                        "point_track_2d_observation_id": stable_id(
                            "point2d", clip_point_id, point.frame_index, prefix="pt2d"
                        ),
                        "video_id": frame["video_id"],
                        "clip_id": clip_id,
                        "frame_id": frame["frame_id"],
                        "frame_index": point.frame_index,
                        "global_point_track_id": global_point_id,
                        "clip_point_track_id": clip_point_id,
                        "global_object_track_id": track_id,
                        "clip_object_track_id": stable_id(
                            "clip_object_track", clip_id, track_id, prefix="cot"
                        ),
                        "point_role": "internal_stable_point",
                        "semantic_keypoint_name": "",
                        "pixel_uv": uv,
                        "tracking_confidence": point.tracking_confidence,
                        "mask_support": bool(point.valid),
                        "observed_independently": True,
                        "is_context_frame": bool(frame["is_context_frame"]),
                        "is_owned_frame": bool(frame["is_owned_frame"]),
                        "valid": bool(point.valid),
                        "missing_reason": point.missing_reason,
                        "source_provider": point.source_tracker,
                        "metadata": {
                            **dict(point.metadata),
                            "formal_mask_only": True,
                            "bbox_fallback": False,
                            "adaptive_point_target": point_target,
                            "cross_clip_identity_confirmed": False,
                        },
                    }
                    point_rows.append(row)
                    if point.valid and point.pixel_uv is not None:
                        local_points[(clip_id, track_id, point.frame_index)].append(
                            (clip_point_id, float(point.pixel_uv[0]), float(point.pixel_uv[1]))
                        )
                ordered_masks = sorted(observations, key=lambda item: item.frame_index)
                for previous, current in zip(ordered_masks, ordered_masks[1:]):
                    assert previous.visible_mask is not None and current.visible_mask is not None
                    iou = _safe_mask_iou(previous.visible_mask, current.visible_mask)
                    area_change = abs(current.mask_area - previous.mask_area) / max(
                        previous.mask_area, 1.0
                    )
                    mask_track_rows.append(
                        {
                            "mask_track_observation_id": stable_id(
                                "mask_track", clip_id, track_id, current.frame_index,
                                prefix="mtrack",
                            ),
                            "video_id": current.video_id,
                            "clip_id": clip_id,
                            "object_track_id": track_id,
                            "previous_frame_index": previous.frame_index,
                            "current_frame_index": current.frame_index,
                            "independently_observed_mask": True,
                            "history_predicted_mask": False,
                            "tracked_mask": True,
                            "observed_vs_predicted_iou": iou,
                            "normalized_boundary_distance": math.nan,
                            "area_change_ratio": area_change,
                            "assignment_consistency": 1.0,
                            "track_switch_count": 0,
                            "valid": True,
                            "missing_reason": "",
                            "metadata": {
                                "current_mask_used_for_prediction": False,
                                "observed_to_observed_continuity_diagnostic": True,
                            },
                        }
                    )
            # Semantic human points are independently observed in each frame;
            # their COCO names provide stable clip-local identity.
            for frame in clip_frames:
                frame_id = str(frame["frame_id"])
                for (candidate_frame, track_id), kp_row in keypoints.items():
                    if candidate_frame != frame_id:
                        continue
                    for point in _json(kp_row.get("keypoints_2d"), []):
                        name = str(point.get("keypoint_name", ""))
                        valid = bool(point.get("valid"))
                        clip_point_id = stable_id(
                            "clip_semantic_keypoint", clip_id, track_id, name, prefix="cpt"
                        )
                        global_point_id = stable_id(
                            "global_semantic_keypoint", track_id, name, prefix="gpt"
                        )
                        uv = [float(point["x"]), float(point["y"])] if valid else None
                        point_rows.append(
                            {
                                "point_track_2d_observation_id": stable_id(
                                    "point2d", clip_point_id, frame["frame_index"], prefix="pt2d"
                                ),
                                "video_id": frame["video_id"],
                                "clip_id": clip_id,
                                "frame_id": frame_id,
                                "frame_index": frame["frame_index"],
                                "global_point_track_id": global_point_id,
                                "clip_point_track_id": clip_point_id,
                                "global_object_track_id": track_id,
                                "clip_object_track_id": stable_id(
                                    "clip_object_track", clip_id, track_id, prefix="cot"
                                ),
                                "point_role": "semantic_keypoint",
                                "semantic_keypoint_name": name,
                                "pixel_uv": uv,
                                "tracking_confidence": float(point.get("confidence") or 0.0),
                                "mask_support": None,
                                "observed_independently": True,
                                "is_context_frame": bool(frame["is_context_frame"]),
                                "is_owned_frame": bool(frame["is_owned_frame"]),
                                "valid": valid,
                                "missing_reason": "" if valid else "keypoint_invalid",
                                "source_provider": kp_row.get("provider_name", ""),
                                "metadata": {
                                    "semantic_identity_fixed": True,
                                    "generated_from_projection": False,
                                    "source_keypoint_observation_id": kp_row[
                                        "keypoint_observation_id"
                                    ],
                                },
                            }
                        )
        handoffs = self._build_clip_handoffs(frames_by_clip, masks_by_key, local_points)
        track_counts = Counter(row["object_track_id"] for row in objects if row.get("valid"))
        track_path = write_parquet(
            self.output_root / "observations/tracks.parquet",
            [
                {
                    "object_track_id": track_id,
                    "observation_count": count,
                    "valid": count > 0,
                    "missing_reason": "" if count > 0 else "no_observations",
                }
                for track_id, count in sorted(track_counts.items())
            ],
            columns=("object_track_id", "observation_count", "valid", "missing_reason"),
        )
        point_path = write_parquet(
            self.output_root / "observations/point_tracks_2d.parquet",
            point_rows,
            columns=POINT_2D_COLUMNS,
        )
        mask_track_path = write_parquet(
            self.output_root / "observations/mask_tracks.parquet",
            mask_track_rows,
            columns=MASK_TRACK_COLUMNS,
        )
        handoff_path = write_parquet(
            self.output_root / "observations/clip_track_handoffs.parquet",
            handoffs,
            columns=HANDOFF_COLUMNS,
        )
        return [track_path, point_path, mask_track_path, handoff_path]

    def _build_clip_handoffs(
        self,
        frames_by_clip: Mapping[str, Sequence[Mapping[str, Any]]],
        masks_by_key: Mapping[tuple[str, str], Mapping[str, Any]],
        local_points: Mapping[tuple[str, str, int], Sequence[tuple[str, float, float]]],
    ) -> list[dict[str, Any]]:
        clips = {row["clip_id"]: row for row in self._read("manifests/clips.parquet")}
        by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for clip in clips.values():
            if clip.get("valid"):
                by_video[str(clip["video_id"])].append(clip)
        output: list[dict[str, Any]] = []
        for video_id, video_clips in by_video.items():
            video_clips.sort(key=lambda row: int(row["start_frame_index"]))
            for source, target in zip(video_clips, video_clips[1:]):
                source_frames = {
                    int(row["frame_index"]): str(row["frame_id"])
                    for row in frames_by_clip.get(str(source["clip_id"]), ())
                }
                target_frames = {
                    int(row["frame_index"]): str(row["frame_id"])
                    for row in frames_by_clip.get(str(target["clip_id"]), ())
                }
                overlap = sorted(set(source_frames) & set(target_frames))
                source_tracks = {
                    track_id
                    for frame_index in source_frames
                    for frame_id, track_id in masks_by_key
                    if frame_id == source_frames[frame_index]
                }
                target_tracks = {
                    track_id
                    for frame_index in target_frames
                    for frame_id, track_id in masks_by_key
                    if frame_id == target_frames[frame_index]
                }
                for track_id in sorted(source_tracks & target_tracks):
                    ious: list[float] = []
                    matched_points = total_points = 0
                    overlap_ids: list[str] = []
                    for frame_index in overlap:
                        frame_id = source_frames[frame_index]
                        mask_row = masks_by_key.get((frame_id, track_id))
                        if mask_row is not None:
                            data = self._load_npz(str(mask_row["array_path"]))
                            if data is not None and "visible_mask" in data:
                                mask = np.asarray(data["visible_mask"], dtype=bool)
                                ious.append(_safe_mask_iou(mask, mask))
                                overlap_ids.append(frame_id)
                        first = local_points.get((str(source["clip_id"]), track_id, frame_index), ())
                        second = local_points.get((str(target["clip_id"]), track_id, frame_index), ())
                        total_points += len(first)
                        for _, x, y in first:
                            if any(math.hypot(x - tx, y - ty) <= 3.0 for _, tx, ty in second):
                                matched_points += 1
                    point_ratio = matched_points / total_points if total_points else 0.0
                    mask_iou = float(np.mean(ious)) if ious else 0.0
                    quality = float(min(mask_iou, point_ratio if total_points else mask_iou))
                    observation = ClipTrackHandoffObservation(
                        handoff_id=stable_id(
                            "clip_handoff", source["clip_id"], target["clip_id"], track_id,
                            prefix="handoff",
                        ),
                        video_id=video_id,
                        source_clip_id=str(source["clip_id"]),
                        target_clip_id=str(target["clip_id"]),
                        global_object_track_id=track_id,
                        source_local_track_id=stable_id(
                            "clip_object_track", source["clip_id"], track_id, prefix="cot"
                        ),
                        target_local_track_id=stable_id(
                            "clip_object_track", target["clip_id"], track_id, prefix="cot"
                        ),
                        overlap_frame_ids=tuple(overlap_ids),
                        mask_iou=mask_iou,
                        point_overlap_ratio=point_ratio,
                        appearance_similarity=math.nan,
                        handoff_quality=quality,
                        alignment_id="",
                        allows_cross_clip_3d=False,
                        valid=bool(overlap_ids),
                        missing_reason="" if overlap_ids else "no_formal_mask_overlap_support",
                        metadata={
                            "identity_handoff_only": True,
                            "cross_clip_geometry_authorized": False,
                        },
                    )
                    output.append(observation.to_dict())
        return output

    @staticmethod
    def _clip_motion_metrics(
        images: Sequence[np.ndarray],
        source_foreground: Optional[np.ndarray] = None,
        target_foreground: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float]:
        if len(images) < 2:
            return math.nan, 0.0, 0.0
        first = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
        last = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
        feature_mask = None
        if source_foreground is not None and source_foreground.shape == first.shape:
            feature_mask = (~np.asarray(source_foreground, dtype=bool)).astype(np.uint8) * 255
        points = cv2.goodFeaturesToTrack(
            first, maxCorners=500, qualityLevel=0.01, minDistance=7, mask=feature_mask
        )
        if points is None or len(points) < 8:
            return math.nan, 0.0, 0.0
        current, status, _ = cv2.calcOpticalFlowPyrLK(first, last, points, None)
        if current is None or status is None:
            return math.nan, 0.0, 0.0
        good = status.reshape(-1).astype(bool)
        source = points.reshape(-1, 2)[good]
        target = current.reshape(-1, 2)[good]
        if target_foreground is not None and target_foreground.shape == last.shape and len(target):
            columns = np.clip(np.rint(target[:, 0]).astype(int), 0, last.shape[1] - 1)
            rows = np.clip(np.rint(target[:, 1]).astype(int), 0, last.shape[0] - 1)
            background = ~np.asarray(target_foreground, dtype=bool)[rows, columns]
            source, target = source[background], target[background]
        ratio = len(source) / len(points)
        if len(source) < 4:
            return math.nan, ratio, 0.0
        motion = float(np.median(np.linalg.norm(target - source, axis=1)))
        _, inlier_mask = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        inliers = float(np.mean(inlier_mask)) if inlier_mask is not None else 0.0
        return motion, ratio, inliers

    def _stage_08_geometry(self) -> list[Path]:
        """Classify every clip without claiming unsupported translation geometry."""

        clips = self._read("manifests/clips.parquet")
        frames_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._read("manifests/frames.parquet"):
            frames_by_clip[str(row["clip_id"])].append(row)
        depths = {str(row["frame_id"]): row for row in self._read("observations/depth.parquet")}
        masks_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._read("observations/masks.parquet"):
            if row.get("valid"):
                masks_by_frame[str(row["frame_id"])].append(row)
        point_rows = self._read("observations/point_tracks_2d.parquet")
        points_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in point_rows:
            if row.get("valid"):
                points_by_clip[str(row["clip_id"])].append(row)
        videos = {row["video_id"]: row for row in self._read("manifests/videos.parquet")}
        camera_rows: list[dict[str, Any]] = []
        readiness_rows: list[dict[str, Any]] = []
        config = self.config.get("stages", {}).get("08_sequence_geometry", {})
        for clip in clips:
            clip_id = str(clip["clip_id"])
            clip_frames = sorted(
                frames_by_clip.get(clip_id, []), key=lambda row: int(row["frame_index"])
            )
            image_rows = [
                (frame, image)
                for frame in clip_frames
                if (image := cv2.imread(str(self.output_root / str(frame["decoded_frame_path"]))))
                is not None
            ]
            images = [item[1] for item in image_rows]
            foreground_ends: list[Optional[np.ndarray]] = [None, None]
            if image_rows:
                for position, (frame, image) in enumerate((image_rows[0], image_rows[-1])):
                    foreground = np.zeros(image.shape[:2], dtype=bool)
                    for mask_row in masks_by_frame.get(str(frame["frame_id"]), []):
                        mask_data = self._load_npz(str(mask_row["array_path"]))
                        if mask_data is not None and "visible_mask" in mask_data:
                            foreground |= np.asarray(mask_data["visible_mask"], dtype=bool)
                    foreground_ends[position] = foreground
            motion, tracked_ratio, homography_ratio = self._clip_motion_metrics(
                images, foreground_ends[0], foreground_ends[1]
            )
            medians: dict[str, float] = {}
            for frame in clip_frames:
                depth_row = depths.get(str(frame["frame_id"]))
                if not depth_row or not depth_row.get("valid"):
                    continue
                data = self._load_npz(str(depth_row["array_path"]))
                if data is None or "depth_map" not in data or "valid_mask" not in data:
                    continue
                depth_map = np.asarray(data["depth_map"], dtype=float)
                support = np.asarray(data["valid_mask"], dtype=bool)
                foreground = np.zeros(depth_map.shape, dtype=bool)
                for mask_row in masks_by_frame.get(str(frame["frame_id"]), []):
                    mask_data = self._load_npz(str(mask_row["array_path"]))
                    if mask_data is not None and "visible_mask" in mask_data:
                        foreground |= np.asarray(mask_data["visible_mask"], dtype=bool)
                support &= ~foreground & np.isfinite(depth_map) & (depth_map > 0)
                if int(np.count_nonzero(support)) >= 64:
                    medians[str(frame["frame_id"])] = float(np.median(depth_map[support]))
            aligned_ratio = len(medians) / len(clip_frames) if clip_frames else 0.0
            decision = classify_clip_geometry(
                median_pixel_motion=motion,
                tracked_transition_ratio=tracked_ratio,
                homography_inlier_ratio=homography_ratio,
                depth_aligned_ratio=aligned_ratio,
                static_motion_threshold=float(config.get("static_motion_threshold_px", 1.5)),
                rotation_motion_limit=float(config.get("rotation_motion_limit_px", 20.0)),
            )
            valid_points = points_by_clip.get(clip_id, [])
            point_lengths = Counter(row["clip_point_track_id"] for row in valid_points)
            mean_length = float(np.mean(list(point_lengths.values()))) if point_lengths else 0.0
            independent_coverage = (
                len({int(row["frame_index"]) for row in valid_points}) / len(clip_frames)
                if clip_frames
                else 0.0
            )
            dynamic_ready = bool(
                decision.valid
                and decision.geometry_mode == "static_camera_3d"
                and independent_coverage >= 0.2
                and mean_length >= 3.0
            )
            if dynamic_ready:
                missing_reason = ""
            elif decision.geometry_mode == "rotation_compensated":
                missing_reason = "rotation_transform_not_materialized"
            else:
                missing_reason = decision.missing_reason or "insufficient_independent_point_tracks"
            readiness_rows.append(
                {
                    "clip_id": clip_id,
                    "video_id": clip["video_id"],
                    "geometry_mode": decision.geometry_mode if dynamic_ready else "unavailable",
                    "sequence_scale_status": decision.sequence_scale_status,
                    "valid": dynamic_ready,
                    "dynamic_3d_ready": dynamic_ready,
                    "quality": decision.quality if dynamic_ready else 0.0,
                    "missing_reason": missing_reason,
                    "median_pixel_motion": motion,
                    "tracked_transition_ratio": tracked_ratio,
                    "homography_inlier_ratio": homography_ratio,
                    "depth_aligned_ratio": aligned_ratio,
                    "independent_track_coverage": independent_coverage,
                    "mean_track_length": mean_length,
                    "coordinate_system_id": clip["coordinate_system_id"],
                    "metadata": {
                        "full_se3_claimed": False,
                        "truth_labels_used": False,
                        "classification_is_geometry_readiness_only": True,
                        "diagnostic_geometry_regime": decision.geometry_mode,
                        "rotation_compensation_materialized": False,
                    },
                }
            )
            valid_medians = [value for value in medians.values() if value > 0]
            reference = float(np.median(valid_medians)) if valid_medians else math.nan
            video = videos[clip["video_id"]]
            width, height = int(video["width"]), int(video["height"])
            focal = float(max(width, height))
            K = [
                [focal, 0.0, (width - 1) / 2.0],
                [0.0, focal, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ]
            for frame in clip_frames:
                current = medians.get(str(frame["frame_id"]), math.nan)
                aligned = math.isfinite(reference) and math.isfinite(current) and current > 0
                scale = reference / current if aligned else math.nan
                frame_depth_valid = bool(depths.get(str(frame["frame_id"]), {}).get("valid"))
                camera_rows.append(
                    {
                        "camera_observation_id": stable_id(
                            "p4b5_camera", clip_id, frame["frame_id"], prefix="camobs"
                        ),
                        "video_id": frame["video_id"],
                        "clip_id": clip_id,
                        "frame_id": frame["frame_id"],
                        "frame_index": frame["frame_index"],
                        "coordinate_system_id": clip["coordinate_system_id"],
                        "K": K if frame_depth_valid else None,
                        "T_world_camera": None,
                        "coordinate_convention": "right_handed_camera_x_right_y_down_z_forward",
                        "intrinsics_source": "approximate_focal_length_relative_geometry",
                        "pose_source": "static_identity_supported_by_image_motion" if dynamic_ready and decision.geometry_mode == "static_camera_3d" else "",
                        "geometry_mode": decision.geometry_mode if dynamic_ready else "unavailable",
                        "depth_alignment_scale": scale,
                        "quality": decision.quality if dynamic_ready and aligned else (0.25 if frame_depth_valid else 0.0),
                        "valid": frame_depth_valid,
                        "missing_reason": "" if frame_depth_valid else "frame_depth_unavailable",
                        "metadata": {
                            "frame_camera_relative_intrinsics_only": True,
                            "metric_intrinsics_calibration": False,
                            "sequence_alignment_valid": aligned,
                            "world_pose_available": False,
                        },
                        "frame_geometry_valid": frame_depth_valid,
                        "sequence_geometry_valid": dynamic_ready and aligned,
                        "sequence_missing_reason": "" if dynamic_ready and aligned else missing_reason,
                    }
                )
        camera_path = self.writer.parquet(
            "observations/camera.parquet", camera_rows, P4B5_CAMERA_COLUMNS
        )
        readiness_path = self.writer.parquet(
            "observations/dynamic_readiness.parquet", readiness_rows, READINESS_COLUMNS
        )
        return [camera_path, readiness_path]

    @staticmethod
    def _backproject(uv: Sequence[float], depth: float, K: np.ndarray) -> list[float]:
        u, v = float(uv[0]), float(uv[1])
        return [
            (u - float(K[0, 2])) * depth / float(K[0, 0]),
            (v - float(K[1, 2])) * depth / float(K[1, 1]),
            depth,
        ]

    @staticmethod
    def _sample_depth(
        depth_map: np.ndarray,
        valid_mask: np.ndarray,
        uv: Sequence[float],
    ) -> float:
        column, row = int(round(float(uv[0]))), int(round(float(uv[1])))
        if not (0 <= row < depth_map.shape[0] and 0 <= column < depth_map.shape[1]):
            return math.nan
        value = float(depth_map[row, column])
        return value if valid_mask[row, column] and math.isfinite(value) and value > 0 else math.nan

    def _stage_09_shared_3d(self) -> list[Path]:
        """Build frame-relative object 3D and gated clip-local point trajectories."""

        frames = self._read("manifests/frames.parquet")
        owned = [row for row in frames if row.get("is_owned_frame")]
        clips = {str(row["clip_id"]): row for row in self._read("manifests/clips.parquet")}
        videos = {str(row["video_id"]): row for row in self._read("manifests/videos.parquet")}
        depths = {str(row["frame_id"]): row for row in self._read("observations/depth.parquet")}
        cameras = {
            (str(row["clip_id"]), str(row["frame_id"])): row
            for row in self._read("observations/camera.parquet")
        }
        readiness = {
            str(row["clip_id"]): row
            for row in self._read("observations/dynamic_readiness.parquet")
        }
        objects_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        object_label: dict[str, str] = {}
        for row in self._read("observations/objects.parquet"):
            objects_by_frame[str(row["frame_id"])].append(row)
            object_label[str(row["object_track_id"])] = str(row["semantic_label"])
        masks = {
            (str(row["frame_id"]), str(row["object_track_id"])): row
            for row in self._read("observations/masks.parquet")
            if row.get("valid")
        }
        shared_rows: list[dict[str, Any]] = []
        artifacts: list[Path] = []
        frame_geometry: dict[str, tuple[str, np.ndarray]] = {}
        for frame in owned:
            frame_id = str(frame["frame_id"])
            clip_id = str(frame["clip_id"])
            depth_row = depths.get(frame_id)
            camera = cameras.get((clip_id, frame_id))
            centers: list[list[float]] = []
            track_ids: list[str] = []
            center_sources: list[str] = []
            reason = "frame_depth_unavailable"
            ref = {"path": "", "shape": {}, "dtype": {}, "sha256": ""}
            if depth_row and depth_row.get("valid") and camera and camera.get("K"):
                data = self._load_npz(str(depth_row["array_path"]))
                if data is not None and "depth_map" in data and "valid_mask" in data:
                    depth_map = np.asarray(data["depth_map"], dtype=float)
                    valid_mask = np.asarray(data["valid_mask"], dtype=bool)
                    K = np.asarray(_json(camera["K"], camera["K"]), dtype=float)
                    frame_geometry[frame_id] = (str(depth_row["array_path"]), K)
                    for obj in objects_by_frame.get(frame_id, []):
                        bbox = _json(obj.get("bbox"), obj.get("bbox"))
                        if not bbox or len(bbox) != 4:
                            continue
                        support: Optional[np.ndarray] = None
                        source = "bbox_depth_support_diagnostic"
                        mask_row = masks.get((frame_id, str(obj["object_track_id"])))
                        if mask_row is not None:
                            mask_data = self._load_npz(str(mask_row["array_path"]))
                            if mask_data is not None and "visible_mask" in mask_data:
                                support = np.asarray(mask_data["visible_mask"], dtype=bool)
                                source = "formal_mask_depth_support"
                        if support is None:
                            support = _bbox_mask(bbox, depth_map.shape)
                        valid_support = support & valid_mask & np.isfinite(depth_map) & (depth_map > 0)
                        if int(np.count_nonzero(valid_support)) < 8:
                            continue
                        z = float(np.median(depth_map[valid_support]))
                        uv = [
                            (float(bbox[0]) + float(bbox[2])) / 2.0,
                            (float(bbox[1]) + float(bbox[3])) / 2.0,
                        ]
                        centers.append(self._backproject(uv, z, K))
                        track_ids.append(str(obj["object_track_id"]))
                        center_sources.append(source)
                    reason = "" if centers else "no_valid_object_reconstruction"
                else:
                    reason = "depth_array_unreadable"
            valid = bool(centers)
            if valid:
                source_name = str(videos[str(frame["video_id"])]["source_name"])
                target = (
                    self.output_root
                    / "arrays/shared_3d_frames"
                    / source_name
                    / f"frame_{int(frame['frame_index']):06d}.npz"
                )
                ref = write_npz_array(
                    target, object_centers_3d=np.asarray(centers, dtype=np.float32)
                )
                artifacts.append(target)
            shared_rows.append(
                {
                    "shared_3d_frame_id": stable_id(
                        "p4b5_frame_shared", frame_id, prefix="s3df"
                    ),
                    "video_id": frame["video_id"],
                    "clip_id": clip_id,
                    "frame_id": frame_id,
                    "frame_index": frame["frame_index"],
                    "coordinate_system_id": f"frame_camera:{frame_id}",
                    "geometry_mode": "frame_camera_relative",
                    "sequence_scale_status": "relative_per_frame",
                    "array_path": ref["path"],
                    "array_shape": ref["shape"],
                    "array_dtype": ref["dtype"],
                    "array_sha256": ref["sha256"],
                    "object_track_ids": track_ids,
                    "valid_object_count": len(track_ids),
                    "quality": min(1.0, len(track_ids) / max(len(objects_by_frame.get(frame_id, [])), 1)) if valid else 0.0,
                    "valid": valid,
                    "missing_reason": reason,
                    "metadata": {
                        "metric_geometry": False,
                        "relative_unit": True,
                        "world_fields": None,
                        "center_support_sources": center_sources,
                        "sequence_geometry_independent": True,
                    },
                    "frame_geometry_scope": "camera_frame_relative_sparse_3d",
                    "world_coordinates_available": False,
                    "cross_frame_subtraction_allowed": False,
                }
            )
        point_2d = sorted(
            self._read("observations/point_tracks_2d.parquet"),
            key=lambda row: (str(row["frame_id"]), str(row["clip_id"]), str(row["clip_point_track_id"])),
        )
        frame_record = {
            (str(row["clip_id"]), str(row["frame_id"])): row for row in frames
        }
        point_3d_rows: list[dict[str, Any]] = []
        keypoint_3d_rows: list[dict[str, Any]] = []
        loaded_frame_id = ""
        loaded_depth: Optional[np.ndarray] = None
        loaded_valid_mask: Optional[np.ndarray] = None
        for point in point_2d:
            clip_id = str(point["clip_id"])
            frame_id = str(point["frame_id"])
            ready = readiness.get(clip_id, {})
            mode = str(ready.get("geometry_mode", "unavailable"))
            scale_status = str(ready.get("sequence_scale_status", "unknown"))
            camera = cameras.get((clip_id, frame_id))
            valid = False
            reason = str(ready.get("missing_reason") or "dynamic_geometry_unavailable")
            point_camera = None
            ray = None
            observed_depth = math.nan
            representation = "unavailable"
            quality = 0.0
            uv = _json(point.get("pixel_uv"), point.get("pixel_uv"))
            if point.get("valid") and uv and camera and camera.get("K") and ready.get("dynamic_3d_ready"):
                geometry = frame_geometry.get(frame_id)
                K = np.asarray(_json(camera["K"], camera["K"]), dtype=float)
                if mode == "rotation_compensated":
                    direction = np.linalg.inv(K) @ np.asarray([float(uv[0]), float(uv[1]), 1.0])
                    direction /= max(float(np.linalg.norm(direction)), 1e-12)
                    ray = direction.tolist()
                    valid = True
                    reason = ""
                    representation = "ray_bearing"
                    quality = float(point.get("tracking_confidence") or 0.0)
                elif mode == "static_camera_3d" and geometry is not None and bool(camera.get("sequence_geometry_valid")):
                    if loaded_frame_id != frame_id:
                        data = self._load_npz(geometry[0])
                        loaded_depth = (
                            np.asarray(data["depth_map"], dtype=float)
                            if data is not None and "depth_map" in data else None
                        )
                        loaded_valid_mask = (
                            np.asarray(data["valid_mask"], dtype=bool)
                            if data is not None and "valid_mask" in data else None
                        )
                        loaded_frame_id = frame_id
                    if loaded_depth is None or loaded_valid_mask is None:
                        reason = "depth_array_unreadable"
                        depth_map = np.empty((0, 0), dtype=float)
                        valid_mask = np.empty((0, 0), dtype=bool)
                    else:
                        depth_map, valid_mask = loaded_depth, loaded_valid_mask
                    observed_depth = self._sample_depth(depth_map, valid_mask, uv)
                    scale = _float(camera.get("depth_alignment_scale"))
                    if math.isfinite(observed_depth) and math.isfinite(scale) and scale > 0:
                        observed_depth *= scale
                        point_camera = self._backproject(uv, observed_depth, K)
                        valid = True
                        reason = ""
                        representation = "clip_local_camera_gauge_3d"
                        quality = min(
                            float(point.get("tracking_confidence") or 0.0),
                            float(ready.get("quality") or 0.0),
                        )
                    else:
                        reason = "invalid_depth_at_tracked_point"
            row_id = stable_id(
                "p4b5_point3d", point["point_track_2d_observation_id"], prefix="pt3d"
            )
            row = {
                "point_track_3d_observation_id": row_id,
                "video_id": point["video_id"],
                "clip_id": clip_id,
                "frame_id": frame_id,
                "frame_index": point["frame_index"],
                "global_point_track_id": point["global_point_track_id"],
                "clip_point_track_id": point["clip_point_track_id"],
                "global_object_track_id": point["global_object_track_id"],
                "point_role": point["point_role"],
                "semantic_keypoint_name": point.get("semantic_keypoint_name", ""),
                "pixel_uv": uv,
                "observed_depth": observed_depth,
                "point_3d_camera": point_camera,
                "point_3d_world": None,
                "ray_bearing": ray,
                "trajectory_representation": representation,
                "coordinate_system_id": clips[clip_id]["coordinate_system_id"],
                "sequence_scale_status": scale_status,
                "geometry_mode": mode,
                "quality": quality,
                "is_owned_frame": bool(point["is_owned_frame"]),
                "valid": valid,
                "missing_reason": "" if valid else reason,
                "source_point_track_2d_id": point["point_track_2d_observation_id"],
                "metadata": {
                    "world_coordinates_available": False,
                    "cross_clip_subtraction_allowed": False,
                    "visualization_depth_used": False,
                },
            }
            point_3d_rows.append(row)
            if point["point_role"] == "semantic_keypoint":
                keypoint_3d_rows.append(
                    {
                        "keypoint_3d_observation_id": stable_id(
                            "keypoint3d", row_id, prefix="kp3d"
                        ),
                        "video_id": point["video_id"],
                        "clip_id": clip_id,
                        "frame_id": frame_id,
                        "frame_index": point["frame_index"],
                        "object_track_id": point["global_object_track_id"],
                        "keypoint_name": point["semantic_keypoint_name"],
                        "pixel_uv": uv,
                        "point_3d_camera": point_camera,
                        "coordinate_system_id": clips[clip_id]["coordinate_system_id"],
                        "quality": quality,
                        "valid": bool(valid and point_camera is not None),
                        "missing_reason": "" if valid and point_camera is not None else (
                            "bearing_only_no_3d_keypoint" if valid else reason
                        ),
                        "source_keypoint_observation_id": _json(
                            point.get("metadata"), {}
                        ).get("source_keypoint_observation_id", ""),
                        "metadata": {
                            "semantic_identity_fixed": True,
                            "relative_not_metric": True,
                        },
                    }
                )
        graph_rows, transition_rows, funnel = self._build_structure_graphs(
            point_3d_rows, object_label, frame_record
        )
        shared_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in shared_rows:
            shared_by_clip[str(row["clip_id"])].append(row)
        clip_rows: list[dict[str, Any]] = []
        for clip_id, clip in clips.items():
            values = shared_by_clip.get(clip_id, [])
            ready = readiness.get(clip_id, {})
            frame_count = len(values)
            frame_valid = sum(bool(row["valid"]) for row in values)
            sequence_valid = sum(
                bool(cameras.get((clip_id, str(row["frame_id"])), {}).get("sequence_geometry_valid"))
                for row in values
            )
            clip_rows.append(
                {
                    "shared_3d_clip_id": stable_id(
                        "p4b5_shared_clip", clip_id, prefix="s3dc"
                    ),
                    "video_id": clip["video_id"],
                    "clip_id": clip_id,
                    "coordinate_system_id": clip["coordinate_system_id"],
                    "reference_frame_index": clip["reference_frame_index"],
                    "geometry_mode": ready.get("geometry_mode", "unavailable"),
                    "sequence_scale_status": ready.get("sequence_scale_status", "unknown"),
                    "depth_alignment_domain": "clip_local_background_median",
                    "pose_graph_id": clip.get("pose_graph_id", ""),
                    "scale_alignment_id": clip.get("scale_alignment_id", ""),
                    "owned_frame_count": frame_count,
                    "valid_shared_frame_count": sequence_valid,
                    "shared_3d_owned_frame_ratio": sequence_valid / frame_count if frame_count else math.nan,
                    "depth_aligned_frame_ratio": float(ready.get("depth_aligned_ratio") or 0.0),
                    "dynamic_readiness_frame_ratio": 1.0 if ready.get("dynamic_3d_ready") else 0.0,
                    "valid": bool(ready.get("dynamic_3d_ready")),
                    "missing_reason": "" if ready.get("dynamic_3d_ready") else ready.get("missing_reason", "dynamic_geometry_unavailable"),
                    "metadata": {
                        "frame_relative_3d_kept_when_sequence_invalid": True,
                        "cross_clip_subtraction_allowed": False,
                        "full_se3_claimed": False,
                    },
                    "frame_depth_coverage": sum(
                        bool(depths.get(str(row["frame_id"]), {}).get("valid")) for row in values
                    ) / frame_count if frame_count else math.nan,
                    "frame_shared_3d_coverage": frame_valid / frame_count if frame_count else math.nan,
                    "sequence_depth_aligned_coverage": sequence_valid / frame_count if frame_count else math.nan,
                    "dynamic_3d_ready_coverage": 1.0 if ready.get("dynamic_3d_ready") else 0.0,
                }
            )
        frame_path = self.writer.parquet(
            "observations/shared_3d_frames.parquet", shared_rows, P4B5_SHARED_FRAME_COLUMNS
        )
        clip_path = self.writer.parquet(
            "observations/shared_3d_clips.parquet", clip_rows, P4B5_SHARED_CLIP_COLUMNS
        )
        point_path = write_parquet(
            self.output_root / "observations/point_tracks_3d.parquet",
            point_3d_rows,
            columns=POINT_3D_COLUMNS,
        )
        keypoint_path = write_parquet(
            self.output_root / "observations/keypoints_3d.parquet",
            keypoint_3d_rows,
            columns=KEYPOINT_3D_COLUMNS,
        )
        graph_path = write_parquet(
            self.output_root / "observations/structure_graphs.parquet",
            graph_rows,
            columns=STRUCTURE_GRAPH_COLUMNS,
        )
        transition_path = write_parquet(
            self.output_root / "observations/structure_transitions.parquet",
            transition_rows,
            columns=STRUCTURE_TRANSITION_COLUMNS,
        )
        funnel_path = self.writer.json("reports/ordinary_structure_funnel.json", funnel)
        return [
            frame_path,
            clip_path,
            point_path,
            keypoint_path,
            graph_path,
            transition_path,
            funnel_path,
            *artifacts,
        ]

    def _build_structure_graphs(
        self,
        points: Sequence[Mapping[str, Any]],
        labels: Mapping[str, str],
        frame_records: Mapping[tuple[str, str], Mapping[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        valid_3d = [
            row for row in points
            if row.get("valid") and row.get("point_3d_camera") is not None
        ]
        grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid_3d:
            grouped[(str(row["clip_id"]), str(row["global_object_track_id"]))].append(row)
        graph_rows: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        candidate_tracks = {
            (str(row["clip_id"]), str(row["global_object_track_id"]))
            for row in points
            if row.get("point_role") == "internal_stable_point"
        }
        graph_tracks: set[tuple[str, str]] = set()
        transition_tracks: set[tuple[str, str]] = set()
        for key in sorted(set(grouped) | candidate_tracks):
            clip_id, track_id = key
            rows = grouped.get(key, [])
            by_frame: dict[int, dict[str, Mapping[str, Any]]] = defaultdict(dict)
            for row in rows:
                by_frame[int(row["frame_index"])][str(row["clip_point_track_id"])] = row
            label = labels.get(track_id, "unknown")
            graph_type = "semantic_person_skeleton" if label == "person" else "formal_mask_internal_fixed_graph"
            point_source = "semantic_keypoints" if label == "person" else "formal_instance_mask_internal"
            minimum_points = 4 if label == "person" else 3
            anchor = next(
                ((index, values) for index, values in sorted(by_frame.items()) if len(values) >= minimum_points),
                None,
            )
            graph_id = stable_id("structure_graph", clip_id, track_id, prefix="sgraph")
            if anchor is None:
                graph_rows.append(
                    {
                        "structure_graph_id": graph_id,
                        "video_id": rows[0]["video_id"] if rows else "",
                        "clip_id": clip_id,
                        "object_track_id": track_id,
                        "semantic_label": label,
                        "graph_type": graph_type,
                        "point_source": point_source,
                        "anchor_frame_index": None,
                        "point_ids": [],
                        "edges": [],
                        "coordinate_system_id": rows[0]["coordinate_system_id"] if rows else "",
                        "fixed_topology": True,
                        "formal_evidence": False,
                        "valid": False,
                        "missing_reason": "insufficient_valid_3d_structure_points",
                        "metadata": {"bbox_graph_used": False},
                    }
                )
                continue
            anchor_index, anchor_values = anchor
            if label == "person":
                name_to_id = {
                    str(row.get("semantic_keypoint_name") or ""): point_id
                    for point_id, row in anchor_values.items()
                }
                skeleton_names = (
                    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
                    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
                    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
                )
                edges = tuple(
                    tuple(sorted((name_to_id[first], name_to_id[second])))
                    for first, second in skeleton_names
                    if first in name_to_id and second in name_to_id
                )
                point_ids = tuple(sorted(anchor_values))
            else:
                point_ids = tuple(sorted(anchor_values))
                xyz = np.asarray(
                    [anchor_values[point_id]["point_3d_camera"] for point_id in point_ids],
                    dtype=float,
                )
                edges = build_fixed_structure_edges(point_ids, xyz)
            valid_graph = bool(edges)
            graph_rows.append(
                {
                    "structure_graph_id": graph_id,
                    "video_id": rows[0]["video_id"],
                    "clip_id": clip_id,
                    "object_track_id": track_id,
                    "semantic_label": label,
                    "graph_type": graph_type,
                    "point_source": point_source,
                    "anchor_frame_index": anchor_index,
                    "point_ids": point_ids,
                    "edges": edges,
                    "coordinate_system_id": rows[0]["coordinate_system_id"],
                    "fixed_topology": True,
                    "formal_evidence": valid_graph,
                    "valid": valid_graph,
                    "missing_reason": "" if valid_graph else "no_fixed_structure_edges",
                    "metadata": {
                        "bbox_graph_used": False,
                        "topology_rebuilt_per_frame": False,
                    },
                }
            )
            if not valid_graph:
                continue
            graph_tracks.add(key)
            previous_index: Optional[int] = None
            for current_index in sorted(index for index in by_frame if index >= anchor_index):
                if previous_index is None:
                    previous_index = current_index
                    continue
                previous_values = by_frame[previous_index]
                current_values = by_frame[current_index]
                residuals: list[float] = []
                for first, second in edges:
                    if not all(
                        point_id in previous_values and point_id in current_values
                        for point_id in (first, second)
                    ):
                        continue
                    previous_length = float(np.linalg.norm(
                        np.asarray(previous_values[first]["point_3d_camera"], dtype=float)
                        - np.asarray(previous_values[second]["point_3d_camera"], dtype=float)
                    ))
                    current_length = float(np.linalg.norm(
                        np.asarray(current_values[first]["point_3d_camera"], dtype=float)
                        - np.asarray(current_values[second]["point_3d_camera"], dtype=float)
                    ))
                    if previous_length > 1e-8 and current_length > 1e-8:
                        residuals.append(abs(math.log(current_length / previous_length)))
                current_sample = next(iter(current_values.values()))
                owned = bool(current_sample.get("is_owned_frame"))
                valid_transition = bool(residuals and owned)
                transitions.append(
                    {
                        "structure_transition_id": stable_id(
                            "structure_transition", graph_id, previous_index, current_index,
                            prefix="strans",
                        ),
                        "video_id": current_sample["video_id"],
                        "clip_id": clip_id,
                        "object_track_id": track_id,
                        "structure_graph_id": graph_id,
                        "previous_frame_index": previous_index,
                        "current_frame_index": current_index,
                        "edge_count": len(edges),
                        "valid_edge_count": len(residuals),
                        "raw_residual": float(np.median(residuals)) if valid_transition else math.nan,
                        "quality": len(residuals) / len(edges) if valid_transition else 0.0,
                        "coordinate_system_id": current_sample["coordinate_system_id"],
                        "is_owned_frame": owned,
                        "valid": valid_transition,
                        "missing_reason": "" if valid_transition else (
                            "context_frame_not_formal_evidence" if not owned else "insufficient_fixed_edge_support"
                        ),
                        "metadata": {
                            "fixed_point_ids": True,
                            "cross_clip_transition": False,
                        },
                    }
                )
                if valid_transition:
                    transition_tracks.add(key)
                previous_index = current_index
        formal_mask_tracks = {
            key for key in candidate_tracks if labels.get(key[1], "unknown") != "person"
        }
        ordinary_graph_tracks = {
            key for key in graph_tracks if labels.get(key[1], "unknown") != "person"
        }
        ordinary_transition_tracks = {
            key for key in transition_tracks if labels.get(key[1], "unknown") != "person"
        }
        funnel = {
            "formal_mask_tracks": len(formal_mask_tracks),
            "tracks_long_enough": len(candidate_tracks),
            "tracks_with_frame_depth": len(grouped),
            "tracks_with_sequence_geometry": len(grouped),
            "tracks_with_internal_points": len(candidate_tracks),
            "tracks_with_stable_point_ids": len(candidate_tracks),
            "tracks_with_valid_3d_points": len(grouped),
            "tracks_with_structure_graph": len(ordinary_graph_tracks),
            "tracks_with_structure_transitions": len(ordinary_transition_tracks),
            "ordinary_structure_graph_track_ids": [list(value) for value in sorted(ordinary_graph_tracks)],
            "ordinary_structure_transition_track_ids": [list(value) for value in sorted(ordinary_transition_tracks)],
            "person_structure_track_count": len(
                {key for key in graph_tracks if labels.get(key[1]) == "person"}
            ),
            "primary_failure_reason": (
                "no_clip_with_valid_sequence_geometry_and_stable_mask_points"
                if not ordinary_graph_tracks
                else ""
            ),
            "bbox_graphs_used_as_formal_evidence": 0,
            "cross_unaligned_clip_transitions_computed": 0,
        }
        return graph_rows, transitions, funnel

    def _stage_11_dynamic(self) -> list[Path]:
        """Materialize traceable owned-frame evidence without a truth decision."""

        points = [
            row for row in self._read("observations/point_tracks_3d.parquet")
            if row.get("valid") and row.get("point_3d_camera") is not None
        ]
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        by_object_frame: dict[tuple[str, str, int], list[np.ndarray]] = defaultdict(list)
        for row in points:
            grouped[(
                str(row["clip_id"]),
                str(row["global_object_track_id"]),
                str(row["clip_point_track_id"]),
            )].append(row)
            by_object_frame[(
                str(row["clip_id"]),
                str(row["global_object_track_id"]),
                int(row["frame_index"]),
            )].append(np.asarray(_json(row["point_3d_camera"], row["point_3d_camera"]), dtype=float))
        object_scale: dict[tuple[str, str, int], float] = {}
        for key, values in by_object_frame.items():
            if len(values) >= 2:
                xyz = np.asarray(values)
                center = np.median(xyz, axis=0)
                distances = np.linalg.norm(xyz - center, axis=1)
                scale = float(2.0 * np.median(distances))
                if math.isfinite(scale) and scale > 1e-8:
                    object_scale[key] = scale
        videos = {row["video_id"]: row for row in self._read("manifests/videos.parquet")}
        cameras = {
            (str(row["clip_id"]), str(row["frame_id"])): row
            for row in self._read("observations/camera.parquet")
        }
        frame_id_by_clip_index = {
            (str(row["clip_id"]), int(row["frame_index"])): str(row["frame_id"])
            for row in self._read("manifests/frames.parquet")
        }
        evidence_rows: list[dict[str, Any]] = []
        for (clip_id, object_id, point_id), samples in sorted(grouped.items()):
            samples.sort(key=lambda row: int(row["frame_index"]))
            for index, (previous, current) in enumerate(zip(samples, samples[1:]), start=1):
                if not current.get("is_owned_frame"):
                    continue
                previous_xyz = np.asarray(_json(previous["point_3d_camera"], previous["point_3d_camera"]), dtype=float)
                current_xyz = np.asarray(_json(current["point_3d_camera"], current["point_3d_camera"]), dtype=float)
                frame_gap = int(current["frame_index"]) - int(previous["frame_index"])
                source_ids = (
                    str(previous["point_track_3d_observation_id"]),
                    str(current["point_track_3d_observation_id"]),
                )
                evidence_rows.append(
                    self._dynamic_evidence_row(
                        branch="track_3d_continuity",
                        point=current,
                        raw_value=float(max(frame_gap - 1, 0)),
                        quality=min(float(previous["quality"]), float(current["quality"])),
                        source_ids=source_ids,
                        metadata={"frame_gap": frame_gap},
                    )
                )
                if index < 2:
                    continue
                earlier = samples[index - 2]
                earlier_xyz = np.asarray(_json(earlier["point_3d_camera"], earlier["point_3d_camera"]), dtype=float)
                first_delta = previous_xyz - earlier_xyz
                second_delta = current_xyz - previous_xyz
                first_norm, second_norm = float(np.linalg.norm(first_delta)), float(np.linalg.norm(second_delta))
                if first_norm <= 1e-8 or second_norm <= 1e-8:
                    direction_value = 0.0
                else:
                    cosine = float(np.clip(np.dot(first_delta, second_delta) / (first_norm * second_norm), -1.0, 1.0))
                    direction_value = float(math.acos(cosine) / math.pi)
                evidence_rows.append(
                    self._dynamic_evidence_row(
                        branch="direction_consistency",
                        point=current,
                        raw_value=direction_value,
                        quality=min(float(earlier["quality"]), float(previous["quality"]), float(current["quality"])),
                        source_ids=(str(earlier["point_track_3d_observation_id"]), *source_ids),
                        metadata={"direction_unit": "angle_over_pi"},
                    )
                )
                previous_scale = object_scale.get((clip_id, object_id, int(previous["frame_index"])))
                current_scale = object_scale.get((clip_id, object_id, int(current["frame_index"])))
                if previous_scale and current_scale:
                    first_speed = first_norm / previous_scale
                    second_speed = second_norm / current_scale
                    evidence_rows.append(
                        self._dynamic_evidence_row(
                            branch="relative_velocity_change",
                            point=current,
                            raw_value=abs(second_speed - first_speed),
                            quality=min(float(earlier["quality"]), float(previous["quality"]), float(current["quality"])),
                            source_ids=(str(earlier["point_track_3d_observation_id"]), *source_ids),
                            metadata={"unit": "object_relative_scale_per_frame", "metric_speed_claimed": False},
                        )
                    )
                camera = cameras.get((clip_id, str(current["frame_id"])))
                if camera and camera.get("K"):
                    predicted = previous_xyz + first_delta
                    K = np.asarray(_json(camera["K"], camera["K"]), dtype=float)
                    projected = K @ predicted
                    observed_uv = np.asarray(_json(current["pixel_uv"], current["pixel_uv"]), dtype=float)
                    if predicted[2] > 1e-8 and projected[2] > 1e-8 and observed_uv.shape == (2,):
                        predicted_uv = projected[:2] / projected[2]
                        video = videos[str(current["video_id"])]
                        diagonal = math.hypot(float(video["width"]), float(video["height"]))
                        error = float(np.linalg.norm(predicted_uv - observed_uv) / diagonal)
                        evidence_rows.append(
                            self._dynamic_evidence_row(
                                branch="dynamic_reprojection",
                                point=current,
                                raw_value=error,
                                quality=min(float(earlier["quality"]), float(previous["quality"]), float(current["quality"])),
                                source_ids=(str(earlier["point_track_3d_observation_id"]), *source_ids),
                                metadata={
                                    "history_only_prediction": True,
                                    "current_observation_independent": True,
                                    "prediction_model": "constant_3d_displacement_clip_local",
                                },
                            )
                        )
        for transition in self._read("observations/structure_transitions.parquet"):
            if not transition.get("valid") or not transition.get("is_owned_frame"):
                continue
            evidence = EvidenceRecord(
                evidence_id=stable_id(
                    "p4b5_evidence", "structure_temporal", transition["structure_transition_id"],
                    prefix="evidence",
                ),
                branch_name="structure_temporal",
                evidence_level="edge",
                video_id=str(transition["video_id"]),
                clip_id=str(transition["clip_id"]),
                frame_id="",
                object_track_id=str(transition["object_track_id"]),
                edge_id=str(transition["structure_graph_id"]),
                raw_value=float(transition["raw_residual"]),
                intrinsic_normalized_value=float(transition["raw_residual"]),
                valid=True,
                quality=float(transition["quality"]),
                applicability=Applicability.APPLICABLE_VALID,
                missing_reason="",
                geometry_mode="static_camera_3d",
                sequence_scale_status="relative_shared_sequence",
                coordinate_system_id=str(transition["coordinate_system_id"]),
                source_evidence_ids=(str(transition["structure_transition_id"]),),
                metadata={"owned_frame_only": True, "classification_output": False},
            ).to_dict()
            evidence["frame_id"] = frame_id_by_clip_index.get(
                (str(transition["clip_id"]), int(transition["current_frame_index"])), ""
            )
            evidence["included_in_formal_aggregation"] = True
            evidence_rows.append(evidence)
        point_evidence = [row for row in evidence_rows if row["evidence_level"] == "point"]
        edge_evidence = [row for row in evidence_rows if row["evidence_level"] == "edge"]
        paths = []
        for level, rows in (
            ("point", point_evidence),
            ("edge", edge_evidence),
            ("object", []),
            ("frame", []),
        ):
            path = self.output_root / f"evidence/{level}_evidence.parquet"
            write_parquet(path, rows, columns=EVIDENCE_OUTPUT_COLUMNS)
            paths.append(path)
        return paths

    def _dynamic_evidence_row(
        self,
        *,
        branch: str,
        point: Mapping[str, Any],
        raw_value: float,
        quality: float,
        source_ids: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        evidence = EvidenceRecord(
            evidence_id=stable_id(
                "p4b5_evidence", branch, point["point_track_3d_observation_id"],
                prefix="evidence",
            ),
            branch_name=branch,
            evidence_level="point",
            video_id=str(point["video_id"]),
            clip_id=str(point["clip_id"]),
            frame_id=str(point["frame_id"]),
            object_track_id=str(point["global_object_track_id"]),
            point_id=str(point["clip_point_track_id"]),
            raw_value=float(raw_value),
            intrinsic_normalized_value=float(raw_value),
            valid=True,
            quality=float(np.clip(quality, 0.0, 1.0)),
            applicability=Applicability.APPLICABLE_VALID,
            missing_reason="",
            geometry_mode=str(point["geometry_mode"]),
            sequence_scale_status=str(point["sequence_scale_status"]),
            coordinate_system_id=str(point["coordinate_system_id"]),
            source_evidence_ids=tuple(source_ids),
            metadata={
                **dict(metadata),
                "owned_frame_only": True,
                "classification_output": False,
                "truth_labels_used": False,
            },
        ).to_dict()
        evidence["included_in_formal_aggregation"] = True
        return evidence

    def _stage_12_occlusion(self) -> list[Path]:
        """Synchronize formal-mask depth order for old and newly scanned candidates."""

        frames = self._owned_frames()
        videos = {row["video_id"]: row for row in self._read("manifests/videos.parquet")}
        source_name_by_id = {video_id: str(row["source_name"]) for video_id, row in videos.items()}
        depths = {
            str(row["frame_id"]): row
            for row in self._read("observations/depth.parquet") if row.get("valid")
        }
        masks_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._read("observations/masks.parquet"):
            if row.get("valid"):
                masks_by_frame[str(row["frame_id"])].append(row)
        audit_root_value = self.config.get("artifacts", {}).get(
            "occlusion_audit_root", "outputs/occlusion_event_coverage_audit"
        )
        audit_root = Path(str(audit_root_value))
        audit_root = audit_root if audit_root.is_absolute() else self.project_root / audit_root
        old_candidates = [
            row for row in _csv_rows(audit_root / "occlusion_window_diagnostics.csv")
            if row.get("status") == "candidate_rejected"
        ]
        old_by_video: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in old_candidates:
            old_by_video[str(row["video_id"])].append(row)
        synchronized_windows: set[tuple[str, int, int]] = set()
        rows: list[dict[str, Any]] = []
        histories: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
        for frame in sorted(frames, key=lambda row: (str(row["video_id"]), int(row["frame_index"]))):
            frame_id = str(frame["frame_id"])
            depth_row = depths.get(frame_id)
            mask_rows = masks_by_frame.get(frame_id, [])
            if depth_row is None or len(mask_rows) < 2:
                continue
            depth_data = self._load_npz(str(depth_row["array_path"]))
            if depth_data is None or "depth_map" not in depth_data:
                continue
            depth_map = np.asarray(depth_data["depth_map"], dtype=float)
            source_name = source_name_by_id[str(frame["video_id"])]
            frame_index = int(frame["frame_index"])
            matching_windows = [
                candidate for candidate in old_by_video.get(source_name, [])
                if int(candidate["start_frame"]) <= frame_index <= int(candidate["end_frame"])
            ]
            loaded: list[tuple[dict[str, Any], np.ndarray]] = []
            for mask_row in mask_rows:
                data = self._load_npz(str(mask_row["array_path"]))
                if data is not None and "visible_mask" in data:
                    loaded.append((mask_row, np.asarray(data["visible_mask"], dtype=bool)))
            for (first_row, first), (second_row, second) in combinations(loaded, 2):
                overlap = int(np.count_nonzero(first & second))
                if overlap == 0 and not matching_windows:
                    continue
                order = synchronized_depth_order(depth_map, first, second)
                predicted: dict[str, Optional[np.ndarray]] = {}
                area_drop: dict[str, bool] = {}
                for mask_row, current_mask in ((first_row, first), (second_row, second)):
                    track_id = str(mask_row["object_track_id"])
                    history = histories.get((str(frame["video_id"]), track_id), [])
                    prediction: Optional[np.ndarray] = None
                    if len(history) >= 2 and history[-1][0] < frame_index:
                        previous2_index, previous2 = history[-2]
                        previous1_index, previous1 = history[-1]
                        if previous2_index < previous1_index and frame_index - previous1_index <= 2:
                            y2, x2 = np.nonzero(previous2)
                            y1, x1 = np.nonzero(previous1)
                            if len(x2) and len(x1):
                                dx = float(np.mean(x1) - np.mean(x2))
                                dy = float(np.mean(y1) - np.mean(y2))
                                transform = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]])
                                prediction = cv2.warpAffine(
                                    previous1.astype(np.uint8), transform,
                                    (current_mask.shape[1], current_mask.shape[0]),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT,
                                ).astype(bool)
                    predicted[track_id] = prediction
                    predicted_area = int(np.count_nonzero(prediction)) if prediction is not None else 0
                    area_drop[track_id] = bool(
                        prediction is not None
                        and predicted_area >= 16
                        and np.count_nonzero(current_mask) < 0.70 * predicted_area
                    )
                first_id, second_id = str(first_row["object_track_id"]), str(second_row["object_track_id"])
                predicted_available = predicted[first_id] is not None or predicted[second_id] is not None
                visible_change_available = predicted_available
                occluded_side = ""
                if area_drop[first_id]:
                    occluded_side = "a"
                elif area_drop[second_id]:
                    occluded_side = "b"
                event = bool(
                    overlap > 0
                    and order.valid
                    and predicted_available
                    and occluded_side
                    and order.background == occluded_side
                )
                if order.valid:
                    for candidate in matching_windows:
                        synchronized_windows.add((
                            source_name,
                            int(candidate["start_frame"]),
                            int(candidate["end_frame"]),
                        ))
                rows.append(
                    {
                        "depth_order_observation_id": stable_id(
                            "depth_order", frame_id, first_row["object_track_id"],
                            second_row["object_track_id"], prefix="dorder",
                        ),
                        "video_id": frame["video_id"],
                        "source_name": source_name,
                        "clip_id": frame["clip_id"],
                        "frame_id": frame_id,
                        "frame_index": frame_index,
                        "object_a_track_id": first_row["object_track_id"],
                        "object_b_track_id": second_row["object_track_id"],
                        "formal_mask_overlap_pixels": overlap,
                        "predicted_support_available": predicted_available,
                        "visible_area_change_available": visible_change_available,
                        "track_continuity_available": True,
                        "depth_a": order.depth_a,
                        "depth_b": order.depth_b,
                        "foreground": order.foreground,
                        "background": order.background,
                        "depth_margin": order.depth_margin,
                        "depth_source": order.depth_source,
                        "quality": order.quality,
                        "depth_order_valid": order.valid,
                        "uncertain": order.uncertain,
                        "old_candidate_window": bool(matching_windows),
                        "new_overlap_candidate": overlap > 0,
                        "formal_occlusion_event": event,
                        "applicability": "applicable_valid" if event else ("not_applicable" if order.valid else "observation_missing"),
                        "missing_reason": (
                            "" if event else ("no_validated_visibility_change_event" if order.valid else order.missing_reason)
                        ),
                        "metadata": {
                            "larger_value_means": "farther",
                            "overlap_boundary_depth_preferred": True,
                            "center_depth_low_quality": order.depth_source == "object_mask_median_low_quality",
                            "event_gate_lowered": False,
                            "history_prediction_uses_current_frame": False,
                            "occluded_side": occluded_side,
                            "background_side": order.background,
                            "area_drop_a": area_drop[first_id],
                            "area_drop_b": area_drop[second_id],
                        },
                    }
                )
            # Update history only after every current-frame candidate was scored.
            for mask_row, current_mask in loaded:
                key = (str(frame["video_id"]), str(mask_row["object_track_id"]))
                histories[key].append((frame_index, current_mask.copy()))
                histories[key] = histories[key][-2:]
        columns = (
            "depth_order_observation_id", "video_id", "source_name", "clip_id", "frame_id",
            "frame_index", "object_a_track_id", "object_b_track_id",
            "formal_mask_overlap_pixels", "predicted_support_available",
            "visible_area_change_available", "track_continuity_available", "depth_a", "depth_b",
            "foreground", "background", "depth_margin", "depth_source", "quality",
            "depth_order_valid", "uncertain", "old_candidate_window", "new_overlap_candidate",
            "formal_occlusion_event", "applicability", "missing_reason", "metadata",
        )
        sync_path = write_parquet(
            self.output_root / "reports/occlusion_depth_order_sync.parquet",
            rows,
            columns=columns,
        )
        old_count = len(old_candidates)
        summary = {
            "previous_candidate_window_count": old_count,
            "previous_candidates_with_synchronized_depth_order": len(synchronized_windows),
            "new_mask_overlap_depth_order_rows": sum(
                bool(row["new_overlap_candidate"] and row["depth_order_valid"]) for row in rows
            ),
            "formal_occlusion_event_count": sum(bool(row["formal_occlusion_event"]) for row in rows),
            "uncertain_depth_order_count": sum(bool(row["uncertain"]) for row in rows),
            "no_event_is_not_zero_residual": True,
            "truth_labels_used": False,
        }
        summary_path = self.writer.json("reports/occlusion_stage_summary.json", summary)
        return [sync_path, summary_path]

    def _stage_13_aggregation(self) -> list[Path]:
        artifacts = super()._stage_13_aggregation()
        metrics = self._coverage_metrics()
        parquet_path = write_parquet(
            self.output_root / "reports/coverage_metrics.parquet",
            [metric.to_dict() for metric in metrics],
            columns=(
                "metric_name", "scope_type", "scope_id", "numerator", "denominator",
                "applicable_count", "observation_missing_count", "invalid_geometry_count",
                "unsupported_mode_count", "unit", "metadata", "ratio",
            ),
        )
        csv_path = self.output_root / "reports/coverage_metrics.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        columns = (
            "metric_name", "scope_type", "scope_id", "numerator", "denominator", "ratio",
            "applicable_count", "observation_missing_count", "invalid_geometry_count",
            "unsupported_mode_count", "unit",
        )
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for metric in metrics:
                writer.writerow({key: metric.to_dict().get(key) for key in columns})
        acceptance = self._acceptance_summary(metrics)
        summary_path = self.writer.json("reports/p4b5_acceptance_summary.json", acceptance)
        return [*artifacts, parquet_path, csv_path, summary_path]

    def _coverage_metrics(self) -> list[CoverageMetric]:
        frames = self._owned_frames()
        depths = self._read("observations/depth.parquet")
        shared_frames = self._read("observations/shared_3d_frames.parquet")
        readiness = self._read("observations/dynamic_readiness.parquet")
        point_evidence = self._read("evidence/point_evidence.parquet")
        edge_evidence = self._read("evidence/edge_evidence.parquet")
        point_3d = self._read("observations/point_tracks_3d.parquet")
        video_ids = sorted({str(row["video_id"]) for row in frames})
        metrics: list[CoverageMetric] = []
        scopes = [("dataset", self.dataset_id, None)] + [
            ("video", video_id, video_id) for video_id in video_ids
        ]
        for scope_type, scope_id, video_id in scopes:
            scoped_frames = [row for row in frames if video_id is None or row["video_id"] == video_id]
            frame_ids = {str(row["frame_id"]) for row in scoped_frames}
            clip_ids = {str(row["clip_id"]) for row in scoped_frames}
            scoped_depth = [row for row in depths if str(row["frame_id"]) in frame_ids]
            scoped_shared = [row for row in shared_frames if str(row["frame_id"]) in frame_ids]
            scoped_ready = [row for row in readiness if str(row["clip_id"]) in clip_ids]
            scoped_point3d = [row for row in point_3d if str(row["clip_id"]) in clip_ids]
            scoped_evidence = [
                row for row in (*point_evidence, *edge_evidence)
                if str(row["clip_id"]) in clip_ids and row.get("valid")
            ]
            frame_denominator = len(frame_ids)
            depth_valid = sum(bool(row["valid"]) for row in scoped_depth)
            shared_valid = sum(bool(row["valid"]) for row in scoped_shared)
            clip_denominator = len(scoped_ready)
            aligned = sum(float(row.get("depth_aligned_ratio") or 0.0) >= 0.8 for row in scoped_ready)
            dynamic = sum(bool(row.get("dynamic_3d_ready")) for row in scoped_ready)
            candidate_transitions = {
                (str(row["clip_id"]), str(row["clip_point_track_id"]), int(row["frame_index"]))
                for row in scoped_point3d if row.get("is_owned_frame")
            }
            evidence_transitions = {
                (str(row["clip_id"]), str(row.get("point_id") or row.get("edge_id")), str(row.get("frame_id")))
                for row in scoped_evidence
            }
            values = (
                ("frame_depth_coverage", depth_valid, frame_denominator, "frame", frame_denominator - depth_valid, 0, 0),
                ("frame_shared_3d_coverage", shared_valid, frame_denominator, "frame", frame_denominator - shared_valid, 0, 0),
                ("sequence_depth_aligned_coverage", aligned, clip_denominator, "clip", 0, clip_denominator - aligned, 0),
                ("dynamic_3d_ready_coverage", dynamic, clip_denominator, "clip", 0, clip_denominator - dynamic, 0),
                (
                    "formal_dynamic_evidence_coverage",
                    min(len(evidence_transitions), len(candidate_transitions)),
                    len(candidate_transitions),
                    "owned_point_transition",
                    max(len(candidate_transitions) - len(evidence_transitions), 0),
                    0,
                    0,
                ),
            )
            for name, numerator, denominator, unit, missing, invalid, unsupported in values:
                metrics.append(
                    CoverageMetric(
                        metric_name=name,
                        scope_type=scope_type,
                        scope_id=scope_id,
                        numerator=int(numerator),
                        denominator=int(denominator),
                        applicable_count=int(denominator),
                        observation_missing_count=int(missing),
                        invalid_geometry_count=int(invalid),
                        unsupported_mode_count=int(unsupported),
                        unit=unit,
                        metadata={"truth_labels_used": False},
                    )
                )
        return metrics

    def _acceptance_summary(self, metrics: Sequence[CoverageMetric]) -> dict[str, Any]:
        dataset_metrics = {
            metric.metric_name: metric.to_dict()
            for metric in metrics if metric.scope_type == "dataset"
        }
        readiness = self._read("observations/dynamic_readiness.parquet")
        modes = Counter(str(row["geometry_mode"]) for row in readiness)
        keypoints = self._read("observations/keypoints.parquet")
        point2d = self._read("observations/point_tracks_2d.parquet")
        graphs = self._read("observations/structure_graphs.parquet")
        transitions = self._read("observations/structure_transitions.parquet")
        occlusion = json.loads(
            (self.output_root / "reports/occlusion_stage_summary.json").read_text(encoding="utf-8")
        )
        funnel = json.loads(
            (self.output_root / "reports/ordinary_structure_funnel.json").read_text(encoding="utf-8")
        )
        return {
            "pipeline_version": P4B5_PIPELINE_VERSION,
            "coverage": dataset_metrics,
            "geometry_mode_clip_counts": dict(modes),
            "valid_person_keypoint_observations": sum(bool(row["valid"]) for row in keypoints),
            "valid_independent_point_2d_observations": sum(bool(row["valid"]) for row in point2d),
            "independent_clip_point_track_count": len({
                row["clip_point_track_id"] for row in point2d if row.get("valid")
            }),
            "ordinary_formal_structure_graph_count": sum(
                bool(row["valid"] and row["graph_type"] == "formal_mask_internal_fixed_graph")
                for row in graphs
            ),
            "formal_structure_transition_count": sum(bool(row["valid"]) for row in transitions),
            "ordinary_structure_funnel": funnel,
            "occlusion_depth_order": occlusion,
            "classification_output": False,
            "truth_labels_used": False,
            "ready_for_p4_c0_dataset_planning": True,
            "ready_for_p4_c_statistical_modeling": False,
            "limitations": [
                "Depth Anything output remains monocular relative depth, not metric depth.",
                "No calibrated full-SE3 camera translation is claimed.",
                "Cross-clip identity handoff does not authorize cross-clip 3D subtraction.",
                "Occlusion events remain gated by observed visibility evidence.",
            ],
        }
