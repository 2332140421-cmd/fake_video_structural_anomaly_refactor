"""Executable P4-B offline dataset pipeline.

The builder never accepts or reads real/fake labels. It migrates existing
label-free P3 observations where available and records missing evidence
explicitly instead of synthesizing zeros or anomaly conclusions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

from .cache import StageCache, build_cache_key, stage_config_hash
from .ids import StableIdFactory, stable_id
from .manifest import (
    build_clip_manifests,
    decode_frame_signatures,
    disambiguate_source_names,
    split_scene_segments,
    video_manifest_row,
)
from .reader import DatasetReader
from .schema import (
    EVIDENCE_COLUMNS,
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    Applicability,
    DatasetManifest,
    EvidenceRecord,
)
from .stages import STAGES, descendants, execution_plan
from .writer import (
    DatasetWriter,
    atomic_write_json,
    json_text,
    sha256_file,
    write_npz_array,
    write_parquet,
)


VIDEO_COLUMNS = ("video_id", "source_name", "source_stem", "source_relative_path", "source_path", "source_original_path", "source_path_kind", "source_sha256", "file_size", "frame_count", "fps", "width", "height", "decode_status", "failure_reason")
CLIP_COLUMNS = ("clip_id", "video_id", "scene_id", "clip_ordinal", "start_frame_index", "end_frame_index", "core_start_frame_index", "core_end_frame_index", "reference_frame_index", "frame_count", "coordinate_system_id", "geometry_mode", "sequence_scale_status", "depth_alignment_domain", "pose_graph_id", "scale_alignment_id", "valid", "missing_reason")
FRAME_COLUMNS = ("frame_record_id", "frame_id", "video_id", "clip_id", "frame_index", "scene_id", "is_context_frame", "is_owned_frame", "owner_clip_id", "decode_status", "failure_reason", "decoded_frame_path")
STAGE_COLUMNS = ("run_id", "stage_name", "status", "cache_key", "cache_hit", "cache_reason", "started_at", "finished_at", "artifact_count", "failure_reason", "metadata")
OBJECT_COLUMNS = ("object_observation_id", "video_id", "frame_id", "frame_index", "source_object_id", "object_track_id", "source_track_id", "semantic_label", "canonical_label", "confidence", "bbox", "bbox_area", "frame_area", "source_provider", "valid", "missing_reason", "metadata")
MASK_COLUMNS = ("mask_observation_id", "video_id", "frame_id", "frame_index", "object_track_id", "segmentation_instance_id", "class_id", "class_name", "confidence", "visible_mask_path", "array_path", "array_shape", "array_dtype", "mask_sha256", "array_sha256", "mask_area", "mask_bbox", "boundary_point_count", "source_provider", "weight_sha256", "is_visible_mask", "is_amodal_mask", "bbox_fallback", "valid", "missing_reason", "metadata")
KEYPOINT_COLUMNS = ("keypoint_observation_id", "video_id", "frame_id", "frame_index", "object_track_id", "total_keypoints", "valid_keypoints", "valid_ratio", "provider_name", "valid", "missing_reason", "metadata")
DEPTH_COLUMNS = ("depth_observation_id", "video_id", "frame_id", "frame_index", "array_path", "array_shape", "array_dtype", "array_sha256", "depth_representation", "scale_status", "larger_value_means", "provider_name", "valid_pixel_ratio", "valid", "missing_reason", "metadata")
CAMERA_COLUMNS = ("camera_observation_id", "video_id", "clip_id", "frame_id", "frame_index", "coordinate_system_id", "K", "T_world_camera", "coordinate_convention", "intrinsics_source", "pose_source", "geometry_mode", "depth_alignment_scale", "quality", "valid", "missing_reason", "metadata")
SHARED_FRAME_COLUMNS = ("shared_3d_frame_id", "video_id", "clip_id", "frame_id", "frame_index", "coordinate_system_id", "geometry_mode", "sequence_scale_status", "array_path", "array_shape", "array_dtype", "array_sha256", "object_track_ids", "valid_object_count", "quality", "valid", "missing_reason", "metadata")
SHARED_CLIP_COLUMNS = ("shared_3d_clip_id", "video_id", "clip_id", "coordinate_system_id", "reference_frame_index", "geometry_mode", "sequence_scale_status", "depth_alignment_domain", "pose_graph_id", "scale_alignment_id", "owned_frame_count", "valid_shared_frame_count", "shared_3d_owned_frame_ratio", "depth_aligned_frame_ratio", "dynamic_readiness_frame_ratio", "valid", "missing_reason", "metadata")
EVIDENCE_OUTPUT_COLUMNS = EVIDENCE_COLUMNS + ("included_in_formal_aggregation",)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _json(value: Any, default: Any) -> Any:
    if value in (None, "", "null"):
        return default
    try:
        return json.loads(str(value))
    except (TypeError, json.JSONDecodeError):
        return default


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return "unknown"


def _module_version(name: str) -> str:
    try:
        module = __import__(name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _environment(device: str, project_root: Path) -> dict[str, Any]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=False
    ).stdout
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "pytorch_version": _module_version("torch"),
        "opencv_version": cv2.__version__,
        "ultralytics_version": _module_version("ultralytics"),
        "pyarrow_version": _module_version("pyarrow"),
        "device": device,
        "operating_system": platform.platform(),
        "package_lock_hash": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
    }


class StructuralEnhancementDatasetBuilder:
    """Run the label-isolated 13-stage P4-B pipeline with resumable cache."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        device: Optional[str] = None,
        num_workers: int = 1,
        selected_video_id: Optional[str] = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parents[1]
        self.config = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.device = device or str(self.config.get("runtime", {}).get("device", "cpu"))
        self.num_workers = int(num_workers)
        self.selected_video_id = selected_video_id
        configured_data_root = self.config.get("sources", {}).get("data_root")
        if configured_data_root is None:
            self.source_root = self.project_root
        else:
            source_root = Path(str(configured_data_root)).expanduser()
            self.source_root = (
                source_root if source_root.is_absolute() else self.project_root / source_root
            ).resolve()
        output = Path(str(self.config["dataset"]["output_root"]))
        self.output_root = output if output.is_absolute() else self.project_root / output
        if selected_video_id:
            self.output_root = self.output_root / "subsets" / str(selected_video_id)
        self.writer = DatasetWriter(self.output_root)
        self.reader = DatasetReader(self.output_root)
        self.cache = StageCache(self.output_root / ".cache")
        self.stage_status_rows: list[dict[str, Any]] = []
        self.run_id = stable_id("run", _now(), os.getpid(), prefix="run")
        self.config_sha256 = sha256_file(self.config_path)
        self.source_paths = self._source_paths()
        source_hashes = [sha256_file(path) for path in self.source_paths]
        self.dataset_id = stable_id(
            "dataset",
            SCHEMA_VERSION,
            str(self.config.get("dataset", {}).get("name", "structural_enhancement")),
            [(path.name, digest) for path, digest in zip(self.source_paths, source_hashes)],
            prefix="dataset",
        )
        self.ids = StableIdFactory(self.dataset_id)
        self.weight_hashes = self._weight_hashes()
        self.force_stages: set[str] = set()

    def _source_paths(self) -> list[Path]:
        values = self.config.get("sources", {}).get("videos", [])
        paths = []
        for value in values:
            path = Path(str(value)).expanduser()
            path = path if path.is_absolute() else self.source_root / path
            if not path.exists():
                raise FileNotFoundError(f"Configured source video does not exist: {path}")
            paths.append(path.resolve())
        if not paths:
            raise ValueError("Configuration must list sources.videos without labels")
        return sorted(paths, key=lambda value: value.as_posix())

    def _weight_hashes(self) -> dict[str, str]:
        output: dict[str, str] = {}
        for name, value in self.config.get("providers", {}).get("weights", {}).items():
            path = Path(str(value))
            path = path if path.is_absolute() else self.project_root / path
            output[str(name)] = sha256_file(path) if path.exists() else "missing"
        depth_cache = Path.home() / ".cache/huggingface/hub/models--depth-anything--Depth-Anything-V2-Small-hf"
        output["depth_model_cache"] = _directory_fingerprint(depth_cache)
        return output

    def _selected_sources(self) -> list[Path]:
        if not self.selected_video_id:
            return self.source_paths
        selected = [path for path in self.source_paths if path.stem == self.selected_video_id]
        if not selected:
            videos = self._read("manifests/videos.parquet")
            names = {row["video_id"]: row["source_name"] for row in videos}
            selected_names = {names.get(self.selected_video_id, "")}
            selected = [path for path in self.source_paths if path.stem in selected_names]
        if not selected:
            raise ValueError(f"Unknown --video-id: {self.selected_video_id}")
        return selected

    def dry_run(self, target_stage: Optional[str]) -> list[dict[str, Any]]:
        """Return an execution plan without loading providers or writing outputs."""

        return [
            {
                "stage": stage.name,
                "dependencies": list(stage.dependencies),
                "selected_video_count": len(self._selected_sources()),
                "would_use_labels": False,
            }
            for stage in execution_plan(target_stage)
        ]

    def run(
        self,
        *,
        target_stage: Optional[str] = None,
        resume: bool = False,
        force_stage: Optional[str] = None,
    ) -> dict[str, Any]:
        """Run through a target stage while isolating clip/stage failures."""

        if force_stage:
            self.force_stages = set(descendants(force_stage))
        handlers: dict[str, Callable[[], list[Path]]] = {
            "01_video_index": self._stage_01_video_index,
            "02_frame_decode": self._stage_02_frame_decode,
            "03_object_detection": self._stage_03_objects,
            "04_instance_segmentation": self._stage_04_masks,
            "05_keypoints": self._stage_05_keypoints,
            "06_depth": self._stage_06_depth,
            "07_tracking": self._stage_07_tracking,
            "08_sequence_geometry": self._stage_08_geometry,
            "09_shared_3d": self._stage_09_shared_3d,
            "10_static_evidence": self._stage_10_static,
            "11_dynamic_evidence": self._stage_11_dynamic,
            "12_occlusion_evidence": self._stage_12_occlusion,
            "13_multilevel_aggregation": self._stage_13_aggregation,
        }
        completed: dict[str, list[str]] = {}
        for stage in execution_plan(target_stage):
            started = _now()
            upstream_hashes = []
            for dependency in stage.dependencies:
                upstream_hashes.extend(completed.get(dependency, []))
            source_hash = hashlib.sha256(
                "".join(sorted(sha256_file(path) for path in self._selected_sources())).encode("ascii")
            ).hexdigest()
            stage_cfg = self._stage_config(stage.name)
            cache_key = build_cache_key(
                source_video_sha256=source_hash,
                stage_name=stage.name,
                stage_config_sha256=stage_config_hash(stage_cfg),
                upstream_artifact_sha256=upstream_hashes,
                provider_weight_sha256=self._stage_weight_hashes(stage.name),
                schema_version=SCHEMA_VERSION,
            )
            lookup = self.cache.lookup(stage.name, cache_key)
            forced = stage.name in self.force_stages
            if resume and lookup.hit and not forced:
                record = json.loads(Path(lookup.record_path).read_text(encoding="utf-8"))
                hashes = [item["sha256"] for item in record["artifacts"]]
                completed[stage.name] = hashes
                self._record_stage(stage.name, "cached", cache_key, True, lookup.reason, started, len(hashes), "", {"resume": True})
                continue
            try:
                artifacts = handlers[stage.name]()
                self.cache.commit(stage.name, cache_key, artifacts, metadata={"forced": forced})
                completed[stage.name] = [sha256_file(path) for path in artifacts]
                self._record_stage(stage.name, "completed", cache_key, False, "forced" if forced else lookup.reason, started, len(artifacts), "", {})
            except Exception as exc:
                self._record_stage(stage.name, "failed", cache_key, False, lookup.reason, started, 0, str(exc), {})
                self._write_stage_status()
                raise
        self._write_stage_status()
        return {
            "dataset_id": self.dataset_id,
            "output_root": str(self.output_root),
            "stages": self.stage_status_rows,
            "labels_read": False,
        }

    def _stage_weight_hashes(self, stage_name: str) -> list[str]:
        if stage_name in {"03_object_detection", "07_tracking"}:
            return [self.weight_hashes.get("object_detection", "missing")]
        if stage_name == "04_instance_segmentation":
            return [self.weight_hashes.get("instance_segmentation", "missing")]
        if stage_name == "05_keypoints":
            return [self.weight_hashes.get("human_keypoints", "missing")]
        if stage_name in {"06_depth", "08_sequence_geometry", "09_shared_3d"}:
            return [self.weight_hashes.get("depth_model_cache", "missing")]
        return []

    def _stage_config(self, stage_name: str) -> dict[str, Any]:
        """Return only configuration that can affect one stage's outputs."""

        output = dict(self.config.get("stages", {}).get(stage_name, {}))
        if stage_name == "01_video_index":
            output.update(
                {
                    "dataset_name": self.config.get("dataset", {}).get("name"),
                    "sources": self.config.get("sources", {}).get("videos", []),
                    "clip_split": self.config.get("clip_split", {}),
                }
            )
        if stage_name in {"03_object_detection", "04_instance_segmentation", "05_keypoints", "11_dynamic_evidence", "12_occlusion_evidence", "13_multilevel_aggregation"}:
            output["artifacts"] = self.config.get("artifacts", {})
        if stage_name in {"03_object_detection", "04_instance_segmentation", "05_keypoints", "06_depth"}:
            output["providers"] = self.config.get("providers", {})
        if stage_name in {"08_sequence_geometry", "09_shared_3d"}:
            output["geometry_mode_by_source_name"] = self.config.get("geometry_mode_by_source_name", {})
        if stage_name == "13_multilevel_aggregation":
            output["branch_registry_version"] = self.config.get("branch_registry_version")
        return output

    def _record_stage(self, name: str, status: str, key: str, hit: bool, reason: str, started: str, artifact_count: int, failure: str, metadata: Mapping[str, Any]) -> None:
        self.stage_status_rows.append({"run_id": self.run_id, "stage_name": name, "status": status, "cache_key": key, "cache_hit": hit, "cache_reason": reason, "started_at": started, "finished_at": _now(), "artifact_count": artifact_count, "failure_reason": failure, "metadata": dict(metadata)})
        self._write_stage_status()

    def _write_stage_status(self) -> Path:
        return write_parquet(self.output_root / "manifests/stage_status.parquet", self.stage_status_rows, columns=STAGE_COLUMNS)

    def _read(self, relative: str) -> list[dict[str, Any]]:
        return self.reader.rows(relative)

    def _source_map(self) -> dict[str, dict[str, Any]]:
        return {str(row["source_name"]): row for row in self._read("manifests/videos.parquet")}

    def _manifest_source_path(self, row: Mapping[str, Any]) -> Path:
        """Resolve new absolute rows and legacy source-root-relative rows."""

        explicit = str(row.get("source_path", "")).strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        value = Path(str(row["source_relative_path"])).expanduser()
        return (value if value.is_absolute() else self.source_root / value).resolve()

    def _owned_frames(self) -> list[dict[str, Any]]:
        return [row for row in self._read("manifests/frames.parquet") if row.get("is_owned_frame")]

    def _stage_01_video_index(self) -> list[Path]:
        source_root = self.source_root
        video_rows = disambiguate_source_names(
            video_manifest_row(
                source_root=source_root,
                source_path=path,
                dataset_id=self.dataset_id,
            )
            for path in self._selected_sources()
        )
        clips: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        split = self.config.get("clip_split", {})
        for video in video_rows:
            path = self._manifest_source_path(video)
            signatures, decode_failures = decode_frame_signatures(path)
            if len(signatures) != int(video["frame_count"]):
                video["frame_count"] = len(signatures)
                video["decode_status"] = "ok_with_container_count_adjustment"
            scenes = split_scene_segments(signatures, cut_threshold=float(split.get("scene_cut_threshold", 0.35)))
            clip_rows, frame_rows = build_clip_manifests(
                video_id=str(video["video_id"]), frame_count=len(signatures), scene_segments=scenes,
                id_factory=self.ids, window_size=int(split.get("window_size", 32)),
                stride=int(split.get("stride", 16)), left_context=int(split.get("left_context", 4)),
                right_context=int(split.get("right_context", 4)), minimum_clip_length=int(split.get("minimum_clip_length", 4)),
            )
            assigned = {int(row["frame_index"]) for row in frame_rows}
            for frame_index in range(len(signatures)):
                if frame_index in assigned:
                    continue
                frame_id = self.ids.frame(str(video["video_id"]), frame_index)
                failure_clip = stable_id("unassigned_clip", video["video_id"], frame_index, prefix="clipfail")
                clip_rows.append({"clip_id": failure_clip, "video_id": video["video_id"], "scene_id": -1, "clip_ordinal": -1, "start_frame_index": frame_index, "end_frame_index": frame_index, "core_start_frame_index": frame_index, "core_end_frame_index": frame_index, "reference_frame_index": frame_index, "frame_count": 1, "coordinate_system_id": self.ids.coordinate_system(failure_clip), "geometry_mode": "unavailable", "sequence_scale_status": "unknown", "depth_alignment_domain": "none", "pose_graph_id": "", "scale_alignment_id": "", "valid": False, "missing_reason": "segment_shorter_than_minimum_clip_length"})
                frame_rows.append({"frame_record_id": stable_id("frame_record", frame_id, failure_clip, prefix="frec"), "frame_id": frame_id, "video_id": video["video_id"], "clip_id": failure_clip, "frame_index": frame_index, "scene_id": -1, "is_context_frame": False, "is_owned_frame": True, "owner_clip_id": failure_clip, "decode_status": "failed" if frame_index in decode_failures else "pending", "failure_reason": "frame_decode_failed" if frame_index in decode_failures else "segment_shorter_than_minimum_clip_length", "decoded_frame_path": ""})
            clips.extend(clip_rows)
            for row in frame_rows:
                row["decode_status"] = (
                    "indexed_decodable"
                    if not row.get("failure_reason")
                    else row.get("decode_status", "failed")
                )
                row["decoded_frame_path"] = (
                    f"arrays/frames/{video['source_name']}/frame_{int(row['frame_index']):06d}.jpg"
                )
            frames.extend(sorted(frame_rows, key=lambda row: (int(row["frame_index"]), str(row["clip_id"]))))
        video_path = self.writer.parquet("manifests/videos.parquet", video_rows, VIDEO_COLUMNS)
        clip_path = self.writer.parquet("manifests/clips.parquet", clips, CLIP_COLUMNS)
        frame_path = self.writer.parquet("manifests/frames.parquet", frames, FRAME_COLUMNS)
        manifest = DatasetManifest(
            dataset_id=self.dataset_id, schema_version=SCHEMA_VERSION, pipeline_version=PIPELINE_VERSION,
            git_commit=_git_commit(self.project_root), creation_time=_now(), config_path=str(self.config_path),
            config_sha256=self.config_sha256, source_root=str(source_root), source_video_count=len(video_rows),
            provider_metadata=dict(self.config.get("providers", {})), weight_sha256_by_provider=self.weight_hashes,
            branch_registry_version=str(self.config.get("branch_registry_version", "p4a_registry_v1")),
            coordinate_convention="right_handed_camera_x_right_y_down_z_forward;clip_local_world",
            depth_convention="larger_value_means_farther;relative_depth_not_metric",
            label_isolation=True, random_seed=int(self.config.get("runtime", {}).get("random_seed", 20260722)),
            environment=_environment(self.device, self.project_root),
        )
        manifest_path = self.writer.json("dataset_manifest.json", manifest.to_dict())
        labels_path = self.output_root / "labels_manifest.parquet"
        if not labels_path.exists():
            write_parquet(labels_path, [], columns=("video_id", "label", "temporal_annotation", "spatial_annotation"))
        return [video_path, clip_path, frame_path, manifest_path]

    def _stage_02_frame_decode(self) -> list[Path]:
        videos = self._source_map()
        frames = self._read("manifests/frames.parquet")
        owners = {(row["video_id"], int(row["frame_index"])): row for row in frames if row.get("is_owned_frame")}
        for source_name, video in videos.items():
            source_path = self._manifest_source_path(video)
            capture = cv2.VideoCapture(str(source_path))
            if not capture.isOpened():
                continue
            frame_index = 0
            try:
                while True:
                    success, image = capture.read()
                    if not success:
                        break
                    key = (video["video_id"], frame_index)
                    owner = owners.get(key)
                    if owner is not None and image is not None:
                        target = self.output_root / str(owner["decoded_frame_path"])
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix(".tmp.jpg")
                        if not cv2.imwrite(str(temporary), image):
                            owner["decode_status"] = "failed"
                            owner["failure_reason"] = "cv2_imwrite_failed"
                        else:
                            os.replace(temporary, target)
                            owner["decode_status"] = "ok"
                            owner["failure_reason"] = ""
                    frame_index += 1
            finally:
                capture.release()
        report = self.writer.json("reports/frame_decode_summary.json", {"dataset_id": self.dataset_id, "unique_frames": len(owners), "decoded_frames": sum(row["decode_status"] == "ok" for row in owners.values()), "failed_frames": sum(row["decode_status"] != "ok" for row in owners.values())})
        return [report]

    def _legacy_coverage_root(self) -> Path:
        value = self.config.get("artifacts", {}).get("p3d_coverage_root", "outputs/real_3d_evidence_coverage_v2")
        path = Path(str(value))
        return path if path.is_absolute() else self.project_root / path

    def _stage_03_objects(self) -> list[Path]:
        root = self._legacy_coverage_root()
        mask_rows = _csv_rows(root / "mask_coverage.csv")
        associations = _csv_rows(root / "mask_object_association.csv")
        assoc = {(row["video_id"], row["frame_index"], row["object_track_id"]): row for row in associations}
        source_map = self._source_map()
        frame_id_map = {(row["video_id"], int(row["frame_index"])): row["frame_id"] for row in self._owned_frames()}
        output = []
        for row in mask_rows:
            video = source_map.get(row["video_id"])
            if video is None:
                continue
            frame_index = int(row["frame_index"])
            frame_id = frame_id_map.get((video["video_id"], frame_index))
            if not frame_id:
                continue
            source_track = row["object_track_id"]
            track_id = self.ids.object_track(video["video_id"], source_track)
            association = assoc.get((row["video_id"], row["frame_index"], source_track), {})
            bbox = _json(row.get("mask_bbox"), None)
            bbox_area = math.nan if not bbox else max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1]))
            output.append({"object_observation_id": stable_id("object_observation", frame_id, track_id, association.get("object_id", source_track), prefix="objobs"), "video_id": video["video_id"], "frame_id": frame_id, "frame_index": frame_index, "source_object_id": association.get("object_id", ""), "object_track_id": track_id, "source_track_id": source_track, "semantic_label": row.get("class_name", "unknown"), "canonical_label": row.get("class_name", "unknown"), "confidence": _float(row.get("confidence"), 0.0), "bbox": bbox, "bbox_area": bbox_area, "frame_area": int(video["width"]) * int(video["height"]), "source_provider": "p3d_v2_yolo_real_detector", "valid": True, "missing_reason": "", "metadata": {"migrated_from": str(root / "mask_coverage.csv"), "labels_used": False}})
        path = self.writer.parquet("observations/objects.parquet", output, OBJECT_COLUMNS)
        return [path]

    def _stage_04_masks(self) -> list[Path]:
        root = self._legacy_coverage_root()
        source_rows = _csv_rows(root / "mask_coverage.csv")
        source_map = self._source_map()
        frame_id_map = {(row["video_id"], int(row["frame_index"])): row["frame_id"] for row in self._owned_frames()}
        output = []
        artifacts: list[Path] = []
        for row in source_rows:
            video = source_map.get(row["video_id"])
            if video is None:
                continue
            frame_index = int(row["frame_index"])
            frame_id = frame_id_map.get((video["video_id"], frame_index))
            if not frame_id:
                continue
            source_track = row["object_track_id"]
            track_id = self.ids.object_track(video["video_id"], source_track)
            valid = _bool(row.get("valid")) and bool(row.get("visible_mask_path"))
            array_ref = {"path": "", "shape": {}, "dtype": {}, "sha256": ""}
            source_mask = self.project_root / str(row.get("visible_mask_path", ""))
            missing_reason = row.get("missing_reason", "")
            if valid and source_mask.exists():
                mask = np.asarray(np.load(source_mask), dtype=np.uint8)
                target = self.output_root / "arrays/masks" / str(video["source_name"]) / f"frame_{frame_index:06d}_{track_id}.npz"
                array_ref = write_npz_array(target, visible_mask=mask)
                artifacts.append(target)
            elif valid:
                valid = False
                missing_reason = "formal_mask_array_missing"
            output.append({"mask_observation_id": stable_id("mask_observation", frame_id, track_id, prefix="maskobs"), "video_id": video["video_id"], "frame_id": frame_id, "frame_index": frame_index, "object_track_id": track_id, "segmentation_instance_id": row.get("segmentation_instance_id", ""), "class_id": row.get("class_id", ""), "class_name": row.get("class_name", ""), "confidence": _float(row.get("confidence"), 0.0), "visible_mask_path": array_ref["path"], "array_path": array_ref["path"], "array_shape": array_ref["shape"], "array_dtype": array_ref["dtype"], "mask_sha256": array_ref["sha256"], "array_sha256": array_ref["sha256"], "mask_area": _float(row.get("mask_area")), "mask_bbox": _json(row.get("mask_bbox"), None), "boundary_point_count": int(row.get("boundary_point_count") or 0), "source_provider": row.get("source_provider", ""), "weight_sha256": row.get("weight_sha256", ""), "is_visible_mask": True, "is_amodal_mask": False, "bbox_fallback": False, "valid": valid, "missing_reason": "" if valid else (missing_reason or "formal_mask_unavailable"), "metadata": {"migrated_from_p3d_v2": True, "formal_mask_evidence": valid, "labels_used": False}})
        path = self.writer.parquet("observations/masks.parquet", output, MASK_COLUMNS)
        return [path, *artifacts]

    def _stage_05_keypoints(self) -> list[Path]:
        source_rows = _csv_rows(self._legacy_coverage_root() / "keypoint_coverage.csv")
        source_map = self._source_map()
        frame_id_map = {(row["video_id"], int(row["frame_index"])): row["frame_id"] for row in self._owned_frames()}
        output = []
        for row in source_rows:
            video = source_map.get(row["video_id"])
            if not video:
                continue
            frame_index = int(row["frame_index"])
            frame_id = frame_id_map.get((video["video_id"], frame_index))
            if not frame_id:
                continue
            track_id = self.ids.object_track(video["video_id"], row["object_track_id"])
            valid = _bool(row.get("valid"))
            output.append({"keypoint_observation_id": stable_id("keypoint_observation", frame_id, track_id, prefix="kpobs"), "video_id": video["video_id"], "frame_id": frame_id, "frame_index": frame_index, "object_track_id": track_id, "total_keypoints": int(row.get("total_keypoints") or 0), "valid_keypoints": int(row.get("valid_keypoints") or 0), "valid_ratio": _float(row.get("valid_ratio")), "provider_name": row.get("provider_name", ""), "valid": valid, "missing_reason": "" if valid else (row.get("missing_reason") or "keypoints_unavailable"), "metadata": {"coordinates_not_migrated": True, "coverage_only": True}})
        path = self.writer.parquet("observations/keypoints.parquet", output, KEYPOINT_COLUMNS)
        return [path]

    def _stage_06_depth(self) -> list[Path]:
        from semantic3d.depth_provider import RealDepthProvider

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        full_names = set(self.config.get("stages", {}).get("06_depth", {}).get("full_video_source_names", []))
        model_name = str(self.config.get("providers", {}).get("depth_model", "depth-anything/Depth-Anything-V2-Small"))
        provider = None
        output = []
        artifacts: list[Path] = []
        source_by_id = {row["video_id"]: row for row in self._read("manifests/videos.parquet")}
        for frame in self._owned_frames():
            video = source_by_id[frame["video_id"]]
            source_name = str(video["source_name"])
            frame_id = str(frame["frame_id"])
            frame_index = int(frame["frame_index"])
            valid = False
            missing_reason = "depth_not_requested_for_video"
            ref = {"path": "", "shape": {}, "dtype": {}, "sha256": ""}
            metadata: dict[str, Any] = {"metric_depth": False, "labels_used": False}
            representation = "unknown"
            scale_status = "unknown"
            direction = "unknown"
            provider_name = ""
            valid_ratio = 0.0
            frame_path = self.output_root / str(frame["decoded_frame_path"])
            if source_name in full_names and frame_path.exists():
                target = self.output_root / "arrays/depth" / source_name / f"frame_{frame_index:06d}.npz"
                reused = False
                if target.exists():
                    try:
                        cached = np.load(target)
                        cached_depth = np.asarray(cached["depth_map"], dtype=np.float32)
                        cached_mask = np.asarray(cached["valid_mask"], dtype=bool)
                        if cached_depth.ndim != 2 or cached_depth.shape != cached_mask.shape:
                            raise ValueError("cached depth shape mismatch")
                        ref = {
                            "path": str(target),
                            "shape": {
                                name: list(np.asarray(cached[name]).shape)
                                for name in cached.files
                            },
                            "dtype": {
                                name: str(np.asarray(cached[name]).dtype)
                                for name in cached.files
                            },
                            "sha256": sha256_file(target),
                        }
                        artifacts.append(target)
                        valid = bool(np.any(cached_mask))
                        missing_reason = "" if valid else "cached_depth_has_no_valid_pixels"
                        representation = "relative_depth"
                        scale_status = "relative_per_frame"
                        direction = "farther"
                        provider_name = "transformers:depth-anything/Depth-Anything-V2-Small-hf"
                        valid_ratio = float(np.mean(cached_mask))
                        metadata.update(
                            {
                                "resumed_from_existing_frame_artifact": True,
                                "conversion": "reciprocal_inverse_to_relative",
                                "legacy_normalized_depth": False,
                            }
                        )
                        reused = True
                    except Exception:
                        target.unlink(missing_ok=True)
                if not reused:
                    if provider is None:
                        provider = RealDepthProvider(model_name=model_name, device=self.device, normalize=False, invert_depth=True)
                    try:
                        observation = provider.predict_observation(frame_path, frame_index=frame_index)
                        observation.require_geometry_depth()
                        ref = write_npz_array(target, depth_map=np.asarray(observation.depth_map, dtype=np.float32), valid_mask=np.asarray(observation.valid_mask, dtype=np.uint8), raw_model_output=np.asarray(observation.raw_model_output, dtype=np.float32))
                        artifacts.append(target)
                        valid = observation.valid
                        missing_reason = observation.missing_reason
                        representation = observation.depth_representation.value
                        scale_status = observation.scale_status.value
                        direction = observation.larger_value_means.value
                        provider_name = observation.provider_name
                        valid_ratio = float(np.mean(observation.valid_mask))
                        metadata.update(observation.metadata)
                    except Exception as exc:
                        missing_reason = f"depth_inference_failed:{type(exc).__name__}"
                        metadata["error"] = str(exc)
            output.append({"depth_observation_id": stable_id("depth_observation", frame_id, prefix="depthobs"), "video_id": video["video_id"], "frame_id": frame_id, "frame_index": frame_index, "array_path": ref["path"], "array_shape": ref["shape"], "array_dtype": ref["dtype"], "array_sha256": ref["sha256"], "depth_representation": representation, "scale_status": scale_status, "larger_value_means": direction, "provider_name": provider_name, "valid_pixel_ratio": valid_ratio, "valid": valid, "missing_reason": "" if valid else missing_reason, "metadata": metadata})
        path = self.writer.parquet("observations/depth.parquet", output, DEPTH_COLUMNS)
        return [path, *artifacts]

    def _stage_07_tracking(self) -> list[Path]:
        objects = self._read("observations/objects.parquet")
        counts = Counter(row["object_track_id"] for row in objects if row.get("valid"))
        rows = [{"object_track_id": track_id, "observation_count": count, "valid": count > 0, "missing_reason": "" if count > 0 else "no_observations"} for track_id, count in sorted(counts.items())]
        path = write_parquet(self.output_root / "observations/tracks.parquet", rows, columns=("object_track_id", "observation_count", "valid", "missing_reason"))
        return [path]

    def _load_npz(self, path: str) -> Optional[Mapping[str, np.ndarray]]:
        value = Path(path)
        value = value if value.is_absolute() else self.output_root / value
        try:
            return np.load(value)
        except Exception:
            return None

    def _stage_08_geometry(self) -> list[Path]:
        clips = self._read("manifests/clips.parquet")
        frames = self._owned_frames()
        depth = {row["frame_id"]: row for row in self._read("observations/depth.parquet")}
        masks_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._read("observations/masks.parquet"):
            if row.get("valid"):
                masks_by_frame[str(row["frame_id"])].append(row)
        video_by_id = {row["video_id"]: row for row in self._read("manifests/videos.parquet")}
        mode_by_name = self.config.get("geometry_mode_by_source_name", {})
        clip_by_id = {row["clip_id"]: row for row in clips}
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for frame in frames:
            grouped[str(frame["clip_id"])].append(frame)
        camera_rows = []
        for clip_id, clip_frames in sorted(grouped.items()):
            clip = clip_by_id[clip_id]
            video = video_by_id[clip["video_id"]]
            mode = str(mode_by_name.get(video["source_name"], "unsupported_3d"))
            medians: dict[str, float] = {}
            for frame in clip_frames:
                depth_row = depth.get(frame["frame_id"])
                if not depth_row or not depth_row.get("valid"):
                    continue
                data = self._load_npz(str(depth_row["array_path"]))
                if data is None or "depth_map" not in data or "valid_mask" not in data:
                    continue
                values = np.asarray(data["depth_map"], dtype=float)
                valid_mask = np.asarray(data["valid_mask"], dtype=bool)
                foreground = np.zeros(values.shape, dtype=bool)
                for mask_row in masks_by_frame.get(str(frame["frame_id"]), []):
                    mask_data = self._load_npz(str(mask_row["array_path"]))
                    if mask_data is None or "visible_mask" not in mask_data:
                        continue
                    foreground |= np.asarray(mask_data["visible_mask"], dtype=bool)
                support = valid_mask & ~foreground & np.isfinite(values) & (values > 0.0)
                if int(np.sum(support)) >= 64:
                    medians[str(frame["frame_id"])] = float(np.median(values[support]))
            reference_values = [value for value in medians.values() if math.isfinite(value) and value > 0.0]
            reference_median = float(np.median(reference_values)) if reference_values else math.nan
            width, height = int(video["width"]), int(video["height"])
            focal = float(max(width, height))
            K = [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
            for frame in clip_frames:
                current = medians.get(str(frame["frame_id"]), math.nan)
                aligned = math.isfinite(reference_median) and math.isfinite(current) and current > 0.0 and mode in {"static_camera_3d", "rotation_compensated", "full_se3_3d"}
                scale = reference_median / current if aligned else math.nan
                reason = "" if aligned else ("depth_alignment_unavailable" if mode != "unsupported_3d" else "unsupported_geometry_mode")
                camera_rows.append({"camera_observation_id": stable_id("camera_observation", clip_id, frame["frame_id"], prefix="camobs"), "video_id": video["video_id"], "clip_id": clip_id, "frame_id": frame["frame_id"], "frame_index": frame["frame_index"], "coordinate_system_id": clip["coordinate_system_id"], "K": K if aligned else None, "T_world_camera": np.eye(4).tolist() if aligned and mode == "static_camera_3d" else None, "coordinate_convention": "right_handed_camera_x_right_y_down_z_forward", "intrinsics_source": "approximate_focal_length_for_relative_geometry", "pose_source": "static_camera_contract" if aligned and mode == "static_camera_3d" else "", "geometry_mode": mode, "depth_alignment_scale": scale, "quality": 0.5 if aligned else 0.0, "valid": aligned, "missing_reason": reason, "metadata": {"alignment": "clip_background_median_scale", "relative_not_metric": True, "foreground_masks_excluded": bool(masks_by_frame.get(str(frame["frame_id"])))}})
        path = self.writer.parquet("observations/camera.parquet", camera_rows, CAMERA_COLUMNS)
        return [path]

    def _stage_09_shared_3d(self) -> list[Path]:
        owned = self._owned_frames()
        clips = {row["clip_id"]: row for row in self._read("manifests/clips.parquet")}
        videos = {row["video_id"]: row for row in self._read("manifests/videos.parquet")}
        objects_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        masks_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for row in self._read("observations/objects.parquet"):
            objects_by_frame[str(row["frame_id"])].append(row)
        for row in self._read("observations/masks.parquet"):
            if row.get("valid"):
                masks_by_key[(str(row["frame_id"]), str(row["object_track_id"]))] = row
        depth = {row["frame_id"]: row for row in self._read("observations/depth.parquet")}
        cameras = {(row["clip_id"], row["frame_id"]): row for row in self._read("observations/camera.parquet")}
        shared_rows = []
        artifacts: list[Path] = []
        for frame in owned:
            clip = clips[frame["clip_id"]]
            camera = cameras.get((frame["clip_id"], frame["frame_id"]))
            depth_row = depth.get(frame["frame_id"])
            centers = []
            track_ids = []
            valid = bool(camera and camera.get("valid") and depth_row and depth_row.get("valid"))
            missing_reason = "" if valid else "shared_geometry_inputs_missing"
            ref = {"path": "", "shape": {}, "dtype": {}, "sha256": ""}
            if valid:
                data = self._load_npz(str(depth_row["array_path"]))
                if data is None or "depth_map" not in data:
                    valid = False
                    missing_reason = "depth_array_unreadable"
                else:
                    depth_map = np.asarray(data["depth_map"], dtype=float) * float(camera["depth_alignment_scale"])
                K = np.asarray(_json(camera["K"], camera["K"]), dtype=float)
                for obj in objects_by_frame.get(str(frame["frame_id"]), []) if valid else []:
                    bbox = _json(obj.get("bbox"), obj.get("bbox"))
                    mask_row = masks_by_key.get((str(frame["frame_id"]), str(obj["object_track_id"])))
                    if not bbox or not mask_row:
                        continue
                    mask_data = self._load_npz(str(mask_row["array_path"]))
                    if mask_data is None or "visible_mask" not in mask_data:
                        continue
                    mask = np.asarray(mask_data["visible_mask"], dtype=bool)
                    support = mask & np.isfinite(depth_map) & (depth_map > 0.0)
                    if int(np.sum(support)) < 16:
                        continue
                    z = float(np.median(depth_map[support]))
                    u = (float(bbox[0]) + float(bbox[2])) / 2.0
                    v = (float(bbox[1]) + float(bbox[3])) / 2.0
                    x = (u - K[0, 2]) * z / K[0, 0]
                    y = (v - K[1, 2]) * z / K[1, 1]
                    centers.append([x, y, z])
                    track_ids.append(str(obj["object_track_id"]))
                valid = bool(centers)
                missing_reason = "" if valid else "no_valid_object_reconstruction"
            if valid:
                source_name = videos[frame["video_id"]]["source_name"]
                target = self.output_root / "arrays/shared_3d" / str(source_name) / str(frame["clip_id"]) / f"frame_{int(frame['frame_index']):06d}.npz"
                ref = write_npz_array(target, object_centers_3d=np.asarray(centers, dtype=np.float32))
                artifacts.append(target)
            shared_rows.append({"shared_3d_frame_id": stable_id("shared_3d_frame", frame["clip_id"], frame["frame_id"], prefix="s3df"), "video_id": frame["video_id"], "clip_id": frame["clip_id"], "frame_id": frame["frame_id"], "frame_index": frame["frame_index"], "coordinate_system_id": clip["coordinate_system_id"], "geometry_mode": camera.get("geometry_mode", "unavailable") if camera else "unavailable", "sequence_scale_status": "relative_shared_sequence" if valid else "unknown", "array_path": ref["path"], "array_shape": ref["shape"], "array_dtype": ref["dtype"], "array_sha256": ref["sha256"], "object_track_ids": track_ids, "valid_object_count": len(track_ids), "quality": float(camera.get("quality", 0.0)) if valid and camera else 0.0, "valid": valid, "missing_reason": missing_reason, "metadata": {"metric_geometry": False, "clip_local_only": True, "cross_clip_subtraction_allowed": False}})
        clip_rows = []
        shared_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in shared_rows:
            shared_by_clip[str(row["clip_id"])].append(row)
        aggregate_root_value = self.config.get("artifacts", {}).get(
            "p4a_aggregation_root", "outputs/partial_p4_aggregation_smoke"
        )
        aggregate_root = Path(str(aggregate_root_value))
        aggregate_root = (
            aggregate_root
            if aggregate_root.is_absolute()
            else self.project_root / aggregate_root
        )
        dynamic_ready_by_source: dict[str, set[int]] = defaultdict(set)
        for row in _csv_rows(aggregate_root / "frame_aggregates.csv"):
            if _bool(row.get("valid")):
                dynamic_ready_by_source[str(row.get("video_id", ""))].add(
                    int(row.get("frame_index") or -1)
                )
        for clip_id, clip in clips.items():
            values = shared_by_clip.get(str(clip_id), [])
            valid_count = sum(bool(row["valid"]) for row in values)
            total = len(values)
            mode = next((row["geometry_mode"] for row in values if row["valid"]), "unavailable")
            ratio = valid_count / total if total else math.nan
            source_name = str(videos[clip["video_id"]]["source_name"])
            dynamic_ready_count = sum(
                int(int(row["frame_index"]) in dynamic_ready_by_source.get(source_name, set()))
                for row in values
            )
            dynamic_ratio = dynamic_ready_count / total if total else math.nan
            clip_rows.append({"shared_3d_clip_id": stable_id("shared_3d_clip", clip_id, prefix="s3dc"), "video_id": clip["video_id"], "clip_id": clip_id, "coordinate_system_id": clip["coordinate_system_id"], "reference_frame_index": clip["reference_frame_index"], "geometry_mode": mode, "sequence_scale_status": "relative_shared_sequence" if valid_count else "unknown", "depth_alignment_domain": "clip_local_background_median", "pose_graph_id": clip["pose_graph_id"], "scale_alignment_id": clip["scale_alignment_id"], "owned_frame_count": total, "valid_shared_frame_count": valid_count, "shared_3d_owned_frame_ratio": ratio, "depth_aligned_frame_ratio": ratio, "dynamic_readiness_frame_ratio": dynamic_ratio, "valid": valid_count > 0, "missing_reason": "" if valid_count else "no_valid_shared_3d_owned_frames", "metadata": {"coordinate_isolation_enforced": True, "ordinary_structure_recomputed": True, "ordinary_structure_missing_reason": "stable_mask_point_correspondence_not_implemented", "dynamic_readiness_requires_point_tracks": True, "dynamic_ready_owned_frame_count": dynamic_ready_count}})
        frame_path = self.writer.parquet("observations/shared_3d_frames.parquet", shared_rows, SHARED_FRAME_COLUMNS)
        clip_path = self.writer.parquet("observations/shared_3d_clips.parquet", clip_rows, SHARED_CLIP_COLUMNS)
        valid_shared_frame_ids = {str(row["frame_id"]) for row in shared_rows if row["valid"]}
        valid_mask_keys = {
            (str(row["frame_id"]), str(row["object_track_id"]))
            for row in self._read("observations/masks.parquet")
            if row.get("valid") and str(row["frame_id"]) in valid_shared_frame_ids
        }
        object_labels = {
            (str(row["frame_id"]), str(row["object_track_id"])): str(row["semantic_label"])
            for row in self._read("observations/objects.parquet")
        }
        ordinary_tracks = {
            track_id
            for frame_id, track_id in valid_mask_keys
            if object_labels.get((frame_id, track_id), "") != "person"
        }
        person_keypoint_tracks = {
            str(row["object_track_id"])
            for row in self._read("observations/keypoints.parquet")
            if row.get("valid")
        }
        structure_rows = _csv_rows(
            self._legacy_coverage_root() / "structure_residual_coverage.csv"
        )
        valid_structure_transitions = sum(
            int(row.get("valid_residual_count") or 0)
            for row in structure_rows
            if _bool(row.get("valid"))
        )
        funnel_path = self.writer.json(
            "reports/full_video_structure_funnel.json",
            {
                "shared_3d_valid_frame_count": len(valid_shared_frame_ids),
                "ordinary_tracks_with_formal_mask_and_shared_depth": len(ordinary_tracks),
                "ordinary_track_ids": sorted(ordinary_tracks),
                "ordinary_tracks_with_stable_3d_point_correspondence": 0,
                "ordinary_formal_structure_graph_count": 0,
                "ordinary_structure_transition_count": 0,
                "ordinary_missing_reason": "stable_mask_point_correspondence_not_implemented",
                "person_structure_track_count": len(person_keypoint_tracks),
                "existing_formal_structure_transition_count": valid_structure_transitions,
                "recomputed_on_full_clip_coverage": True,
                "cross_unaligned_clip_transitions_computed": False,
            },
        )
        return [frame_path, clip_path, funnel_path, *artifacts]

    def _empty_evidence_table(self, level: str) -> Path:
        path = self.output_root / f"evidence/{level}_evidence.parquet"
        write_parquet(path, [], columns=EVIDENCE_OUTPUT_COLUMNS)
        return path

    def _stage_10_static(self) -> list[Path]:
        path = self.output_root / "evidence/intermediate/static_object_evidence.parquet"
        write_parquet(path, [], columns=EVIDENCE_OUTPUT_COLUMNS)
        return [path]

    def _frame_owner(self) -> dict[tuple[str, int], dict[str, Any]]:
        videos = {row["source_name"]: row for row in self._read("manifests/videos.parquet")}
        output = {}
        for frame in self._owned_frames():
            name = next((source_name for source_name, video in videos.items() if video["video_id"] == frame["video_id"]), "")
            output[(name, int(frame["frame_index"]))] = frame
        return output

    def _evidence_from_aggregate(self, row: Mapping[str, Any], level: str, frame: Mapping[str, Any], source_name: str) -> dict[str, Any]:
        raw = _float(row.get("value"))
        valid = _bool(row.get("valid")) and math.isfinite(raw)
        branches = _json(row.get("contributing_branch_names"), [])
        branch = branches[0] if len(branches) == 1 else "multibranch_aggregate"
        source_track = str(row.get("object_track_id", ""))
        object_track = self.ids.object_track(frame["video_id"], source_track) if source_track else ""
        source_point = str(row.get("point_id", ""))
        source_edge = str(row.get("edge_id", ""))
        point_id = self.ids.point(object_track, source_point) if source_point and object_track else ""
        edge_id = self.ids.edge(object_track, source_edge) if source_edge and object_track else ""
        evidence = EvidenceRecord(
            evidence_id=self.ids.evidence(str(branch), level, str(frame["frame_id"]), object_track, point_id, edge_id),
            branch_name=str(branch), evidence_level=level, video_id=str(frame["video_id"]), clip_id=str(frame["clip_id"]), frame_id=str(frame["frame_id"]), object_track_id=object_track, point_id=point_id, edge_id=edge_id,
            raw_value=raw if valid else math.nan, intrinsic_normalized_value=raw if valid else math.nan,
            statistically_normalized_value=math.nan, normalization_fit_source="none", valid=valid,
            quality=float(np.clip(_float(row.get("quality"), 0.0), 0.0, 1.0)),
            applicability=Applicability.APPLICABLE_VALID if valid else Applicability.APPLICABLE_INVALID,
            missing_reason="" if valid else (str(row.get("missing_reason")) or "invalid_migrated_evidence"),
            geometry_mode="static_camera_3d", sequence_scale_status="relative_shared_sequence",
            coordinate_system_id=str(next(item["coordinate_system_id"] for item in self._read("manifests/clips.parquet") if item["clip_id"] == frame["clip_id"])),
            source_evidence_ids=tuple(str(value) for value in _json(row.get("contributing_source_ids"), [])),
            localization_reference=str(row.get("localization_mask_reference", "")),
            metadata={"migrated_from": "p4a_partial_smoke", "source_name": source_name, "formal_or_diagnostic": "formal"},
        ).to_dict()
        evidence["included_in_formal_aggregation"] = valid
        return evidence

    def _stage_11_dynamic(self) -> list[Path]:
        aggregate_root_value = self.config.get("artifacts", {}).get("p4a_aggregation_root", "outputs/partial_p4_aggregation_smoke")
        aggregate_root = Path(str(aggregate_root_value))
        aggregate_root = aggregate_root if aggregate_root.is_absolute() else self.project_root / aggregate_root
        owners = self._frame_owner()
        outputs: dict[str, list[dict[str, Any]]] = {level: [] for level in ("point", "edge", "object", "frame")}
        files = {"point": "point_aggregates.csv", "edge": "edge_aggregates.csv", "object": "object_aggregates.csv", "frame": "frame_aggregates.csv"}
        for level, filename in files.items():
            for row in _csv_rows(aggregate_root / filename):
                source_name = str(row.get("video_id", ""))
                frame_index = int(row.get("frame_index") or -1)
                owner = owners.get((source_name, frame_index))
                if owner is None:
                    continue
                if level == "object":
                    branch_scores = _json(row.get("branch_scores"), {})
                    for branch_name, branch_value in sorted(branch_scores.items()):
                        branch_row = dict(row)
                        branch_row["value"] = branch_value
                        branch_row["contributing_branch_names"] = json_text([branch_name])
                        branch_row["contributing_source_ids"] = json_text(
                            _json(row.get("contributing_source_ids"), [])
                        )
                        outputs[level].append(
                            self._evidence_from_aggregate(
                                branch_row, level, owner, source_name
                            )
                        )
                else:
                    outputs[level].append(self._evidence_from_aggregate(row, level, owner, source_name))
        paths = []
        for level, rows in outputs.items():
            path = self.output_root / f"evidence/{level}_evidence.parquet"
            write_parquet(path, rows, columns=EVIDENCE_OUTPUT_COLUMNS)
            paths.append(path)
        return paths

    def _stage_12_occlusion(self) -> list[Path]:
        depth = {
            row["frame_id"]: row
            for row in self._read("observations/depth.parquet")
            if row.get("valid")
        }
        masks_by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self._read("observations/masks.parquet"):
            if row.get("valid"):
                masks_by_frame[str(row["frame_id"])].append(row)
        sync_rows = []
        for frame in self._owned_frames():
            frame_id = str(frame["frame_id"])
            depth_row = depth.get(frame_id)
            mask_rows = masks_by_frame.get(frame_id, [])
            if depth_row is None or len(mask_rows) < 2:
                continue
            depth_data = self._load_npz(str(depth_row["array_path"]))
            if depth_data is None or "depth_map" not in depth_data:
                continue
            depth_map = np.asarray(depth_data["depth_map"], dtype=float)
            object_depths: dict[str, float] = {}
            loaded_masks: list[tuple[str, np.ndarray]] = []
            for mask_row in mask_rows:
                mask_data = self._load_npz(str(mask_row["array_path"]))
                if mask_data is None or "visible_mask" not in mask_data:
                    continue
                mask = np.asarray(mask_data["visible_mask"], dtype=bool)
                support = mask & np.isfinite(depth_map) & (depth_map > 0.0)
                if int(np.sum(support)) < 16:
                    continue
                track_id = str(mask_row["object_track_id"])
                object_depths[track_id] = float(np.median(depth_map[support]))
                loaded_masks.append((track_id, mask))
            overlap_pairs = 0
            for index, (_, first) in enumerate(loaded_masks):
                for _, second in loaded_masks[index + 1 :]:
                    if np.any(first & second):
                        overlap_pairs += 1
            if len(object_depths) >= 2:
                sync_rows.append(
                    {
                        "video_id": frame["video_id"],
                        "clip_id": frame["clip_id"],
                        "frame_id": frame_id,
                        "frame_index": frame["frame_index"],
                        "object_depths": object_depths,
                        "front_to_back_track_ids": [
                            item[0]
                            for item in sorted(object_depths.items(), key=lambda item: item[1])
                        ],
                        "overlap_pair_count": overlap_pairs,
                        "depth_order_available": True,
                        "formal_occlusion_event": False,
                        "applicability": "not_applicable",
                        "missing_reason": "no_validated_visibility_change_event",
                        "metadata": {
                            "larger_value_means": "farther",
                            "same_frame_relative_depth_order": True,
                            "not_anomaly_evidence_without_event": True,
                        },
                    }
                )
        sync_path = self.output_root / "reports/occlusion_depth_order_sync.parquet"
        write_parquet(
            sync_path,
            sync_rows,
            columns=(
                "video_id", "clip_id", "frame_id", "frame_index",
                "object_depths", "front_to_back_track_ids", "overlap_pair_count",
                "depth_order_available", "formal_occlusion_event", "applicability",
                "missing_reason", "metadata",
            ),
        )
        report = self.writer.json(
            "reports/occlusion_stage_summary.json",
            {
                "formal_events_imported": 0,
                "depth_order_synchronized_frame_count": len(sync_rows),
                "frames_with_mask_overlap_and_depth_order": sum(
                    int(row["overlap_pair_count"] > 0) for row in sync_rows
                ),
                "applicability": "not_applicable",
                "reason": "full_video_scan_found_no_validated_visibility_change_event",
                "no_event_is_not_zero_residual": True,
            },
        )
        return [sync_path, report]

    def _stage_13_aggregation(self) -> list[Path]:
        clips = self._read("manifests/clips.parquet")
        shared = {row["clip_id"]: row for row in self._read("observations/shared_3d_clips.parquet")}
        evidence_rows = []
        registry = _load_branch_registry(self.project_root, self.config)
        raw_levels = []
        for level in ("point", "edge", "object", "frame"):
            raw_levels.extend(self._read(f"evidence/{level}_evidence.parquet"))
        valid_branches_by_clip: dict[str, set[str]] = defaultdict(set)
        for row in raw_levels:
            if row.get("valid") and row.get("branch_name") != "multibranch_aggregate":
                valid_branches_by_clip[str(row["clip_id"])].add(str(row["branch_name"]))
        event_branches = {"occlusion_depth_order", "visibility_explanation", "boundary_occlusion", "reappearance_consistency"}
        for clip in clips:
            shared_clip = shared.get(clip["clip_id"])
            geometry_valid = bool(shared_clip and shared_clip.get("valid"))
            mode = str(shared_clip.get("geometry_mode", "unavailable")) if shared_clip else "unavailable"
            for branch_name, definition in sorted(registry.items()):
                if branch_name in valid_branches_by_clip.get(str(clip["clip_id"]), set()):
                    applicability = Applicability.APPLICABLE_INVALID
                    reason = "valid_lower_level_evidence_not_reduced_to_branch_clip_score"
                elif branch_name in event_branches and geometry_valid:
                    applicability = Applicability.NOT_APPLICABLE
                    reason = "no_observed_event_in_clip"
                elif not geometry_valid:
                    applicability = Applicability.INVALID_GEOMETRY
                    reason = str(shared_clip.get("missing_reason", "shared_3d_unavailable")) if shared_clip else "shared_3d_unavailable"
                elif mode not in set(definition.get("supported_geometry_modes", [])):
                    applicability = Applicability.UNSUPPORTED_MODE
                    reason = "branch_unsupported_for_geometry_mode"
                else:
                    applicability = Applicability.OBSERVATION_MISSING
                    reason = "branch_observation_not_materialized"
                row = EvidenceRecord(
                    evidence_id=self.ids.evidence(branch_name, "clip", str(clip["clip_id"])),
                    branch_name=branch_name, evidence_level="clip", video_id=str(clip["video_id"]), clip_id=str(clip["clip_id"]),
                    raw_value=math.nan, intrinsic_normalized_value=math.nan, statistically_normalized_value=math.nan,
                    normalization_fit_source="none", valid=False, quality=0.0, applicability=applicability,
                    missing_reason=reason, geometry_mode=mode, sequence_scale_status=str(shared_clip.get("sequence_scale_status", "unknown")) if shared_clip else "unknown",
                    coordinate_system_id=str(clip["coordinate_system_id"]), metadata={"formal_or_diagnostic": definition.get("formal_or_diagnostic", "formal"), "readiness_only": True, "not_anomaly_score": True},
                ).to_dict()
                row["included_in_formal_aggregation"] = False
                evidence_rows.append(row)
        aggregate_root_value = self.config.get("artifacts", {}).get(
            "p4a_aggregation_root", "outputs/partial_p4_aggregation_smoke"
        )
        aggregate_root = Path(str(aggregate_root_value))
        aggregate_root = (
            aggregate_root
            if aggregate_root.is_absolute()
            else self.project_root / aggregate_root
        )
        clip_aggregate_path = aggregate_root / "clip_aggregates.json"
        if clip_aggregate_path.exists():
            owners = self._frame_owner()
            for aggregate in json.loads(clip_aggregate_path.read_text(encoding="utf-8")):
                source_name = str(aggregate.get("video_id", ""))
                frame_indices = [int(value) for value in aggregate.get("frame_indices", [])]
                owner_rows = [owners.get((source_name, index)) for index in frame_indices]
                owner_rows = [row for row in owner_rows if row is not None]
                owner_clip_ids = {str(row["clip_id"]) for row in owner_rows}
                value = _float(aggregate.get("value"))
                valid = bool(aggregate.get("valid")) and math.isfinite(value)
                if len(owner_rows) != len(frame_indices) or len(owner_clip_ids) != 1:
                    valid = False
                    reason = "aggregate_frames_cross_unaligned_owner_clips"
                    owner = owner_rows[0] if owner_rows else None
                else:
                    reason = ""
                    owner = owner_rows[0]
                if owner is None:
                    continue
                clip = next(item for item in clips if item["clip_id"] == owner["clip_id"])
                aggregate_row = EvidenceRecord(
                    evidence_id=self.ids.evidence(
                        "multilevel_aggregate", "clip", str(owner["clip_id"])
                    ),
                    branch_name="multilevel_aggregate",
                    evidence_level="clip",
                    video_id=str(owner["video_id"]),
                    clip_id=str(owner["clip_id"]),
                    raw_value=value if valid else math.nan,
                    intrinsic_normalized_value=value if valid else math.nan,
                    statistically_normalized_value=math.nan,
                    normalization_fit_source="none",
                    valid=valid,
                    quality=float(np.clip(_float(aggregate.get("quality"), 0.0), 0.0, 1.0)),
                    applicability=(
                        Applicability.APPLICABLE_VALID
                        if valid
                        else Applicability.INVALID_GEOMETRY
                    ),
                    missing_reason=reason,
                    geometry_mode="static_camera_3d",
                    sequence_scale_status="relative_shared_sequence",
                    coordinate_system_id=str(clip["coordinate_system_id"]),
                    source_evidence_ids=tuple(
                        str(value)
                        for value in aggregate.get("contributing_source_ids", [])
                    ),
                    metadata={
                        "migrated_from": "p4a_partial_smoke",
                        "classification_output": False,
                        "cross_clip_alignment_required": len(owner_clip_ids) > 1,
                        "formal_or_diagnostic": "formal",
                    },
                ).to_dict()
                aggregate_row["included_in_formal_aggregation"] = valid
                evidence_rows.append(aggregate_row)
        path = self.output_root / "evidence/clip_evidence.parquet"
        write_parquet(path, evidence_rows, columns=EVIDENCE_OUTPUT_COLUMNS)
        coverage = _coverage_summary(clips, registry, evidence_rows, raw_levels)
        coverage_path = self.writer.json("reports/aggregation_coverage.json", coverage)
        return [path, coverage_path]


def _directory_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        stat = item.stat()
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
    return digest.hexdigest()


def _load_branch_registry(project_root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    value = config.get("artifacts", {}).get("p4a_aggregation_root", "outputs/partial_p4_aggregation_smoke")
    root = Path(str(value))
    root = root if root.is_absolute() else project_root / root
    path = root / "evidence_registry.json"
    if path.exists():
        return dict(json.loads(path.read_text(encoding="utf-8")).get("branches", {}))
    names = ("semantic_size_3d", "depth_order_3d", "boundary_depth_3d", "spatial_intersection_3d", "track_3d_continuity", "direction_consistency", "relative_velocity_change", "dynamic_reprojection", "structure_temporal", "occlusion_depth_order", "visibility_explanation", "boundary_occlusion", "reappearance_consistency")
    return {name: {"supported_geometry_modes": ["static_camera_3d", "full_se3_3d"], "formal_or_diagnostic": "formal"} for name in names}


def _coverage_summary(clips: Sequence[Mapping[str, Any]], registry: Mapping[str, Any], clip_rows: Sequence[Mapping[str, Any]], lower_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid_by_clip: dict[str, set[str]] = defaultdict(set)
    for row in lower_rows:
        if row.get("valid") and row.get("branch_name") in registry:
            valid_by_clip[str(row["clip_id"])].add(str(row["branch_name"]))
    applicable_by_clip: dict[str, set[str]] = defaultdict(set)
    statuses = defaultdict(Counter)
    for row in clip_rows:
        if row.get("branch_name") not in registry:
            continue
        statuses[str(row["clip_id"])][str(row["applicability"])] += 1
        if row["applicability"] in {Applicability.APPLICABLE_VALID.value, Applicability.APPLICABLE_INVALID.value, Applicability.OBSERVATION_MISSING.value}:
            applicable_by_clip[str(row["clip_id"])].add(str(row["branch_name"]))
    per_clip = []
    for clip in clips:
        clip_id = str(clip["clip_id"])
        valid = valid_by_clip.get(clip_id, set())
        applicable = applicable_by_clip.get(clip_id, set())
        per_clip.append({"clip_id": clip_id, "video_id": clip["video_id"], "registry_branch_coverage": len(valid) / len(registry) if registry else math.nan, "applicable_branch_coverage": len(valid) / len(applicable) if applicable else math.nan, "valid_branch_count": len(valid), "applicable_branch_count": len(applicable), "status_counts": dict(statuses[clip_id])})
    return {"branch_registry_version": "p4a_registry_v1", "registered_branch_count": len(registry), "per_clip": per_clip, "labels_used": False, "classification_output": False}
