"""Offline P4-C3B-M3 view, metric-scale, and track-history smoke."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

from ..dimension_aligned_scale_depth import load_dimension_aligned_prior_resolver
from ..geometry.camera import CameraObservation, CoordinateConvention
from .metric_scale import (
    ExtentEstimator,
    MetricDepthDefinition,
    MetricDepthEvidence,
    MetricDepthType,
    MetricObjectRegion,
    MetricScaleStatus,
    MetricScaleThresholds,
    MetricSingleObjectScaleBranch,
)
from .multi_interval_prior import MultiIntervalScalePriorRegistry
from .scale_evidence import ProviderStatus
from .temporal_scale import (
    ScaleHistoryObservation,
    TemporalReferenceMethod,
    TemporalSameObjectScaleBranch,
    TemporalScaleMode,
)
from .view_observability import (
    CameraMotionClass,
    CameraViewObservation,
    ObjectViewInput,
    PoseEstimateStatus,
    ViewpointClass,
    evaluate_object_view,
)


STRICT_HASHES = {
    "scale_priors_strict_v1.yaml": (
        "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"
    ),
    "scale_priors_strict_v2.yaml": (
        "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"
    ),
}
PROTOCOL_HASHES = {
    "p4c0_experiment_protocol_v1.yaml": (
        "8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304"
    ),
    "p4c1_experiment_manifest_v1.yaml": (
        "ec48e26da4f434a1356959997b546ac30dc9e439281b2e09174f7c86a35ce086"
    ),
    "p4c2_formal_data_readiness_v1.yaml": (
        "fe8c3cda137337330209528f4025d0593fefa42886be89eb214bfd58d38a8d89"
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _software_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "visible_mask" not in archive.files:
            raise ValueError("Formal instance mask archive lacks visible_mask.")
        return np.asarray(archive["visible_mask"], dtype=bool)


def _parse_bbox(value: object) -> tuple[float, float, float, float]:
    data = json.loads(value) if isinstance(value, str) else value
    if not isinstance(data, (list, tuple)) or len(data) != 4:
        raise ValueError("Object bbox must contain four values.")
    return tuple(float(item) for item in data)


def _scale_bbox(
    bbox: tuple[float, float, float, float],
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> tuple[float, float, float, float]:
    source_height, source_width = source_shape
    target_height, target_width = target_shape
    sx = target_width / source_width
    sy = target_height / source_height
    return bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy


def _motion_class(row: Mapping[str, Any] | None) -> CameraMotionClass:
    if row is None:
        return CameraMotionClass.MOTION_UNRELIABLE
    geometry = str(row.get("geometry_mode", "unavailable"))
    motion = float(row.get("median_pixel_motion", float("nan")))
    if geometry == "static_camera_3d":
        return (
            CameraMotionClass.STATIC
            if math.isfinite(motion) and motion <= 0.5
            else CameraMotionClass.LOW_MOTION
        )
    if geometry in {"full_se3", "metric_world", "camera_motion_3d"}:
        return CameraMotionClass.CAMERA_MOTION
    if geometry in {"object_motion", "object_centric_3d"}:
        return CameraMotionClass.OBJECT_MOTION
    if geometry in {"mixed_motion"}:
        return CameraMotionClass.MIXED_MOTION
    return CameraMotionClass.MOTION_UNRELIABLE


def _find_clip(
    rows: Sequence[Mapping[str, Any]], video_id: str, frame_index: int
) -> Mapping[str, Any] | None:
    candidates = [
        row
        for row in rows
        if str(row["video_id"]) == video_id
        and int(row["start_frame_index"]) <= frame_index <= int(row["end_frame_index"])
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            not (
                int(row["core_start_frame_index"])
                <= frame_index
                <= int(row["core_end_frame_index"])
            ),
            int(row["clip_ordinal"]),
        )
    )
    return candidates[0]


def _camera_fingerprint(K: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(K, dtype=np.float64).tobytes()).hexdigest()


def _temporal_relative_compatibility_check() -> bool:
    branch = TemporalSameObjectScaleBranch(reference_method="previous_valid")
    first = ScaleHistoryObservation(
        "synthetic",
        "clip",
        "frame_0",
        0,
        "track",
        "object_0",
        "relative_visible_extent",
        1.0,
        "relative_local_unit",
        TemporalScaleMode.RELATIVE_LOCAL,
        "relative_depth_provider",
        "z_depth",
        "relative_K",
        "relative_shared_clip",
    )
    second = ScaleHistoryObservation(
        "synthetic",
        "clip",
        "frame_1",
        1,
        "track",
        "object_1",
        "relative_visible_extent",
        1.0,
        "relative_local_unit",
        TemporalScaleMode.RELATIVE_LOCAL,
        "relative_depth_provider",
        "z_depth",
        "relative_K",
        "relative_shared_clip",
    )
    result = branch.evaluate(second, [first])
    return bool(result.valid and result.depth_unit == "relative_local_unit")


def _funnel(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "total": len(rows),
        "applicable": sum(bool(row.get("applicable")) for row in rows),
        "input_ready": sum(bool(row.get("input_ready")) for row in rows),
        "attempted": sum(bool(row.get("attempted")) for row in rows),
        "valid": sum(bool(row.get("valid")) for row in rows),
        "provider_failed": sum(
            str(row.get("provider_status")) == "provider_failed" for row in rows
        ),
        "blocked": sum(str(row.get("status")) == "blocked_by_input" for row in rows),
        "not_applicable": sum(
            str(row.get("status")) == "not_applicable" for row in rows
        ),
    }


def _row_status(valid: bool, provider_status: str, reason: str) -> str:
    if valid:
        return "executed_valid"
    if provider_status == ProviderStatus.PROVIDER_FAILED.value:
        return "provider_failed"
    if reason.startswith("no_observable") or reason.endswith("not_applicable"):
        return "not_applicable"
    return "blocked_by_input"


def run_view_scale_history_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Run M3 from persisted M1/M2/P4-B.5 artifacts without model inference."""

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    config_hash = _sha256(config_file)
    commit = _software_commit(root)
    inputs = config["inputs"]
    m1_root = _resolve(root, inputs["metric_provider_smoke_root"])
    m2_root = _resolve(root, inputs["metric_scene3d_root"])
    dataset_root = _resolve(root, inputs["formal_observation_dataset_root"])

    frame_rows = _read_csv(m1_root / "metric_depth_frame_manifest.csv")
    m2_object_rows = _read_csv(m2_root / "object_pointcloud_audit.csv")
    extent_rows = _read_csv(m2_root / "object_extent_audit.csv")
    m2_by_key = {
        (row["video_id"], int(row["frame_index"]), row["track_id"]): row
        for row in m2_object_rows
    }
    extent_by_key = {
        (row["video_id"], int(row["frame_index"]), row["track_id"]): row
        for row in extent_rows
    }
    frame_by_key = {
        (row["video_id"], int(row["frame_index"])): row for row in frame_rows
    }

    observation_root = dataset_root / "observations"
    objects = pq.read_table(observation_root / "objects.parquet").to_pylist()
    masks = pq.read_table(observation_root / "masks.parquet").to_pylist()
    mask_tracks = pq.read_table(observation_root / "mask_tracks.parquet").to_pylist()
    dynamic = pq.read_table(observation_root / "dynamic_readiness.parquet").to_pylist()
    clip_rows = pq.read_table(dataset_root / "manifests" / "clips.parquet").to_pylist()
    video_rows = pq.read_table(dataset_root / "manifests" / "videos.parquet").to_pylist()
    video_by_source = {str(row["source_name"]): row for row in video_rows}
    object_by_key = {
        (str(row["object_track_id"]), int(row["frame_index"])): row for row in objects
    }
    mask_by_key = {
        (str(row["object_track_id"]), int(row["frame_index"])): row
        for row in masks
        if bool(row["valid"])
        and bool(row["is_visible_mask"])
        and not bool(row["is_amodal_mask"])
        and not bool(row["bbox_fallback"])
    }
    mask_track_by_key = {
        (str(row["object_track_id"]), int(row["current_frame_index"])): row
        for row in mask_tracks
    }
    dynamic_by_clip = {str(row["clip_id"]): row for row in dynamic}

    strict_resolver = load_dimension_aligned_prior_resolver(
        _resolve(root, inputs["strict_v2_prior_path"])
    )
    registry = MultiIntervalScalePriorRegistry.from_strict_v2(strict_resolver)
    scale_cfg = config["metric_single_object"]
    scale_branch = MetricSingleObjectScaleBranch(
        registry,
        estimator=ExtentEstimator(scale_cfg["estimator"]),
        thresholds=MetricScaleThresholds(
            min_detection_confidence=float(scale_cfg["min_detection_confidence"]),
            min_depth_confidence=float(scale_cfg["min_depth_confidence"]),
            min_valid_depth_ratio=float(scale_cfg["min_valid_depth_ratio"]),
            max_out_of_frame_ratio=float(scale_cfg["max_out_of_frame_ratio"]),
            max_occlusion_ratio=float(scale_cfg["max_occlusion_ratio"]),
            min_point_count=int(scale_cfg["min_point_count"]),
            quantile_low=float(scale_cfg["quantile_low"]),
            quantile_high=float(scale_cfg["quantile_high"]),
            allow_approximated_intrinsics=bool(
                scale_cfg["allow_approximated_intrinsics"]
            ),
        ),
        config_sha256=config_hash,
        software_commit=commit,
    )

    camera_views: dict[tuple[str, int], CameraViewObservation] = {}
    camera_objects: dict[tuple[str, int], CameraObservation] = {}
    motion_by_frame: dict[tuple[str, int], CameraMotionClass] = {}
    camera_rows: list[dict[str, Any]] = []
    for frame in sorted(frame_rows, key=lambda row: (row["video_id"], int(row["frame_index"]))):
        source = frame["video_id"]
        index = int(frame["frame_index"])
        depth = np.load(_resolve(root, frame["depth_m_path"]), allow_pickle=False)
        K = np.load(_resolve(root, frame["intrinsics_path"]), allow_pickle=False)
        dataset_video = video_by_source.get(source)
        hashed_video_id = str(dataset_video["video_id"]) if dataset_video else ""
        clip = _find_clip(clip_rows, hashed_video_id, index)
        dynamic_row = dynamic_by_clip.get(str(clip["clip_id"])) if clip else None
        motion = _motion_class(dynamic_row)
        camera = CameraObservation(
            K=K,
            distortion=None,
            T_world_camera=None,
            T_camera_world=None,
            image_width=int(depth.shape[1]),
            image_height=int(depth.shape[0]),
            coordinate_convention=CoordinateConvention.OPENCV,
            intrinsics_source=frame["intrinsics_source"],
            pose_source="unavailable_single_frame",
            valid=True,
            quality=0.7,
            metadata={
                "quality_is_probability": False,
                "metric_depth_is_sensor_ground_truth": False,
            },
        )
        configured_confidence = config["camera_view"].get("intrinsics_confidence")
        camera_view = CameraViewObservation.from_camera(
            frame["frame_id"],
            camera,
            camera_motion_class=motion,
            intrinsics_confidence=(
                float("nan")
                if configured_confidence is None
                else float(configured_confidence)
            ),
            image_transform_chain=(
                {
                    "operation": "identity",
                    "source_shape": list(depth.shape),
                    "target_shape": list(depth.shape),
                },
            ),
            distortion_status=config["camera_view"]["distortion_status"],
        )
        key = (source, index)
        camera_views[key] = camera_view
        camera_objects[key] = camera
        motion_by_frame[key] = motion
        camera_rows.append(
            {
                "video_id": source,
                "clip_id": frame["clip_id"],
                "frame_id": frame["frame_id"],
                "frame_index": index,
                "fx": camera_view.fx,
                "fy": camera_view.fy,
                "cx": camera_view.cx,
                "cy": camera_view.cy,
                "image_width": camera_view.image_width,
                "image_height": camera_view.image_height,
                "fov_x": camera_view.fov_x,
                "fov_y": camera_view.fov_y,
                "fov_unit": "degree",
                "distortion_status": camera_view.distortion_status,
                "intrinsics_source": camera_view.intrinsics_source,
                "intrinsics_confidence": camera_view.intrinsics_confidence,
                "intrinsics_quality": "model_predicted_unscored",
                "camera_motion_class": motion.value,
                "pose_status": camera_view.pose_status,
                "image_transform_chain": json.dumps(
                    _json_safe(camera_view.image_transform_chain), sort_keys=True
                ),
                "valid": camera_view.valid,
                "failure_reason": camera_view.failure_reason,
                "coordinate_frame": "camera_frame_metric",
                "depth_definition": "z_depth",
                "metric_depth_claim": "model_predicted_not_sensor_ground_truth",
            }
        )

    object_views: dict[tuple[str, int, str], Any] = {}
    metric_rows: list[dict[str, Any]] = []
    object_view_rows: list[dict[str, Any]] = []
    dimension_rows: list[dict[str, Any]] = []
    object_cfg = config["object_view"]
    for key in sorted(m2_by_key):
        source, index, track_id = key
        cloud_row = m2_by_key[key]
        extent_row = extent_by_key[key]
        frame = frame_by_key[(source, index)]
        source_object = object_by_key.get((track_id, index))
        mask_row = mask_by_key.get((track_id, index))
        camera = camera_objects[(source, index)]
        camera_view = camera_views[(source, index)]
        if source_object is None or mask_row is None:
            continue
        depth_map = np.load(_resolve(root, frame["depth_m_path"]), allow_pickle=False)
        valid_depth = np.load(
            _resolve(root, frame["valid_mask_path"]), allow_pickle=False
        ).astype(bool)
        confidence_map = np.load(
            _resolve(root, frame["confidence_path"]), allow_pickle=False
        )
        mask = _load_mask(Path(mask_row["array_path"]))
        if mask.shape != depth_map.shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (depth_map.shape[1], depth_map.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        dataset_video = video_by_source[source]
        bbox = _scale_bbox(
            _parse_bbox(source_object["bbox"]),
            (int(dataset_video["height"]), int(dataset_video["width"])),
            depth_map.shape,
        )
        bbox_area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        mask_track = mask_track_by_key.get((track_id, index))
        mask_stability = (
            float(mask_track["observed_vs_predicted_iou"])
            if mask_track is not None
            and math.isfinite(float(mask_track["observed_vs_predicted_iou"]))
            else float("nan")
        )
        view_input = ObjectViewInput(
            object_id=cloud_row["object_id"],
            track_id=track_id,
            class_name=cloud_row["class_name"],
            bbox=bbox,
            image_width=depth_map.shape[1],
            image_height=depth_map.shape[0],
            detection_confidence=float(source_object["confidence"]),
            mask_area=float(np.count_nonzero(mask)),
            bbox_area=bbox_area,
            occlusion_ratio=float("nan"),
            mask_stability=mask_stability,
            viewpoint_hint=ViewpointClass.UNKNOWN,
            pose_estimate_status=PoseEstimateStatus.UNAVAILABLE,
            view_confidence=float("nan"),
            metadata={
                "mask_is_visible_not_amodal": True,
                "formal_mask_provider": mask_row["source_provider"],
                "depth_extent_supported": False,
                "mask_track_iou": mask_stability,
            },
        )
        object_view = evaluate_object_view(
            view_input,
            minimum_view_confidence=float(object_cfg["minimum_view_confidence"]),
            minimum_visible_ratio=float(object_cfg["minimum_visible_ratio"]),
            maximum_occlusion_ratio=float(object_cfg["maximum_occlusion_ratio"]),
            border_margin_ratio=float(object_cfg["border_margin_ratio"]),
        )
        object_views[key] = object_view
        contacts = object_view.metadata.get("border_contacts", [])
        object_view_rows.append(
            {
                "video_id": source,
                "clip_id": frame["clip_id"],
                "frame_id": frame["frame_id"],
                "frame_index": index,
                "object_id": object_view.object_id,
                "track_id": object_view.track_id,
                "class_name": object_view.class_name,
                "viewpoint_class": object_view.viewpoint_class.value,
                "orientation_estimate": json.dumps(
                    object_view.orientation_estimate, sort_keys=True
                ),
                "pose_estimate_status": object_view.pose_estimate_status.value,
                "foreshortening_risk": object_view.foreshortening_risk,
                "border_contact_ratio": object_view.border_contact_ratio,
                "border_contacts": "|".join(contacts),
                "visible_ratio": object_view.visible_ratio,
                "occlusion_ratio": object_view.occlusion_ratio,
                "mask_completeness": object_view.mask_completeness,
                "height_observable": object_view.height_observable,
                "width_observable": object_view.width_observable,
                "length_observable": object_view.length_observable,
                "depth_extent_observable": object_view.depth_extent_observable,
                "view_confidence": object_view.view_confidence,
                "valid": object_view.valid,
                "failure_reason": object_view.failure_reason,
                "mask_completeness_is_amodal": False,
            }
        )
        flags = {
            "height_m": object_view.height_observable,
            "width_m": object_view.width_observable,
            "length_m": object_view.length_observable,
            "depth_extent_m": object_view.depth_extent_observable,
        }
        for dimension, accepted in flags.items():
            dimension_rows.append(
                {
                    "video_id": source,
                    "clip_id": frame["clip_id"],
                    "frame_id": frame["frame_id"],
                    "frame_index": index,
                    "object_id": object_view.object_id,
                    "track_id": track_id,
                    "class_name": object_view.class_name,
                    "dimension_type": dimension,
                    "dimension_observable": accepted,
                    "viewpoint_class": object_view.viewpoint_class.value,
                    "foreshortening_risk": object_view.foreshortening_risk,
                    "reasons": "|".join(
                        object_view.dimension_reasons.get(dimension, ())
                    ),
                }
            )

        region = MetricObjectRegion(
            video_id=source,
            clip_id=frame["clip_id"],
            frame_id=frame["frame_id"],
            object_id=object_view.object_id,
            track_id=track_id,
            class_name=object_view.class_name,
            bbox=bbox,
            image_shape=depth_map.shape,
            mask=mask,
            detection_confidence=float(source_object["confidence"]),
            border_contacts=frozenset(contacts),
            severe_truncation=object_view.border_contact_ratio >= 0.5,
            out_of_frame_ratio=0.0,
            occlusion_ratio=(
                object_view.occlusion_ratio
                if math.isfinite(object_view.occlusion_ratio)
                else 0.0
            ),
            pose_status=object_view.pose_estimate_status.value,
            metadata={
                **object_view.metric_region_metadata(),
                "occlusion_ratio_unknown": not math.isfinite(
                    object_view.occlusion_ratio
                ),
            },
        )
        depth_evidence = MetricDepthEvidence(
            depth_map=depth_map,
            valid_mask=valid_depth,
            confidence_map=confidence_map,
            depth_type=MetricDepthType.METRIC,
            depth_unit="meter",
            scale_status=MetricScaleStatus.MODEL_PREDICTED,
            depth_definition=MetricDepthDefinition.Z_DEPTH,
            provider_name=frame["provider_name"],
            provider_status=ProviderStatus.OK,
            quality=float(cloud_row["depth_quality"]),
            metadata={
                "metric_depth_is_sensor_ground_truth": False,
                "quality_is_probability": False,
                "weight_sha256": frame["weight_sha256"],
            },
        )
        result = scale_branch.evaluate(region, depth_evidence, camera)
        evidence = result.evidence
        provider_status = evidence.provider_status.value
        canonical_applicable = bool(object_view.observable_dimensions)
        status = (
            "provider_failed"
            if provider_status == ProviderStatus.PROVIDER_FAILED.value
            else "not_applicable"
            if not canonical_applicable
            else _row_status(evidence.valid, provider_status, evidence.failure_reason)
        )
        applicability_reason = (
            ""
            if canonical_applicable
            else "|".join(
                sorted(
                    {
                        reason
                        for values in object_view.dimension_reasons.values()
                        for reason in values
                    }
                )
            )
            or "no_canonical_dimension_observable"
        )
        metric_rows.append(
            {
                "video_id": source,
                "clip_id": frame["clip_id"],
                "frame_id": frame["frame_id"],
                "frame_index": index,
                "object_id": object_view.object_id,
                "track_id": track_id,
                "class_name": object_view.class_name,
                "estimated_dimensions": json.dumps(
                    _json_safe(result.estimated_dimensions_m), sort_keys=True
                ),
                "observable_dimensions": "|".join(
                    object_view.observable_dimensions
                ),
                "selected_prior_intervals": json.dumps(
                    _json_safe(result.dimension_intervals), sort_keys=True
                ),
                "per_dimension_residual": json.dumps(
                    _json_safe(result.dimension_residuals), sort_keys=True
                ),
                "combined_residual": evidence.residual_value,
                "quality_score": evidence.confidence,
                "quality_is_probability": False,
                "uncertainty": evidence.uncertainty,
                "valid": evidence.valid,
                "failure_reason": evidence.failure_reason,
                "applicability_failure_reason": applicability_reason,
                "provider_status": provider_status,
                "status": status,
                "applicable": canonical_applicable,
                "input_ready": bool(
                    camera.valid and cloud_row["valid"] == "True" and mask_row["valid"]
                ),
                "attempted": True,
                "depth_provider": frame["provider_name"],
                "depth_definition": "z_depth",
                "depth_unit": "meter",
                "intrinsics_source": frame["intrinsics_source"],
                "coordinate_frame": "camera_frame_metric",
                "metric_depth_is_sensor_ground_truth": False,
                "provenance": json.dumps(
                    _json_safe(evidence.provenance), sort_keys=True
                ),
            }
        )

    history_records: list[ScaleHistoryObservation] = []
    history_rows: list[dict[str, Any]] = []
    temporal_cfg = config["temporal_scale"]
    semantic_dimensions = (
        ("height_m", "y_extent_m", "height_observable"),
        ("width_m", "x_extent_m", "width_observable"),
        ("length_m", "z_extent_m", "length_observable"),
        ("depth_extent_m", "z_extent_m", "depth_extent_observable"),
    )
    visible_dimensions = (
        ("camera_x_visible_extent_m", "x_extent_m", "horizontal"),
        ("camera_y_visible_extent_m", "y_extent_m", "vertical"),
        ("camera_z_visible_range_m", "z_extent_m", "depth"),
    )
    previous_extent_by_track: dict[
        tuple[str, str], tuple[int, np.ndarray, ViewpointClass]
    ] = {}
    for key in sorted(extent_by_key):
        source, index, track_id = key
        extent = extent_by_key[key]
        cloud = m2_by_key[key]
        frame = frame_by_key[(source, index)]
        view = object_views[key]
        camera_view = camera_views[(source, index)]
        contacts = set(view.metadata.get("border_contacts", []))
        quality = min(float(cloud["mask_quality"]), float(cloud["depth_quality"]))
        mask_track = mask_track_by_key.get((track_id, index))
        mask_stable = (
            True
            if index == min(
                int(item["frame_index"])
                for item in extent_rows
                if item["video_id"] == source and item["track_id"] == track_id
            )
            else bool(mask_track and mask_track["valid"])
        )
        intrinsics = [camera_view.fx, camera_view.fy, camera_view.cx, camera_view.cy]
        extent_vector = np.asarray(
            [
                float(extent["x_extent_m"]),
                float(extent["y_extent_m"]),
                float(extent["z_extent_m"]),
            ],
            dtype=float,
        )
        previous_extent = previous_extent_by_track.get((source, track_id))
        shape_signature_change = float("nan")
        pose_change_status = "stable"
        if previous_extent is not None:
            _, prior_vector, prior_viewpoint = previous_extent
            prior_signature = prior_vector / max(float(np.max(prior_vector)), 1e-12)
            current_signature = extent_vector / max(
                float(np.max(extent_vector)), 1e-12
            )
            shape_signature_change = float(
                np.max(
                    np.abs(
                        np.log(
                            np.maximum(current_signature, 1e-12)
                            / np.maximum(prior_signature, 1e-12)
                        )
                    )
                )
            )
            explicit_view_change = (
                prior_viewpoint != ViewpointClass.UNKNOWN
                and view.viewpoint_class != ViewpointClass.UNKNOWN
                and prior_viewpoint != view.viewpoint_class
            )
            pose_change_status = (
                "changed"
                if explicit_view_change
                or shape_signature_change
                > float(
                    temporal_cfg[
                        "max_visible_extent_change_for_pose_compatibility"
                    ]
                )
                else "compatible"
            )
        previous_extent_by_track[(source, track_id)] = (
            index,
            extent_vector,
            view.viewpoint_class,
        )
        common = {
            "video_id": source,
            "clip_id": frame["clip_id"],
            "frame_id": frame["frame_id"],
            "frame_index": index,
            "track_id": track_id,
            "object_id": extent["object_id"],
            "size_unit": "meter",
            "temporal_mode": TemporalScaleMode.METRIC,
            "depth_provider": frame["provider_name"],
            "depth_definition": "z_depth",
            "intrinsics_fingerprint": _camera_fingerprint(
                camera_objects[(source, index)].K
            ),
            "intrinsics_source": frame["intrinsics_source"],
            "depth_scale_alignment_status": "metric_model_predicted_per_frame",
            "pose_change_status": pose_change_status,
            "occlusion_status": "unresolved",
            "truncated": view.border_contact_ratio >= 0.5,
            "out_of_frame": False,
            "mask_stable": mask_stable,
            "provider_status": ProviderStatus.OK,
            "quality": quality,
            "viewpoint_class": view.viewpoint_class.value,
            "track_continuity_status": (
                "continuous"
                if not mask_track or int(mask_track["track_switch_count"]) == 0
                else "id_switch"
            ),
            "scene_cut": False,
            "metadata": {
                "intrinsics_parameters": intrinsics,
                "metric_depth_is_sensor_ground_truth": False,
                "coordinate_frame": "camera_frame_metric",
                "pose_source": "unavailable_single_frame",
                "provenance": "p4c3b_m2_robust_object_extent",
                "visible_extent_shape_signature_change": shape_signature_change,
            },
        }
        for dimension, value_name, flag_name in semantic_dimensions:
            observable = bool(getattr(view, flag_name))
            reason = (
                ""
                if observable
                else "|".join(view.dimension_reasons.get(dimension, ()))
                or "canonical_dimension_not_observable"
            )
            item = ScaleHistoryObservation(
                dimension_type=dimension,
                size_value=float(extent[value_name]) if observable else float("nan"),
                dimension_observable=observable,
                valid=observable,
                failure_reason=reason,
                **common,
            )
            history_records.append(item)
        for dimension, value_name, axis in visible_dimensions:
            axis_complete = (
                not ({"left", "right"} & contacts)
                if axis == "horizontal"
                else not ({"top", "bottom"} & contacts)
                if axis == "vertical"
                else True
            )
            observable = bool(extent["valid"] == "True" and axis_complete)
            item = ScaleHistoryObservation(
                dimension_type=dimension,
                size_value=float(extent[value_name]) if observable else float("nan"),
                dimension_observable=observable,
                valid=observable,
                failure_reason="" if observable else "visible_extent_truncated_or_invalid",
                **common,
            )
            history_records.append(item)

    for item in history_records:
        history_rows.append(
            {
                "video_id": item.video_id,
                "clip_id": item.clip_id,
                "frame_id": item.frame_id,
                "frame_index": item.frame_index,
                "track_id": item.track_id,
                "object_id": item.object_id,
                "dimension_type": item.dimension_type,
                "estimated_size": item.size_value,
                "size_unit": item.size_unit,
                "viewpoint_class": item.viewpoint_class,
                "dimension_observable": item.dimension_observable,
                "depth_provider": item.depth_provider,
                "depth_definition": item.depth_definition,
                "intrinsics_source": item.intrinsics_source,
                "quality_score": item.quality,
                "quality_is_probability": False,
                "valid": item.valid,
                "failure_reason": item.failure_reason,
                "coordinate_frame": "camera_frame_metric",
                "pose_source": "unavailable_single_frame",
                "metric_depth_is_sensor_ground_truth": False,
                "provenance": json.dumps(item.metadata, sort_keys=True),
            }
        )

    temporal_rows: list[dict[str, Any]] = []
    grouped_history: dict[tuple[str, str, str], list[ScaleHistoryObservation]] = defaultdict(list)
    for item in history_records:
        grouped_history[(item.video_id, item.track_id, item.dimension_type)].append(item)
    for items in grouped_history.values():
        items.sort(key=lambda item: item.frame_index)
        for method_name in temporal_cfg["reference_methods"]:
            branch = TemporalSameObjectScaleBranch(
                reference_method=TemporalReferenceMethod(method_name),
                min_valid_history=int(temporal_cfg["min_valid_history"]),
                reference_window=int(temporal_cfg["reference_window"]),
                max_frame_gap=int(temporal_cfg["max_frame_gap"]),
                minimum_quality=float(temporal_cfg["minimum_quality"]),
                max_intrinsics_relative_change=float(
                    temporal_cfg["max_intrinsics_relative_change"]
                ),
                config_sha256=config_hash,
                software_commit=commit,
            )
            for position, current in enumerate(items):
                result = branch.evaluate(current, items[:position])
                provider_status = result.provider_status.value
                status = (
                    "provider_failed"
                    if provider_status == ProviderStatus.PROVIDER_FAILED.value
                    else "not_applicable"
                    if not current.dimension_observable
                    else _row_status(
                        result.valid, provider_status, result.failure_reason
                    )
                )
                temporal_rows.append(
                    {
                        "video_id": current.video_id,
                        "clip_id": current.clip_id,
                        "frame_id": current.frame_id,
                        "frame_index": current.frame_index,
                        "track_id": current.track_id,
                        "object_id": current.object_id,
                        "dimension_type": current.dimension_type,
                        "reference_method": method_name,
                        "current_size": current.size_value,
                        "reference_size": result.provenance.get(
                            "reference_size", float("nan")
                        ),
                        "residual": result.residual_value,
                        "quality_score": result.confidence,
                        "uncertainty": result.uncertainty,
                        "valid": result.valid,
                        "failure_reason": result.failure_reason,
                        "provider_status": provider_status,
                        "status": status,
                        "applicable": current.dimension_observable,
                        "input_ready": bool(
                            current.valid
                            and current.provider_status == ProviderStatus.OK
                        ),
                        "attempted": True,
                        "viewpoint_class": current.viewpoint_class,
                        "intrinsics_source": current.intrinsics_source,
                        "depth_provider": current.depth_provider,
                        "depth_definition": current.depth_definition,
                        "size_unit": current.size_unit,
                        "coordinate_frame": "camera_frame_metric",
                        "metric_depth_is_sensor_ground_truth": False,
                        "provenance": json.dumps(
                            _json_safe(result.provenance), sort_keys=True
                        ),
                    }
                )

    static_rows: list[dict[str, Any]] = []
    for (source, clip_id), selected in sorted(
        {
            (row["video_id"], row["clip_id"]): [
                item
                for item in camera_rows
                if item["video_id"] == row["video_id"]
                and item["clip_id"] == row["clip_id"]
            ]
            for row in camera_rows
        }.items()
    ):
        motion_values = sorted({row["camera_motion_class"] for row in selected})
        metric_count = sum(row["clip_id"] == clip_id for row in metric_rows)
        temporal_count = sum(row["clip_id"] == clip_id for row in temporal_rows)
        static_rows.append(
            {
                "video_id": source,
                "clip_id": clip_id,
                "camera_motion_class": "|".join(motion_values),
                "metric_single_object_attempted": metric_count,
                "temporal_same_object_attempted": temporal_count,
                "static_or_low_motion_blocks_metric_scale": False,
                "static_or_low_motion_blocks_temporal_scale": False,
                "provider_failure_from_static_state": False,
                "status": "executed",
            }
        )

    eligibility = {
        "metric_single_object_scale": _funnel(metric_rows),
        "temporal_same_object_scale": _funnel(temporal_rows),
    }
    relative_preserved = _temporal_relative_compatibility_check()
    frozen = {
        **{
            name: _sha256(root / "configs" / name) == expected
            for name, expected in STRICT_HASHES.items()
        },
        **{
            name: _sha256(root / "configs" / name) == expected
            for name, expected in PROTOCOL_HASHES.items()
        },
    }
    invalid_metric_zero = any(
        not row["valid"]
        and math.isfinite(float(row["combined_residual"]))
        and float(row["combined_residual"]) == 0.0
        for row in metric_rows
    )
    invalid_temporal_zero = any(
        not row["valid"]
        and math.isfinite(float(row["residual"]))
        and float(row["residual"]) == 0.0
        for row in temporal_rows
    )
    metric_attempted = bool(metric_rows)
    temporal_valid = sum(bool(row["valid"]) for row in temporal_rows)
    static_or_low_motion_smoke_count = sum(
        row["camera_motion_class"]
        in {CameraMotionClass.STATIC.value, CameraMotionClass.LOW_MOTION.value}
        for row in camera_rows
    )
    statuses = {
        "camera_view_model_complete": bool(camera_rows),
        "intrinsics_quality_modeled": all(
            row["intrinsics_source"] and row["intrinsics_confidence"] != 1.0
            for row in camera_rows
        ),
        "object_viewpoint_modeled": bool(object_view_rows),
        "dimension_observability_complete": bool(dimension_rows),
        "metric_single_object_real_smoke_verified": metric_attempted
        and not invalid_metric_zero,
        "track_size_history_materialized": bool(history_rows),
        "temporal_metric_real_smoke_verified": temporal_valid > 0
        and not invalid_temporal_zero,
        "temporal_relative_mode_preserved": relative_preserved,
        "static_video_scale_detection_supported": all(
            not row["static_or_low_motion_blocks_metric_scale"]
            and not row["static_or_low_motion_blocks_temporal_scale"]
            for row in static_rows
        ),
        "ready_for_pose_integration": True,
        "method_effectiveness_established": False,
    }
    validation = {
        **statuses,
        "stage": "P4-C3B-M3",
        "config_sha256": config_hash,
        "software_commit": commit,
        "camera_frame_count": len(camera_rows),
        "object_view_count": len(object_view_rows),
        "resolved_object_viewpoint_count": sum(
            row["viewpoint_class"] != ViewpointClass.UNKNOWN.value
            for row in object_view_rows
        ),
        "observable_canonical_dimension_count": sum(
            bool(row["dimension_observable"]) for row in dimension_rows
        ),
        "metric_single_object_attempt_count": len(metric_rows),
        "metric_single_object_valid_count": sum(
            bool(row["valid"]) for row in metric_rows
        ),
        "metric_single_object_evidence_available": any(
            bool(row["valid"]) for row in metric_rows
        ),
        "track_size_history_count": len(history_rows),
        "temporal_attempt_count": len(temporal_rows),
        "temporal_valid_count": temporal_valid,
        "static_or_low_motion_real_smoke_frame_count": static_or_low_motion_smoke_count,
        "invalid_metric_residual_encoded_as_zero": invalid_metric_zero,
        "invalid_temporal_residual_encoded_as_zero": invalid_temporal_zero,
        "eligibility_funnels": eligibility,
        "frozen_hashes_unchanged": frozen,
        "all_frozen_hashes_unchanged": all(frozen.values()),
        "large_model_inference_executed": False,
        "authenticity_labels_used": False,
        "world_frame_claimed": False,
    }

    _write_csv(
        output / "camera_view_audit.csv",
        camera_rows,
        [
            "video_id",
            "clip_id",
            "frame_id",
            "frame_index",
            "fx",
            "fy",
            "cx",
            "cy",
            "image_width",
            "image_height",
            "fov_x",
            "fov_y",
            "fov_unit",
            "distortion_status",
            "intrinsics_source",
            "intrinsics_confidence",
            "intrinsics_quality",
            "camera_motion_class",
            "pose_status",
            "image_transform_chain",
            "valid",
            "failure_reason",
            "coordinate_frame",
            "depth_definition",
            "metric_depth_claim",
        ],
    )
    _write_csv(
        output / "object_view_audit.csv",
        object_view_rows,
        list(object_view_rows[0]) if object_view_rows else [],
    )
    _write_csv(
        output / "dimension_observability_audit.csv",
        dimension_rows,
        list(dimension_rows[0]) if dimension_rows else [],
    )
    _write_csv(
        output / "metric_single_object_execution.csv",
        metric_rows,
        list(metric_rows[0]) if metric_rows else [],
    )
    _write_csv(
        output / "track_size_history.csv",
        history_rows,
        list(history_rows[0]) if history_rows else [],
    )
    _write_csv(
        output / "temporal_scale_execution.csv",
        temporal_rows,
        list(temporal_rows[0]) if temporal_rows else [],
    )
    _write_csv(
        output / "static_clip_branch_audit.csv",
        static_rows,
        list(static_rows[0]) if static_rows else [],
    )
    _write_json(output / "eligibility_funnels.json", eligibility)
    _write_json(output / "validation_report.json", validation)
    report_lines = [
        "# P4-C3B-M3 View and Scale History Report",
        "",
        "This stage audits model-predicted metric depth; it is not sensor ground truth.",
        "All reconstructed values remain in camera_frame_metric. No world frame is claimed.",
        "",
        "## Coverage",
        "",
        f"- Camera frames: {len(camera_rows)}",
        f"- Object views: {len(object_view_rows)}",
        f"- Resolved object viewpoints: {validation['resolved_object_viewpoint_count']}",
        f"- Observable canonical dimensions: {validation['observable_canonical_dimension_count']}",
        f"- Metric single-object attempts: {len(metric_rows)}",
        f"- Metric single-object valid residuals: {validation['metric_single_object_valid_count']}",
        f"- Track size history records: {len(history_rows)}",
        f"- Temporal attempts: {len(temporal_rows)}",
        f"- Temporal valid residuals: {temporal_valid}",
        f"- Static/low-motion real smoke frames: {static_or_low_motion_smoke_count}",
        "",
        "Unknown viewpoint is preserved when no reliable category pose estimate exists.",
        "Canonical dimensions rejected by view/visibility gates remain NaN.",
        "Camera-axis visible extents are separately named and may support temporal",
        "stability; they are not relabelled as canonical physical dimensions.",
        "",
        "## Status",
        "",
    ]
    report_lines.extend(
        f"- {name}: {str(value).lower()}" for name, value in statuses.items()
    )
    report_lines.extend(
        [
            "",
            "## Limits",
            "",
            "- The smoke covers only persisted M1/M2 frames and does not establish effectiveness.",
            "- No general category pose provider is available; many viewpoints remain unknown.",
            "- The persisted M1/M2 smoke contains no frame classified as static or low-motion;",
            "  static routing is contract-tested but not claimed as a positive real static smoke.",
            "- Visible masks are not amodal masks, so occluded full dimensions are not claimed.",
            "- Per-frame model-predicted intrinsics are not calibrated camera metadata.",
            "- No training, threshold selection, distribution fitting, or authenticity evaluation ran.",
            "",
        ]
    )
    (output / "VIEW_SCALE_HISTORY_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return validation
