"""Offline P4-C3B-M2 smoke using saved M1 metric depth and formal masks."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
import pyarrow.parquet as pq
import yaml

from ..occlusion.mask_observation import InstanceMaskObservation
from ..shared_3d_observation import CoordinateFrame
from .boundary import reconstruct_boundary_points
from .contracts import MetricPointType, MetricSurfacePoint
from .image_geometry import ImageCoordinateTransform, align_binary_mask
from .reconstruction import (
    build_object_surface_pointcloud,
    build_scene_surface_points,
    canonicalize_z_depth,
    to_shared_object_observation,
)
from .structure_graph import build_single_frame_structure_graph
from .structure_points import select_geometric_track_points


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _resolve(project_root: Path, value: str) -> Path:
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


def _load_formal_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "visible_mask" not in archive.files:
            raise ValueError("Formal mask archive lacks visible_mask.")
        return np.asarray(archive["visible_mask"], dtype=bool)


def _point_row(point: MetricSurfacePoint) -> dict[str, Any]:
    return {
        "point_id": point.point_id,
        "point_type": point.point_type.value,
        "frame_id": point.frame_id,
        "object_id": point.object_id or "",
        "track_id": point.track_id or "",
        "u": point.u,
        "v": point.v,
        "x_m": point.x_m,
        "y_m": point.y_m,
        "z_m": point.z_m,
        "depth_confidence": point.depth_confidence,
        "confidence": point.confidence,
        "uncertainty": point.uncertainty,
        "uncertainty_definition": point.uncertainty_definition,
        "visibility": point.visibility.value,
        "coordinate_frame": point.coordinate_frame.value,
        "depth_unit": point.depth_unit,
        "depth_definition": point.depth_definition,
        "intrinsics_source": point.intrinsics_source,
        "pose_source": point.pose_source,
        "provider_name": point.provider_name,
        "valid": point.valid,
        "failure_reason": point.failure_reason,
        "provenance": json.dumps(_json_safe(point.provenance), sort_keys=True),
    }


def _save_points(path: Path, points: Iterable[MetricSurfacePoint]) -> None:
    selected = list(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        point_id=np.asarray([point.point_id for point in selected]),
        point_type=np.asarray([point.point_type.value for point in selected]),
        frame_id=np.asarray([point.frame_id for point in selected]),
        object_id=np.asarray([point.object_id or "" for point in selected]),
        track_id=np.asarray([point.track_id or "" for point in selected]),
        uv=np.asarray([[point.u, point.v] for point in selected], dtype=np.float32),
        xyz_m=np.asarray(
            [[point.x_m, point.y_m, point.z_m] for point in selected], dtype=np.float32
        ),
        confidence=np.asarray([point.confidence for point in selected], dtype=np.float32),
        depth_confidence=np.asarray(
            [point.depth_confidence for point in selected], dtype=np.float32
        ),
        uncertainty=np.asarray(
            [point.uncertainty for point in selected], dtype=np.float32
        ),
        valid=np.asarray([point.valid for point in selected], dtype=bool),
        coordinate_frame=np.asarray(
            [CoordinateFrame.CAMERA_FRAME_METRIC.value for _ in selected]
        ),
        depth_unit=np.asarray([point.depth_unit for point in selected]),
        depth_definition=np.asarray([point.depth_definition for point in selected]),
        intrinsics_source=np.asarray([point.intrinsics_source for point in selected]),
        pose_source=np.asarray([point.pose_source for point in selected]),
        source_depth_provider=np.asarray([point.provider_name for point in selected]),
    )


def run_metric_scene3d_smoke(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build M2 single-frame metric 2.5D artifacts without model inference."""

    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    m1_root = _resolve(root, inputs["metric_provider_smoke_root"])
    dataset_root = _resolve(root, inputs["formal_observation_dataset_root"])
    selected_sources = set(inputs.get("source_names", []))
    selected_frames = {
        source: {int(value) for value in values}
        for source, values in inputs.get("frame_indices", {}).items()
    }
    m1_manifest = _read_csv(m1_root / "metric_depth_frame_manifest.csv")
    object_audit = _read_csv(m1_root / "object_region_depth_audit.csv")
    mask_rows = pq.read_table(
        dataset_root / "observations" / "masks.parquet"
    ).to_pylist()
    object_rows = pq.read_table(
        dataset_root / "observations" / "objects.parquet"
    ).to_pylist()
    mask_by_key = {
        (str(row["object_track_id"]), int(row["frame_index"])): row
        for row in mask_rows
        if bool(row["valid"])
        and bool(row["is_visible_mask"])
        and not bool(row["is_amodal_mask"])
        and not bool(row["bbox_fallback"])
    }
    object_by_key = {
        (str(row["object_track_id"]), int(row["frame_index"])): row
        for row in object_rows
    }
    objects_by_frame: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in object_audit:
        objects_by_frame[(row["video_id"], int(row["frame_index"]))].append(row)

    scene_rows: list[dict[str, Any]] = []
    object_cloud_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    internal_rows: list[dict[str, Any]] = []
    extent_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    valid_track_frames: dict[str, set[int]] = defaultdict(set)
    coordinate_counts: Counter[str] = Counter()

    reconstruction = config["reconstruction"]
    frame_candidates = [
        row
        for row in m1_manifest
        if row["provider_status"] == "executed_valid"
        and (not selected_sources or row["video_id"] in selected_sources)
        and (
            row["video_id"] not in selected_frames
            or int(row["frame_index"]) in selected_frames[row["video_id"]]
        )
    ]
    frame_candidates.sort(key=lambda row: (row["video_id"], int(row["frame_index"])))
    for frame_row in frame_candidates:
        video_id = frame_row["video_id"]
        frame_index = int(frame_row["frame_index"])
        frame_id = frame_row["frame_id"]
        image_path = m1_root / "frames" / video_id / f"frame_{frame_index:06d}.jpg"
        if not image_path.exists():
            failures["missing_saved_source_frame"] += 1
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures["saved_source_frame_decode_failed"] += 1
            continue
        depth = np.load(_resolve(root, frame_row["depth_m_path"]), allow_pickle=False)
        valid_mask = np.load(
            _resolve(root, frame_row["valid_mask_path"]), allow_pickle=False
        ).astype(bool)
        confidence = np.load(
            _resolve(root, frame_row["confidence_path"]), allow_pickle=False
        )
        uncertainty = np.load(
            _resolve(root, frame_row["uncertainty_path"]), allow_pickle=False
        )
        K_source = np.load(
            _resolve(root, frame_row["intrinsics_path"]), allow_pickle=False
        )
        source_height, source_width = image.shape[:2]
        depth_transform = ImageCoordinateTransform(
            source_width=source_width,
            source_height=source_height,
            target_width=depth.shape[1],
            target_height=depth.shape[0],
            scale_x=depth.shape[1] / source_width,
            scale_y=depth.shape[0] / source_height,
            operation="identity" if depth.shape == image.shape[:2] else "full_frame_resize",
            distortion_status="model_input_undistortion_not_reported",
        )
        K_depth = depth_transform.transform_intrinsics(K_source)
        z_depth = canonicalize_z_depth(
            depth, K_depth, depth_definition=frame_row["depth_definition"]
        )
        scene_points = build_scene_surface_points(
            frame_id=frame_id,
            depth_map=z_depth,
            valid_mask=valid_mask,
            K=K_depth,
            confidence_map=confidence,
            uncertainty_map=uncertainty,
            provider_name=frame_row["provider_name"],
            intrinsics_source=frame_row["intrinsics_source"],
            stride=int(reconstruction["scene_stride"]),
        )
        scene_path = output / "arrays" / video_id / frame_id / "scene_surface_points.npz"
        _save_points(scene_path, scene_points)
        coordinate_counts.update(point.coordinate_frame.value for point in scene_points)
        scene_rows.append(
            {
                "video_id": video_id,
                "clip_id": frame_row["clip_id"],
                "frame_id": frame_id,
                "frame_index": frame_index,
                "scene_surface_candidate_count": int(np.count_nonzero(valid_mask)),
                "scene_surface_point_count": len(scene_points),
                "scene_surface_points_path": str(scene_path.relative_to(root)),
                "coordinate_frame": CoordinateFrame.CAMERA_FRAME_METRIC.value,
                "depth_unit": "meter",
                "depth_definition": "z_depth",
                "intrinsics_source": frame_row["intrinsics_source"],
                "pose_source": "unavailable_single_frame",
                "provider_name": frame_row["provider_name"],
                "metric_scale_status": frame_row["metric_scale_status"],
                "structure_description": "single_frame_camera_metric_visible_surface_2p5d",
                "complete_scene_claim": False,
                "valid": bool(scene_points),
                "failure_reason": "" if scene_points else "no_valid_scene_surface_points",
            }
        )
        for object_row in sorted(
            objects_by_frame.get((video_id, frame_index), []),
            key=lambda row: row["object_track_id"],
        ):
            track_id = object_row["object_track_id"]
            key = (track_id, frame_index)
            mask_row = mask_by_key.get(key)
            source_object = object_by_key.get(key, {})
            object_id = str(
                source_object.get("source_object_id")
                or mask_row.get("segmentation_instance_id") if mask_row else track_id
            )
            if mask_row is None:
                failures["formal_mask_missing_for_object"] += 1
                continue
            mask_native = _load_formal_mask(Path(mask_row["array_path"]))
            mask_transform = ImageCoordinateTransform(
                source_width=source_width,
                source_height=source_height,
                target_width=mask_native.shape[1],
                target_height=mask_native.shape[0],
                scale_x=mask_native.shape[1] / source_width,
                scale_y=mask_native.shape[0] / source_height,
                operation=(
                    "identity"
                    if mask_native.shape == image.shape[:2]
                    else "full_frame_resize"
                ),
                distortion_status="segmentation_output_resized_to_original",
            )
            aligned_mask = align_binary_mask(
                mask_native,
                mask_transform=mask_transform,
                target_transform=depth_transform,
            )
            aligned_image = (
                image
                if image.shape[:2] == z_depth.shape
                else cv2.resize(
                    image,
                    (z_depth.shape[1], z_depth.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            )
            alignment_rows.append(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "track_id": track_id,
                    "source_image_shape": [source_height, source_width],
                    "mask_shape_before": list(mask_native.shape),
                    "depth_shape": list(z_depth.shape),
                    "mask_shape_after": list(aligned_mask.shape),
                    "mask_transform": asdict(mask_transform),
                    "depth_transform": asdict(depth_transform),
                    "aligned_mask_pixels": int(np.count_nonzero(aligned_mask)),
                    "alignment_status": "executed_valid",
                    "distortion_status": depth_transform.distortion_status,
                }
            )
            cloud = build_object_surface_pointcloud(
                frame_id=frame_id,
                object_id=object_id,
                track_id=track_id,
                class_name=object_row["class_name"],
                mask=aligned_mask,
                mask_quality=float(mask_row["confidence"]),
                depth_map=z_depth,
                valid_mask=valid_mask,
                K=K_depth,
                confidence_map=confidence,
                uncertainty_map=uncertainty,
                provider_name=frame_row["provider_name"],
                intrinsics_source=frame_row["intrinsics_source"],
                quantile_low=float(reconstruction["extent_quantile_low"]),
                quantile_high=float(reconstruction["extent_quantile_high"]),
                max_recorded_points=int(reconstruction["max_recorded_object_points"]),
            )
            object_points_path = (
                output / "arrays" / video_id / frame_id / f"{track_id}_surface_points.npz"
            )
            _save_points(object_points_path, cloud.points)
            shared = to_shared_object_observation(
                cloud, video_id=video_id, frame_index=frame_index
            )
            object_cloud_rows.append(
                {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "object_id": object_id,
                    "track_id": track_id,
                    "class_name": object_row["class_name"],
                    "point_count": cloud.point_count,
                    "recorded_point_count": len(cloud.points),
                    "valid_point_ratio": cloud.valid_point_ratio,
                    "point_uncertainty": cloud.point_uncertainty,
                    "point_uncertainty_definition": (
                        "provider_native_uncertainty_not_meter_calibrated"
                    ),
                    "mask_quality": cloud.mask_quality,
                    "depth_quality": cloud.depth_quality,
                    "object_surface_points_path": str(object_points_path.relative_to(root)),
                    "coordinate_frame": CoordinateFrame.CAMERA_FRAME_METRIC.value,
                    "depth_unit": "meter",
                    "depth_definition": "z_depth",
                    "provider_name": frame_row["provider_name"],
                    "shared_object_valid": shared.valid,
                    "valid": cloud.valid,
                    "failure_reason": cloud.failure_reason,
                }
            )
            extent_rows.append(
                {
                    "video_id": video_id,
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "object_id": object_id,
                    "track_id": track_id,
                    "class_name": object_row["class_name"],
                    "x_extent_m": cloud.x_extent_m,
                    "y_extent_m": cloud.y_extent_m,
                    "z_extent_m": cloud.z_extent_m,
                    "robust_centroid_m": json.dumps(_json_safe(cloud.robust_centroid_m)),
                    "robust_covariance": json.dumps(_json_safe(cloud.robust_covariance)),
                    "quantile_low": cloud.quantile_low,
                    "quantile_high": cloud.quantile_high,
                    "extent_method": "per_axis_robust_quantiles",
                    "semantic_characteristic_scale": False,
                    "valid": cloud.valid,
                    "failure_reason": cloud.failure_reason,
                }
            )
            mask_observation = InstanceMaskObservation.from_visible_mask(
                video_id=video_id,
                frame_index=frame_index,
                object_track_id=track_id,
                semantic_label=object_row["class_name"],
                mask=aligned_mask,
                confidence=float(mask_row["confidence"]),
                source_provider=str(mask_row["source_provider"]),
                metadata={
                    "formal_mask_evidence": True,
                    "legacy_bbox_fallback": False,
                    "is_amodal_mask": False,
                    "weight_sha256": mask_row["weight_sha256"],
                },
            )
            boundaries = reconstruct_boundary_points(
                frame_id=frame_id,
                object_id=object_id,
                track_id=track_id,
                mask=aligned_mask,
                mask_quality=mask_observation.confidence,
                depth_map=z_depth,
                valid_mask=valid_mask,
                K=K_depth,
                confidence_map=confidence,
                uncertainty_map=uncertainty,
                provider_name=frame_row["provider_name"],
                intrinsics_source=frame_row["intrinsics_source"],
                sample_count=int(reconstruction["boundary_sample_count"]),
                min_spacing_px=float(reconstruction["boundary_min_spacing_px"]),
                side_radius_px=int(reconstruction["boundary_side_radius_px"]),
            )
            for boundary in boundaries:
                boundary_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        **_point_row(boundary.point),
                        "foreground_depth_m": boundary.foreground_depth_m,
                        "background_depth_m": boundary.background_depth_m,
                        "boundary_depth_jump_m": boundary.boundary_depth_jump_m,
                        "boundary_order": boundary.boundary_order,
                    }
                )
            internal = select_geometric_track_points(
                image=aligned_image,
                mask_observation=mask_observation,
                depth_map=z_depth,
                valid_mask=valid_mask,
                K=K_depth,
                confidence_map=confidence,
                uncertainty_map=uncertainty,
                frame_id=frame_id,
                object_id=object_id,
                provider_name=frame_row["provider_name"],
                intrinsics_source=frame_row["intrinsics_source"],
                max_points=int(reconstruction["internal_max_points"]),
                erosion_pixels=int(reconstruction["internal_erosion_pixels"]),
                min_distance_px=float(reconstruction["internal_min_distance_px"]),
                max_local_depth_mad_ratio=float(
                    reconstruction["internal_max_depth_mad_ratio"]
                ),
            )
            if internal:
                valid_track_frames[track_id].add(frame_index)
                internal_rows.extend(
                    {"video_id": video_id, "frame_index": frame_index, **_point_row(point)}
                    for point in internal
                )
            else:
                failures["no_geometric_track_points"] += 1
                internal_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "frame_id": frame_id,
                        "object_id": object_id,
                        "track_id": track_id,
                        "point_type": MetricPointType.GEOMETRIC_TRACK_POINT.value,
                        "valid": False,
                        "failure_reason": "no_texture_depth_stable_internal_points",
                        "coordinate_frame": CoordinateFrame.CAMERA_FRAME_METRIC.value,
                    }
                )
            graph = build_single_frame_structure_graph(
                frame_id=frame_id,
                object_id=object_id,
                track_id=track_id,
                boundary_points=[item.point for item in boundaries],
                internal_points=internal,
                knn_k=int(reconstruction["graph_knn_k"]),
                radius_m=(
                    None
                    if reconstruction.get("graph_radius_m") is None
                    else float(reconstruction["graph_radius_m"])
                ),
            )
            if graph.edges:
                for edge in graph.edges:
                    graph_rows.append(
                        {
                            "video_id": video_id,
                            "frame_index": frame_index,
                            "graph_id": graph.graph_id,
                            "object_id": object_id,
                            "track_id": track_id,
                            "node_count": len(graph.nodes),
                            "edge_id": edge.edge_id,
                            "source_point_id": edge.source_point_id,
                            "target_point_id": edge.target_point_id,
                            "edge_length_m": edge.edge_length_m,
                            "relative_depth_m": edge.relative_depth_m,
                            "direction_vector": json.dumps(edge.direction_vector),
                            "edge_type": edge.edge_type,
                            "confidence": edge.confidence,
                            "coordinate_frame": graph.coordinate_frame.value,
                            "valid": edge.valid,
                            "failure_reason": edge.failure_reason,
                            "temporal_d3_residual_computed": False,
                        }
                    )
            else:
                graph_rows.append(
                    {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "graph_id": graph.graph_id,
                        "object_id": object_id,
                        "track_id": track_id,
                        "node_count": len(graph.nodes),
                        "coordinate_frame": graph.coordinate_frame.value,
                        "valid": False,
                        "failure_reason": graph.failure_reason,
                        "temporal_d3_residual_computed": False,
                    }
                )
            coordinate_counts.update(
                point.point.coordinate_frame.value for point in boundaries
            )
            coordinate_counts.update(point.coordinate_frame.value for point in internal)
            coordinate_counts.update(point.coordinate_frame.value for point in cloud.points)

    scene_fields = [
        "video_id", "clip_id", "frame_id", "frame_index",
        "scene_surface_candidate_count", "scene_surface_point_count",
        "scene_surface_points_path", "coordinate_frame", "depth_unit",
        "depth_definition", "intrinsics_source", "pose_source", "provider_name",
        "metric_scale_status", "structure_description", "complete_scene_claim",
        "valid", "failure_reason",
    ]
    object_fields = [
        "video_id", "frame_id", "frame_index", "object_id", "track_id",
        "class_name", "point_count", "recorded_point_count", "valid_point_ratio",
        "point_uncertainty", "point_uncertainty_definition", "mask_quality", "depth_quality",
        "object_surface_points_path", "coordinate_frame", "depth_unit",
        "depth_definition", "provider_name", "shared_object_valid", "valid",
        "failure_reason",
    ]
    point_fields = [
        "video_id", "frame_index", "point_id", "point_type", "frame_id",
        "object_id", "track_id", "u", "v", "x_m", "y_m", "z_m",
        "depth_confidence", "confidence", "uncertainty", "uncertainty_definition",
        "visibility", "coordinate_frame", "depth_unit", "depth_definition",
        "intrinsics_source", "pose_source", "provider_name", "valid",
        "failure_reason", "provenance",
    ]
    _write_csv(output / "scene3d_frame_manifest.csv", scene_rows, scene_fields)
    _write_csv(output / "object_pointcloud_audit.csv", object_cloud_rows, object_fields)
    _write_csv(
        output / "boundary_point_audit.csv",
        boundary_rows,
        point_fields
        + [
            "foreground_depth_m", "background_depth_m",
            "boundary_depth_jump_m", "boundary_order",
        ],
    )
    _write_csv(output / "internal_structure_point_audit.csv", internal_rows, point_fields)
    _write_csv(
        output / "object_extent_audit.csv",
        extent_rows,
        [
            "video_id", "frame_id", "frame_index", "object_id", "track_id",
            "class_name", "x_extent_m", "y_extent_m", "z_extent_m",
            "robust_centroid_m", "robust_covariance", "quantile_low",
            "quantile_high", "extent_method", "semantic_characteristic_scale",
            "valid", "failure_reason",
        ],
    )
    _write_csv(
        output / "single_frame_structure_graph_audit.csv",
        graph_rows,
        [
            "video_id", "frame_index", "graph_id", "object_id", "track_id",
            "node_count", "edge_id", "source_point_id", "target_point_id",
            "edge_length_m", "relative_depth_m", "direction_vector", "edge_type",
            "confidence", "coordinate_frame", "valid", "failure_reason",
            "temporal_d3_residual_computed",
        ],
    )
    _write_json(
        output / "coordinate_frame_audit.json",
        {
            "coordinate_convention": "opencv_x_right_y_down_z_forward",
            "pixel_center_convention": "integer_coordinates_are_pixel_centers_no_half_offset",
            "allowed_real_output_frame": CoordinateFrame.CAMERA_FRAME_METRIC.value,
            "coordinate_frame_counts": dict(sorted(coordinate_counts.items())),
            "world_frame_point_count": coordinate_counts.get(
                CoordinateFrame.WORLD_FRAME.value, 0
            ),
            "depth_unit": "meter",
            "depth_definition": "z_depth",
            "metric_scale_status": "model_predicted",
            "sensor_ground_truth": False,
            "distortion_status": "model_input_undistortion_not_reported",
        },
    )
    _write_json(
        output / "depth_mask_alignment_audit.json",
        {
            "record_count": len(alignment_rows),
            "all_executed_valid": all(
                row["alignment_status"] == "executed_valid" for row in alignment_rows
            ),
            "records": alignment_rows,
        },
    )
    blocked = {
        "world_frame_reconstruction": {
            "status": "blocked_by_input",
            "reason": "camera_pose_and_cross_frame_fusion_not_in_m2_scope",
        },
        "semantic_keypoints": {
            "status": "not_applicable",
            "reason": "no_category_specific_keypoint_provider_executed",
        },
        "temporal_d3_residual": {
            "status": "interface_only",
            "reason": "m2_builds_single_frame_graph_only",
        },
        "complete_scene_geometry": {
            "status": "not_applicable",
            "reason": "monocular_single_frame_depth_observes_visible_surface_only",
        },
    }
    _write_json(output / "blocked_features.json", blocked)
    valid_scene = sum(bool(row["valid"]) for row in scene_rows)
    valid_clouds = sum(bool(row["valid"]) for row in object_cloud_rows)
    valid_boundaries = sum(bool(row.get("valid")) for row in boundary_rows)
    valid_internal = sum(bool(row.get("valid")) for row in internal_rows)
    valid_graphs = len({row["graph_id"] for row in graph_rows if row.get("valid")})
    temporal_ready_tracks = sum(len(frames) >= 2 for frames in valid_track_frames.values())
    status = {
        "camera_frame_metric_scene_complete": bool(scene_rows) and valid_scene == len(scene_rows),
        "scene_surface_points_verified": valid_scene > 0,
        "object_surface_points_verified": valid_clouds > 0,
        "metric_boundary_points_verified": valid_boundaries > 0,
        "geometric_track_points_verified": valid_internal > 0,
        "semantic_keypoints_available": False,
        "single_frame_structure_graph_complete": valid_graphs > 0,
        "world_frame_reconstruction_complete": False,
        "ready_for_view_observability": valid_clouds > 0 and valid_boundaries > 0,
        "ready_for_temporal_size_materialization": temporal_ready_tracks > 0,
        "method_effectiveness_established": False,
    }
    validation = {
        "schema_version": "semantic3d_p4c3b_metric_scene3d_validation_v1",
        "stage": "P4-C3B-M2",
        "config_path": str(config_file.relative_to(root)),
        "config_sha256": _sha256(config_file),
        "software_commit": _software_commit(root),
        "input_metric_manifest_sha256": _sha256(
            m1_root / "metric_depth_frame_manifest.csv"
        ),
        "counts": {
            "frames_attempted": len(frame_candidates),
            "frames_valid": valid_scene,
            "objects_attempted": len(object_cloud_rows),
            "object_pointclouds_valid": valid_clouds,
            "boundary_points_total": len(boundary_rows),
            "boundary_points_valid": valid_boundaries,
            "geometric_track_points_valid": valid_internal,
            "single_frame_graphs_valid": valid_graphs,
            "single_frame_graph_edges_valid": sum(
                bool(row.get("valid")) for row in graph_rows
            ),
            "temporal_ready_tracks": temporal_ready_tracks,
        },
        "failure_reasons": dict(sorted(failures.items())),
        **status,
    }
    _write_json(output / "validation_report.json", validation)
    report = [
        "# P4-C3B-M2 Metric Scene3D Report",
        "",
        "本阶段构建的是“单帧相机坐标系下的米制可见表面 2.5D 结构”，"
        "不代表完整三维场景，也不代表传感器真值。",
        "",
        "## Scope",
        "",
        f"- Frames: {valid_scene}/{len(frame_candidates)} valid",
        f"- Object metric point clouds: {valid_clouds}/{len(object_cloud_rows)} valid",
        f"- Valid metric boundary points: {valid_boundaries}",
        f"- Valid geometric track-point candidates: {valid_internal}",
        f"- Valid single-frame structure graphs: {valid_graphs}",
        "- Coordinate frame: `camera_frame_metric` only",
        "- Depth: model-predicted metric `z_depth` in meters, not sensor ground truth",
        "- Intrinsics: model-predicted; not calibrated",
        "- World-frame reconstruction: not performed",
        "- Temporal D3 residual: not performed",
        "",
        "## Status",
        "",
    ]
    report.extend(f"- `{key}`: `{str(value).lower()}`" for key, value in status.items())
    report.extend(
        [
            "",
            "## Audit Notes",
            "",
            "- Object extents use per-axis robust quantiles, not raw min/max.",
            "- Visible masks are not interpreted as amodal object geometry.",
            "- Generic internal points are `geometric_track_point` candidates, not semantic keypoints.",
            "- Missing geometry remains invalid/NaN and is never encoded as a zero point.",
            "- Provider failure, unavailable pose, and absent semantic keypoints are not anomaly evidence.",
            "",
        ]
    )
    (output / "METRIC_SCENE3D_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return validation
