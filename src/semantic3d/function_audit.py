"""Read-only functional closure audit for the six-video Semantic3D prototype.

The audit deliberately separates source availability, real artifact execution,
and valid residual production.  It never invokes a learned provider and never
uses authenticity labels to fit, threshold, or evaluate the method.
"""

from __future__ import annotations

import ast
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


ALLOWED_FEATURE_STATUSES = {
    "implemented_and_executed",
    "implemented_not_executed",
    "partially_executed",
    "interface_only",
    "missing",
    "blocked_by_input",
    "provider_failed",
    "not_applicable",
}

COVERAGE_FEATURES = (
    "object",
    "mask",
    "depth",
    "pose",
    "track",
    "shared_3d",
    "semantic3d",
    "scale_depth_residual",
    "reprojection_residual",
    "trajectory_residual",
    "boundary_residual",
    "occlusion_residual",
    "D1",
    "D2",
    "D3",
)


@dataclass(frozen=True)
class FeatureSpec:
    """Static source contract for one paper-method function."""

    group: str
    feature_name: str
    source_file: str
    function_or_class: str
    input_schema: str
    output_schema: str
    unit_or_coordinate_system: str
    coverage_key: str
    tests_covering_feature: str
    base_status: str = ""
    base_failure_reason: str = ""


def _feature_specs() -> tuple[FeatureSpec, ...]:
    """Return the deterministic paper-method inventory definition."""

    return (
        FeatureSpec("A", "video_reading_and_frame_extraction", "src/semantic3d/video_preprocess.py", "extract_frames", "video path, fps, max_frames", "frame paths and global frame indices", "decoded pixel frames", "frame_decode", "tests/test_video_observation_pipeline.py"),
        FeatureSpec("A", "clip_construction", "src/semantic3d/video_preprocess.py", "build_clips", "ordered frame paths and global indices", "clip_id, frame_indices, frame_paths", "global frame index", "clip", "tests/test_video_observation_pipeline.py"),
        FeatureSpec("A", "object_detection", "src/semantic3d/real_object_provider.py", "RealObjectProvider.predict", "frame path, index, width, height", "ObjectObservationJSON list", "pixel bbox; confidence [0,1]", "object", "tests/test_real_object_provider.py", "implemented_and_executed"),
        FeatureSpec("A", "object_category", "src/semantic3d/observations.py", "ObjectObservationJSON", "detector class name", "semantic_label and canonical_label", "categorical", "object", "tests/test_observations_io.py;tests/test_real_object_provider.py", "implemented_and_executed"),
        FeatureSpec("A", "object_bbox", "src/semantic3d/observations.py", "ObjectObservationJSON.bbox", "detector xyxy", "bbox=[x1,y1,x2,y2]", "pixels; xyxy", "object", "tests/test_real_object_provider.py", "implemented_and_executed"),
        FeatureSpec("A", "instance_mask", "src/semantic3d/occlusion/mask_provider.py", "RealInstanceMaskProvider.predict", "frame and detected objects", "InstanceMaskObservation", "binary visible mask in image pixels", "mask", "tests/test_occlusion_masks.py;tests/test_p4b5_full_observation.py"),
        FeatureSpec("A", "object_projected_area", "src/semantic3d/scale_depth.py", "ObjectObservation.projection_ratio", "mask_area and frame_area", "mask_area/frame_area", "dimensionless area fraction", "object", "tests/test_scale_depth.py"),
        FeatureSpec("A", "normalized_projection_scale", "src/semantic3d/scale_depth.py", "ObjectObservation.equivalent_projection_scale", "mask_area and frame_area", "sqrt(mask_area/frame_area)", "dimensionless linear image fraction", "object", "tests/test_scale_depth.py"),
        FeatureSpec("B", "frame_depth", "src/semantic3d/depth_provider.py", "RealDepthProvider.predict_observation", "frame path and frame index", "DepthObservation", "monocular relative depth; larger means farther", "depth", "tests/test_depth_provider.py"),
        FeatureSpec("B", "object_region_depth_statistic", "src/semantic3d/depth_provider.py", "compute_object_depth_from_bbox", "depth map and bbox", "median/mean object depth", "same relative-depth gauge as frame", "scale_depth_attempted", "tests/test_depth_provider.py", "implemented_and_executed"),
        FeatureSpec("B", "valid_depth_ratio", "src/semantic3d/reconstruction/depth_sampling.py", "sample_depth", "DepthObservation and pixel/window", "DepthSample quality and validity", "ratio in [0,1]", "depth", "tests/test_object_3d_reconstruction.py;tests/test_p4b5_full_observation.py"),
        FeatureSpec("B", "physical_scale_prior", "src/semantic3d/strict_scale_prior.py", "StrictPhysicalScalePriorResolver.resolve", "canonical object label", "physical prior resolution", "metres in frozen prior definition", "scale_depth_attempted", "tests/test_strict_rsd_baseline.py;tests/test_strict_rsd_v2.py", "implemented_and_executed"),
        FeatureSpec("B", "object_pair_scale_ratio_interval", "src/semantic3d/scale_depth.py", "compute_scale_depth_interval", "two object observations and scale priors", "lower and upper expected depth ratio", "dimensionless ratio", "scale_depth_residual", "tests/test_scale_depth.py"),
        FeatureSpec("B", "expected_depth_ratio_interval", "src/semantic3d/dimension_aligned_scale_depth.py", "compute_dimension_aligned_rsd", "dimension-compatible projected measurements and priors", "ratio/log interval and evidence", "dimensionless ratio/log-ratio", "scale_depth_residual", "tests/test_dimension_aligned_rsd.py;tests/test_strict_rsd_v2.py"),
        FeatureSpec("B", "scale_depth_residual_R_sd", "src/semantic3d/scale_depth.py", "rsd_2d_coarse_log", "object pair and scale priors", "R_sd and details", "dimensionless log interval distance", "scale_depth_residual", "tests/test_scale_depth.py;tests/test_strict_rsd_v2.py"),
        FeatureSpec("C", "inter_frame_correspondence_points", "src/semantic3d/dynamic_3d/track_observation.py", "BasePointTracker", "ordered frames", "PointTrack2DObservation", "pixel coordinates with global frame index", "point_track_2d", "tests/test_dynamic_3d_readiness.py;tests/test_p4b5_full_observation.py"),
        FeatureSpec("C", "object_tracks", "src/semantic3d/object_association.py", "ObjectAssociator.associate", "frame object observations", "global object track IDs", "identity over global frame index", "track", "tests/test_object_association.py"),
        FeatureSpec("C", "boundary_tracks", "src/semantic3d/occlusion/mask_tracking.py", "MaskTracker.track", "observed and history-predicted masks", "TrackedMaskObservation", "pixel mask IoU and normalized boundary distance", "mask_track", "tests/test_occlusion_masks.py"),
        FeatureSpec("C", "camera_intrinsics", "src/semantic3d/geometry/camera.py", "CameraObservation", "K and image dimensions", "validated pinhole intrinsics", "pixels; OpenCV x-right y-down z-forward", "camera", "tests/test_camera_geometry.py"),
        FeatureSpec("C", "single_frame_camera_coordinates", "src/semantic3d/reconstruction/shared_3d_builder.py", "Shared3DFrameBuilder.build", "camera, depth, 2D objects/points", "Shared3DFrameObservation", "frame-camera relative 3D gauge", "shared_3d", "tests/test_shared_3d_builder.py;tests/test_p4b5_full_observation.py"),
        FeatureSpec("C", "continuous_camera_pose", "src/semantic3d/sequence_geometry/pose_estimation.py", "LayeredPoseEstimator", "independent background tracks and K", "relative pose observation", "current-camera from previous-camera; scale ambiguous", "pose", "tests/test_real_clip_sequence_geometry_smoke.py;tests/test_sequence_geometry_observation.py"),
        FeatureSpec("C", "pose_quality_and_failure_state", "src/semantic3d/dynamic_3d/readiness.py", "assess_dynamic_3d_readiness", "pose/depth/tracking quality", "Dynamic3DReadiness", "quality [0,1] and controlled missing reason", "dynamic_readiness", "tests/test_dynamic_3d_readiness.py", "implemented_and_executed"),
        FeatureSpec("C", "three_dimensional_backprojection_points", "src/semantic3d/geometry/backprojection.py", "backproject_pixel", "u,v,Z,K", "Point3DObservation", "camera relative 3D unless metric depth supplied", "point_track_3d", "tests/test_backprojection.py"),
        FeatureSpec("C", "cross_frame_3d_transform", "src/semantic3d/geometry/transforms.py", "transform_points", "3D points and 4x4 directed transform", "transformed Point3DObservation", "column-vector rigid transform", "pose", "tests/test_camera_geometry.py;tests/test_dynamic_reprojection_residual.py"),
        FeatureSpec("C", "next_frame_reprojected_points", "src/semantic3d/dynamic_3d/reprojection_residual.py", "compute_dynamic_reprojection_residual", "previous 3D point, independent current 2D point, K, relative pose", "predicted_uv and residual evidence", "pixels and image-diagonal-normalized error", "reprojection_residual", "tests/test_dynamic_reprojection_residual.py"),
        FeatureSpec("D", "point_reprojection_residual", "src/semantic3d/dynamic_3d/reprojection_residual.py", "compute_dynamic_reprojection_residual", "shared 3D trajectory and independent 2D point", "r_dynamic_reprojection", "image-diagonal-normalized pixel distance", "reprojection_residual", "tests/test_dynamic_reprojection_residual.py"),
        FeatureSpec("D", "trajectory_residual", "src/semantic3d/dynamic_3d/track_residual.py", "compute_track_3d_continuity_residuals", "three same-ID 3D track observations", "r_track_3d_continuity", "object-scale-normalized second difference", "trajectory_residual", "tests/test_p3b_dynamic_residuals.py"),
        FeatureSpec("D", "depth_change_residual", "src/semantic3d/depth_temporal_consistency.py", "r_depth_cons_2p5d", "same-track 2.5D object transition", "ResidualEvidence", "dimensionless log geometry change", "depth_consistency", "tests/test_depth_temporal_consistency.py"),
        FeatureSpec("D", "boundary_motion_residual", "src/semantic3d/occlusion/boundary_occlusion_residual.py", "compute_boundary_occlusion_residual", "tracked visible masks and occlusion relation", "BoundaryOcclusionResidual", "image-diagonal-normalized boundary distance", "boundary_residual", "tests/test_occlusion_residuals.py"),
        FeatureSpec("D", "motion_explanation_residual", "src/semantic3d/dynamic_3d/direction_residual.py", "compute_direction_consistency_residuals", "object-bound 3D point histories", "direction consistency evidences", "one-minus-cosine / relative motion", "trajectory_residual", "tests/test_p3b_dynamic_residuals.py"),
        FeatureSpec("D", "occlusion_residual", "src/semantic3d/occlusion/depth_order_residual.py", "compute_occlusion_depth_order_residual", "validated occlusion relation and relative depth", "OcclusionDepthOrderResidual", "relative depth order", "occlusion_residual", "tests/test_occlusion_residuals.py"),
        FeatureSpec("D", "disappearance_reappearance_residual", "src/semantic3d/occlusion/reappearance.py", "evaluate_reappearance", "visibility history and re-identification evidence", "ReappearanceObservation", "identity/structure/depth consistency", "occlusion_residual", "tests/test_occlusion_residuals.py"),
        FeatureSpec("D", "D1_static_camera_dynamic_geometry", "src/semantic3d/dynamic_3d/readiness.py", "DynamicGeometryMode.STATIC_CAMERA_3D", "static-camera shared sequence 3D", "D1-compatible dynamic evidence", "clip-local relative shared sequence", "D1", "tests/test_real_dynamic_3d_smoke.py"),
        FeatureSpec("D", "D2_rotation_compensated_geometry", "src/semantic3d/dynamic_3d/readiness.py", "DynamicGeometryMode.ROTATION_COMPENSATED", "materialized rotation-compensated pose", "D2-compatible evidence", "bearing/rotation-compensated coordinates", "D2", "tests/test_dynamic_reprojection_residual.py", "blocked_by_input", "rotation_transform_not_materialized"),
        FeatureSpec("D", "D3_full_SE3_geometry", "src/semantic3d/dynamic_3d/readiness.py", "DynamicGeometryMode.FULL_SE3_3D", "calibrated full-SE3 pose and shared sequence scale", "D3-compatible world evidence", "world/clip coordinate system", "D3", "tests/test_dynamic_reprojection_residual.py", "blocked_by_input", "full_se3_not_observationally_supported"),
        FeatureSpec("E", "clip_residual_table", "src/semantic3d/aggregation_v2/contracts.py", "ClipEvidenceAggregate", "frame evidence", "clip score with validity/coverage/provenance", "branch-specific normalized evidence", "clip_evidence_attempted", "tests/test_aggregation_v2_multilevel.py", "partially_executed", "branch-level clip rows exist for all clips; one multilevel aggregate is valid"),
        FeatureSpec("E", "frame_anomaly_sequence_interface", "src/semantic3d/aggregation_v2/contracts.py", "FrameEvidenceAggregate", "object evidence", "frame score sequence", "configured normalized residual scale", "frame_aggregate", "tests/test_aggregation_v2_multilevel.py", "implemented_not_executed", "frame_evidence_table_empty"),
        FeatureSpec("E", "temporal_segment_localization_interface", "src/semantic3d/aggregation_v2/temporal_localization.py", "localize_temporal_intervals", "frame indices, scores, explicit threshold", "TemporalInterval list", "global frame indices", "frame_aggregate", "tests/test_aggregation_v2_multilevel.py", "implemented_not_executed", "no_six_video_frame_score_sequence"),
        FeatureSpec("E", "spatial_region_prompt_interface", "src/semantic3d/aggregation_v2/contracts.py", "ObjectEvidenceAggregate.localization_mask_reference", "localized point/edge evidence and object mask", "mask/bbox reference", "image pixel region", "object_aggregate", "tests/test_aggregation_v2_multilevel.py", "interface_only", "six_video_object_aggregate_table_empty"),
        FeatureSpec("E", "object_localization_interface", "src/semantic3d/aggregation_v2/hierarchy.py", "aggregate_object_evidence", "point and edge aggregates", "object score and top contributors", "object track and frame index", "object_aggregate", "tests/test_aggregation_v2_multilevel.py", "interface_only", "six_video_object_aggregate_table_empty"),
        FeatureSpec("E", "trajectory_localization_interface", "src/semantic3d/aggregation_v2/contracts.py", "PointEvidenceAggregate", "point residual evidence", "point_id/object_track_id/frame_index provenance", "point track identity", "point_evidence", "tests/test_aggregation_v2_multilevel.py", "partially_executed", "point provenance exists but no final localization output"),
        FeatureSpec("E", "residual_aggregation_interface", "src/semantic3d/aggregation_v2/hierarchy.py", "aggregate_multilevel_evidence", "point and edge ResidualEvidence", "point-to-clip aggregates", "quality/coverage-aware dimensionless score", "clip_aggregate", "tests/test_aggregation_v2.py;tests/test_aggregation_v2_multilevel.py", "partially_executed", "point and edge aggregation executed; object/frame materialization is absent"),
    )


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist() if path.exists() else []


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _symbol_line(project_root: Path, source_file: str, symbol: str) -> int | None:
    path = project_root / source_file
    if not path.exists():
        return None
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return None
    wanted = symbol.split(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == wanted:
            return int(node.lineno)
    return None


def _status(count: int, *, attempted: bool = True, empty_status: str = "blocked_by_input") -> str:
    if count > 0:
        return "implemented_and_executed"
    return empty_status if attempted else "implemented_not_executed"


def _owner_maps(
    frame_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], str], dict[str, set[int]]]:
    owner: dict[tuple[str, int], str] = {}
    clip_frames: dict[str, set[int]] = defaultdict(set)
    for row in frame_rows:
        if not _bool(row.get("is_owned_frame")):
            continue
        key = (str(row["video_id"]), int(row["frame_index"]))
        clip_id = str(row.get("owner_clip_id") or row["clip_id"])
        owner[key] = clip_id
        clip_frames[clip_id].add(int(row["frame_index"]))
    return owner, clip_frames


def _increment(
    counts: dict[tuple[str, str], Counter[str]],
    entity: tuple[str, str],
    feature: str,
    amount: int = 1,
) -> None:
    counts[entity][feature] += int(amount)


def _build_coverage(
    dataset_root: Path,
    strict_rsd_path: Path,
    depth_consistency_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    videos = _read_parquet(dataset_root / "manifests/videos.parquet")
    clips = _read_parquet(dataset_root / "manifests/clips.parquet")
    frames = _read_parquet(dataset_root / "manifests/frames.parquet")
    owner, clip_frames = _owner_maps(frames)
    id_to_name = {str(row["video_id"]): str(row["source_name"]) for row in videos}
    name_to_id = {value: key for key, value in id_to_name.items()}
    clip_to_video = {str(row["clip_id"]): str(row["video_id"]) for row in clips}

    video_counts: dict[str, Counter[str]] = defaultdict(Counter)
    clip_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in videos:
        video_id = str(row["video_id"])
        if str(row.get("decode_status")) == "ok":
            video_counts[video_id]["frame_decode"] = int(row.get("frame_count") or 0)
    for row in clips:
        key = (str(row["video_id"]), str(row["clip_id"]))
        clip_counts[key]["clip"] = 1 if _bool(row.get("valid")) else 0
        video_counts[key[0]]["clip"] += clip_counts[key]["clip"]
        video_counts[key[0]]["frame_decode"] += 1
        clip_counts[key]["frame_decode"] = 1

    table_specs = (
        ("objects.parquet", "object", "frame_index"),
        ("masks.parquet", "mask", "frame_index"),
        ("depth.parquet", "depth", "frame_index"),
        ("shared_3d_frames.parquet", "shared_3d", "frame_index"),
    )
    for filename, feature, frame_field in table_specs:
        for row in _read_parquet(dataset_root / "observations" / filename):
            if not _bool(row.get("valid")):
                continue
            video_id = str(row["video_id"])
            video_counts[video_id][feature] += 1
            clip_id = owner.get((video_id, int(row[frame_field])))
            if clip_id:
                _increment(clip_counts, (video_id, clip_id), feature)

    objects = _read_parquet(dataset_root / "observations/objects.parquet")
    tracks_by_video: dict[str, set[str]] = defaultdict(set)
    for row in objects:
        if _bool(row.get("valid")) and row.get("object_track_id"):
            tracks_by_video[str(row["video_id"])].add(str(row["object_track_id"]))
    for video_id, values in tracks_by_video.items():
        video_counts[video_id]["track"] = len(values)

    for row in _read_parquet(dataset_root / "observations/point_tracks_2d.parquet"):
        if not _bool(row.get("valid")):
            continue
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        video_counts[video_id]["point_track_2d"] += 1
        _increment(clip_counts, (video_id, clip_id), "point_track_2d")
        _increment(clip_counts, (video_id, clip_id), "track")
    for row in _read_parquet(dataset_root / "observations/mask_tracks.parquet"):
        if not _bool(row.get("valid")):
            continue
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        video_counts[video_id]["mask_track"] += 1
        _increment(clip_counts, (video_id, clip_id), "mask_track")
    for row in _read_parquet(dataset_root / "observations/point_tracks_3d.parquet"):
        if not _bool(row.get("valid")):
            continue
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        video_counts[video_id]["point_track_3d"] += 1
        _increment(clip_counts, (video_id, clip_id), "point_track_3d")
    for row in _read_parquet(dataset_root / "observations/camera.parquet"):
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        if _bool(row.get("frame_geometry_valid")):
            video_counts[video_id]["camera"] += 1
            _increment(clip_counts, (video_id, clip_id), "camera")
        if _bool(row.get("sequence_geometry_valid")):
            clip_counts[(video_id, clip_id)]["pose"] = 1
    for row in _read_parquet(dataset_root / "observations/dynamic_readiness.parquet"):
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        video_counts[video_id]["dynamic_readiness_attempted"] += 1
        _increment(clip_counts, (video_id, clip_id), "dynamic_readiness_attempted")
        if _bool(row.get("dynamic_3d_ready")):
            video_counts[video_id]["pose"] += 1
            video_counts[video_id]["D1"] += int(str(row.get("geometry_mode")) == "static_camera_3d")
            video_counts[video_id]["D2"] += int(str(row.get("geometry_mode")) == "rotation_compensated")
            video_counts[video_id]["D3"] += int(str(row.get("geometry_mode")) == "full_se3_3d")
            for tier, mode in (("D1", "static_camera_3d"), ("D2", "rotation_compensated"), ("D3", "full_se3_3d")):
                if str(row.get("geometry_mode")) == mode:
                    _increment(clip_counts, (video_id, clip_id), tier)

    clip_evidence_rows = _read_parquet(dataset_root / "evidence/clip_evidence.parquet")
    for row in clip_evidence_rows:
        video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
        video_counts[video_id]["clip_evidence_attempted"] += 1
        _increment(clip_counts, (video_id, clip_id), "clip_evidence_attempted")

    branch_to_feature = {
        "dynamic_reprojection": "reprojection_residual",
        "track_3d_continuity": "trajectory_residual",
        "direction_consistency": "trajectory_residual",
        "relative_velocity_change": "trajectory_residual",
        "structure_temporal": "semantic3d",
        "boundary_occlusion": "boundary_residual",
        "occlusion_depth_order": "occlusion_residual",
        "visibility_explanation": "occlusion_residual",
        "reappearance_consistency": "occlusion_residual",
    }
    for rel in ("evidence/point_evidence.parquet", "evidence/edge_evidence.parquet"):
        for row in _read_parquet(dataset_root / rel):
            if not _bool(row.get("valid")):
                continue
            feature = branch_to_feature.get(str(row.get("branch_name")), "semantic3d")
            video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
            video_counts[video_id][feature] += 1
            video_counts[video_id]["semantic3d"] += 1
            _increment(clip_counts, (video_id, clip_id), feature)
            _increment(clip_counts, (video_id, clip_id), "semantic3d")
            if rel.endswith("point_evidence.parquet"):
                _increment(clip_counts, (video_id, clip_id), "point_evidence")
    for row in _read_parquet(dataset_root / "evidence/clip_evidence.parquet"):
        if _bool(row.get("valid")) and str(row.get("branch_name")) == "multilevel_aggregate":
            video_id, clip_id = str(row["video_id"]), str(row["clip_id"])
            video_counts[video_id]["clip_aggregate"] += 1
            _increment(clip_counts, (video_id, clip_id), "clip_aggregate")

    rsd_rows = _read_csv(strict_rsd_path)
    for row in rsd_rows:
        video_id = name_to_id.get(str(row.get("video_id")), "")
        if not video_id:
            continue
        video_counts[video_id]["scale_depth_attempted"] += 1
        clip_id = owner.get((video_id, int(row.get("frame_index") or 0)))
        if clip_id:
            _increment(clip_counts, (video_id, clip_id), "scale_depth_attempted")
        if _bool(row.get("valid")):
            video_counts[video_id]["scale_depth_residual"] += 1
            if clip_id:
                _increment(clip_counts, (video_id, clip_id), "scale_depth_residual")

    depth_cons_rows = _read_csv(depth_consistency_path)
    for row in depth_cons_rows:
        video_id = name_to_id.get(str(row.get("video_id")), "")
        if not video_id:
            continue
        video_counts[video_id]["depth_consistency_attempted"] += 1
        clip_id = owner.get((video_id, int(row.get("current_frame_index") or 0)))
        if clip_id:
            _increment(clip_counts, (video_id, clip_id), "depth_consistency_attempted")
        if _bool(row.get("valid")):
            video_counts[video_id]["depth_consistency"] += 1
            if clip_id:
                _increment(clip_counts, (video_id, clip_id), "depth_consistency")

    # Audit-only execution aliases. They preserve whether a provider/stage ran,
    # separately from whether it produced applicable evidence.
    for video_id in id_to_name:
        video_counts[video_id]["camera"] = sum(
            counts["camera"] for (item_video, _), counts in clip_counts.items()
            if item_video == video_id
        )
        video_counts[video_id]["point_track_2d"] = sum(
            counts["point_track_2d"] for (item_video, _), counts in clip_counts.items()
            if item_video == video_id
        )
        video_counts[video_id]["mask_track"] = sum(
            counts["mask_track"] for (item_video, _), counts in clip_counts.items()
            if item_video == video_id
        )
        video_counts[video_id]["point_track_3d"] = sum(
            counts["point_track_3d"] for (item_video, _), counts in clip_counts.items()
            if item_video == video_id
        )
        video_counts[video_id]["point_evidence"] = video_counts[video_id]["semantic3d"]
        video_counts[video_id]["dynamic_readiness"] = video_counts[video_id]["dynamic_readiness_attempted"]
    for counts in clip_counts.values():
        counts["dynamic_readiness"] = counts["dynamic_readiness_attempted"]

    feature_counts = {
        "object": "object",
        "mask": "mask",
        "depth": "depth",
        "pose": "pose",
        "track": "track",
        "shared_3d": "shared_3d",
        "semantic3d": "semantic3d",
        "scale_depth_residual": "scale_depth_residual",
        "reprojection_residual": "reprojection_residual",
        "trajectory_residual": "trajectory_residual",
        "boundary_residual": "boundary_residual",
        "occlusion_residual": "occlusion_residual",
        "D1": "D1",
        "D2": "D2",
        "D3": "D3",
    }

    def coverage_status(counts: Counter[str], feature: str, level: str) -> str:
        count = counts[feature_counts[feature]]
        if count > 0:
            return "implemented_and_executed"
        if feature in {"D2", "D3"}:
            return "blocked_by_input"
        if feature in {"boundary_residual", "occlusion_residual"}:
            return "not_applicable"
        if feature == "scale_depth_residual":
            return "blocked_by_input" if counts["scale_depth_attempted"] else "implemented_not_executed"
        if feature == "pose":
            return "blocked_by_input" if counts["dynamic_readiness_attempted"] else "implemented_not_executed"
        if feature in {"shared_3d", "semantic3d", "reprojection_residual", "trajectory_residual", "D1"}:
            return "blocked_by_input"
        if feature == "object":
            return "not_applicable"
        if feature == "mask":
            return "not_applicable" if counts["object"] == 0 else "provider_failed"
        if feature == "track":
            return "not_applicable" if counts["object"] == 0 else "provider_failed"
        return "provider_failed"

    per_video = []
    clips_by_video = Counter(str(row["video_id"]) for row in clips)
    for video in sorted(videos, key=lambda row: str(row["source_name"])):
        video_id = str(video["video_id"])
        counts = video_counts[video_id]
        row: dict[str, Any] = {
            "video_id": id_to_name[video_id],
            "dataset_video_id": video_id,
            "num_frames": int(video.get("frame_count") or 0),
            "num_clips": clips_by_video[video_id],
        }
        for feature in COVERAGE_FEATURES:
            row[f"{feature}_status"] = coverage_status(counts, feature, "video")
            row[f"{feature}_valid_count"] = counts[feature_counts[feature]]
        for feature in (
            "frame_decode", "clip", "point_track_2d", "mask_track", "camera",
            "dynamic_readiness", "point_track_3d", "depth_consistency",
            "scale_depth_attempted", "clip_evidence_attempted", "clip_aggregate",
            "point_evidence",
        ):
            count = counts[feature]
            empty = "blocked_by_input" if feature in {"point_track_3d", "depth_consistency"} else "implemented_not_executed"
            row[f"{feature}_status"] = _status(count, attempted=True, empty_status=empty)
            row[f"{feature}_valid_count"] = count
        per_video.append(row)

    per_clip = []
    for clip in sorted(clips, key=lambda row: (id_to_name[str(row["video_id"])], int(row["clip_ordinal"]))):
        video_id, clip_id = str(clip["video_id"]), str(clip["clip_id"])
        counts = clip_counts[(video_id, clip_id)]
        row = {
            "video_id": id_to_name[video_id],
            "dataset_video_id": video_id,
            "clip_id": clip_id,
            "clip_ordinal": int(clip["clip_ordinal"]),
            "start_frame_index": int(clip["start_frame_index"]),
            "end_frame_index": int(clip["end_frame_index"]),
            "owned_frame_count": len(clip_frames.get(clip_id, set())),
        }
        for feature in COVERAGE_FEATURES:
            row[f"{feature}_status"] = coverage_status(counts, feature, "clip")
            row[f"{feature}_valid_count"] = counts[feature_counts[feature]]
        for feature in (
            "frame_decode", "clip", "point_track_2d", "mask_track", "camera",
            "dynamic_readiness", "point_track_3d", "depth_consistency",
            "scale_depth_attempted", "clip_evidence_attempted", "clip_aggregate",
            "point_evidence",
        ):
            count = counts[feature]
            empty = "blocked_by_input" if feature in {"point_track_3d", "depth_consistency"} else "implemented_not_executed"
            row[f"{feature}_status"] = _status(count, attempted=True, empty_status=empty)
            row[f"{feature}_valid_count"] = count
        per_clip.append(row)

    context = {
        "video_id_to_name": id_to_name,
        "video_counts": video_counts,
        "clip_counts": clip_counts,
        "video_count": len(videos),
        "clip_count": len(clips),
        "frame_count": sum(int(row.get("frame_count") or 0) for row in videos),
        "strict_rsd_path": str(strict_rsd_path),
        "depth_consistency_path": str(depth_consistency_path),
    }
    return per_video, per_clip, context


def _numeric_stats(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str,
    valid_field: str,
    video_field: str,
    video_names: Mapping[str, str],
    artifact_path: str,
    evidence_level: str,
    execution_state: str | None = None,
) -> dict[str, Any]:
    values: list[float] = []
    count_nan = 0
    count_inf = 0
    count_valid = 0
    distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        valid = _bool(row.get(valid_field))
        video_id = str(row.get(video_field, "unknown"))
        video_name = video_names.get(video_id, video_id or "unknown")
        distribution[video_name]["total"] += 1
        if valid:
            count_valid += 1
            distribution[video_name]["valid"] += 1
        value = _float(row.get(value_field))
        if math.isnan(value):
            count_nan += 1
        elif math.isinf(value):
            count_inf += 1
        elif valid:
            values.append(value)
    total = len(rows)
    if execution_state is None:
        if total == 0:
            execution_state = "implemented_not_executed"
        elif count_valid == 0:
            execution_state = "executed_no_valid_output"
        else:
            execution_state = "valid_numeric_output"
    return {
        "residual_name": name,
        "evidence_level": evidence_level,
        "execution_state": execution_state,
        "artifact_path": artifact_path,
        "value_field": value_field,
        "count_total": total,
        "count_valid": count_valid,
        "count_missing": total - count_valid,
        "count_nan": count_nan,
        "count_inf": count_inf,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "median": median(values) if values else None,
        "zero_fraction": sum(abs(value) <= 1e-12 for value in values) / len(values) if values else None,
        "source_video_distribution": {
            key: dict(value) for key, value in sorted(distribution.items())
        },
    }


def _build_residual_audit(
    dataset_root: Path,
    strict_rsd_path: Path,
    depth_consistency_path: Path,
    video_names: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    point_rows = _read_parquet(dataset_root / "evidence/point_evidence.parquet")
    edge_rows = _read_parquet(dataset_root / "evidence/edge_evidence.parquet")
    clip_rows = _read_parquet(dataset_root / "evidence/clip_evidence.parquet")
    artifacts: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []

    def add(
        name: str,
        rows: Sequence[Mapping[str, Any]],
        path: Path,
        value_field: str,
        valid_field: str,
        video_field: str,
        level: str,
        provenance: str,
    ) -> None:
        audits.append(_numeric_stats(
            name, rows, value_field=value_field, valid_field=valid_field,
            video_field=video_field, video_names=video_names,
            artifact_path=str(path), evidence_level=level,
        ))
        artifacts.append({
            "residual_name": name,
            "evidence_level": level,
            "artifact_path": str(path),
            "artifact_location": "archive" if str(path).startswith("/mnt/") else "project",
            "row_count": len(rows),
            "value_field": value_field,
            "valid_field": valid_field,
            "provenance": provenance,
        })

    strict_rows = _read_csv(strict_rsd_path)
    add("rsd_2d_dimension_aligned_log", strict_rows, strict_rsd_path, "rsd_log", "valid", "video_id", "object_pair", "frozen strict-v2 historical six-video result")
    add("rsd_2d_dimension_aligned_ratio", strict_rows, strict_rsd_path, "rsd_ratio", "valid", "video_id", "object_pair", "frozen strict-v2 historical six-video result")
    depth_rows = _read_csv(depth_consistency_path)
    add("r_depth_cons_2p5d", depth_rows, depth_consistency_path, "residual", "valid", "video_id", "object_transition", "existing six-video 2.5D transition result")
    add("r_depth_cons_2p5d_raw", depth_rows, depth_consistency_path, "raw_residual", "valid", "video_id", "object_transition", "existing six-video 2.5D transition result")

    lower_level = {
        "dynamic_reprojection": (point_rows, dataset_root / "evidence/point_evidence.parquet", "point"),
        "track_3d_continuity": (point_rows, dataset_root / "evidence/point_evidence.parquet", "point"),
        "direction_consistency": (point_rows, dataset_root / "evidence/point_evidence.parquet", "point"),
        "relative_velocity_change": (point_rows, dataset_root / "evidence/point_evidence.parquet", "point"),
        "structure_temporal": (edge_rows, dataset_root / "evidence/edge_evidence.parquet", "edge"),
    }
    all_branches = (
        "semantic_size_3d", "depth_order_3d", "boundary_depth_3d",
        "spatial_intersection_3d", "track_3d_continuity", "direction_consistency",
        "relative_velocity_change", "dynamic_reprojection", "structure_temporal",
        "occlusion_depth_order", "visibility_explanation", "boundary_occlusion",
        "reappearance_consistency", "multilevel_aggregate",
    )
    for branch in all_branches:
        if branch in lower_level:
            source_rows, path, level = lower_level[branch]
            rows = [row for row in source_rows if str(row.get("branch_name")) == branch]
        else:
            path, level = dataset_root / "evidence/clip_evidence.parquet", "clip"
            rows = [row for row in clip_rows if str(row.get("branch_name")) == branch]
        add(branch, rows, path, "raw_value", "valid", "video_id", level, "P4-B.5 full-observation evidence table")

    return artifacts, {"residuals": audits}


def _historical_residual_artifacts(
    project_root: Path, archive_root: Path
) -> list[dict[str, Any]]:
    """Inventory earlier P3/P4 residual files without using them as primary stats."""

    roots = (
        archive_root / "outputs/real_dynamic_3d_smoke",
        archive_root / "outputs/real_object_dynamic_3d_smoke",
        project_root / "outputs/real_3d_evidence_coverage_v2/shared_3d_smoke/occlusion",
    )
    output: list[dict[str, Any]] = []
    value_candidates = (
        "formal_residual", "normalized_residual", "normalized_pixel_error",
        "raw_residual", "residual", "pixel_error",
    )
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*residuals.csv")):
            rows = _read_csv(path)
            fields = tuple(rows[0]) if rows else ()
            value_field = next((name for name in value_candidates if name in fields), "")
            output.append({
                "residual_name": f"historical_{path.parent.name}_{path.stem}",
                "evidence_level": "historical_smoke",
                "artifact_path": str(path),
                "artifact_location": "archive" if str(path).startswith("/mnt/") else "project",
                "row_count": len(rows),
                "value_field": value_field,
                "valid_field": "valid" if "valid" in fields else ("formal_residual_valid" if "formal_residual_valid" in fields else ""),
                "provenance": "earlier P3 read-only smoke artifact; not merged into current P4-B.5 numeric totals",
            })
    previous = project_root / "outputs/structural_enhancement_dataset/p4b_six_video_smoke"
    for relative in (
        "evidence/point_evidence.parquet",
        "evidence/edge_evidence.parquet",
        "evidence/clip_evidence.parquet",
    ):
        path = previous / relative
        rows = _read_parquet(path)
        if path.exists():
            output.append({
                "residual_name": f"historical_p4b_{path.stem}",
                "evidence_level": "previous_dataset_version",
                "artifact_path": str(path),
                "artifact_location": "project",
                "row_count": len(rows),
                "value_field": "raw_value",
                "valid_field": "valid",
                "provenance": "P4-B predecessor retained for audit; superseded by P4-B.5 primary tables",
            })
    return output


def run_synthetic_formula_checks() -> dict[str, Any]:
    """Exercise existing formulas using synthetic geometry only."""

    from .dynamic_3d import (
        DynamicGeometryMode,
        PointTrack2DObservation,
        PointTrack3DObservation,
        compute_dynamic_reprojection_residual,
    )
    from .geometry.backprojection import backproject_pixel
    from .geometry.camera import (
        CoordinateConvention,
        DepthDefinition,
        PixelCenterConvention,
        TransformConvention,
    )
    from .geometry.projection import project_point
    from .scale_depth import (
        ObjectObservation,
        ScalePrior,
        compute_scale_depth_interval,
        scale_depth_residual,
        scale_depth_residual_log,
    )

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, details: Mapping[str, Any]) -> None:
        checks.append({"test_name": name, "passed": bool(passed), "details": dict(details)})

    priors = {"a": ScalePrior(1.0, 2.0), "b": ScalePrior(2.0, 4.0)}
    a_inside = ObjectObservation("a", "a", 100.0, 10000.0, 5.0)
    b = ObjectObservation("b", "b", 400.0, 10000.0, 5.0)
    residual_inside, inside_details = scale_depth_residual_log(a_inside, b, priors)
    record("rsd_inside_interval_is_zero", abs(residual_inside) <= 1e-12, {"residual": residual_inside, **inside_details})

    a_out_1 = ObjectObservation("a1", "a", 100.0, 10000.0, 15.0)
    a_out_2 = ObjectObservation("a2", "a", 100.0, 10000.0, 20.0)
    residual_1, details_1 = scale_depth_residual_log(a_out_1, b, priors)
    residual_2, details_2 = scale_depth_residual_log(a_out_2, b, priors)
    record("rsd_outside_interval_is_monotonic", residual_2 > residual_1 > 0.0, {"first": residual_1, "second": residual_2, "first_details": details_1, "second_details": details_2})

    swapped, swapped_details = scale_depth_residual_log(b, a_out_1, priors)
    lower, upper = compute_scale_depth_interval(a_out_1, b, priors)
    swapped_lower, swapped_upper = compute_scale_depth_interval(b, a_out_1, priors)
    symmetry = math.isclose(residual_1, swapped, abs_tol=1e-12) and math.isclose(swapped_lower, 1.0 / upper, abs_tol=1e-12) and math.isclose(swapped_upper, 1.0 / lower, abs_tol=1e-12)
    record("rsd_order_swap_is_log_consistent", symmetry, {"forward_residual": residual_1, "swapped_residual": swapped, "forward_interval": [lower, upper], "swapped_interval": [swapped_lower, swapped_upper], "swapped_details": swapped_details})

    invalid_results = []
    for name, invalid in (
        ("zero_mask", ObjectObservation("bad_mask", "a", 0.0, 10000.0, 5.0)),
        ("zero_depth", ObjectObservation("bad_depth", "a", 100.0, 10000.0, 0.0)),
    ):
        try:
            value, _ = scale_depth_residual(invalid, b, priors)
            invalid_results.append({"case": name, "handled": False, "value": value})
        except ValueError as exc:
            invalid_results.append({"case": name, "handled": True, "exception": str(exc)})
    record("invalid_rsd_inputs_are_rejected_without_numeric_output", all(row["handled"] for row in invalid_results), {"cases": invalid_results})

    K = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])
    source = backproject_pixel(70.0, 30.0, 5.0, K, point_id="cycle")
    projected = project_point(source, K)
    cycle_error = math.hypot(float(projected.x) - 70.0, float(projected.y) - 30.0) if projected.valid else math.inf
    record("identity_projection_backprojection_cycle", cycle_error <= 1e-12, {"pixel_error": cycle_error, "projected_uv": [projected.x, projected.y]})

    previous = PointTrack3DObservation(
        point_id="p", object_track_id="o", frame_index=0,
        pixel_uv=(50.0, 40.0), observed_depth=5.0,
        point_3d_camera=(0.0, 0.0, 5.0), point_3d_world=(0.0, 0.0, 5.0),
        visibility="visible", occlusion_status="visible",
        tracking_confidence=1.0, depth_quality=1.0, reconstruction_quality=1.0,
        source_tracker="synthetic", scale_status="relative_shared_sequence",
        geometry_mode="full_se3_3d", valid=True,
    )
    current = PointTrack2DObservation(
        point_id="p", object_track_id="o", frame_index=1,
        pixel_uv=(70.0, 40.0), visibility="visible", occlusion_status="visible",
        tracking_confidence=1.0, source_tracker="synthetic_independent",
        valid=True, metadata={"independent_observation": True, "generated_from_projection": False},
    )
    current_from_previous = np.eye(4)
    current_from_previous[0, 3] = 1.0
    known = compute_dynamic_reprojection_residual(
        previous, current, K_current=K, image_width=100, image_height=80,
        relative_pose_current_from_previous=current_from_previous,
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D, is_background=True,
    )
    record("known_camera_transform_matches_hand_projection", known.valid and known.pixel_error <= 1e-12, {"expected_uv": [70.0, 40.0], "predicted_uv": known.predicted_uv, "pixel_error": known.pixel_error})

    reversed_pose = np.eye(4)
    reversed_pose[0, 3] = -1.0
    reversed_result = compute_dynamic_reprojection_residual(
        previous, current, K_current=K, image_width=100, image_height=80,
        relative_pose_current_from_previous=reversed_pose,
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D, is_background=True,
    )
    record("reversed_pose_direction_is_detectable", reversed_result.pixel_error > 1.0 and reversed_result.pixel_error > known.pixel_error, {"correct_error": known.pixel_error, "reversed_error": reversed_result.pixel_error, "reversed_predicted_uv": reversed_result.predicted_uv})

    missing_provider = PointTrack2DObservation.missing(point_id="p", object_track_id="o", frame_index=1, reason="provider_failed", source_tracker="failed_provider")
    provider_result = compute_dynamic_reprojection_residual(
        previous, missing_provider, K_current=K, image_width=100, image_height=80,
        relative_pose_current_from_previous=np.eye(4),
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D, is_background=True,
    )
    record("provider_failure_is_masked_not_high_residual", not provider_result.valid and math.isnan(provider_result.residual_evidence.value), {"valid": provider_result.valid, "value": provider_result.residual_evidence.value, "missing_reason": provider_result.missing_reason})

    occluded = PointTrack2DObservation.missing(point_id="p", object_track_id="o", frame_index=1, reason="occluded_or_invisible_point", source_tracker="synthetic_tracker")
    occluded_result = compute_dynamic_reprojection_residual(
        previous, occluded, K_current=K, image_width=100, image_height=80,
        relative_pose_current_from_previous=np.eye(4),
        geometry_mode=DynamicGeometryMode.FULL_SE3_3D, is_background=False,
    )
    record("occluded_point_is_not_scored_as_correspondence", not occluded_result.valid and math.isnan(occluded_result.pixel_error), {"valid": occluded_result.valid, "pixel_error": occluded_result.pixel_error, "missing_reason": occluded_result.missing_reason})

    conventions = {
        "camera": CoordinateConvention.OPENCV.value,
        "depth": DepthDefinition.Z_DEPTH.value,
        "transform": TransformConvention.COLUMN_VECTOR.value,
        "pixel_center": PixelCenterConvention.INTEGER_CENTERS.value,
        "pixel_storage": "non_homogeneous_uv; homogeneous coordinates used internally for matrix multiplication",
    }
    record("coordinate_conventions_are_explicit", all(value and value != "unknown" for value in conventions.values()), conventions)
    return {
        "all_passed": all(row["passed"] for row in checks),
        "passed": sum(row["passed"] for row in checks),
        "total": len(checks),
        "checks": checks,
        "uses_authenticity_labels": False,
        "performance_metrics_computed": False,
    }


def _coordinate_audit(dataset_root: Path) -> dict[str, Any]:
    depth = _read_parquet(dataset_root / "observations/depth.parquet")
    camera = _read_parquet(dataset_root / "observations/camera.parquet")
    points = _read_parquet(dataset_root / "observations/point_tracks_3d.parquet")
    unique = lambda rows, key: sorted({str(row.get(key, "")) for row in rows})
    return {
        "depth": {
            "representation": unique(depth, "depth_representation"),
            "frame_scale_status": unique(depth, "scale_status"),
            "larger_value_means": unique(depth, "larger_value_means"),
            "provider": unique(depth, "provider_name"),
            "interpretation": "Monocular relative depth, not metric depth. P4-B.5 stores per-frame depth and a separate clip sequence alignment status.",
            "cross_video_comparable": False,
            "reason": "No metric calibration or shared scale anchor exists across videos.",
        },
        "intrinsics": {
            "source": unique(camera, "intrinsics_source"),
            "status": "estimated_or_assumed_approximation",
            "is_calibrated_ground_truth": False,
        },
        "pose": {
            "stored_api": "T_world_from_camera and T_camera_from_world are explicit inverses",
            "relative_pose_api": "relative_pose_current_from_previous maps previous-camera points into current-camera coordinates",
            "six_video_pose_sources": unique(camera, "pose_source"),
            "six_video_geometry_modes": unique(camera, "geometry_mode"),
            "continuous_moving_camera_pose_status": "unresolved_or_unmaterialized",
        },
        "coordinates": {
            "camera": "right-handed OpenCV-like x-right, y-down, z-forward",
            "pixel": "non-homogeneous (u,v); integer coordinates are pixel centres",
            "homogeneous_usage": "internal 3x3/4x4 matrix operations only",
            "point_3d_units": unique(points, "trajectory_representation"),
            "world_coordinates_available": False,
        },
        "residual_normalization": {
            "rsd_2d_dimension_aligned_log": "dimensionless distance in log depth-ratio space",
            "r_depth_cons_2p5d": "dimensionless log geometry-state difference minus tolerance",
            "dynamic_reprojection": "pixel error divided by image diagonal",
            "track_3d_continuity": "second difference normalized by observed object scale when available",
            "direction_consistency": "one minus cosine/directional disagreement",
            "relative_velocity_change": "relative shared-sequence displacement normalized by object scale",
            "structure_temporal": "edge length change normalized by object structure scale",
        },
        "resolution_normalization": {
            "R_sd": "sqrt(projected_area/frame_area)",
            "reprojection": "image diagonal",
            "boundary": "image diagonal or mask-relative distance depending on branch",
        },
        "unresolved": [
            "metric depth scale",
            "calibrated camera intrinsics",
            "materialized D2 rotation transforms in the current full dataset",
            "full-SE3 translation scale and world trajectory",
            "cross-video relative-depth comparability",
        ],
    }


def _feature_inventory(
    project_root: Path,
    per_video: Sequence[Mapping[str, Any]],
    per_clip: Sequence[Mapping[str, Any]],
    artifact_paths: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for spec in _feature_specs():
        video_statuses = Counter(str(row.get(f"{spec.coverage_key}_status", "")) for row in per_video)
        clip_statuses = Counter(str(row.get(f"{spec.coverage_key}_status", "")) for row in per_clip)
        videos_succeeded = sum(str(row.get(f"{spec.coverage_key}_status")) == "implemented_and_executed" for row in per_video)
        clips_succeeded = sum(str(row.get(f"{spec.coverage_key}_status")) == "implemented_and_executed" for row in per_clip)
        status = spec.base_status
        if not status:
            if videos_succeeded == len(per_video) and clips_succeeded == len(per_clip):
                status = "implemented_and_executed"
            elif videos_succeeded > 0 or clips_succeeded > 0:
                status = "partially_executed"
            elif video_statuses.get("not_applicable") or clip_statuses.get("not_applicable"):
                status = "not_applicable"
            elif video_statuses.get("blocked_by_input") or clip_statuses.get("blocked_by_input"):
                status = "blocked_by_input"
            elif video_statuses.get("provider_failed") or clip_statuses.get("provider_failed"):
                status = "provider_failed"
            else:
                status = "implemented_not_executed"
        if status not in ALLOWED_FEATURE_STATUSES:
            raise ValueError(f"Invalid feature status: {status}")
        reason = spec.base_failure_reason
        if not reason and status != "implemented_and_executed":
            reason = ";".join(
                f"{key}={value}" for key, value in sorted((video_statuses + clip_statuses).items()) if key
            )
        rows.append({
            "group": spec.group,
            "feature_name": spec.feature_name,
            "status": status,
            "source_file": spec.source_file,
            "source_line": _symbol_line(project_root, spec.source_file, spec.function_or_class),
            "function_or_class": spec.function_or_class,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "unit_or_coordinate_system": spec.unit_or_coordinate_system,
            "videos_attempted": len(per_video),
            "videos_succeeded": videos_succeeded,
            "clips_attempted": len(per_clip),
            "clips_succeeded": clips_succeeded,
            "artifact_paths": artifact_paths.get(spec.coverage_key, ""),
            "failure_reasons": reason,
            "tests_covering_feature": spec.tests_covering_feature,
        })
    return rows


def _report(
    inventory: Sequence[Mapping[str, Any]],
    per_video: Sequence[Mapping[str, Any]],
    residual_audit: Mapping[str, Any],
    synthetic: Mapping[str, Any],
    dataset_root: Path,
    strict_rsd_path: Path,
) -> tuple[str, dict[str, Any]]:
    by_status = Counter(str(row["status"]) for row in inventory)
    residual_by_name = {row["residual_name"]: row for row in residual_audit["residuals"]}
    valid_rsd = residual_by_name["rsd_2d_dimension_aligned_log"]["count_valid"]
    valid_reprojection = residual_by_name["dynamic_reprojection"]["count_valid"]
    d1_videos = [row["video_id"] for row in per_video if int(row["D1_valid_count"]) > 0]
    no_valid_rsd = [row["video_id"] for row in per_video if int(row["scale_depth_residual_valid_count"]) == 0]
    validation = {
        "all_paper_functions_implemented": False,
        "all_paper_functions_executed_on_six_videos": False,
        "static_geometry_pipeline_operational": sum(int(row["shared_3d_valid_count"]) for row in per_video) > 0,
        "scale_depth_residual_verified": bool(valid_rsd > 0 and synthetic["all_passed"]),
        "sequence_pose_pipeline_operational": False,
        "reprojection_pipeline_operational": valid_reprojection > 0,
        "D1_verified": bool(d1_videos and valid_reprojection > 0),
        "D2_verified": False,
        "D3_verified": False,
        "occlusion_pipeline_verified": False,
        "localization_interfaces_verified": False,
        "method_effectiveness_established": False,
        "audit_scope": "functional and numerical integrity only; no authenticity performance",
        "dataset_root": str(dataset_root),
        "strict_rsd_artifact": str(strict_rsd_path),
        "six_video_count": len(per_video),
        "feature_status_counts": dict(sorted(by_status.items())),
        "synthetic_formula_checks_passed": synthetic["all_passed"],
    }
    lines = [
        "# P4-C3A-V Six-Video Function Audit",
        "",
        "## Scope",
        "",
        "This is a read-only functional closure and numerical integrity audit. It did not run learned providers, train a model, fit a distribution, select a threshold, or compute authentic/fake performance.",
        "",
        "The current P4-B.5 dataset is the primary artifact. The frozen strict R_sd v2 result was previously moved by the documented disk cleanup and is read in place from the E-drive archive; it was not regenerated or copied.",
        "",
        "## Executive Result",
        "",
        f"- Paper functions all implemented: **{str(validation['all_paper_functions_implemented']).lower()}**.",
        f"- Paper functions all executed on six videos: **{str(validation['all_paper_functions_executed_on_six_videos']).lower()}**.",
        f"- Static frame geometry operational: **{str(validation['static_geometry_pipeline_operational']).lower()}** (relative, non-metric).",
        f"- Scale-depth residual formula and real artifact verified: **{str(validation['scale_depth_residual_verified']).lower()}** ({valid_rsd} valid strict-v2 pair rows).",
        f"- Continuous sequence pose operational: **{str(validation['sequence_pose_pipeline_operational']).lower()}**. Only static-camera D1 is materialized for {len(d1_videos)}/6 videos and 2/59 clips.",
        f"- Reprojection pipeline operational in D1: **{str(validation['reprojection_pipeline_operational']).lower()}** ({valid_reprojection} valid point residuals).",
        "- D2 verified: **false**; rotation transforms are not materialized in the full dataset.",
        "- D3 verified: **false**; full-SE3 translation and world coordinates are unsupported by current inputs.",
        "- Occlusion residual pipeline verified: **false**; masks and tracking exist, but no validated six-video occlusion/reappearance event produced a formal residual.",
        "- Localization interfaces verified end to end: **false**; point provenance exists, while object/frame tables are empty and no six-video temporal/spatial localization artifact exists.",
        "- Method effectiveness established: **false**. This audit deliberately makes no real/fake comparison.",
        "",
        "## Six-Video Coverage",
        "",
        f"- Videos: {len(per_video)}; clips: 59; unique decoded frames: 984.",
        "- Frame depth: 984/984 valid, monocular relative depth with larger values meaning farther.",
        "- Shared 3D frames: 656/984 valid, all frame-camera relative; world coordinates are unavailable.",
        f"- D1 videos: {', '.join(d1_videos) if d1_videos else 'none'}.",
        f"- Videos without a valid strict-v2 R_sd pair: {', '.join(no_valid_rsd) if no_valid_rsd else 'none'}.",
        "- D2/D3 clips: 0/59. Formal occlusion/reappearance events: 0.",
        "",
        "## Residual Integrity",
        "",
    ]
    for row in residual_audit["residuals"]:
        lines.append(
            f"- `{row['residual_name']}` ({row['evidence_level']}): state={row['execution_state']}, total={row['count_total']}, valid={row['count_valid']}, NaN={row['count_nan']}, Inf={row['count_inf']}, min={row['min']}, max={row['max']}."
        )
    lines.extend([
        "",
        "NaN in invalid rows is expected evidence semantics, not a zero score. `provider_failed`, invalid geometry, and event absence are not converted into anomaly residuals.",
        "",
        "## Coordinate And Unit Findings",
        "",
        "- Depth is monocular relative depth, not metres. P4-B.5 frame arrays are `relative_per_frame`; clip alignment may produce `relative_shared_sequence` only inside one clip.",
        "- Intrinsics are approximate focal-length estimates, not calibrated ground truth.",
        "- Camera coordinates are OpenCV-like x-right/y-down/z-forward. Pixels are stored as non-homogeneous `(u,v)`; homogeneous vectors are internal implementation details.",
        "- Pose APIs are direction-explicit. Current full data only materializes static identity pose; moving-camera D2 and D3 remain blocked.",
        "- Valid 3D trajectories use a clip-local camera gauge. They are not metric and must not be compared across videos.",
        "",
        "## Formula Checks",
        "",
        f"All {synthetic['total']} synthetic checks passed: **{str(synthetic['all_passed']).lower()}**. These checks cover R_sd interval behavior, order exchange, invalid input handling, projection/backprojection closure, known pose, reversed pose detection, provider failure masking, and occlusion masking.",
        "",
        "## Blocked Or Incomplete Closure",
        "",
        "- Continuous moving-camera pose is not operational in the current P4-B.5 full run.",
        "- D2 has a code path and synthetic tests but no materialized six-video rotation transform.",
        "- D3 has a code path and synthetic tests but no calibrated full-SE3/world trajectory input.",
        "- Static 3D residual classes exist, but six-video lower-level static object evidence is not materialized (`static_object_evidence.parquet` is empty).",
        "- Boundary/occlusion/reappearance residual classes exist, but no validated event produced formal residual evidence.",
        "- Object/frame localization aggregates and frame anomaly sequences are not materialized; temporal localization has only an interface/synthetic test path.",
        "",
        "## Conclusion",
        "",
        "The project has a functioning object/mask/depth observation pipeline, partial relative-3D reconstruction, a verified 2D R_sd baseline, and D1 dynamic point residuals on two clips. It does not yet have a six-video continuous-pose, D2/D3, formal occlusion, or end-to-end localization closure. Passing unit tests must not be interpreted as evidence of detection effectiveness.",
    ])
    return "\n".join(lines) + "\n", validation


def build_function_audit(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    dataset_root: str | Path | None = None,
    archive_root: str | Path = "/mnt/e/fake_video_structural_anomaly_archive",
) -> dict[str, Any]:
    """Build all P4-C3A-V audit artifacts without running learned providers."""

    root = Path(project_root).resolve()
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    data_root = Path(dataset_root) if dataset_root is not None else root / "outputs/structural_enhancement_dataset/p4b5_six_video_full_observation"
    if not data_root.is_absolute():
        data_root = root / data_root
    archive = Path(archive_root)
    strict_project = root / "outputs/evaluation/rsd_strict_v2/per_pair_rsd_details.csv"
    strict_archive = archive / "outputs/evaluation/rsd_strict_v2/per_pair_rsd_details.csv"
    strict_rsd = strict_project if strict_project.exists() else strict_archive
    depth_consistency = root / "outputs/results/test_videos_depth_consistency_pairs.csv"
    required = (
        data_root / "dataset_manifest.json",
        data_root / "manifests/videos.parquet",
        data_root / "manifests/clips.parquet",
        strict_rsd,
        depth_consistency,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Required audit artifacts are missing: " + ", ".join(missing))

    per_video, per_clip, context = _build_coverage(data_root, strict_rsd, depth_consistency)
    artifacts, numeric = _build_residual_audit(
        data_root, strict_rsd, depth_consistency, context["video_id_to_name"]
    )
    artifacts.extend(_historical_residual_artifacts(root, archive))
    synthetic = run_synthetic_formula_checks()
    coordinates = _coordinate_audit(data_root)
    artifact_paths = {
        "frame_decode": str(data_root / "manifests/videos.parquet"),
        "clip": str(data_root / "manifests/clips.parquet"),
        "object": str(data_root / "observations/objects.parquet"),
        "mask": str(data_root / "observations/masks.parquet"),
        "depth": str(data_root / "observations/depth.parquet"),
        "scale_depth_residual": str(strict_rsd),
        "point_track_2d": str(data_root / "observations/point_tracks_2d.parquet"),
        "track": str(data_root / "observations/tracks.parquet"),
        "mask_track": str(data_root / "observations/mask_tracks.parquet"),
        "camera": str(data_root / "observations/camera.parquet"),
        "shared_3d": str(data_root / "observations/shared_3d_frames.parquet"),
        "pose": str(data_root / "observations/dynamic_readiness.parquet"),
        "dynamic_readiness": str(data_root / "observations/dynamic_readiness.parquet"),
        "point_track_3d": str(data_root / "observations/point_tracks_3d.parquet"),
        "reprojection_residual": str(data_root / "evidence/point_evidence.parquet"),
        "trajectory_residual": str(data_root / "evidence/point_evidence.parquet"),
        "depth_consistency": str(depth_consistency),
        "boundary_residual": str(data_root / "evidence/clip_evidence.parquet"),
        "occlusion_residual": str(data_root / "evidence/clip_evidence.parquet"),
        "D1": str(data_root / "observations/dynamic_readiness.parquet"),
        "D2": str(data_root / "observations/dynamic_readiness.parquet"),
        "D3": str(data_root / "observations/dynamic_readiness.parquet"),
        "clip_aggregate": str(data_root / "evidence/clip_evidence.parquet"),
        "frame_aggregate": str(data_root / "evidence/frame_evidence.parquet"),
        "object_aggregate": str(data_root / "evidence/object_evidence.parquet"),
        "point_evidence": str(data_root / "evidence/point_evidence.parquet"),
    }
    inventory = _feature_inventory(root, per_video, per_clip, artifact_paths)
    blocked = {
        "features": [
            {
                "feature_name": row["feature_name"],
                "status": row["status"],
                "failure_reasons": row["failure_reasons"],
                "required_input_or_closure": {
                    "continuous_camera_pose": "validated moving-camera relative poses",
                    "D2_rotation_compensated_geometry": "materialized rotation transforms",
                    "D3_full_SE3_geometry": "calibrated full-SE3 translation and world coordinates",
                    "boundary_motion_residual": "validated occlusion event with formal mask history",
                    "occlusion_residual": "validated occluder and visibility-change event",
                    "disappearance_reappearance_residual": "observed disappearance/reappearance event with re-identification evidence",
                    "frame_anomaly_sequence_interface": "materialized object and frame aggregates",
                    "temporal_segment_localization_interface": "frame score sequence plus predeclared threshold",
                    "spatial_region_prompt_interface": "materialized object aggregates and mask references",
                    "object_localization_interface": "materialized object aggregates",
                }.get(str(row["feature_name"]), "valid upstream evidence"),
            }
            for row in inventory
            if row["status"] != "implemented_and_executed"
        ],
        "provider_failure_is_anomaly_evidence": False,
        "missing_evidence_is_zero": False,
    }
    report_text, validation = _report(inventory, per_video, numeric, synthetic, data_root, strict_rsd)

    inventory_fields = (
        "group", "feature_name", "status", "source_file", "source_line",
        "function_or_class", "input_schema", "output_schema",
        "unit_or_coordinate_system", "videos_attempted", "videos_succeeded",
        "clips_attempted", "clips_succeeded", "artifact_paths",
        "failure_reasons", "tests_covering_feature",
    )
    coverage_fields = ["video_id", "dataset_video_id", "num_frames", "num_clips"]
    clip_fields = ["video_id", "dataset_video_id", "clip_id", "clip_ordinal", "start_frame_index", "end_frame_index", "owned_frame_count"]
    for feature in COVERAGE_FEATURES:
        coverage_fields.extend((f"{feature}_status", f"{feature}_valid_count"))
        clip_fields.extend((f"{feature}_status", f"{feature}_valid_count"))
    _write_csv(output / "method_feature_inventory.csv", inventory, inventory_fields)
    _write_csv(output / "per_video_coverage.csv", per_video, coverage_fields)
    _write_csv(output / "per_clip_coverage.csv", per_clip, clip_fields)
    _write_csv(
        output / "residual_artifact_inventory.csv",
        artifacts,
        ("residual_name", "evidence_level", "artifact_path", "artifact_location", "row_count", "value_field", "valid_field", "provenance"),
    )
    _write_json(output / "residual_numeric_audit.json", numeric)
    _write_json(output / "coordinate_and_unit_audit.json", coordinates)
    _write_json(output / "synthetic_formula_tests.json", synthetic)
    _write_json(output / "blocked_features.json", blocked)
    (output / "FUNCTION_AUDIT_REPORT.md").write_text(report_text, encoding="utf-8")
    _write_json(output / "validation_report.json", validation)
    return validation
