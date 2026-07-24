"""Offline P4-C3B-M4 pose and real D2 smoke over persisted M1/M2 artifacts."""

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
import pandas as pd
import yaml

from .alignment import build_clip_local_alignment
from .contracts import (
    D2ResidualObservation,
    D2VisibilityStatus,
    PairwisePoseObservation,
    PoseProviderStatus,
)
from .provider import ShortBaselinePoseProvider, ShortBaselinePoseThresholds
from .residuals import (
    aggregate_object_d2_residual,
    compute_d2_projection_residual,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    records = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in records:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in records:
            encoded = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(_json_safe(value), sort_keys=True)
                encoded[key] = value
            writer.writerow(encoded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_manifest(root: Path, metric_root: Path) -> dict[tuple[str, int], dict[str, str]]:
    path = metric_root / "metric_depth_frame_manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["video_id"], int(row["frame_index"])): row
            for row in csv.DictReader(handle)
        }


def _load_mask_index(
    structural_root: Path,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int, str], dict[str, Any]]]:
    videos = pd.read_parquet(structural_root / "manifests/videos.parquet")
    video_names = dict(zip(videos["video_id"], videos["source_name"], strict=True))
    masks = pd.read_parquet(structural_root / "observations/masks.parquet")
    unions: dict[tuple[str, int], np.ndarray] = {}
    object_masks: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in masks.to_dict("records"):
        if not bool(row.get("valid", False)) or bool(row.get("bbox_fallback", False)):
            continue
        source_name = video_names.get(str(row["video_id"]))
        if source_name is None:
            continue
        path = Path(str(row.get("array_path", "")))
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as payload:
            if "visible_mask" not in payload:
                continue
            mask = np.asarray(payload["visible_mask"], dtype=bool)
        frame_index = int(row["frame_index"])
        key = (source_name, frame_index)
        if key not in unions:
            unions[key] = np.zeros(mask.shape, dtype=bool)
        unions[key] |= mask
        object_masks[(source_name, frame_index, str(row["object_track_id"]))] = {
            "mask": mask,
            "object_id": str(row["segmentation_instance_id"]),
            "track_id": str(row["object_track_id"]),
            "class_name": str(row["class_name"]),
            "confidence": float(row["confidence"]),
        }
    return unions, object_masks


def _frame_inputs(
    root: Path,
    metric_root: Path,
    row: Mapping[str, str],
) -> dict[str, Any]:
    video_id = row["video_id"]
    frame_index = int(row["frame_index"])
    image_path = metric_root / "frames" / video_id / f"frame_{frame_index:06d}.jpg"
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Persisted M1 frame cannot be decoded: {image_path}")
    return {
        "image": image,
        "depth": np.load(_resolve(root, row["depth_m_path"]), allow_pickle=False),
        "valid_mask": np.load(
            _resolve(root, row["valid_mask_path"]), allow_pickle=False
        ).astype(bool),
        "confidence": np.load(
            _resolve(root, row["confidence_path"]), allow_pickle=False
        ),
        "K": np.load(_resolve(root, row["intrinsics_path"]), allow_pickle=False),
        "manifest": dict(row),
        "image_path": str(image_path),
    }


def _pose_row(
    clip_id: str,
    video_id: str,
    requested_motion_class: str,
    pose: PairwisePoseObservation,
) -> dict[str, Any]:
    return {
        "video_id": video_id,
        "clip_id": clip_id,
        "requested_motion_class": requested_motion_class,
        "frame_t": pose.frame_t,
        "frame_t1": pose.frame_t1,
        "rotation": _json_safe(pose.rotation),
        "translation": _json_safe(pose.translation),
        "translation_norm_m": (
            float(np.linalg.norm(pose.translation))
            if pose.translation is not None
            else float("nan")
        ),
        "pose_convention": pose.pose_convention,
        "camera_to_world_or_world_to_camera": pose.camera_to_world_or_world_to_camera,
        "translation_scale_status": pose.translation_scale_status,
        "inlier_count": pose.inlier_count,
        "inlier_ratio": pose.inlier_ratio,
        "reprojection_error": pose.reprojection_error,
        "static_background_ratio": pose.static_background_ratio,
        "dynamic_foreground_ratio": pose.dynamic_foreground_ratio,
        "confidence": pose.confidence,
        "provider_status": pose.provider_status.value,
        "failure_reason": pose.failure_reason,
        "background_candidates": pose.background_candidates,
        "foreground_rejected": pose.foreground_rejected,
        "geometric_inliers": pose.geometric_inliers,
        "degeneracy_status": pose.degeneracy_status,
        "provider_name": pose.provider_name,
        "valid": pose.valid,
        "coordinate_frame_source": "camera_frame_metric",
        "coordinate_frame_target": "camera_frame_metric",
        "depth_unit": "meter",
        "depth_definition": "z_depth",
        "metric_depth_status": "model_predicted_not_sensor_truth",
        "authenticity_label_used": False,
    }


def _static_row(
    clip_id: str,
    video_id: str,
    pose: PairwisePoseObservation,
) -> dict[str, Any]:
    item = pose.static_verification
    if item is None:
        return {
            "video_id": video_id,
            "clip_id": clip_id,
            "frame_t": pose.frame_t,
            "frame_t1": pose.frame_t1,
            "verified_static": False,
            "valid": False,
            "failure_reason": "static_verification_not_attempted",
        }
    row = asdict(item)
    row.update({"video_id": video_id, "clip_id": clip_id, "valid": True})
    return row


def _backproject(
    uv: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    column = int(round(float(uv[0])))
    row = int(round(float(uv[1])))
    if (
        row < 0
        or column < 0
        or row >= depth.shape[0]
        or column >= depth.shape[1]
        or not valid_mask[row, column]
    ):
        return None, float("nan")
    z = float(depth[row, column])
    if not math.isfinite(z) or z <= 0.0:
        return None, float("nan")
    ray = np.linalg.inv(K) @ np.asarray([uv[0], uv[1], 1.0], dtype=float)
    return ray * z, z


def _d2_row(item: D2ResidualObservation) -> dict[str, Any]:
    row = asdict(item)
    row["visibility_status"] = item.visibility_status.value
    row["provider_status"] = item.provider_status.value
    return row


def _nearest_target_boundary(
    target_rows: list[dict[str, str]],
    predicted_uv: np.ndarray,
) -> tuple[np.ndarray | None, str]:
    if not target_rows:
        return None, "target_boundary_missing"
    points = np.asarray(
        [[float(row["u"]), float(row["v"])] for row in target_rows],
        dtype=float,
    )
    index = int(np.argmin(np.linalg.norm(points - predicted_uv[None, :], axis=1)))
    return points[index], target_rows[index]["point_id"]


def _funnel_template() -> dict[str, int]:
    return {
        "total": 0,
        "applicable": 0,
        "input_ready": 0,
        "attempted": 0,
        "valid": 0,
        "provider_failed": 0,
        "blocked": 0,
        "not_applicable": 0,
    }


def run_pose_d2_smoke(
    project_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Run M4 using persisted small M1/M2 artifacts and no model inference."""

    root = Path(project_root).resolve()
    config_file = _resolve(root, config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = _resolve(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    metric_root = _resolve(root, config["inputs"]["metric_provider_root"])
    scene_root = _resolve(root, config["inputs"]["metric_scene3d_root"])
    structural_root = _resolve(root, config["inputs"]["structural_dataset_root"])
    manifest = _load_manifest(root, metric_root)
    foreground_unions, _ = _load_mask_index(structural_root)
    thresholds = ShortBaselinePoseThresholds(**config["pose_thresholds"])
    provider = ShortBaselinePoseProvider(thresholds)
    boundary_rows = list(
        csv.DictReader(
            (scene_root / "boundary_point_audit.csv").open(
                newline="", encoding="utf-8"
            )
        )
    )
    boundaries: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in boundary_rows:
        if row.get("valid") == "True":
            boundaries[(row["video_id"], int(row["frame_index"]), row["track_id"])].append(
                row
            )

    pose_rows: list[dict[str, Any]] = []
    static_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    boundary_output_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    pair_poses: dict[tuple[str, int, int], PairwisePoseObservation] = {}
    frame_cache: dict[tuple[str, int], dict[str, Any]] = {}
    clip_pose_map: dict[str, list[PairwisePoseObservation]] = defaultdict(list)
    funnel_by_class: dict[str, dict[str, int]] = defaultdict(_funnel_template)

    for clip in config["smoke_clips"]:
        video_id = str(clip["video_id"])
        clip_id = str(clip["clip_id"])
        motion_class = str(clip["requested_motion_class"])
        frames = [int(value) for value in clip["frame_indices"]]
        for frame_t, frame_t1 in zip(frames, frames[1:]):
            funnel = funnel_by_class[motion_class]
            funnel["total"] += 1
            funnel["applicable"] += 1
            source_row = manifest.get((video_id, frame_t))
            target_row = manifest.get((video_id, frame_t1))
            if source_row is None or target_row is None:
                pose = provider.estimate_pair(
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    np.zeros((8, 8, 3), dtype=np.uint8),
                    frame_t=frame_t,
                    frame_t1=frame_t1,
                    K_source=None,
                    K_target=None,
                    source_depth_m=None,
                    source_depth_valid_mask=None,
                )
            else:
                funnel["input_ready"] += 1
                source = frame_cache.setdefault(
                    (video_id, frame_t), _frame_inputs(root, metric_root, source_row)
                )
                target = frame_cache.setdefault(
                    (video_id, frame_t1), _frame_inputs(root, metric_root, target_row)
                )
                pose = provider.estimate_pair(
                    source["image"],
                    target["image"],
                    frame_t=frame_t,
                    frame_t1=frame_t1,
                    K_source=source["K"],
                    K_target=target["K"],
                    source_depth_m=source["depth"],
                    source_depth_valid_mask=source["valid_mask"],
                    source_foreground_mask=foreground_unions.get((video_id, frame_t)),
                    target_foreground_mask=foreground_unions.get((video_id, frame_t1)),
                )
            funnel["attempted"] += 1
            if pose.valid:
                funnel["valid"] += 1
            elif pose.provider_status == PoseProviderStatus.PROVIDER_FAILED:
                funnel["provider_failed"] += 1
            elif pose.provider_status in {
                PoseProviderStatus.BLOCKED_BY_INTRINSICS,
                PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE,
            }:
                funnel["blocked"] += 1
            elif pose.provider_status == PoseProviderStatus.NOT_APPLICABLE:
                funnel["not_applicable"] += 1
            pair_poses[(video_id, frame_t, frame_t1)] = pose
            clip_pose_map[clip_id].append(pose)
            pose_rows.append(_pose_row(clip_id, video_id, motion_class, pose))
            static_rows.append(_static_row(clip_id, video_id, pose))

        alignments = build_clip_local_alignment(
            clip_id, frames, clip_pose_map[clip_id]
        )
        for item in alignments:
            alignment_rows.append(
                {
                    **asdict(item),
                    "T_clip_from_camera": _json_safe(item.T_clip_from_camera),
                    "world_frame_claimed": False,
                }
            )

    max_points = int(config["d2"]["maximum_background_points_per_pair"])
    max_boundaries = int(config["d2"]["maximum_boundary_points_per_object_pair"])
    occlusion_margin = float(config["d2"]["depth_occlusion_margin_m"])
    conflict_relative = float(
        config["d2"]["depth_conflict_relative_threshold"]
    )
    for clip in config["smoke_clips"]:
        video_id = str(clip["video_id"])
        clip_id = str(clip["clip_id"])
        frames = [int(value) for value in clip["frame_indices"]]
        for frame_t, frame_t1 in zip(frames, frames[1:]):
            pose = pair_poses[(video_id, frame_t, frame_t1)]
            source = frame_cache.get((video_id, frame_t))
            target = frame_cache.get((video_id, frame_t1))
            if source is None or target is None:
                item = compute_d2_projection_residual(
                    evidence_id=f"{clip_id}:{frame_t}:{frame_t1}:blocked",
                    evidence_type="point",
                    video_id=video_id,
                    clip_id=clip_id,
                    pose=pose,
                    source_point_camera_m=[0.0, 0.0, 1.0],
                    target_observed_uv=None,
                    K_target=np.eye(3),
                    image_width=8,
                    image_height=8,
                    point_id="blocked_input_control",
                )
                point_rows.append(_d2_row(item))
                continue

            track_rows = list(pose.metadata.get("track_rows", []))
            pnp_mask = np.asarray(
                pose.metadata.get("pnp_inlier_mask", [0] * len(track_rows)),
                dtype=bool,
            )
            if pnp_mask.shape[0] != len(track_rows):
                pnp_mask = np.zeros(len(track_rows), dtype=bool)
            selected_indices = np.flatnonzero(pnp_mask)[:max_points]
            if not pose.valid and not selected_indices.size:
                selected_indices = np.arange(min(len(track_rows), 1))
            for index in selected_indices:
                track = track_rows[int(index)]
                source_uv = np.asarray(
                    [track["source_x"], track["source_y"]], dtype=float
                )
                target_uv = np.asarray(
                    [track["target_x"], track["target_y"]], dtype=float
                )
                xyz, _ = _backproject(
                    source_uv, source["depth"], source["valid_mask"], source["K"]
                )
                if xyz is None:
                    xyz = np.asarray([float("nan")] * 3)
                item = compute_d2_projection_residual(
                    evidence_id=f"{clip_id}:bg:{frame_t}:{frame_t1}:{index}",
                    evidence_type="point",
                    video_id=video_id,
                    clip_id=clip_id,
                    pose=pose,
                    source_point_camera_m=xyz,
                    target_observed_uv=target_uv,
                    K_target=target["K"],
                    image_width=target["image"].shape[1],
                    image_height=target["image"].shape[0],
                    target_depth_m=target["depth"],
                    target_depth_valid_mask=target["valid_mask"],
                    point_id=str(track["track_id"]),
                    point_confidence=pose.confidence,
                    depth_occlusion_margin_m=occlusion_margin,
                    depth_conflict_relative_threshold=conflict_relative,
                )
                point_rows.append(_d2_row(item))

            tracks = sorted(
                {
                    key[2]
                    for key in boundaries
                    if key[0] == video_id and key[1] == frame_t
                }
                & {
                    key[2]
                    for key in boundaries
                    if key[0] == video_id and key[1] == frame_t1
                }
            )
            for track_id in tracks:
                source_boundaries = boundaries[(video_id, frame_t, track_id)][
                    :max_boundaries
                ]
                target_boundaries = boundaries[(video_id, frame_t1, track_id)]
                object_evidence: list[D2ResidualObservation] = []
                for boundary in source_boundaries:
                    xyz = np.asarray(
                        [boundary["x_m"], boundary["y_m"], boundary["z_m"]],
                        dtype=float,
                    )
                    target_xyz = (
                        pose.T_target_from_source
                        @ np.asarray([xyz[0], xyz[1], xyz[2], 1.0])
                    )[:3] if pose.T_target_from_source is not None else None
                    predicted_uv = None
                    if target_xyz is not None and target_xyz[2] > 0:
                        projected = target["K"] @ target_xyz
                        predicted_uv = projected[:2] / projected[2]
                    target_uv, target_point_id = (
                        _nearest_target_boundary(target_boundaries, predicted_uv)
                        if predicted_uv is not None
                        else (None, "")
                    )
                    item = compute_d2_projection_residual(
                        evidence_id=f"{clip_id}:boundary:{boundary['point_id']}",
                        evidence_type="boundary",
                        video_id=video_id,
                        clip_id=clip_id,
                        pose=pose,
                        source_point_camera_m=xyz,
                        target_observed_uv=target_uv,
                        K_target=target["K"],
                        image_width=target["image"].shape[1],
                        image_height=target["image"].shape[0],
                        target_depth_m=target["depth"],
                        target_depth_valid_mask=target["valid_mask"],
                        object_id=boundary["object_id"],
                        track_id=track_id,
                        point_id=boundary["point_id"],
                        point_confidence=float(boundary["confidence"]),
                        depth_occlusion_margin_m=occlusion_margin,
                        depth_conflict_relative_threshold=conflict_relative,
                    )
                    item.metadata["target_boundary_point_id"] = target_point_id
                    object_evidence.append(item)
                    boundary_output_rows.append(_d2_row(item))
                aggregate = aggregate_object_d2_residual(
                    object_evidence,
                    evidence_id=f"{clip_id}:object:{track_id}:{frame_t}:{frame_t1}",
                    video_id=video_id,
                    clip_id=clip_id,
                    pose=pose,
                    object_id=source_boundaries[0]["object_id"] if source_boundaries else "",
                    track_id=track_id,
                )
                object_rows.append(_d2_row(aggregate))

    _write_csv(output / "pairwise_pose_audit.csv", pose_rows)
    _write_csv(output / "static_verification_audit.csv", static_rows)
    _write_csv(output / "clip_local_alignment_audit.csv", alignment_rows)
    _write_csv(output / "d2_point_residual_audit.csv", point_rows)
    _write_csv(output / "d2_boundary_residual_audit.csv", boundary_output_rows)
    _write_csv(output / "d2_object_residual_audit.csv", object_rows)

    valid_pose_rows = [row for row in pose_rows if row["valid"]]
    status_counts = Counter(row["provider_status"] for row in pose_rows)
    pose_quality = {
        "pair_count": len(pose_rows),
        "valid_pair_count": len(valid_pose_rows),
        "provider_status_counts": dict(sorted(status_counts.items())),
        "mean_inlier_ratio": (
            float(np.mean([row["inlier_ratio"] for row in valid_pose_rows]))
            if valid_pose_rows
            else None
        ),
        "mean_reprojection_error_px": (
            float(np.mean([row["reprojection_error"] for row in valid_pose_rows]))
            if valid_pose_rows
            else None
        ),
        "mean_pose_confidence": (
            float(np.mean([row["confidence"] for row in valid_pose_rows]))
            if valid_pose_rows
            else None
        ),
        "metric_depth_is_sensor_ground_truth": False,
        "authenticity_performance_evaluated": False,
    }
    _write_json(output / "pose_quality_audit.json", pose_quality)
    _write_json(
        output / "d2_eligibility_funnel.json",
        {
            "by_requested_motion_class": dict(sorted(funnel_by_class.items())),
            "status_semantics": {
                "blocked": "required geometric input unavailable",
                "provider_failed": "provider attempted but failed",
                "not_applicable": "geometry branch does not apply",
                "valid": "pose accepted for geometric diagnostics only",
            },
        },
    )
    _write_json(
        output / "pose_provider_manifest.json",
        {
            "stage": "P4-C3B-M4",
            "provider_name": provider.provider_name,
            "provider_version": provider.provider_version,
            "algorithm": "foreground-filtered LK correspondences plus source metric-depth PnP",
            "pose_convention": "X_target_camera=T_target_from_source@X_source_camera",
            "translation_scale_status": "metric_model_depth_or_zero_static",
            "depth_scale_semantics": "model_predicted_metric_not_sensor_truth",
            "intrinsics_source": "model_predicted_per_frame",
            "foreground_policy": "formal visible instance masks excluded before pose estimation",
            "identity_policy": "allowed only after multi-evidence static verification",
            "thresholds": asdict(thresholds),
            "config_path": str(config_file.relative_to(root)),
            "config_sha256": _sha256(config_file),
            "software_commit": _git_commit(root),
            "model_inference_executed": False,
            "authenticity_labels_used": False,
        },
    )

    d2_all = point_rows + boundary_output_rows + object_rows
    valid_point = sum(bool(row["valid"]) for row in point_rows)
    valid_boundary = sum(bool(row["valid"]) for row in boundary_output_rows)
    valid_object = sum(bool(row["valid"]) for row in object_rows)
    camera_motion_verified = any(
        row["requested_motion_class"] == "camera_motion"
        and row["provider_status"] == PoseProviderStatus.ESTIMATED_VALID.value
        for row in pose_rows
    ) and any(
        row["video_id"] == "real_1" and bool(row["valid"]) for row in d2_all
    )
    static_verified = any(
        row["requested_motion_class"] == "verified_static"
        and row["provider_status"] == PoseProviderStatus.VERIFIED_STATIC.value
        for row in pose_rows
    ) and any(
        row["video_id"] == "fake_1" and bool(row["valid"]) for row in d2_all
    )
    unreliable_rows = [
        row for row in pose_rows if row["requested_motion_class"] == "motion_unreliable"
    ]
    unreliable_handled = bool(unreliable_rows) and all(
        not bool(row["valid"]) for row in unreliable_rows
    )
    statuses = {
        "pose_provider_real_execution": len(pose_rows) > len(unreliable_rows),
        "verified_static_supported": any(
            row["provider_status"] == PoseProviderStatus.VERIFIED_STATIC.value
            for row in pose_rows
        ),
        "identity_pose_guard_verified": all(
            row["provider_status"] != PoseProviderStatus.VERIFIED_STATIC.value
            or next(
                (
                    bool(item.get("verified_static"))
                    and int(item.get("evidence_count", 0)) >= 3
                    for item in static_rows
                    if item["video_id"] == row["video_id"]
                    and item.get("source_frame_index") == row["frame_t"]
                    and item.get("target_frame_index") == row["frame_t1"]
                ),
                False,
            )
            for row in pose_rows
        ),
        "pairwise_pose_quality_verified": bool(valid_pose_rows),
        "clip_local_alignment_complete": any(
            bool(row["valid"]) and row["frame_index"] != row["reference_frame_index"]
            for row in alignment_rows
        ),
        "d2_real_video_executed": bool(d2_all),
        "d2_camera_motion_verified": camera_motion_verified,
        "d2_static_verified": static_verified,
        "d2_motion_unreliable_handled": unreliable_handled,
        "world_frame_reconstruction_complete": False,
        "ready_for_d3_implementation": camera_motion_verified and static_verified,
        "method_effectiveness_established": False,
    }
    validation = {
        **statuses,
        "valid_pose_pairs": len(valid_pose_rows),
        "valid_d2_point_count": valid_point,
        "valid_d2_boundary_count": valid_boundary,
        "valid_d2_object_count": valid_object,
        "d2_visibility_counts": dict(
            sorted(Counter(row["visibility_status"] for row in d2_all).items())
        ),
        "no_invalid_residual_encoded_as_zero": all(
            bool(row["valid"])
            or all(
                isinstance(row[name], float) and math.isnan(row[name])
                for name in (
                    "point_reprojection_residual",
                    "boundary_reprojection_residual",
                    "depth_reprojection_residual",
                    "object_reprojection_residual",
                )
            )
            for row in d2_all
        ),
        "config_sha256": _sha256(config_file),
        "software_commit": _git_commit(root),
    }
    _write_json(output / "validation_report.json", validation)
    _write_json(
        output / "blocked_features.json",
        {
            "world_frame_reconstruction": {
                "status": "not_implemented",
                "reason": "M4 is limited to short clip_local_aligned coordinates.",
            },
            "long_term_mapping": {
                "status": "not_applicable",
                "reason": "Explicitly outside P4-C3B-M4 scope.",
            },
            "motion_unreliable_control": {
                "status": "blocked_by_intrinsics",
                "reason": "No trustworthy persisted metric depth and K; no pose or high residual fabricated.",
            },
            "method_effectiveness": {
                "status": "not_established",
                "reason": "No labels, training, thresholding, or authenticity performance evaluation used.",
            },
        },
    )
    report_lines = [
        "# P4-C3B-M4 Pose and D2 Smoke Report",
        "",
        "This smoke estimates adjacent-frame pose from foreground-filtered background "
        "tracks and model-predicted metric depth. It is geometry QA, not a forged-video "
        "decision and not sensor-ground-truth metrology.",
        "",
        "## Scope",
        "",
        "- Coordinates: `camera_frame_metric` inputs and `clip_local_aligned` output.",
        "- Pose convention: `X_target_camera = T_target_from_source @ X_source_camera`.",
        "- Identity is accepted only after multiple static checks; provider failure never "
        "falls back to identity.",
        "- `world_frame` reconstruction is not implemented.",
        "",
        "## Results",
        "",
        f"- Pose pairs: {len(pose_rows)} attempted, {len(valid_pose_rows)} valid.",
        f"- D2 points: {valid_point}/{len(point_rows)} valid.",
        f"- D2 boundaries: {valid_boundary}/{len(boundary_output_rows)} valid.",
        f"- D2 objects: {valid_object}/{len(object_rows)} valid.",
        f"- Pose states: `{dict(sorted(status_counts.items()))}`.",
        f"- Visibility states: `{validation['d2_visibility_counts']}`.",
        "",
        "Invalid, occluded, out-of-frame, depth-conflicting and missing-correspondence "
        "measurements retain NaN and are not interpreted as high residuals.",
        "",
        "## Status",
        "",
    ]
    report_lines.extend(
        f"- `{name}`: `{str(value).lower()}`"
        for name, value in statuses.items()
    )
    report_lines.extend(
        [
            "",
            "## Limits",
            "",
            "- Metric depth and intrinsics are model predictions, not calibrated sensor truth.",
            "- Instance masks are visible masks, not amodal object support.",
            "- The smoke covers only adjacent pairs in two persisted short clips plus one "
            "blocked-input control.",
            "- No authenticity labels, learned distribution, threshold, or performance metric "
            "was used.",
            "",
        ]
    )
    (output / "POSE_D2_SMOKE_REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return validation
