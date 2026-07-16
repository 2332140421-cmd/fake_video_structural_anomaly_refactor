"""Load the immutable geometry cache emitted by the P3-0.5 smoke pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..depth_provider import (
    DepthObservation,
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from ..geometry.camera import CameraObservation
from ..sequence_geometry import (
    RelativePoseObservation,
    SequenceScaleStatus,
    Shared3DClipObservation,
)
from ..shared_3d_observation import Shared3DFrameObservation


@dataclass(frozen=True)
class SharedGeometryCache:
    """Rehydrated shared clip plus unaligned depth for QA comparisons."""

    clip: Shared3DClipObservation
    per_frame_geometry_depth: Mapping[int, np.ndarray]
    foreground_masks: Mapping[int, np.ndarray]
    frame_paths: Mapping[int, Path]
    geometry_quality: Mapping[str, Any]
    selected_pose_edges_by_target: Mapping[int, Mapping[str, Any]]
    manifest_path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "per_frame_geometry_depth",
            {int(key): np.asarray(value, dtype=float) for key, value in self.per_frame_geometry_depth.items()},
        )
        object.__setattr__(
            self,
            "foreground_masks",
            {int(key): np.asarray(value, dtype=bool) for key, value in self.foreground_masks.items()},
        )
        object.__setattr__(self, "frame_paths", {int(key): Path(value) for key, value in self.frame_paths.items()})
        object.__setattr__(
            self,
            "selected_pose_edges_by_target",
            {int(key): dict(value) for key, value in self.selected_pose_edges_by_target.items()},
        )
        object.__setattr__(self, "geometry_quality", dict(self.geometry_quality))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "metadata", dict(self.metadata))


def load_shared_geometry_cache(path: str | Path) -> SharedGeometryCache:
    """Load aligned depth, K, and pose without invoking any estimator."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("cache_version") != "p3_0_5_shared_geometry_v1":
        raise ValueError("Unsupported or missing shared geometry cache version.")
    indices = tuple(int(value) for value in payload["frame_indices"])
    K = np.asarray(payload["K"], dtype=float)
    width, height = int(payload["image_width"]), int(payload["image_height"])
    status = SequenceScaleStatus(payload["sequence_scale_status"])
    depth_scale = (
        DepthScaleStatus.METRIC_CALIBRATED
        if status == SequenceScaleStatus.METRIC_SEQUENCE
        else DepthScaleStatus.RELATIVE_SHARED_SEQUENCE
    )
    depth_representation = (
        DepthRepresentation.METRIC_DEPTH
        if status == SequenceScaleStatus.METRIC_SEQUENCE
        else DepthRepresentation.RELATIVE_DEPTH
    )
    twc_map = {
        int(key): None if value is None else np.asarray(value, dtype=float)
        for key, value in payload["T_world_from_camera_by_frame"].items()
    }
    tcw_map = {
        int(key): None if value is None else np.asarray(value, dtype=float)
        for key, value in payload["T_camera_from_world_by_frame"].items()
    }
    frames = []
    per_frame_depth = {}
    foreground_masks = {}
    for index in indices:
        record = payload["depth_records"][str(index)]
        aligned = np.load(record["aligned_geometry_depth_path"], allow_pickle=False)
        unaligned = np.load(record["per_frame_geometry_depth_path"], allow_pickle=False)
        saved_mask = np.load(record["valid_mask_path"], allow_pickle=False).astype(bool)
        foreground_masks[index] = np.load(
            record["foreground_mask_path"], allow_pickle=False
        ).astype(bool)
        valid = saved_mask & np.isfinite(aligned) & (aligned > 0.0)
        if not np.any(valid):
            raise ValueError(f"Cached aligned depth has no valid pixels for frame {index}.")
        depth = DepthObservation(
            depth_map=aligned,
            raw_model_output=None,
            visualization_depth=None,
            depth_representation=depth_representation,
            scale_status=depth_scale,
            larger_value_means=LargerValueMeans.FARTHER,
            valid_mask=valid,
            confidence_map=None,
            provider_name=f"cached:{record['provider_name']}",
            frame_index=index,
            valid=True,
            quality=float(record["quality"]),
            metadata={
                "cache_manifest": str(manifest_path),
                "sequence_aligned": True,
                "visualization_depth": False,
                "depth_reestimated": False,
            },
        )
        camera = CameraObservation.from_parameters(
            K=K,
            image_width=width,
            image_height=height,
            intrinsics_source=str(payload["intrinsics_source"]),
            quality=float(payload["camera_quality"]),
            T_world_from_camera=twc_map[index],
            T_camera_from_world=tcw_map[index],
            pose_source="cached_p3_0_5_pose_graph",
            metadata={
                "cache_manifest": str(manifest_path),
                "pose_reestimated": False,
            },
        )
        frames.append(
            Shared3DFrameObservation(
                video_id=str(payload["video_id"]),
                frame_index=index,
                image_width=width,
                image_height=height,
                camera=camera,
                depth=depth,
                objects=(),
                valid=True,
                quality=min(camera.quality, depth.quality),
                source_frame_id=Path(payload["frame_paths"][str(index)]).stem,
                metadata={"shared_geometry_cache": True},
            )
        )
        per_frame_depth[index] = unaligned
    edge_map = {
        int(key): dict(value)
        for key, value in payload.get("selected_pose_edges_by_target", {}).items()
    }
    relative_poses = []
    for position, index in enumerate(indices):
        twc = twc_map[index]
        tcw = tcw_map[index]
        if twc is None or tcw is None:
            relative_poses.append(
                RelativePoseObservation.missing(
                    target_frame_index=index,
                    source_frame_index=(indices[position - 1] if position else None),
                    reason="cached_pose_graph_disconnected",
                )
            )
            continue
        previous = indices[position - 1] if position else None
        relative = np.eye(4) if previous is None else tcw @ twc_map[previous]
        edge = edge_map.get(index, {})
        quality = float(edge.get("quality", payload["sequence_geometry_quality"]))
        support = int(edge.get("support_count", 0 if previous is None else 1))
        if previous is not None:
            support = max(support, 1)
        reprojection_error = float(
            edge.get(
                "reprojection_error",
                payload["geometry_quality"].get(
                    "background_reprojection_after_compensation_px", 0.0
                ),
            )
        )
        if not math.isfinite(reprojection_error):
            reprojection_error = 0.0
        relative_poses.append(
            RelativePoseObservation.from_transforms(
                source_frame_index=previous,
                target_frame_index=index,
                T_world_from_camera=twc,
                relative_pose_from_previous=relative,
                pose_source="cached_reference_gauge" if previous is None else "cached_p3_0_5_pose_graph",
                pose_quality=quality,
                background_support_count=support,
                background_inlier_ratio=float(edge.get("inlier_ratio", 1.0)),
                reprojection_error=reprojection_error,
                metadata={
                    "pose_model_type": edge.get("pose_model_type", "reference_gauge"),
                    "translation_scale_status": edge.get(
                        "translation_scale_status", "not_available"
                    ),
                    "pose_reestimated": False,
                },
            )
        )
    cuts = {int(key): bool(value) for key, value in payload["scene_cut_flags"].items()}
    clip = Shared3DClipObservation(
        video_id=str(payload["video_id"]),
        clip_id=str(payload["clip_id"]),
        frame_indices=indices,
        frames=tuple(frames),
        reference_frame_index=int(payload["reference_frame_index"]),
        T_world_from_camera_by_frame=twc_map,
        T_camera_from_world_by_frame=tcw_map,
        relative_poses=tuple(relative_poses),
        sequence_scale_status=status,
        depth_alignment_observations=(),
        scene_cut_flags=cuts,
        background_track_ids=(),
        foreground_object_ids=(),
        provider_name="p3_0_5_shared_geometry_cache",
        valid=True,
        quality=float(payload["sequence_geometry_quality"]),
        metadata={
            "pose_scale_compatible_with_depth": bool(
                payload["geometry_quality"].get("full_se3_selected_edge_count", 0)
            ),
            "cache_manifest": str(manifest_path),
            "depth_reestimated": False,
            "intrinsics_reestimated": False,
            "pose_reestimated": False,
        },
    )
    return SharedGeometryCache(
        clip=clip,
        per_frame_geometry_depth=per_frame_depth,
        foreground_masks=foreground_masks,
        frame_paths={int(key): Path(value) for key, value in payload["frame_paths"].items()},
        geometry_quality=payload["geometry_quality"],
        selected_pose_edges_by_target=edge_map,
        manifest_path=manifest_path,
        metadata=dict(payload.get("provenance", {})),
    )
