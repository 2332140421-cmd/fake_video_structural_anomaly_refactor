"""Offline M5 D3, occlusion, and reappearance smoke and audit generation."""

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

import numpy as np
import pandas as pd
import yaml

from ..method_completion.d3_relations import D3RelationType
from ..pose_d2.contracts import PoseProviderStatus
from .contracts import (
    D3FrameGraph,
    D3GraphNode,
    D3NodeType,
    D3RelationResidual,
    D3TransitionContext,
)
from .events import (
    OcclusionEventEvidence,
    OcclusionEventInputsV2,
    OcclusionEventType,
    ReappearanceResidual,
    classify_occlusion_event,
    compute_reappearance_residual,
)
from .graph import build_d3_frame_graph
from .residuals import D3StructureResidualExecutor


def _safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, ensure_ascii=False)
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
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: (
                        json.dumps(_safe(row.get(key)), sort_keys=True)
                        if isinstance(row.get(key), (dict, list, tuple))
                        else row.get(key, "")
                    )
                    for key in fields
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _parse_vector(value: Any) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(item) for item in json.loads(value))
    return tuple(float(item) for item in value)


def _transform_point(transform: np.ndarray, xyz: Iterable[float]) -> tuple[float, float, float]:
    point = np.asarray([*xyz, 1.0], dtype=float)
    transformed = transform @ point
    return tuple(float(value) for value in transformed[:3])


def _node_row_counts(graph: D3FrameGraph) -> dict[str, int]:
    return {
        node_type.value: sum(node.node_type == node_type for node in graph.nodes)
        for node_type in D3NodeType
    }


def _relation_row_counts(graph: D3FrameGraph) -> dict[str, int]:
    return {
        relation_type.value: sum(
            relation.relation_type == relation_type for relation in graph.relations
        )
        for relation_type in D3RelationType
    }


def _load_persisted_graphs(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, int], D3FrameGraph],
    dict[tuple[str, int, int], D3TransitionContext],
    list[dict[str, Any]],
    set[tuple[str, int, str]],
]:
    scene_root = root / config["inputs"]["metric_scene3d_root"]
    pose_root = root / config["inputs"]["pose_d2_root"]
    extents = pd.read_csv(scene_root / "object_extent_audit.csv")
    pointclouds = pd.read_csv(scene_root / "object_pointcloud_audit.csv")
    boundaries = pd.read_csv(scene_root / "boundary_point_audit.csv")
    internals = pd.read_csv(scene_root / "internal_structure_point_audit.csv")
    poses = pd.read_csv(pose_root / "pairwise_pose_audit.csv")
    alignments = pd.read_csv(pose_root / "clip_local_alignment_audit.csv")
    quality_index = {
        (row.video_id, int(row.frame_index), row.track_id): (
            float(row.mask_quality),
            float(row.depth_quality),
        )
        for row in pointclouds.itertuples()
        if bool(row.valid)
    }
    alignment_index: dict[tuple[str, int], tuple[np.ndarray, str, float]] = {}
    for row in alignments.itertuples():
        if not bool(row.valid) or not isinstance(row.T_clip_from_camera, str):
            continue
        alignment_index[(row.clip_id, int(row.frame_index))] = (
            np.asarray(json.loads(row.T_clip_from_camera), dtype=float),
            str(row.pose_source),
            float(row.confidence),
        )
    pose_index = {
        (row.clip_id, int(row.frame_t), int(row.frame_t1)): row
        for row in poses.itertuples()
    }
    graphs: dict[tuple[str, int], D3FrameGraph] = {}
    graph_rows: list[dict[str, Any]] = []
    unstable_tracks: set[tuple[str, int, str]] = set()
    max_boundary = int(config["graph"]["maximum_boundary_nodes_per_object"])
    max_internal = int(config["graph"]["maximum_internal_nodes_per_object"])
    for clip in config["persisted_video_clips"]:
        video_id = str(clip["video_id"])
        clip_id = str(clip["clip_id"])
        for frame_index in (int(value) for value in clip["frame_indices"]):
            alignment = alignment_index.get((clip_id, frame_index))
            if alignment is None:
                continue
            transform, pose_source, alignment_quality = alignment
            frame_extent = extents[
                (extents.video_id == video_id)
                & (extents.frame_index == frame_index)
                & (extents.valid == True)  # noqa: E712
            ]
            nodes: list[D3GraphNode] = []
            for row in frame_extent.itertuples():
                centroid = _parse_vector(row.robust_centroid_m)
                mask_quality, depth_quality = quality_index.get(
                    (video_id, frame_index, row.track_id), (0.0, 0.0)
                )
                confidence = min(mask_quality, depth_quality, alignment_quality)
                nodes.append(
                    D3GraphNode(
                        node_id=f"object:{row.track_id}",
                        node_type=D3NodeType.OBJECT_NODE,
                        frame_index=frame_index,
                        object_id=row.object_id,
                        track_id=row.track_id,
                        semantic_label=row.class_name,
                        xyz_m=_transform_point(transform, centroid),
                        coordinate_frame="clip_local_aligned",
                        source_observation_id=f"{row.frame_id}:{row.object_id}",
                        confidence=confidence,
                        identity_reliable=True,
                        visibility="visible",
                        valid=True,
                        localization_reference={
                            "level": "object_mask",
                            "object_id": row.object_id,
                            "track_id": row.track_id,
                            "frame_index": frame_index,
                        },
                        provenance={
                            "source_coordinate_frame": "camera_frame_metric",
                            "pose_source": pose_source,
                            "depth_provider": "unidepth_v2_vits14",
                            "depth_unit": "meter",
                            "depth_definition": "z_depth",
                            "sensor_ground_truth": False,
                        },
                    )
                )
                selected_boundaries = boundaries[
                    (boundaries.video_id == video_id)
                    & (boundaries.frame_index == frame_index)
                    & (boundaries.track_id == row.track_id)
                    & (boundaries.valid == True)  # noqa: E712
                ].sort_values("boundary_order").head(max_boundary)
                for boundary in selected_boundaries.itertuples():
                    nodes.append(
                        D3GraphNode(
                            node_id=f"boundary:{row.track_id}:{int(boundary.boundary_order):04d}",
                            node_type=D3NodeType.BOUNDARY_NODE,
                            frame_index=frame_index,
                            object_id=row.object_id,
                            track_id=row.track_id,
                            semantic_label=row.class_name,
                            xyz_m=_transform_point(
                                transform, (boundary.x_m, boundary.y_m, boundary.z_m)
                            ),
                            coordinate_frame="clip_local_aligned",
                            source_observation_id=boundary.point_id,
                            confidence=min(float(boundary.confidence), alignment_quality),
                            identity_reliable=False,
                            visibility="visible",
                            valid=True,
                            localization_reference={
                                "level": "boundary_point",
                                "u": float(boundary.u),
                                "v": float(boundary.v),
                                "object_id": row.object_id,
                            },
                            provenance={
                                "visible_mask_boundary": True,
                                "amodal_boundary": False,
                                "cross_frame_identity_verified": False,
                            },
                        )
                    )
                selected_internal = internals[
                    (internals.video_id == video_id)
                    & (internals.frame_index == frame_index)
                    & (internals.track_id == row.track_id)
                    & (internals.valid == True)  # noqa: E712
                ].head(max_internal)
                for index, point in enumerate(selected_internal.itertuples()):
                    nodes.append(
                        D3GraphNode(
                            node_id=f"geometric:{row.track_id}:{index:04d}",
                            node_type=D3NodeType.GEOMETRIC_TRACK_NODE,
                            frame_index=frame_index,
                            object_id=row.object_id,
                            track_id=row.track_id,
                            semantic_label=row.class_name,
                            xyz_m=_transform_point(
                                transform, (point.x_m, point.y_m, point.z_m)
                            ),
                            coordinate_frame="clip_local_aligned",
                            source_observation_id=point.point_id,
                            confidence=min(float(point.confidence), alignment_quality),
                            identity_reliable=False,
                            visibility="visible",
                            valid=True,
                            localization_reference={
                                "level": "geometric_track_candidate",
                                "u": float(point.u),
                                "v": float(point.v),
                                "object_id": row.object_id,
                            },
                            provenance={
                                "semantic_keypoint": False,
                                "cross_frame_trackability_verified": False,
                            },
                        )
                    )
                    unstable_tracks.add((video_id, frame_index, row.track_id))
            graph = build_d3_frame_graph(
                graph_id=f"{clip_id}:{frame_index}:d3_graph",
                video_id=video_id,
                clip_id=clip_id,
                frame_index=frame_index,
                nodes=nodes,
                pose_source=pose_source,
                fixed_structure_edges=(),
            )
            graphs[(clip_id, frame_index)] = graph
            graph_rows.append(
                {
                    "source_kind": "persisted_video",
                    "graph_id": graph.graph_id,
                    "video_id": video_id,
                    "clip_id": clip_id,
                    "frame_index": frame_index,
                    "coordinate_frame": graph.coordinate_frame,
                    "pose_source": graph.pose_source,
                    "node_count": len(graph.nodes),
                    "relation_count": len(graph.relations),
                    "node_type_counts": _node_row_counts(graph),
                    "relation_type_counts": _relation_row_counts(graph),
                    "quality": graph.quality,
                    "valid": graph.valid,
                    "failure_reason": graph.failure_reason,
                    "world_frame_claimed": False,
                }
            )
    contexts: dict[tuple[str, int, int], D3TransitionContext] = {}
    for clip in config["persisted_video_clips"]:
        clip_id = str(clip["clip_id"])
        video_id = str(clip["video_id"])
        frames = [int(value) for value in clip["frame_indices"]]
        for frame_t, frame_t1 in zip(frames, frames[1:]):
            pose = pose_index.get((clip_id, frame_t, frame_t1))
            pose_valid = bool(pose is not None and bool(pose.valid))
            status = (
                PoseProviderStatus(str(pose.provider_status))
                if pose is not None
                else PoseProviderStatus.UNKNOWN
            )
            graph_ready = (
                (clip_id, frame_t) in graphs and (clip_id, frame_t1) in graphs
            )
            valid = bool(pose_valid and status.usable_for_geometry and graph_ready)
            contexts[(clip_id, frame_t, frame_t1)] = D3TransitionContext(
                video_id=video_id,
                clip_id=clip_id,
                frame_t=frame_t,
                frame_t1=frame_t1,
                pose_status=status,
                pose_confidence=float(pose.confidence) if pose is not None else 0.0,
                pose_valid=pose_valid,
                correspondence_identity_reliable=graph_ready,
                source_coordinate_frame="clip_local_aligned",
                target_coordinate_frame="clip_local_aligned",
                valid=valid,
                failure_reason="" if valid else "blocked_by_pose_or_correspondence",
                metadata={"authenticity_label_used": False},
            )
    return graphs, contexts, graph_rows, unstable_tracks


def _synthetic_nodes(
    frame_index: int,
    *,
    object_b_xyz: tuple[float, float, float] = (1.0, 0.0, 4.0),
    object_b_orientation: tuple[float, float, float] = (1.0, 0.0, 0.0),
    deform: bool = False,
) -> list[D3GraphNode]:
    values = {
        "object:a": (0.0, 0.0, 3.0),
        "object:b": object_b_xyz,
        "p1": (-0.2, 0.0, 3.0),
        "p2": ((0.5 if deform else 0.2), 0.0, 3.0),
        "p3": (0.0, 0.3, 3.0),
        "boundary:a:0": (-0.3, 0.0, 3.0),
        "boundary:a:1": (0.3, 0.0, 3.0),
    }
    output = []
    for node_id, xyz in values.items():
        if node_id.startswith("object"):
            node_type = D3NodeType.OBJECT_NODE
            track_id = node_id.split(":")[1]
        elif node_id.startswith("boundary"):
            node_type = D3NodeType.BOUNDARY_NODE
            track_id = "a"
        else:
            node_type = D3NodeType.GEOMETRIC_TRACK_NODE
            track_id = "a"
        output.append(
            D3GraphNode(
                node_id=node_id,
                node_type=node_type,
                frame_index=frame_index,
                object_id=f"obj_{track_id}",
                track_id=track_id,
                semantic_label="synthetic_object",
                xyz_m=xyz,
                coordinate_frame="clip_local_aligned",
                source_observation_id=f"synthetic:{frame_index}:{node_id}",
                confidence=1.0,
                identity_reliable=True,
                visibility="visible",
                valid=True,
                orientation_vector=(
                    (1.0, 0.0, 0.0)
                    if node_id == "object:a"
                    else object_b_orientation
                    if node_id == "object:b"
                    else None
                ),
                localization_reference={"node_id": node_id, "frame_index": frame_index},
                provenance={"synthetic_geometry_truth": True},
            )
        )
    return output


def _synthetic_graph(
    graph_name: str,
    frame_index: int,
    *,
    clip_id: str = "synthetic_reference",
    object_b_xyz: tuple[float, float, float] = (1.0, 0.0, 4.0),
    object_b_orientation: tuple[float, float, float] = (1.0, 0.0, 0.0),
    deform: bool = False,
    overlap: tuple[float, float] = (0.1, 0.2),
    contact: tuple[float, float] = (0.02, 0.9),
) -> D3FrameGraph:
    return build_d3_frame_graph(
        graph_id=f"{graph_name}:{frame_index}",
        video_id="synthetic_d3",
        clip_id=clip_id,
        frame_index=frame_index,
        nodes=_synthetic_nodes(
            frame_index,
            object_b_xyz=object_b_xyz,
            object_b_orientation=object_b_orientation,
            deform=deform,
        ),
        pose_source="synthetic_known_pose",
        fixed_structure_edges=(("edge_12", "p1", "p2"), ("edge_13", "p1", "p3")),
        containment_relations={("a", "b"): overlap},
        support_contact_relations={("a", "b"): contact},
    )


def _context(
    clip_id: str,
    *,
    valid: bool = True,
) -> D3TransitionContext:
    return D3TransitionContext(
        video_id="synthetic_d3",
        clip_id=clip_id,
        frame_t=0,
        frame_t1=1,
        pose_status=(
            PoseProviderStatus.ESTIMATED_VALID
            if valid
            else PoseProviderStatus.BLOCKED_BY_CORRESPONDENCE
        ),
        pose_confidence=1.0 if valid else 0.0,
        pose_valid=valid,
        correspondence_identity_reliable=valid,
        source_coordinate_frame="clip_local_aligned",
        target_coordinate_frame="clip_local_aligned",
        valid=valid,
        failure_reason="" if valid else "blocked_by_pose_or_correspondence",
    )


def _event_inputs(name: str) -> OcclusionEventInputsV2:
    common = dict(
        video_id="synthetic_events",
        clip_id=name,
        frame_index=3,
        object_track_id="track_a",
        previous_event_type=OcclusionEventType.UNKNOWN,
        formal_visible_mask_available=True,
        history_prediction_available=True,
        observed_object_available=False,
        candidate_object_available=False,
        identity_consistent=True,
        mask_overlap_ratio=0.0,
        depth_order_supported=False,
        visible_ratio=float("nan"),
        predicted_in_frame_ratio=1.0,
        trajectory_prediction_quality=0.9,
        d2_reprojection_supported=True,
        detector_attempted=True,
        detector_reliable=True,
        detection_confirmed_absent=False,
        tracker_failed=False,
        persistent_absence_frames=0,
        possible_occluder_ids=(),
    )
    overrides: dict[str, Any] = {}
    if name == "partial_occlusion":
        overrides = {
            "observed_object_available": True,
            "candidate_object_available": True,
            "visible_ratio": 0.45,
            "mask_overlap_ratio": 0.55,
            "depth_order_supported": True,
            "possible_occluder_ids": ("track_b",),
        }
    elif name == "full_occlusion":
        overrides = {
            "mask_overlap_ratio": 0.90,
            "depth_order_supported": True,
            "possible_occluder_ids": ("track_b",),
        }
    elif name == "out_of_frame":
        overrides = {"predicted_in_frame_ratio": 0.05}
    elif name == "detector_miss":
        overrides = {"detector_reliable": False}
    elif name == "true_disappearance":
        overrides = {
            "detection_confirmed_absent": True,
            "persistent_absence_frames": 3,
        }
    elif name == "reappearance":
        overrides = {
            "previous_event_type": OcclusionEventType.FULL_OCCLUSION,
            "observed_object_available": True,
            "candidate_object_available": True,
            "visible_ratio": 0.9,
        }
    elif name == "id_switch":
        overrides = {
            "candidate_object_available": True,
            "observed_object_available": True,
            "identity_consistent": False,
            "visible_ratio": 0.9,
        }
    elif name == "no_event":
        overrides = {
            "observed_object_available": True,
            "candidate_object_available": True,
            "visible_ratio": 1.0,
        }
    return OcclusionEventInputsV2(**{**common, **overrides})


def _residual_row(item: D3RelationResidual, source_kind: str) -> dict[str, Any]:
    row = asdict(item)
    row["relation_type"] = item.relation_type.value
    row["source_kind"] = source_kind
    return row


def _event_row(item: OcclusionEventEvidence, source_kind: str) -> dict[str, Any]:
    row = asdict(item)
    row["event_type"] = item.event_type.value
    row["source_kind"] = source_kind
    return row


def _reappearance_row(item: ReappearanceResidual, source_kind: str) -> dict[str, Any]:
    return {**asdict(item), "source_kind": source_kind}


def _funnel(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    records = list(rows)
    def is_not_applicable(row: Mapping[str, Any]) -> bool:
        return bool(
            str(row.get("failure_reason", "")) == "not_applicable_no_event"
            or str(row.get("status", "")) == "not_applicable"
        )

    def is_provider_failed(row: Mapping[str, Any]) -> bool:
        return str(row.get("status", "")) == "provider_failed"

    def is_blocked(row: Mapping[str, Any]) -> bool:
        return bool(
            str(row.get("failure_reason", "")).startswith("blocked_by")
            or str(row.get("status", "")) == "blocked_by_input"
        )

    return {
        "total": len(records),
        "applicable": sum(not is_not_applicable(row) for row in records),
        "input_ready": sum(
            not is_not_applicable(row)
            and not is_provider_failed(row)
            and not is_blocked(row)
            for row in records
        ),
        "attempted": sum(
            not is_not_applicable(row)
            and not is_provider_failed(row)
            and not is_blocked(row)
            for row in records
        ),
        "valid": sum(_bool(row.get("valid", False)) for row in records),
        "provider_failed": sum(is_provider_failed(row) for row in records),
        "blocked": sum(is_blocked(row) for row in records),
        "not_applicable": sum(is_not_applicable(row) for row in records),
    }


def run_d3_occlusion_smoke(
    project_root: str | Path,
    config_path: str | Path = "configs/p4c3b_d3_occlusion_v1.yaml",
) -> dict[str, Any]:
    """Run persisted-video D3 plus controlled synthetic event validation."""

    root = Path(project_root).resolve()
    config_file = root / config_path
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    graphs, contexts, graph_rows, unstable_tracks = _load_persisted_graphs(
        root, config
    )
    executor = D3StructureResidualExecutor()
    residual_rows: list[dict[str, Any]] = []
    for key, context in sorted(contexts.items()):
        previous = graphs.get((context.clip_id, context.frame_t))
        current = graphs.get((context.clip_id, context.frame_t1))
        if previous is None or current is None:
            continue
        residuals = executor.compare_graphs(previous, current, context)
        residual_rows.extend(
            _residual_row(item, "persisted_video") for item in residuals
        )
        shared_unstable = {
            track
            for video, frame, track in unstable_tracks
            if video == context.video_id and frame in {context.frame_t, context.frame_t1}
        }
        for track_id in sorted(shared_unstable):
            for relation_type, name in (
                (D3RelationType.STRUCTURE_EDGE_LENGTH, "R_edge_length"),
                (D3RelationType.LOCAL_RIGIDITY, "R_local_rigidity"),
            ):
                residual_rows.append(
                    _residual_row(
                        D3RelationResidual(
                            residual_id=(
                                f"{context.clip_id}:{context.frame_t}:"
                                f"{context.frame_t1}:{relation_type.value}:{track_id}"
                            ),
                            residual_name=name,
                            relation_type=relation_type,
                            video_id=context.video_id,
                            clip_id=context.clip_id,
                            frame_t=context.frame_t,
                            frame_t1=context.frame_t1,
                            source_nodes=(track_id,),
                            source_edge=f"unverified_structure_graph:{track_id}",
                            coordinate_frame="clip_local_aligned",
                            reference_relation=(float("nan"),),
                            observed_relation=(float("nan"),),
                            value=float("nan"),
                            confidence=0.0,
                            valid=False,
                            failure_reason="blocked_by_pose_or_correspondence",
                            localization_reference={
                                "level": "object_structure_graph",
                                "track_id": track_id,
                            },
                            metadata={
                                "reason_detail": (
                                    "M2 geometric points are not verified across frames"
                                ),
                                "missing_is_zero": False,
                            },
                        ),
                        "persisted_video",
                    )
                )

    synthetic_current_graphs = {
        "rigid": {},
        "relative_distance_change": {"object_b_xyz": (2.0, 0.0, 4.0)},
        "depth_order_change": {"object_b_xyz": (1.0, 0.0, 2.0)},
        "edge_and_rigidity_change": {"deform": True},
        "orientation_change": {"object_b_orientation": (0.0, 1.0, 0.0)},
        "overlap_contact_change": {
            "overlap": (0.5, 0.7),
            "contact": (0.15, 0.4),
        },
    }
    synthetic_cases: dict[str, tuple[D3FrameGraph, D3FrameGraph]] = {}
    for name, kwargs in synthetic_current_graphs.items():
        clip_id = f"synthetic_{name}"
        synthetic_cases[name] = (
            _synthetic_graph(
                f"{clip_id}_reference", 0, clip_id=clip_id
            ),
            _synthetic_graph(
                f"{clip_id}_observed", 1, clip_id=clip_id, **kwargs
            ),
        )
    for graph_pair in synthetic_cases.values():
        for graph in graph_pair:
            graph_rows.append(
                {
                    "source_kind": "synthetic_control",
                    "graph_id": graph.graph_id,
                    "video_id": graph.video_id,
                    "clip_id": graph.clip_id,
                    "frame_index": graph.frame_index,
                    "coordinate_frame": graph.coordinate_frame,
                    "pose_source": graph.pose_source,
                    "node_count": len(graph.nodes),
                    "relation_count": len(graph.relations),
                    "node_type_counts": _node_row_counts(graph),
                    "relation_type_counts": _relation_row_counts(graph),
                    "quality": graph.quality,
                    "valid": graph.valid,
                    "failure_reason": graph.failure_reason,
                    "world_frame_claimed": False,
                }
            )
    synthetic_results: dict[str, list[D3RelationResidual]] = {}
    for name, (reference_graph, current_graph) in synthetic_cases.items():
        rows = list(
            executor.compare_graphs(
                reference_graph,
                current_graph,
                _context(reference_graph.clip_id),
            )
        )
        synthetic_results[name] = rows
        residual_rows.extend(_residual_row(item, "synthetic_control") for item in rows)
    rigid_reference, rigid_current = synthetic_cases["rigid"]
    blocked_rows = list(
        executor.compare_graphs(
            rigid_reference,
            rigid_current,
            _context("synthetic_pose_blocked", valid=False),
        )
    )
    synthetic_results["pose_blocked"] = blocked_rows
    residual_rows.extend(_residual_row(item, "synthetic_control") for item in blocked_rows)

    event_names = (
        "partial_occlusion",
        "full_occlusion",
        "out_of_frame",
        "detector_miss",
        "true_disappearance",
        "reappearance",
        "id_switch",
        "no_event",
    )
    events = {
        name: classify_occlusion_event(
            _event_inputs(name),
            **config["events"],
        )
        for name in event_names
    }
    event_rows = [_event_row(item, "synthetic_control") for item in events.values()]
    for context in contexts.values():
        previous_graph = graphs.get((context.clip_id, context.frame_t))
        current_graph = graphs.get((context.clip_id, context.frame_t1))
        if previous_graph is None or current_graph is None:
            continue
        previous_tracks = {
            node.track_id
            for node in previous_graph.nodes
            if node.node_type == D3NodeType.OBJECT_NODE
        }
        current_tracks = {
            node.track_id
            for node in current_graph.nodes
            if node.node_type == D3NodeType.OBJECT_NODE
        }
        for track_id in sorted(previous_tracks | current_tracks):
            observed_in_previous = track_id in previous_tracks
            observed_in_current = track_id in current_tracks
            event = classify_occlusion_event(
                OcclusionEventInputsV2(
                    video_id=context.video_id,
                    clip_id=context.clip_id,
                    frame_index=context.frame_t1,
                    object_track_id=track_id,
                    previous_event_type=OcclusionEventType.UNKNOWN,
                    formal_visible_mask_available=observed_in_previous,
                    history_prediction_available=bool(
                        context.valid and observed_in_previous
                    ),
                    observed_object_available=observed_in_current,
                    candidate_object_available=observed_in_current,
                    identity_consistent=bool(
                        observed_in_previous and observed_in_current
                    ),
                    mask_overlap_ratio=float("nan"),
                    depth_order_supported=False,
                    visible_ratio=(
                        1.0 if observed_in_previous and observed_in_current
                        else float("nan")
                    ),
                    predicted_in_frame_ratio=float("nan"),
                    trajectory_prediction_quality=context.pose_confidence,
                    d2_reprojection_supported=context.valid,
                    detector_attempted=True,
                    detector_reliable=True,
                    detection_confirmed_absent=False,
                    tracker_failed=False,
                    persistent_absence_frames=0,
                    metadata={
                        "derived_from_persisted_d3_object_tracks": True,
                        "event_claim_without_support": False,
                    },
                ),
                **config["events"],
            )
            event_rows.append(_event_row(event, "persisted_video"))

    valid_reappearance = compute_reappearance_residual(
        event=events["reappearance"],
        previous_track_id="track_a",
        current_track_id="track_a",
        identity_consistent=True,
        predicted_position_error_normalized=0.02,
        previous_depth_m=3.0,
        current_depth_m=3.05,
        previous_physical_scale_m=1.5,
        current_physical_scale_m=1.48,
        structure_change=0.03,
        motion_trend_change=0.04,
        confidence=0.9,
    )
    no_event_reappearance = compute_reappearance_residual(
        event=events["no_event"],
        previous_track_id="track_a",
        current_track_id="track_a",
        identity_consistent=True,
        predicted_position_error_normalized=0.0,
        previous_depth_m=3.0,
        current_depth_m=3.0,
        previous_physical_scale_m=1.5,
        current_physical_scale_m=1.5,
        structure_change=0.0,
        motion_trend_change=0.0,
        confidence=1.0,
    )
    reappearance_rows = [
        _reappearance_row(valid_reappearance, "synthetic_control"),
        _reappearance_row(no_event_reappearance, "synthetic_control"),
    ]

    def values(name: str, residual_name: str) -> list[float]:
        return [
            item.value
            for item in synthetic_results[name]
            if item.valid and item.residual_name == residual_name
        ]

    synthetic_validation = {
        "rigid_relations_zero": all(
            item.value <= 1e-12
            for item in synthetic_results["rigid"]
            if item.valid
        ),
        "relative_distance_change_positive": bool(
            values("relative_distance_change", "R_relative_distance")
            and max(values("relative_distance_change", "R_relative_distance")) > 0.0
        ),
        "depth_order_change_positive": bool(
            values("depth_order_change", "R_depth_order")
            and max(values("depth_order_change", "R_depth_order")) > 0.0
        ),
        "edge_length_change_positive": bool(
            values("edge_and_rigidity_change", "R_edge_length")
            and max(values("edge_and_rigidity_change", "R_edge_length")) > 0.0
        ),
        "local_rigidity_change_positive": bool(
            values("edge_and_rigidity_change", "R_local_rigidity")
            and max(values("edge_and_rigidity_change", "R_local_rigidity")) > 0.0
        ),
        "orientation_change_positive": bool(
            values("orientation_change", "R_relative_orientation")
            and max(values("orientation_change", "R_relative_orientation")) > 0.0
        ),
        "pose_blocked_all_nan": all(
            not item.valid and math.isnan(item.value) for item in blocked_rows
        ),
        "event_classifications": {
            name: item.event_type.value for name, item in events.items()
        },
        "no_event_not_applicable": (
            not events["no_event"].valid
            and events["no_event"].failure_reason == "not_applicable_no_event"
        ),
        "reappearance_valid": valid_reappearance.valid,
        "reappearance_no_event_nan": (
            not no_event_reappearance.valid
            and math.isnan(no_event_reappearance.combined_residual)
        ),
        "authenticity_performance_evaluated": False,
    }

    _write_csv(output / "d3_graph_manifest.csv", graph_rows)
    _write_csv(output / "d3_relation_residuals.csv", residual_rows)
    _write_csv(output / "occlusion_event_manifest.csv", event_rows)
    _write_csv(output / "reappearance_event_manifest.csv", reappearance_rows)
    _write_json(
        output / "event_classification_audit.json",
        {
            "event_counts": dict(
                sorted(Counter(row["event_type"] for row in event_rows).items())
            ),
            "valid_event_count": sum(_bool(row["valid"]) for row in event_rows),
            "persisted_video_formal_occlusion_event_count": sum(
                row["source_kind"] == "persisted_video"
                and row["event_type"]
                in {
                    OcclusionEventType.PARTIAL_OCCLUSION.value,
                    OcclusionEventType.FULL_OCCLUSION.value,
                }
                and _bool(row["valid"])
                for row in event_rows
            ),
            "no_event_is_zero_residual": False,
            "authenticity_labels_used": False,
        },
    )
    _write_json(
        output / "d3_eligibility_funnel.json",
        {
            "d3_relations": _funnel(residual_rows),
            "occlusion_events": _funnel(event_rows),
            "reappearance": _funnel(reappearance_rows),
        },
    )
    _write_json(output / "synthetic_event_validation.json", synthetic_validation)
    persisted_valid = [
        row
        for row in residual_rows
        if row["source_kind"] == "persisted_video" and _bool(row["valid"])
    ]
    formal_real_occlusions = sum(
        row["source_kind"] == "persisted_video"
        and row["event_type"]
        in {
            OcclusionEventType.PARTIAL_OCCLUSION.value,
            OcclusionEventType.FULL_OCCLUSION.value,
        }
        and _bool(row["valid"])
        for row in event_rows
    )
    statuses = {
        "d3_graph_implementation_complete": True,
        "d3_relation_residuals_synthetic_verified": all(
            bool(value)
            for key, value in synthetic_validation.items()
            if key
            in {
                "rigid_relations_zero",
                "relative_distance_change_positive",
                "depth_order_change_positive",
                "edge_length_change_positive",
                "local_rigidity_change_positive",
                "orientation_change_positive",
                "pose_blocked_all_nan",
            }
        ),
        "d3_real_video_executed": bool(persisted_valid),
        "occlusion_event_classifier_complete": True,
        "reappearance_residual_complete": valid_reappearance.valid,
        "formal_occlusion_events_available": formal_real_occlusions > 0,
        "ready_for_evidence_fusion": bool(persisted_valid)
        and synthetic_validation["reappearance_valid"],
        "method_effectiveness_established": False,
    }
    validation = {
        **statuses,
        "persisted_video_valid_d3_relation_count": len(persisted_valid),
        "persisted_video_blocked_structure_relation_count": sum(
            row["source_kind"] == "persisted_video"
            and row["failure_reason"] == "blocked_by_pose_or_correspondence"
            for row in residual_rows
        ),
        "synthetic_valid_d3_relation_count": sum(
            row["source_kind"] == "synthetic_control" and _bool(row["valid"])
            for row in residual_rows
        ),
        "formal_persisted_video_occlusion_event_count": formal_real_occlusions,
        "world_frame_reconstruction_complete": False,
        "config_sha256": _sha256(config_file),
        "software_commit": _commit(root),
    }
    _write_json(
        output / "blocked_features.json",
        {
            "persisted_video_internal_edge_identity": {
                "status": "blocked_by_pose_or_correspondence",
                "reason": (
                    "M2 geometric_track_point candidates do not have verified "
                    "cross-frame point identities."
                ),
            },
            "formal_real_occlusion_events": {
                "status": (
                    "executed_valid"
                    if formal_real_occlusions
                    else "not_applicable_no_event"
                ),
                "reason": (
                    ""
                    if formal_real_occlusions
                    else "No strict object-level partial/full event exists in the persisted M4 clips."
                ),
            },
            "world_frame": {
                "status": "not_implemented",
                "reason": "M5 remains clip_local_aligned.",
            },
            "method_effectiveness": {
                "status": "not_established",
                "reason": "No training, threshold selection, or authenticity evaluation.",
            },
        },
    )
    _write_json(output / "validation_report.json", validation)
    report = [
        "# P4-C3B-M5 D3, Occlusion, and Reappearance Report",
        "",
        "M5 implements pose-gated higher-order 3D relation diagnostics and conservative "
        "visibility events. It does not train a detector or evaluate authenticity performance.",
        "",
        "## D3",
        "",
        f"- Persisted-video valid D3 relations: {len(persisted_valid)}.",
        "- Real short-clip relations use `clip_local_aligned`; no world frame is claimed.",
        "- Object-track relations execute when M4 pose and track identity are valid.",
        "- Ordinary M2 structure points lack verified cross-frame identity, so real "
        "edge-length and local-rigidity rows remain blocked with NaN.",
        "",
        "## Visibility events",
        "",
        f"- Formal persisted-video partial/full occlusion events: {formal_real_occlusions}.",
        "- Controlled scenarios cover partial/full occlusion, out-of-frame, detector miss, "
        "true disappearance, reappearance, ID switch/track failure, and no-event.",
        "- No-event is `not_applicable_no_event`, never residual zero.",
        "",
        "## Status",
        "",
        *[
            f"- `{name}`: `{str(value).lower()}`"
            for name, value in statuses.items()
        ],
        "",
        "## Limits",
        "",
        "- Metric depth and intrinsics are model predictions, not sensor truth.",
        "- Masks are visible instance masks, not amodal masks.",
        "- Synthetic controls verify formulas and event logic only; they do not establish "
        "forged-video detection effectiveness.",
        "- Evidence fusion may consume valid/quality/localization fields, but missing, "
        "blocked, provider-failed, and not-applicable branches must remain masked.",
        "",
    ]
    (output / "D3_OCCLUSION_REPORT.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    return validation
