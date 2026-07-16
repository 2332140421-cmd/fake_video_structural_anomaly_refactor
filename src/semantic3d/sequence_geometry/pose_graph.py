"""Lightweight relative-pose graph with explicit disconnected-frame handling."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .pose_estimation import PoseEstimateCandidate, PoseModelType


@dataclass(frozen=True)
class PoseGraphResult:
    """Connected pose graph rooted at one sequence reference frame."""

    frame_indices: tuple[int, ...]
    reference_frame: int
    edges: tuple[PoseEstimateCandidate, ...]
    selected_edges: tuple[PoseEstimateCandidate, ...]
    T_world_from_camera_by_frame: Mapping[int, Optional[np.ndarray]]
    T_camera_from_world_by_frame: Mapping[int, Optional[np.ndarray]]
    connected_component_id: Mapping[int, int]
    selected_reference_frame: Mapping[int, Optional[int]]
    connected_frame_ratio: float
    valid_edge_ratio: float
    pose_chain_length: int
    disconnected_frames: tuple[int, ...]
    pose_graph_quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.frame_indices)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("Pose graph frame_indices must be non-empty and unique.")
        if self.reference_frame not in indices:
            raise ValueError("Pose graph reference_frame must belong to frame_indices.")
        for name in ("connected_frame_ratio", "valid_edge_ratio", "pose_graph_quality"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
            object.__setattr__(self, name, value)
        if self.pose_chain_length < 0:
            raise ValueError("pose_chain_length must be non-negative.")
        if self.valid and self.missing_reason:
            raise ValueError("Valid pose graph cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid pose graph requires missing_reason.")
        twc: dict[int, Optional[np.ndarray]] = {}
        tcw: dict[int, Optional[np.ndarray]] = {}
        for index in indices:
            world = self.T_world_from_camera_by_frame.get(index)
            camera = self.T_camera_from_world_by_frame.get(index)
            if (world is None) != (camera is None):
                raise ValueError("Pose graph world/camera transforms must be missing together.")
            if world is not None and camera is not None:
                world = np.asarray(world, dtype=float)
                camera = np.asarray(camera, dtype=float)
                if world.shape != (4, 4) or camera.shape != (4, 4):
                    raise ValueError("Pose graph transforms must be 4x4.")
                if not np.allclose(world @ camera, np.eye(4), atol=1e-6):
                    raise ValueError("Pose graph transforms must be mutual inverses.")
            twc[index], tcw[index] = world, camera
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "selected_edges", tuple(self.selected_edges))
        object.__setattr__(self, "T_world_from_camera_by_frame", twc)
        object.__setattr__(self, "T_camera_from_world_by_frame", tcw)
        object.__setattr__(self, "connected_component_id", dict(self.connected_component_id))
        object.__setattr__(self, "selected_reference_frame", dict(self.selected_reference_frame))
        object.__setattr__(self, "disconnected_frames", tuple(self.disconnected_frames))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the graph without converting missing transforms to identity."""

        return {
            "frame_indices": list(self.frame_indices),
            "reference_frame": self.reference_frame,
            "edges": [edge.to_dict() for edge in self.edges],
            "selected_edges": [edge.to_dict() for edge in self.selected_edges],
            "T_world_from_camera_by_frame": {
                str(index): None if value is None else value.tolist()
                for index, value in self.T_world_from_camera_by_frame.items()
            },
            "T_camera_from_world_by_frame": {
                str(index): None if value is None else value.tolist()
                for index, value in self.T_camera_from_world_by_frame.items()
            },
            "connected_component_id": {
                str(index): value for index, value in self.connected_component_id.items()
            },
            "selected_reference_frame": {
                str(index): value for index, value in self.selected_reference_frame.items()
            },
            "connected_frame_ratio": self.connected_frame_ratio,
            "valid_edge_ratio": self.valid_edge_ratio,
            "pose_chain_length": self.pose_chain_length,
            "disconnected_frames": list(self.disconnected_frames),
            "pose_graph_quality": self.pose_graph_quality,
            "valid": self.valid,
            "missing_reason": self.missing_reason,
            "metadata": dict(self.metadata),
        }


def _component_ids(
    frame_indices: Sequence[int], edges: Sequence[PoseEstimateCandidate]
) -> dict[int, int]:
    adjacency: dict[int, set[int]] = {index: set() for index in frame_indices}
    for edge in edges:
        if not edge.valid:
            continue
        adjacency[edge.source_frame_index].add(edge.target_frame_index)
        adjacency[edge.target_frame_index].add(edge.source_frame_index)
    components: dict[int, int] = {}
    component_id = 0
    for start in frame_indices:
        if start in components:
            continue
        queue = deque([start])
        components[start] = component_id
        while queue:
            current = queue.popleft()
            for neighbour in adjacency[current]:
                if neighbour not in components:
                    components[neighbour] = component_id
                    queue.append(neighbour)
        component_id += 1
    return components


def build_pose_graph(
    frame_indices: Sequence[int],
    candidates: Sequence[PoseEstimateCandidate],
    *,
    reference_frame: Optional[int] = None,
    minimum_edge_quality: float = 0.15,
) -> PoseGraphResult:
    """Propagate valid adjacent or skip-frame edges from a reference gauge."""

    indices = tuple(int(index) for index in frame_indices)
    if not indices:
        raise ValueError("frame_indices cannot be empty.")
    reference = indices[0] if reference_frame is None else int(reference_frame)
    if reference not in indices:
        raise ValueError("reference_frame must belong to frame_indices.")
    candidate_tuple = tuple(candidates)
    valid_edges = tuple(
        edge
        for edge in candidate_tuple
        if edge.valid
        and edge.quality >= minimum_edge_quality
        and edge.source_frame_index in indices
        and edge.target_frame_index in indices
    )
    twc: dict[int, Optional[np.ndarray]] = {index: None for index in indices}
    tcw: dict[int, Optional[np.ndarray]] = {index: None for index in indices}
    selected_reference: dict[int, Optional[int]] = {index: None for index in indices}
    distance: dict[int, int] = {reference: 0}
    twc[reference] = np.eye(4, dtype=float)
    tcw[reference] = np.eye(4, dtype=float)
    selected_edges: list[PoseEstimateCandidate] = []
    unresolved = {index for index in indices if index != reference}
    while unresolved:
        progress = False
        for target in tuple(index for index in indices if index in unresolved):
            options: list[tuple[int, float, int, np.ndarray, PoseEstimateCandidate]] = []
            for edge in valid_edges:
                assert edge.T_target_from_source is not None
                if edge.target_frame_index == target and twc[edge.source_frame_index] is not None:
                    options.append(
                        (
                            abs(edge.target_frame_index - edge.source_frame_index),
                            -edge.quality,
                            edge.source_frame_index,
                            edge.T_target_from_source,
                            edge,
                        )
                    )
                elif edge.source_frame_index == target and twc[edge.target_frame_index] is not None:
                    options.append(
                        (
                            abs(edge.target_frame_index - edge.source_frame_index),
                            -edge.quality,
                            edge.target_frame_index,
                            np.linalg.inv(edge.T_target_from_source),
                            edge,
                        )
                    )
            if not options:
                continue
            _, _, source, target_from_source, edge = min(
                options, key=lambda item: (item[0], item[1])
            )
            assert twc[source] is not None
            # X_target = T_target_from_source X_source, therefore
            # T_world_from_target = T_world_from_source inv(T_target_from_source).
            target_twc = twc[source] @ np.linalg.inv(target_from_source)
            twc[target] = target_twc
            tcw[target] = np.linalg.inv(target_twc)
            selected_reference[target] = source
            distance[target] = distance[source] + 1
            selected_edges.append(edge)
            unresolved.remove(target)
            progress = True
        if not progress:
            break
    connected = tuple(index for index in indices if twc[index] is not None)
    disconnected = tuple(index for index in indices if twc[index] is None)
    components = _component_ids(indices, valid_edges)
    connected_ratio = len(connected) / len(indices)
    valid_edge_ratio = len(valid_edges) / len(candidate_tuple) if candidate_tuple else 0.0
    chain_length = max(distance.values()) if distance else 0
    selected_quality = (
        float(np.mean([edge.quality for edge in selected_edges]))
        if selected_edges
        else (1.0 if len(indices) == 1 else 0.0)
    )
    graph_quality = float(connected_ratio * selected_quality)
    valid = connected_ratio == 1.0
    return PoseGraphResult(
        frame_indices=indices,
        reference_frame=reference,
        edges=candidate_tuple,
        selected_edges=tuple(selected_edges),
        T_world_from_camera_by_frame=twc,
        T_camera_from_world_by_frame=tcw,
        connected_component_id=components,
        selected_reference_frame=selected_reference,
        connected_frame_ratio=connected_ratio,
        valid_edge_ratio=valid_edge_ratio,
        pose_chain_length=chain_length,
        disconnected_frames=disconnected,
        pose_graph_quality=graph_quality,
        valid=valid,
        missing_reason="" if valid else "pose_graph_disconnected",
        metadata={
            "reference_is_coordinate_gauge": True,
            "missing_edges_filled_with_identity": False,
            "robust_optimization_applied": False,
            "selected_edge_count": len(selected_edges),
            "rotation_only_edge_count": sum(
                edge.pose_model_type == PoseModelType.ROTATION_HOMOGRAPHY
                for edge in selected_edges
            ),
            "full_se3_edge_count": sum(edge.full_se3 for edge in selected_edges),
        },
    )
