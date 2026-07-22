"""Fixed object structure graphs built once from stable point identities."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .object_track_binding import ObjectPointBinding, ObjectPointTrack3D, PointRole
from .readiness import DynamicGeometryMode
from .track_observation import PointTrack3DObservation


HUMAN_SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


@dataclass(frozen=True)
class StructureEdge:
    """One fixed edge whose point identities persist over time."""

    point_id_a: str
    point_id_b: str
    reference_length: float
    reference_normalized_length: float
    edge_quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        length = float(self.reference_length)
        normalized = float(self.reference_normalized_length)
        quality = float(self.edge_quality)
        if self.point_id_a == self.point_id_b:
            raise ValueError("A structure edge requires two different point IDs.")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("edge_quality must be in [0, 1].")
        if self.valid:
            if not all(math.isfinite(value) and value > 0.0 for value in (length, normalized)):
                raise ValueError("Valid structure edge lengths must be positive.")
            if self.missing_reason:
                raise ValueError("Valid structure edge cannot have missing_reason.")
        else:
            if not math.isnan(length) or not math.isnan(normalized) or not self.missing_reason:
                raise ValueError("Invalid structure edge requires NaN lengths and a reason.")
        object.__setattr__(self, "reference_length", length)
        object.__setattr__(self, "reference_normalized_length", normalized)
        object.__setattr__(self, "edge_quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ObjectStructureGraph:
    """Object graph fixed at one reference frame and reused without rewiring."""

    object_track_id: str
    semantic_label: str
    reference_frame_index: Optional[int]
    point_ids: tuple[str, ...]
    edges: tuple[StructureEdge, ...]
    graph_type: str
    geometry_mode: DynamicGeometryMode | str
    valid: bool
    quality: float
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = DynamicGeometryMode(self.geometry_mode)
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Graph quality must be in [0, 1].")
        if len(self.point_ids) != len(set(self.point_ids)):
            raise ValueError("Structure graph point IDs must be unique.")
        if self.valid:
            if self.reference_frame_index is None or not self.edges or self.missing_reason:
                raise ValueError("Valid graph requires a reference frame and edges.")
            point_ids = set(self.point_ids)
            if any(edge.point_id_a not in point_ids or edge.point_id_b not in point_ids for edge in self.edges):
                raise ValueError("Structure edge references an unknown point ID.")
        elif not self.missing_reason:
            raise ValueError("Invalid graph requires missing_reason.")
        object.__setattr__(self, "point_ids", tuple(self.point_ids))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "geometry_mode", mode)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


def _coordinate(point: PointTrack3DObservation) -> Optional[np.ndarray]:
    if not point.valid:
        return None
    if point.geometry_mode == DynamicGeometryMode.FULL_SE3_3D:
        value = point.point_3d_world
    elif point.geometry_mode == DynamicGeometryMode.STATIC_CAMERA_3D:
        value = point.point_3d_camera
    else:
        return None
    return None if value is None else np.asarray(value, dtype=float)


def _invalid_graph(
    object_track_id: str,
    semantic_label: str,
    mode: DynamicGeometryMode,
    reason: str,
) -> ObjectStructureGraph:
    return ObjectStructureGraph(
        object_track_id=object_track_id,
        semantic_label=semantic_label,
        reference_frame_index=None,
        point_ids=(),
        edges=(),
        graph_type="unavailable",
        geometry_mode=mode,
        valid=False,
        quality=0.0,
        missing_reason=reason,
    )


def build_object_structure_graph(
    tracks: Sequence[ObjectPointTrack3D],
    *,
    object_track_id: str,
    semantic_label: str,
    k_neighbors: int = 2,
) -> ObjectStructureGraph:
    """Build a fixed semantic skeleton or kNN graph at the first common frame."""

    if k_neighbors < 1:
        raise ValueError("k_neighbors must be positive.")
    selected = [item for item in tracks if item.binding.object_track_id == object_track_id and item.valid]
    if not selected:
        return _invalid_graph(object_track_id, semantic_label, DynamicGeometryMode.UNAVAILABLE, "no_bound_object_points")
    modes = {point.geometry_mode for item in selected for point in item.points_3d if point.valid}
    if len(modes) != 1:
        return _invalid_graph(object_track_id, semantic_label, DynamicGeometryMode.UNAVAILABLE, "inconsistent_geometry_modes")
    mode = next(iter(modes))
    if mode not in {DynamicGeometryMode.STATIC_CAMERA_3D, DynamicGeometryMode.FULL_SE3_3D}:
        return _invalid_graph(object_track_id, semantic_label, mode, "geometry_mode_has_no_structure_3d")
    by_frame: dict[int, dict[str, np.ndarray]] = {}
    valid_frames_by_point: dict[str, set[int]] = {}
    binding_by_point: dict[str, ObjectPointBinding] = {}
    quality_by_point: dict[str, float] = {}
    scale_by_frame: dict[int, float] = {}
    for item in selected:
        binding_by_point[item.binding.point_id] = item.binding
        quality_by_point[item.binding.point_id] = item.quality
        for frame_index, scale in item.object_scale_by_frame.items():
            if scale is not None and math.isfinite(float(scale)) and float(scale) > 0.0:
                scale_by_frame[int(frame_index)] = float(scale)
        for point in item.points_3d:
            coordinate = _coordinate(point)
            if coordinate is not None:
                by_frame.setdefault(point.frame_index, {})[point.point_id] = coordinate
                valid_frames_by_point.setdefault(point.point_id, set()).add(point.frame_index)
    eligible = [(frame, values) for frame, values in sorted(by_frame.items()) if len(values) >= 2]
    if not eligible:
        return _invalid_graph(object_track_id, semantic_label, mode, "insufficient_common_structure_points")
    reference_frame, coordinates = eligible[0]
    scale = scale_by_frame.get(reference_frame)
    if scale is None:
        distances = [
            float(np.linalg.norm(coordinates[a] - coordinates[b]))
            for index, a in enumerate(coordinates)
            for b in list(coordinates)[index + 1 :]
        ]
        scale = float(np.median(distances)) if distances else float("nan")
    if not math.isfinite(scale) or scale <= 0.0:
        return _invalid_graph(object_track_id, semantic_label, mode, "invalid_object_scale")
    pairs: set[tuple[str, str]] = set()
    graph_type = "fixed_knn"
    if semantic_label == "person":
        names = {
            binding.semantic_keypoint_name: point_id
            for point_id, binding in binding_by_point.items()
            if binding.point_role == PointRole.SEMANTIC_KEYPOINT and binding.semantic_keypoint_name
        }
        for name_a, name_b in HUMAN_SKELETON_EDGES:
            if name_a in names and name_b in names and names[name_a] in coordinates and names[name_b] in coordinates:
                pairs.add(tuple(sorted((names[name_a], names[name_b]))))
        graph_type = "semantic_human_skeleton"
    else:
        ids = sorted(coordinates)
        for point_id in ids:
            neighbours = sorted(
                (float(np.linalg.norm(coordinates[point_id] - coordinates[other])), other)
                for other in ids if other != point_id
            )[:k_neighbors]
            for _, other in neighbours:
                pairs.add(tuple(sorted((point_id, other))))
    if not pairs:
        return _invalid_graph(object_track_id, semantic_label, mode, "no_valid_fixed_edges")
    pairs = {
        pair for pair in pairs
        if len(valid_frames_by_point.get(pair[0], set()) & valid_frames_by_point.get(pair[1], set())) >= 3
    }
    if not pairs:
        return _invalid_graph(object_track_id, semantic_label, mode, "no_edges_with_temporal_support")
    edges = []
    for point_a, point_b in sorted(pairs):
        length = float(np.linalg.norm(coordinates[point_a] - coordinates[point_b]))
        if not math.isfinite(length) or length <= 0.0:
            continue
        edges.append(StructureEdge(
            point_id_a=point_a,
            point_id_b=point_b,
            reference_length=length,
            reference_normalized_length=length / scale,
            edge_quality=min(quality_by_point[point_a], quality_by_point[point_b]),
            valid=True,
            metadata={
                "reference_frame_index": reference_frame,
                "fixed_adjacency": True,
                "common_valid_frame_count": len(
                    valid_frames_by_point[point_a] & valid_frames_by_point[point_b]
                ),
            },
        ))
    if not edges:
        return _invalid_graph(object_track_id, semantic_label, mode, "no_valid_fixed_edges")
    return ObjectStructureGraph(
        object_track_id=object_track_id,
        semantic_label=semantic_label,
        reference_frame_index=reference_frame,
        point_ids=tuple(sorted(coordinates)),
        edges=tuple(edges),
        graph_type=graph_type,
        geometry_mode=mode,
        valid=True,
        quality=float(np.mean([edge.edge_quality for edge in edges])),
        metadata={"adjacency_fixed_after_reference": True, "object_scale": scale},
    )
