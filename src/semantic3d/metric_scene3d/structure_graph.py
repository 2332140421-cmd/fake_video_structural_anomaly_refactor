"""Single-frame local metric graph construction."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from ..shared_3d_observation import CoordinateFrame
from .contracts import (
    MetricPointType,
    MetricSurfacePoint,
    SingleFrameStructureGraph,
    StructureEdge3D,
)


def build_single_frame_structure_graph(
    *,
    frame_id: str,
    object_id: str,
    track_id: str | None,
    boundary_points: Sequence[MetricSurfacePoint],
    internal_points: Sequence[MetricSurfacePoint],
    semantic_keypoints: Sequence[MetricSurfacePoint] = (),
    knn_k: int = 3,
    radius_m: float | None = None,
) -> SingleFrameStructureGraph:
    """Build boundary-adjacency and geometric-neighbour edges for one frame."""

    groups = (
        tuple(boundary_points),
        tuple(internal_points),
        tuple(semantic_keypoints),
    )
    nodes = tuple(point for group in groups for point in group if point.valid)
    if any(
        point.coordinate_frame != CoordinateFrame.CAMERA_FRAME_METRIC
        for point in nodes
    ):
        raise ValueError("All M2 graph nodes must use camera_frame_metric.")
    allowed = {
        MetricPointType.BOUNDARY_POINT,
        MetricPointType.GEOMETRIC_TRACK_POINT,
        MetricPointType.SEMANTIC_KEYPOINT,
    }
    if any(point.point_type not in allowed for point in nodes):
        raise ValueError("Dense surface points cannot be graph structure nodes.")
    if len(nodes) < 2:
        return SingleFrameStructureGraph(
            graph_id=f"{frame_id}:{object_id}:single_frame_graph",
            frame_id=frame_id,
            object_id=object_id,
            track_id=track_id,
            nodes=nodes,
            edges=(),
            coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC,
            valid=False,
            quality=0.0,
            failure_reason="insufficient_valid_structure_nodes",
        )
    by_id = {point.point_id: point for point in nodes}
    edge_types: dict[tuple[str, str], set[str]] = defaultdict(set)
    valid_boundary = [point for point in boundary_points if point.valid]
    if len(valid_boundary) >= 2:
        for index, source in enumerate(valid_boundary):
            target = valid_boundary[(index + 1) % len(valid_boundary)]
            key = tuple(sorted((source.point_id, target.point_id)))
            edge_types[key].add("boundary_adjacent")
    xyz = np.asarray([point.xyz() for point in nodes])
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    for source_index, source in enumerate(nodes):
        order = np.argsort(distances[source_index])
        neighbours = [
            index
            for index in order
            if index != source_index and distances[source_index, index] > 1e-12
        ][: max(knn_k, 0)]
        for target_index in neighbours:
            target = nodes[target_index]
            key = tuple(sorted((source.point_id, target.point_id)))
            edge_types[key].add("knn")
        if radius_m is not None:
            for target_index, distance in enumerate(distances[source_index]):
                if source_index != target_index and 1e-12 < distance <= radius_m:
                    target = nodes[target_index]
                    key = tuple(sorted((source.point_id, target.point_id)))
                    edge_types[key].add("radius")
    edges = []
    for edge_index, (key, kinds) in enumerate(sorted(edge_types.items())):
        source, target = by_id[key[0]], by_id[key[1]]
        delta = target.xyz() - source.xyz()
        length = float(np.linalg.norm(delta))
        if length <= 1e-12:
            continue
        edges.append(
            StructureEdge3D(
                edge_id=f"{frame_id}:{object_id}:edge:{edge_index:05d}",
                source_point_id=source.point_id,
                target_point_id=target.point_id,
                edge_length_m=length,
                relative_depth_m=float(delta[2]),
                direction_vector=tuple(float(value) for value in delta / length),
                edge_type="+".join(sorted(kinds)),
                confidence=min(source.confidence, target.confidence),
                valid=True,
            )
        )
    quality = (
        float(np.median([point.confidence for point in nodes])) if nodes else 0.0
    )
    return SingleFrameStructureGraph(
        graph_id=f"{frame_id}:{object_id}:single_frame_graph",
        frame_id=frame_id,
        object_id=object_id,
        track_id=track_id,
        nodes=nodes,
        edges=tuple(edges),
        coordinate_frame=CoordinateFrame.CAMERA_FRAME_METRIC,
        valid=bool(edges),
        quality=quality,
        failure_reason="" if edges else "no_valid_structure_edges",
        metadata={
            "single_frame_only": True,
            "temporal_d3_residual_computed": False,
            "node_type_counts": {
                point_type.value: sum(
                    point.point_type == point_type for point in nodes
                )
                for point_type in (
                    MetricPointType.BOUNDARY_POINT,
                    MetricPointType.GEOMETRIC_TRACK_POINT,
                    MetricPointType.SEMANTIC_KEYPOINT,
                )
            },
        },
    )
