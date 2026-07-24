"""Build pose-aligned D3 frame graphs from shared M2/M4 observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from ..method_completion.d3_relations import D3RelationType
from .contracts import D3FrameGraph, D3GraphNode, D3GraphRelation, D3NodeType


def _relation(
    *,
    relation_id: str,
    relation_type: D3RelationType,
    frame_index: int,
    source_node_ids: Sequence[str],
    source_edge_id: str,
    values: Sequence[float],
    unit: str,
    confidence: float,
    identity_reliable: bool,
    localization_reference: Mapping[str, object],
    metadata: Mapping[str, object] | None = None,
) -> D3GraphRelation:
    return D3GraphRelation(
        relation_id=relation_id,
        relation_type=relation_type,
        frame_index=frame_index,
        source_node_ids=tuple(source_node_ids),
        source_edge_id=source_edge_id,
        values=tuple(float(value) for value in values),
        unit=unit,
        coordinate_frame="clip_local_aligned",
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        identity_reliable=identity_reliable,
        valid=True,
        localization_reference=dict(localization_reference),
        metadata=dict(metadata or {}),
    )


def build_d3_frame_graph(
    *,
    graph_id: str,
    video_id: str,
    clip_id: str,
    frame_index: int,
    nodes: Sequence[D3GraphNode],
    pose_source: str,
    fixed_structure_edges: Sequence[tuple[str, str, str]] = (),
    containment_relations: Mapping[tuple[str, str], tuple[float, float]] | None = None,
    support_contact_relations: Mapping[tuple[str, str], tuple[float, float]] | None = None,
) -> D3FrameGraph:
    """Build object, boundary, fixed-edge, and optional physical relations.

    ``fixed_structure_edges`` must reference stable point node IDs. No temporal
    nearest-neighbour rewiring is performed.
    """

    valid_nodes = tuple(node for node in nodes if node.valid)
    if not valid_nodes:
        return D3FrameGraph(
            graph_id=graph_id,
            video_id=video_id,
            clip_id=clip_id,
            frame_index=frame_index,
            nodes=(),
            relations=(),
            coordinate_frame="clip_local_aligned",
            pose_source=pose_source,
            quality=0.0,
            valid=False,
            failure_reason="no_valid_d3_nodes",
        )
    if any(node.frame_index != frame_index for node in valid_nodes):
        raise ValueError("All D3 graph nodes must belong to the graph frame.")
    by_id = {node.node_id: node for node in valid_nodes}
    object_nodes = sorted(
        (node for node in valid_nodes if node.node_type == D3NodeType.OBJECT_NODE),
        key=lambda node: node.track_id,
    )
    boundaries: dict[str, list[D3GraphNode]] = defaultdict(list)
    for node in valid_nodes:
        if node.node_type == D3NodeType.BOUNDARY_NODE:
            boundaries[node.track_id].append(node)
    relations: list[D3GraphRelation] = []

    for index, first in enumerate(object_nodes):
        for second in object_nodes[index + 1 :]:
            pair = tuple(sorted((first.track_id, second.track_id)))
            first_xyz = np.asarray(first.xyz_m)
            second_xyz = np.asarray(second.xyz_m)
            delta = second_xyz - first_xyz
            distance = float(np.linalg.norm(delta))
            reliable = first.identity_reliable and second.identity_reliable
            confidence = min(first.confidence, second.confidence)
            localization = {
                "level": "object_pair",
                "track_ids": list(pair),
                "object_ids": [first.object_id, second.object_id],
                "frame_index": frame_index,
            }
            if distance > 1e-12:
                relations.append(
                    _relation(
                        relation_id=f"object_distance:{pair[0]}:{pair[1]}",
                        relation_type=D3RelationType.OBJECT_RELATIVE_DISTANCE,
                        frame_index=frame_index,
                        source_node_ids=(first.node_id, second.node_id),
                        source_edge_id=f"object_pair:{pair[0]}:{pair[1]}",
                        values=(distance,),
                        unit="meter",
                        confidence=confidence,
                        identity_reliable=reliable,
                        localization_reference=localization,
                    )
                )
                relations.append(
                    _relation(
                        relation_id=f"bearing_relation:{pair[0]}:{pair[1]}",
                        relation_type=D3RelationType.BEARING_RELATION,
                        frame_index=frame_index,
                        source_node_ids=(first.node_id, second.node_id),
                        source_edge_id=f"object_pair:{pair[0]}:{pair[1]}",
                        values=delta / distance,
                        unit="unit_vector",
                        confidence=confidence,
                        identity_reliable=reliable,
                        localization_reference=localization,
                    )
                )
            if (
                first.orientation_vector is not None
                and second.orientation_vector is not None
            ):
                first_orientation = np.asarray(first.orientation_vector, dtype=float)
                second_orientation = np.asarray(second.orientation_vector, dtype=float)
                first_orientation /= np.linalg.norm(first_orientation)
                second_orientation /= np.linalg.norm(second_orientation)
                relative_angle = float(
                    np.arccos(
                        np.clip(
                            np.dot(first_orientation, second_orientation), -1.0, 1.0
                        )
                    )
                    / np.pi
                )
                relations.append(
                    _relation(
                        relation_id=f"relative_orientation:{pair[0]}:{pair[1]}",
                        relation_type=D3RelationType.RELATIVE_ORIENTATION,
                        frame_index=frame_index,
                        source_node_ids=(first.node_id, second.node_id),
                        source_edge_id=f"object_orientation_pair:{pair[0]}:{pair[1]}",
                        values=(relative_angle,),
                        unit="normalized_angle",
                        confidence=confidence,
                        identity_reliable=reliable,
                        localization_reference=localization,
                        metadata={
                            "orientation_source_a": first.provenance.get(
                                "orientation_source", "unknown"
                            ),
                            "orientation_source_b": second.provenance.get(
                                "orientation_source", "unknown"
                            ),
                        },
                    )
                )
            relations.append(
                _relation(
                    relation_id=f"depth_order:{pair[0]}:{pair[1]}",
                    relation_type=D3RelationType.DEPTH_ORDER,
                    frame_index=frame_index,
                    source_node_ids=(first.node_id, second.node_id),
                    source_edge_id=f"object_pair:{pair[0]}:{pair[1]}",
                    values=(float(first_xyz[2] - second_xyz[2]),),
                    unit="signed_meter",
                    confidence=confidence,
                    identity_reliable=reliable,
                    localization_reference=localization,
                )
            )
            key = pair
            if containment_relations and key in containment_relations:
                relations.append(
                    _relation(
                        relation_id=f"containment_overlap:{pair[0]}:{pair[1]}",
                        relation_type=D3RelationType.CONTAINMENT_OR_OVERLAP,
                        frame_index=frame_index,
                        source_node_ids=(first.node_id, second.node_id),
                        source_edge_id=f"projected_mask_pair:{pair[0]}:{pair[1]}",
                        values=containment_relations[key],
                        unit="ratio",
                        confidence=confidence,
                        identity_reliable=reliable,
                        localization_reference=localization,
                        metadata={"relation_source": "formal_visible_mask_projection"},
                    )
                )
            if support_contact_relations and key in support_contact_relations:
                relations.append(
                    _relation(
                        relation_id=f"support_contact:{pair[0]}:{pair[1]}",
                        relation_type=D3RelationType.SUPPORT_OR_CONTACT,
                        frame_index=frame_index,
                        source_node_ids=(first.node_id, second.node_id),
                        source_edge_id=f"contact_pair:{pair[0]}:{pair[1]}",
                        values=support_contact_relations[key],
                        unit="meter_and_confidence",
                        confidence=confidence,
                        identity_reliable=reliable,
                        localization_reference=localization,
                        metadata={"estimated_only_when_physical_contact_supported": True},
                    )
                )

    object_by_track = {node.track_id: node for node in object_nodes}
    for track_id, boundary_nodes in sorted(boundaries.items()):
        object_node = object_by_track.get(track_id)
        if object_node is None or not boundary_nodes:
            continue
        object_xyz = np.asarray(object_node.xyz_m)
        radii = np.asarray(
            [
                np.linalg.norm(np.asarray(node.xyz_m) - object_xyz)
                for node in boundary_nodes
            ],
            dtype=float,
        )
        if not radii.size or not np.isfinite(radii).all() or np.median(radii) <= 0.0:
            continue
        relations.append(
            _relation(
                relation_id=f"object_boundary:{track_id}",
                relation_type=D3RelationType.OBJECT_BOUNDARY_RELATION,
                frame_index=frame_index,
                source_node_ids=(
                    object_node.node_id,
                    *(node.node_id for node in boundary_nodes),
                ),
                source_edge_id=f"object_boundary_aggregate:{track_id}",
                values=(float(np.median(radii)), float(np.percentile(radii, 90))),
                unit="meter",
                confidence=min(
                    object_node.confidence,
                    float(np.median([node.confidence for node in boundary_nodes])),
                ),
                identity_reliable=object_node.identity_reliable,
                localization_reference={
                    "level": "object_boundary",
                    "track_id": track_id,
                    "object_id": object_node.object_id,
                    "boundary_node_ids": [node.node_id for node in boundary_nodes],
                    "frame_index": frame_index,
                },
                metadata={
                    "aggregation": "median_and_p90_radial_distance",
                    "boundary_point_identity_required": False,
                    "object_track_identity_required": True,
                },
            )
        )

    fixed_lengths: dict[str, list[float]] = defaultdict(list)
    fixed_sources: dict[str, list[str]] = defaultdict(list)
    fixed_quality: dict[str, list[float]] = defaultdict(list)
    fixed_identity_reliability: dict[str, list[bool]] = defaultdict(list)
    for edge_id, source_id, target_id in fixed_structure_edges:
        source = by_id.get(source_id)
        target = by_id.get(target_id)
        if source is None or target is None:
            continue
        delta = np.asarray(target.xyz_m) - np.asarray(source.xyz_m)
        length = float(np.linalg.norm(delta))
        if not math_is_positive(length):
            continue
        track_id = source.track_id
        reliable = (
            source.track_id == target.track_id
            and source.identity_reliable
            and target.identity_reliable
        )
        confidence = min(source.confidence, target.confidence)
        relations.append(
            _relation(
                relation_id=f"structure_edge:{track_id}:{edge_id}",
                relation_type=D3RelationType.STRUCTURE_EDGE_LENGTH,
                frame_index=frame_index,
                source_node_ids=(source.node_id, target.node_id),
                source_edge_id=edge_id,
                values=(length,),
                unit="meter",
                confidence=confidence,
                identity_reliable=reliable,
                localization_reference={
                    "level": "structure_edge",
                    "track_id": track_id,
                    "point_ids": [source.node_id, target.node_id],
                    "edge_id": edge_id,
                    "frame_index": frame_index,
                },
                metadata={"fixed_adjacency": True},
            )
        )
        fixed_lengths[track_id].append(length)
        fixed_sources[track_id].extend((source.node_id, target.node_id))
        fixed_quality[track_id].append(confidence)
        fixed_identity_reliability[track_id].append(reliable)
    for track_id, lengths in sorted(fixed_lengths.items()):
        median = float(np.median(lengths))
        if median <= 0.0:
            continue
        normalized = tuple(float(value / median) for value in lengths)
        relations.append(
            _relation(
                relation_id=f"local_rigidity:{track_id}",
                relation_type=D3RelationType.LOCAL_RIGIDITY,
                frame_index=frame_index,
                source_node_ids=tuple(dict.fromkeys(fixed_sources[track_id])),
                source_edge_id=f"fixed_structure_graph:{track_id}",
                values=normalized,
                unit="normalized_edge_length",
                confidence=float(np.median(fixed_quality[track_id])),
                identity_reliable=all(fixed_identity_reliability[track_id]),
                localization_reference={
                    "level": "object_structure_graph",
                    "track_id": track_id,
                    "edge_count": len(lengths),
                    "frame_index": frame_index,
                },
                metadata={
                    "normalization": "within_frame_median_edge_length",
                    "fixed_adjacency": True,
                },
            )
        )

    valid = bool(relations)
    quality = (
        float(np.median([relation.confidence for relation in relations]))
        if relations
        else 0.0
    )
    return D3FrameGraph(
        graph_id=graph_id,
        video_id=video_id,
        clip_id=clip_id,
        frame_index=frame_index,
        nodes=valid_nodes,
        relations=tuple(relations),
        coordinate_frame="clip_local_aligned",
        pose_source=pose_source,
        quality=quality,
        valid=valid,
        failure_reason="" if valid else "no_valid_d3_relations",
        metadata={
            "world_frame_claimed": False,
            "fixed_edges_rewired_per_frame": False,
            "node_type_counts": {
                node_type.value: sum(node.node_type == node_type for node in valid_nodes)
                for node_type in D3NodeType
            },
        },
    )


def math_is_positive(value: float) -> bool:
    """Return whether a scalar is finite and strictly positive."""

    return bool(np.isfinite(value) and value > 1e-12)
