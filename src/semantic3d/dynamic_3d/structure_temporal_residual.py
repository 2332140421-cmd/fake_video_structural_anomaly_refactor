"""Temporal residuals on fixed object structure edges."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..validity import ResidualEvidence
from .motion_model import trajectory_coordinate
from .structure_graph import ObjectStructureGraph, StructureEdge
from .track_observation import PointTrack3DObservation


@dataclass(frozen=True)
class EdgeTemporalResidual:
    """Length change evidence for one fixed edge in one frame."""

    object_track_id: str
    frame_index: int
    point_id_a: str
    point_id_b: str
    current_length: float
    current_normalized_length: float
    edge_length_change: ResidualEvidence
    normalized_edge_length_change: ResidualEvidence
    local_angle_change: ResidualEvidence
    valid: bool
    missing_reason: str = ""

    def __post_init__(self) -> None:
        if self.valid and (not self.edge_length_change.valid or not self.normalized_edge_length_change.valid or self.missing_reason):
            raise ValueError("Valid edge residual requires raw and normalized evidence.")
        if not self.valid and (not self.missing_reason or math.isfinite(float(self.current_length)) or math.isfinite(float(self.current_normalized_length))):
            raise ValueError("Invalid edge residual requires NaN values and a reason.")


@dataclass(frozen=True)
class ObjectStructureTemporalResidual:
    """Robust frame-level aggregate retaining anomalous point and edge IDs."""

    object_track_id: str
    frame_index: int
    object_structure_residual: ResidualEvidence
    edge_residuals: tuple[EdgeTemporalResidual, ...]
    anomalous_edge_ids: tuple[str, ...]
    anomalous_point_ids: tuple[str, ...]
    valid_edge_ratio: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _missing_edge(graph: ObjectStructureGraph, edge: StructureEdge, frame: int, reason: str) -> EdgeTemporalResidual:
    ids = (graph.object_track_id, edge.point_id_a, edge.point_id_b, str(frame))
    return EdgeTemporalResidual(
        graph.object_track_id, frame, edge.point_id_a, edge.point_id_b,
        float("nan"), float("nan"),
        ResidualEvidence.missing("r_edge_length_change", reason, source_ids=ids),
        ResidualEvidence.missing("r_normalized_edge_length_change", reason, source_ids=ids),
        ResidualEvidence.missing("r_local_angle_change", "local_angle_not_computed", source_ids=ids),
        False, reason,
    )


def compute_structure_temporal_residuals(
    graph: ObjectStructureGraph,
    observations: Sequence[PointTrack3DObservation],
    object_scale_by_frame: Mapping[int, Optional[float]],
    *,
    anomaly_top_fraction: float = 0.25,
) -> tuple[ObjectStructureTemporalResidual, ...]:
    """Evaluate fixed edge lengths; missing points remain invalid, never zero."""

    if not 0.0 < anomaly_top_fraction <= 1.0:
        raise ValueError("anomaly_top_fraction must be in (0, 1].")
    if not graph.valid:
        return ()
    lookup = {(point.point_id, point.frame_index): point for point in observations if point.object_track_id == graph.object_track_id}
    frames = sorted({point.frame_index for point in observations if point.object_track_id == graph.object_track_id and graph.reference_frame_index is not None and point.frame_index > graph.reference_frame_index})
    output = []
    for frame in frames:
        rows = []
        scale = object_scale_by_frame.get(frame)
        for edge in graph.edges:
            point_a, point_b = lookup.get((edge.point_id_a, frame)), lookup.get((edge.point_id_b, frame))
            if point_a is None or point_b is None:
                rows.append(_missing_edge(graph, edge, frame, "missing_structure_point"))
                continue
            xyz_a, xyz_b = trajectory_coordinate(point_a), trajectory_coordinate(point_b)
            if xyz_a is None or xyz_b is None:
                rows.append(_missing_edge(graph, edge, frame, "invalid_structure_point"))
                continue
            if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0:
                rows.append(_missing_edge(graph, edge, frame, "invalid_object_scale"))
                continue
            length = float(np.linalg.norm(xyz_a - xyz_b))
            normalized = length / float(scale)
            ids = (graph.object_track_id, edge.point_id_a, edge.point_id_b, str(frame))
            quality = min(edge.edge_quality, point_a.reconstruction_quality, point_b.reconstruction_quality)
            rows.append(EdgeTemporalResidual(
                graph.object_track_id, frame, edge.point_id_a, edge.point_id_b,
                length, normalized,
                ResidualEvidence.observed("r_edge_length_change", abs(length - edge.reference_length), quality=quality, source_ids=ids),
                ResidualEvidence.observed("r_normalized_edge_length_change", abs(normalized - edge.reference_normalized_length), quality=quality, source_ids=ids),
                ResidualEvidence.missing("r_local_angle_change", "local_angle_not_computed", source_ids=ids),
                True,
            ))
        valid_rows = [row for row in rows if row.valid]
        if valid_rows:
            values = np.asarray([row.normalized_edge_length_change.value for row in valid_rows])
            top_count = max(1, int(math.ceil(len(values) * anomaly_top_fraction)))
            top_indices = np.argsort(values)[-top_count:]
            anomalous = [valid_rows[int(index)] for index in top_indices]
            evidence = ResidualEvidence.observed("r_object_structure_temporal", float(np.mean(values[top_indices])), quality=float(np.mean([row.normalized_edge_length_change.quality for row in valid_rows])), source_ids=(graph.object_track_id, str(frame)), metadata={"aggregation": "top_fraction_mean", "top_fraction": anomaly_top_fraction})
            edge_ids = tuple(f"{row.point_id_a}:{row.point_id_b}" for row in anomalous)
            point_ids = tuple(sorted({point for row in anomalous for point in (row.point_id_a, row.point_id_b)}))
            reason = ""
        else:
            evidence = ResidualEvidence.missing("r_object_structure_temporal", "no_valid_structure_edges", source_ids=(graph.object_track_id, str(frame)))
            edge_ids, point_ids, reason = (), (), "no_valid_structure_edges"
        output.append(ObjectStructureTemporalResidual(
            graph.object_track_id, frame, evidence, tuple(rows), edge_ids, point_ids,
            len(valid_rows) / len(rows) if rows else 0.0, bool(valid_rows), reason,
            {"deformation_model": "articulated_bone_length" if graph.semantic_label == "person" else "rigid_fixed_graph", "missing_edges_are_zero": False},
        ))
    return tuple(output)
