"""Validity-aware contracts for M5 D3 graphs and relation residuals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..method_completion.d3_relations import D3RelationType
from ..pose_d2.contracts import PoseProviderStatus


class D3NodeType(str, Enum):
    """Semantic role of a node in a D3 frame graph."""

    OBJECT_NODE = "object_node"
    BOUNDARY_NODE = "boundary_node"
    GEOMETRIC_TRACK_NODE = "geometric_track_node"
    SEMANTIC_KEYPOINT_NODE = "semantic_keypoint_node"


def _xyz_or_none(value: Optional[Sequence[float]]) -> Optional[tuple[float, float, float]]:
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("D3 xyz must contain three finite values.")
    return tuple(float(item) for item in array)


@dataclass(frozen=True)
class D3GraphNode:
    """One pose-aligned object or structure node with localization provenance."""

    node_id: str
    node_type: D3NodeType | str
    frame_index: int
    object_id: str
    track_id: str
    semantic_label: str
    xyz_m: Optional[tuple[float, float, float]]
    coordinate_frame: str
    source_observation_id: str
    confidence: float
    identity_reliable: bool
    visibility: str
    valid: bool
    orientation_vector: Optional[tuple[float, float, float]] = None
    failure_reason: str = ""
    localization_reference: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        node_type = D3NodeType(self.node_type)
        xyz = _xyz_or_none(self.xyz_m)
        orientation = _xyz_or_none(self.orientation_vector)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("D3 node confidence must be in [0, 1].")
        if self.coordinate_frame != "clip_local_aligned":
            raise ValueError("M5 D3 nodes must use clip_local_aligned.")
        if self.valid:
            if xyz is None or self.failure_reason:
                raise ValueError("Valid D3 node requires xyz and no failure reason.")
            if not self.source_observation_id:
                raise ValueError("Valid D3 node requires source provenance.")
            if orientation is not None and np.linalg.norm(orientation) <= 1e-12:
                raise ValueError("D3 node orientation_vector must be non-zero.")
        elif xyz is not None or not self.failure_reason:
            raise ValueError("Invalid D3 node requires missing xyz and a reason.")
        object.__setattr__(self, "node_type", node_type)
        object.__setattr__(self, "xyz_m", xyz)
        object.__setattr__(self, "orientation_vector", orientation)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "localization_reference", dict(self.localization_reference))
        object.__setattr__(self, "provenance", dict(self.provenance))


@dataclass(frozen=True)
class D3GraphRelation:
    """One relation measured inside a pose-aligned frame graph."""

    relation_id: str
    relation_type: D3RelationType | str
    frame_index: int
    source_node_ids: tuple[str, ...]
    source_edge_id: str
    values: tuple[float, ...]
    unit: str
    coordinate_frame: str
    confidence: float
    identity_reliable: bool
    valid: bool
    failure_reason: str = ""
    localization_reference: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_type = D3RelationType(self.relation_type)
        values = tuple(float(value) for value in self.values)
        confidence = float(self.confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("D3 relation confidence must be in [0, 1].")
        if self.coordinate_frame != "clip_local_aligned":
            raise ValueError("M5 D3 relations must use clip_local_aligned.")
        if self.valid:
            if not values or not all(math.isfinite(value) for value in values):
                raise ValueError("Valid D3 relation requires finite values.")
            if not self.source_node_ids or not self.unit or self.failure_reason:
                raise ValueError("Valid D3 relation requires sources, unit, and no reason.")
        elif not self.failure_reason or any(math.isfinite(value) for value in values):
            raise ValueError("Invalid D3 relation requires NaN values and a reason.")
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "source_node_ids", tuple(self.source_node_ids))
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "localization_reference", dict(self.localization_reference))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class D3FrameGraph:
    """One clip-local frame graph assembled from M2/M4 observations."""

    graph_id: str
    video_id: str
    clip_id: str
    frame_index: int
    nodes: tuple[D3GraphNode, ...]
    relations: tuple[D3GraphRelation, ...]
    coordinate_frame: str
    pose_source: str
    quality: float
    valid: bool
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coordinate_frame != "clip_local_aligned":
            raise ValueError("D3FrameGraph cannot claim world_frame.")
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("D3 graph quality must be in [0, 1].")
        if self.valid and (not self.nodes or not self.relations or self.failure_reason):
            raise ValueError("Valid D3 frame graph requires nodes and relations.")
        if not self.valid and not self.failure_reason:
            raise ValueError("Invalid D3 frame graph requires failure_reason.")
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "relations", tuple(self.relations))
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class D3TransitionContext:
    """Eligibility gate shared by every D3 relation in one frame transition."""

    video_id: str
    clip_id: str
    frame_t: int
    frame_t1: int
    pose_status: PoseProviderStatus | str
    pose_confidence: float
    pose_valid: bool
    correspondence_identity_reliable: bool
    source_coordinate_frame: str
    target_coordinate_frame: str
    valid: bool
    failure_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = PoseProviderStatus(self.pose_status)
        confidence = float(self.pose_confidence)
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("D3 transition pose confidence must be in [0, 1].")
        expected = bool(
            self.pose_valid
            and status.usable_for_geometry
            and self.correspondence_identity_reliable
            and self.source_coordinate_frame == "clip_local_aligned"
            and self.target_coordinate_frame == "clip_local_aligned"
        )
        if self.valid != expected:
            raise ValueError("D3 transition valid must match pose/correspondence gate.")
        if self.valid and self.failure_reason:
            raise ValueError("Valid D3 transition cannot have a failure reason.")
        if not self.valid and not self.failure_reason:
            raise ValueError("Invalid D3 transition requires failure_reason.")
        object.__setattr__(self, "pose_status", status)
        object.__setattr__(self, "pose_confidence", confidence)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class D3RelationResidual:
    """One auditable higher-order relation residual and localization pointer."""

    residual_id: str
    residual_name: str
    relation_type: D3RelationType | str
    video_id: str
    clip_id: str
    frame_t: int
    frame_t1: int
    source_nodes: tuple[str, ...]
    source_edge: str
    coordinate_frame: str
    reference_relation: tuple[float, ...]
    observed_relation: tuple[float, ...]
    value: float
    confidence: float
    valid: bool
    failure_reason: str
    localization_reference: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        relation_type = D3RelationType(self.relation_type)
        confidence = float(self.confidence)
        value = float(self.value)
        if self.coordinate_frame != "clip_local_aligned":
            raise ValueError("D3 residual must use clip_local_aligned.")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("D3 residual confidence must be in [0, 1].")
        if self.valid:
            if not math.isfinite(value) or value < 0.0 or self.failure_reason:
                raise ValueError("Valid D3 residual must be finite and non-negative.")
        elif not math.isnan(value) or not self.failure_reason:
            raise ValueError("Invalid D3 residual must keep NaN and a reason.")
        object.__setattr__(self, "relation_type", relation_type)
        object.__setattr__(self, "source_nodes", tuple(self.source_nodes))
        object.__setattr__(
            self, "reference_relation", tuple(float(v) for v in self.reference_relation)
        )
        object.__setattr__(
            self, "observed_relation", tuple(float(v) for v in self.observed_relation)
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "localization_reference", dict(self.localization_reference))
        object.__setattr__(self, "metadata", dict(self.metadata))
