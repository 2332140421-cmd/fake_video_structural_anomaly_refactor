"""P4-C3B-M5 higher-order 3D relations and visibility events."""

from .contracts import (
    D3FrameGraph,
    D3GraphNode,
    D3GraphRelation,
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

__all__ = [
    "D3FrameGraph",
    "D3GraphNode",
    "D3GraphRelation",
    "D3NodeType",
    "D3RelationResidual",
    "D3StructureResidualExecutor",
    "D3TransitionContext",
    "OcclusionEventEvidence",
    "OcclusionEventInputsV2",
    "OcclusionEventType",
    "ReappearanceResidual",
    "build_d3_frame_graph",
    "classify_occlusion_event",
    "compute_reappearance_residual",
]
