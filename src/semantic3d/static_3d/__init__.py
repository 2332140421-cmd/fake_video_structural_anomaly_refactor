"""Evidence-aware static 3D diagnostics and residuals."""

from .boundary_depth import BoundaryDepth3DResidual
from .depth_order import DepthOrder3DResidual
from .reconstruction_quality import ReconstructionQualityEvidence
from .residual_types import EvidenceRole, Static3DContext, reprojection_cycle_evidence
from .semantic_size import SemanticSize3DResidual
from .spatial_intersection import SpatialIntersection3DResidual

__all__ = [
    "BoundaryDepth3DResidual",
    "DepthOrder3DResidual",
    "EvidenceRole",
    "ReconstructionQualityEvidence",
    "SemanticSize3DResidual",
    "SpatialIntersection3DResidual",
    "Static3DContext",
    "reprojection_cycle_evidence",
]
