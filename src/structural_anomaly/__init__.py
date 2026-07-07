"""Object-aware structural residuals for forged video anomaly prototypes."""

from .scale_depth import (
    ObjectObservation,
    ScaleDepthResidualResult,
    compute_equivalent_projection_scale,
    compute_scale_depth_residual,
    compute_scale_depth_residual_log,
)
from .fusion import ResidualBundle, fuse_residuals

__all__ = [
    "ObjectObservation",
    "ResidualBundle",
    "ScaleDepthResidualResult",
    "compute_equivalent_projection_scale",
    "compute_scale_depth_residual",
    "compute_scale_depth_residual_log",
    "fuse_residuals",
]
