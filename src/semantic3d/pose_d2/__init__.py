"""P4-C3B-M4 short-baseline camera pose and real D2 contracts."""

from .alignment import ClipLocalAlignment, build_clip_local_alignment
from .contracts import (
    D2ResidualObservation,
    D2VisibilityStatus,
    PairwisePoseObservation,
    PoseProviderStatus,
    StaticVerificationObservation,
)
from .provider import (
    ShortBaselinePoseProvider,
    ShortBaselinePoseThresholds,
    estimate_metric_transform_from_correspondences,
)
from .residuals import (
    aggregate_object_d2_residual,
    compute_d2_projection_residual,
)

__all__ = [
    "ClipLocalAlignment",
    "D2ResidualObservation",
    "D2VisibilityStatus",
    "PairwisePoseObservation",
    "PoseProviderStatus",
    "ShortBaselinePoseProvider",
    "ShortBaselinePoseThresholds",
    "StaticVerificationObservation",
    "aggregate_object_d2_residual",
    "build_clip_local_alignment",
    "compute_d2_projection_residual",
    "estimate_metric_transform_from_correspondences",
]
