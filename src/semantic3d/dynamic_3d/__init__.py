"""Evidence-aware dynamic 3D readiness, tracks, and residual interfaces."""

from .cache import SharedGeometryCache, load_shared_geometry_cache
from .readiness import (
    Dynamic3DReadiness,
    Dynamic3DReadinessThresholds,
    DynamicGeometryMode,
    assess_dynamic_3d_readiness,
    relative_improvement,
)
from .reprojection_residual import (
    DynamicReprojectionResidual,
    ReprojectionEvidenceType,
    compute_dynamic_reprojection_residual,
)
from .track_observation import (
    BasePointTracker,
    ExistingInterfaceAdapter,
    MockPointTracker,
    ObjectTrack3DObservation,
    PointTrack2DObservation,
    PointTrack3DObservation,
    SyntheticPointTracker,
    reconstruct_point_tracks_3d,
    summarize_point_track_coverage,
)
from .track_residual import (
    Track3DContinuityResidual,
    compute_track_3d_continuity_residuals,
)

__all__ = [
    "BasePointTracker",
    "Dynamic3DReadiness",
    "Dynamic3DReadinessThresholds",
    "DynamicGeometryMode",
    "DynamicReprojectionResidual",
    "ExistingInterfaceAdapter",
    "MockPointTracker",
    "ObjectTrack3DObservation",
    "PointTrack2DObservation",
    "PointTrack3DObservation",
    "ReprojectionEvidenceType",
    "SharedGeometryCache",
    "SyntheticPointTracker",
    "Track3DContinuityResidual",
    "assess_dynamic_3d_readiness",
    "compute_dynamic_reprojection_residual",
    "compute_track_3d_continuity_residuals",
    "reconstruct_point_tracks_3d",
    "load_shared_geometry_cache",
    "relative_improvement",
    "summarize_point_track_coverage",
]
