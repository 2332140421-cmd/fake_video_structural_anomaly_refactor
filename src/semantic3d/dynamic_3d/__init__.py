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
from .direction_residual import (
    DirectionConsistencyResidual,
    compute_direction_consistency_residuals,
)
from .motion_model import (
    BaseObjectMotionModel,
    ObjectMedianTranslationModel,
    ObjectMotionPrediction,
    PointConstantVelocityModel,
    trajectory_coordinate,
)
from .object_dynamic_aggregation import (
    ObjectDynamicAggregate,
    aggregate_object_dynamic_evidence,
)
from .object_track_binding import (
    ObjectBindingResult,
    ObjectDynamicObservation,
    ObjectPointBinding,
    ObjectPointTrack3D,
    PointRole,
    assemble_object_point_tracks_3d,
    bind_point_tracks_to_objects,
    select_stable_object_point_tracks,
)
from .relative_velocity import (
    RelativeVelocityResidual,
    compute_relative_velocity_residuals,
)
from .structure_graph import (
    HUMAN_SKELETON_EDGES,
    ObjectStructureGraph,
    StructureEdge,
    build_object_structure_graph,
)
from .structure_temporal_residual import (
    EdgeTemporalResidual,
    ObjectStructureTemporalResidual,
    compute_structure_temporal_residuals,
)
from .person_keypoint_binding import (
    PersonKeypointBindingResult,
    PersonKeypointCoverage,
    bind_person_keypoints_to_shared_3d,
)

__all__ = [
    "BasePointTracker",
    "BaseObjectMotionModel",
    "DirectionConsistencyResidual",
    "Dynamic3DReadiness",
    "Dynamic3DReadinessThresholds",
    "DynamicGeometryMode",
    "DynamicReprojectionResidual",
    "ExistingInterfaceAdapter",
    "MockPointTracker",
    "ObjectTrack3DObservation",
    "ObjectBindingResult",
    "ObjectDynamicAggregate",
    "ObjectDynamicObservation",
    "ObjectMedianTranslationModel",
    "ObjectMotionPrediction",
    "ObjectPointBinding",
    "ObjectPointTrack3D",
    "ObjectStructureGraph",
    "ObjectStructureTemporalResidual",
    "EdgeTemporalResidual",
    "PointConstantVelocityModel",
    "PointRole",
    "PointTrack2DObservation",
    "PointTrack3DObservation",
    "ReprojectionEvidenceType",
    "SharedGeometryCache",
    "StructureEdge",
    "SyntheticPointTracker",
    "Track3DContinuityResidual",
    "assess_dynamic_3d_readiness",
    "aggregate_object_dynamic_evidence",
    "assemble_object_point_tracks_3d",
    "bind_point_tracks_to_objects",
    "select_stable_object_point_tracks",
    "build_object_structure_graph",
    "compute_direction_consistency_residuals",
    "compute_dynamic_reprojection_residual",
    "compute_track_3d_continuity_residuals",
    "compute_relative_velocity_residuals",
    "compute_structure_temporal_residuals",
    "reconstruct_point_tracks_3d",
    "load_shared_geometry_cache",
    "relative_improvement",
    "summarize_point_track_coverage",
    "trajectory_coordinate",
    "RelativeVelocityResidual",
    "HUMAN_SKELETON_EDGES",
    "PersonKeypointBindingResult",
    "PersonKeypointCoverage",
    "bind_person_keypoints_to_shared_3d",
]
