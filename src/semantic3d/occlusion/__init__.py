"""Validity-aware instance visibility and occlusion evidence interfaces."""

from .mask_observation import (
    InstanceMaskObservation,
    MaskBoundaryObservation,
    PredictedObjectSupport,
    TrackedMaskObservation,
    mask_bbox,
    mask_boundary_points,
)
from .mask_provider import (
    BaseInstanceMaskProvider,
    ExistingDetectionMaskAdapter,
    InstanceMaskCandidate,
    InstanceSegmentationModelMetadata,
    MockInstanceMaskProvider,
    RealInstanceMaskProvider,
    SyntheticInstanceMaskProvider,
)
from .mask_object_association import (
    MaskAssociationDiagnostic,
    MaskObjectAssociationResult,
    associate_instance_masks,
)
from .mask_tracking import MaskTracker, TrackedMaskProvider
from .mask_structure_points import (
    adaptive_erosion_pixels,
    eroded_mask_interior,
    group_mask_points_by_track,
    select_formal_mask_internal_points,
    track_formal_mask_internal_points,
)
from .support_prediction import predict_object_support
from .boundary_occlusion_residual import BoundaryOcclusionResidual, compute_boundary_occlusion_residual
from .depth_order_residual import OcclusionDepthOrderResidual, compute_occlusion_depth_order_residual
from .occlusion_graph import OcclusionGraph, OcclusionRelation, build_occlusion_graph
from .reappearance import ReappearanceConsistencyResidual, ReappearanceObservation, evaluate_reappearance
from .visibility_residual import VisibilityExplanation, VisibilityExplanationResidual, compute_visibility_explanation_residual
from .visibility_state import ObjectVisibilityObservation, VisibilityState, infer_visibility_state
from .scene_cut_statistics import SceneCutStatistics, compute_scene_cut_statistics
from .event_validation import (
    OcclusionEventInputs,
    OcclusionEventValidation,
    validate_occlusion_event,
)

__all__ = [
    "BaseInstanceMaskProvider",
    "ExistingDetectionMaskAdapter",
    "InstanceMaskCandidate",
    "InstanceSegmentationModelMetadata",
    "InstanceMaskObservation",
    "MaskBoundaryObservation",
    "mask_bbox",
    "mask_boundary_points",
    "MaskTracker",
    "MockInstanceMaskProvider",
    "PredictedObjectSupport",
    "RealInstanceMaskProvider",
    "SyntheticInstanceMaskProvider",
    "TrackedMaskObservation",
    "TrackedMaskProvider",
    "adaptive_erosion_pixels",
    "eroded_mask_interior",
    "group_mask_points_by_track",
    "select_formal_mask_internal_points",
    "track_formal_mask_internal_points",
    "MaskAssociationDiagnostic",
    "MaskObjectAssociationResult",
    "predict_object_support",
    "BoundaryOcclusionResidual",
    "OcclusionDepthOrderResidual",
    "OcclusionGraph",
    "OcclusionRelation",
    "ObjectVisibilityObservation",
    "ReappearanceConsistencyResidual",
    "ReappearanceObservation",
    "VisibilityExplanation",
    "VisibilityExplanationResidual",
    "VisibilityState",
    "build_occlusion_graph",
    "compute_boundary_occlusion_residual",
    "compute_occlusion_depth_order_residual",
    "compute_visibility_explanation_residual",
    "evaluate_reappearance",
    "infer_visibility_state",
    "associate_instance_masks",
    "SceneCutStatistics",
    "compute_scene_cut_statistics",
    "OcclusionEventInputs",
    "OcclusionEventValidation",
    "validate_occlusion_event",
]
