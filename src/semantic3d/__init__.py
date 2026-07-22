"""Semantic 3D structural residuals for forged video anomaly prototypes."""

from __future__ import annotations

__all__ = [
    "BaseKeypointProvider",
    "BaseDepthProvider",
    "BaseObjectProvider",
    "ClipWindow",
    "ClipObservation",
    "ClipObservationJSON",
    "ClipResidualResultJSON",
    "ClipResidualSummary",
    "ApplicabilityGateResult",
    "AssociationDiagnostics",
    "CupHeightApplicabilityConfig",
    "CupHeightApplicabilityGate",
    "DepthProvider",
    "DepthObservation",
    "DepthRepresentation",
    "DepthScaleStatus",
    "DepthDefinition",
    "DepthSample",
    "DepthSamplingMethod",
    "DepthStatistic",
    "DepthTemporalResidualResult",
    "DepthAlignmentMode",
    "DepthAlignmentObservation",
    "DimensionAlignedPriorEntry",
    "DimensionAlignedPriorResolver",
    "FlowProvider",
    "FrameMockDepthProvider",
    "GeometryScaleStatus",
    "GeometryScaleUnit",
    "LargerValueMeans",
    "LegacyDepthProviderAdapter",
    "PhysicalPriorGateResult",
    "EmpiricalPairPrior",
    "EmpiricalPairPriorMatch",
    "EmpiricalPairPriorResolver",
    "FrameObservation",
    "FrameObservationJSON",
    "ForegroundMaskObservation",
    "Keypoint2D",
    "KeypointPrediction",
    "MockDepthProvider",
    "MockKeypointProvider",
    "MockObjectProvider",
    "MockSequenceGeometryProvider",
    "ObjectAssociator",
    "ObjectLevelResidual",
    "ObjectLevelResidualEvidence",
    "ObjectMaskObservation",
    "ObjectObservation",
    "ObjectObservationJSON",
    "Object3DReconstructor",
    "Object3DReconstructorConfig",
    "ObjectPairResidual",
    "ObjectTrack",
    "ResidualReport",
    "ResidualEvidence",
    "BoundaryDepth3DResidual",
    "DepthOrder3DResidual",
    "EvidenceRole",
    "ReconstructionQualityEvidence",
    "SemanticSize3DResidual",
    "SpatialIntersection3DResidual",
    "Static3DContext",
    "ResidualValues",
    "ResidualWeights",
    "RelativePoseObservation",
    "RealDepthProvider",
    "ProjectedMeasurementResult",
    "ProjectedMeasurementRule",
    "ProjectedMeasurementRules",
    "ResolvedScalePrior",
    "ScalePrior",
    "ScalePriorRecord",
    "ScalePriorResolver",
    "SceneCutDecision",
    "SequenceGeometryObservation",
    "SequenceGeometryQuality",
    "SequenceGeometryStabilizationResult",
    "SequenceDepthAlignmentResult",
    "SequenceScaleStatus",
    "CameraMotionRegime",
    "MotionRegimeObservation",
    "MotionRegimeThresholds",
    "PoseEstimateCandidate",
    "PoseGraphResult",
    "PoseModelType",
    "TranslationScaleStatus",
    "DepthAlignmentModelSelection",
    "Dynamic3DFrameValidity",
    "StrictPhysicalPriorEntry",
    "StrictPhysicalScalePriorResolver",
    "StrictResolvedPhysicalPrior",
    "CameraObservation",
    "CoordinateConvention",
    "Object3DObservation",
    "Point2DObservation",
    "Point3DObservation",
    "PixelCenterConvention",
    "ReconstructionFrame",
    "Shared3DFrameObservation",
    "Shared3DClipObservation",
    "Shared3DFrameBuilder",
    "TransformConvention",
    "VisibilityStatus",
    "MissingReason",
    "SegmentationProvider",
    "TrackerProvider",
    "BaseSceneCutDetector",
    "BaseSequenceGeometryProvider",
    "HistogramFeatureSceneCutDetector",
    "LegacyDepthPoseSequenceAdapter",
    "SyntheticSequenceGeometryProvider",
    "UnifiedSequenceGeometryProvider",
    "aggregate_map_by_mask",
    "aggregate_points_by_mask",
    "aggregate_clip_depth_residuals",
    "aggregate_track_depth_residuals",
    "aggregate_depth_transition_evidence",
    "aggregate_residual_evidence",
    "aggregate_values",
    "applicability_skip_reason",
    "build_clip_observation",
    "build_clips",
    "build_frame_observation",
    "bbox_area_to_mask_area",
    "bbox_area_ratio",
    "bbox_iou",
    "backproject_pixel",
    "backproject_points",
    "build_shared_3d_frame_observation",
    "build_foreground_mask",
    "build_object_level_residuals",
    "build_object_level_residuals_with_details",
    "build_object_level_residual_evidence",
    "build_object_pair_residuals",
    "compute_depth_temporal_residual",
    "compute_frame_depth_reference",
    "compute_dimension_aligned_rsd",
    "compute_depth_strategy",
    "compute_projected_measurement",
    "compute_scale_depth_interval",
    "camera_center_world",
    "camera_to_world",
    "deduplicate_frames_by_index",
    "depth_consistency_plot_series_from_csv",
    "depth_transition_evidence",
    "draw_multilevel_summary",
    "draw_object_residual_map",
    "draw_object_summary",
    "draw_pairwise_residual_graph",
    "draw_residual_heatmap_from_masks",
    "default_scale_prior_resolver",
    "extract_frames",
    "evaluate_physical_prior_gate",
    "estimate_depth_alignment",
    "estimate_depth_alignment_from_correspondences",
    "estimate_depth_alignment_with_holdout",
    "select_depth_alignment_model",
    "select_depth_alignment_from_correspondences",
    "align_depth_sequence_to_reference",
    "build_pose_graph",
    "classify_motion_regime",
    "estimate_adaptive_pose_candidates",
    "stabilize_sequence_geometry",
    "fuse_residuals",
    "fuse_residual_evidence",
    "load_clip_observation",
    "load_clip_residual_result",
    "load_frame_observation",
    "load_scale_prior_resolver",
    "load_dimension_aligned_prior_resolver",
    "load_projected_measurement_rules",
    "load_strict_physical_prior_resolver",
    "normalize_object_label",
    "normalize_residuals",
    "normalized_center_distance",
    "pairwise_scale_depth_residuals",
    "project_point",
    "project_points",
    "reprojection_cycle_evidence",
    "residual_values_to_dict",
    "save_clip_observation",
    "save_clip_residual_result",
    "save_depth_consistency_tracks_plot_from_csv",
    "save_raw_and_thresholded_residual_plots_from_csv",
    "save_depth_consistency_tracks_plot",
    "save_frame_observation",
    "scale_depth_residual",
    "scale_depth_residual_log",
    "sample_depth",
    "rsd_2d_coarse",
    "rsd_2d_coarse_log",
    "rsd_2d_dimension_aligned",
    "r_depth_cons_2p5d",
    "make_frame_pair_id",
    "make_track_pair_id",
    "recompute_scale_depth_formula",
    "safe_bbox_bounds",
    "swapped_log_residual_consistent",
    "summarize_clip_residuals",
    "topk_mean",
    "transform_points",
    "video_risk_score",
    "world_to_camera",
    "apply_depth_alignment",
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
    "SyntheticPointTracker",
    "Track3DContinuityResidual",
    "assess_dynamic_3d_readiness",
    "compute_dynamic_reprojection_residual",
    "compute_track_3d_continuity_residuals",
    "reconstruct_point_tracks_3d",
    "relative_improvement",
    "summarize_point_track_coverage",
    "BaseObjectMotionModel",
    "DirectionConsistencyResidual",
    "EdgeTemporalResidual",
    "ObjectBindingResult",
    "ObjectDynamicAggregate",
    "ObjectDynamicObservation",
    "ObjectMedianTranslationModel",
    "ObjectMotionPrediction",
    "ObjectPointBinding",
    "ObjectPointTrack3D",
    "ObjectStructureGraph",
    "ObjectStructureTemporalResidual",
    "PointConstantVelocityModel",
    "PointRole",
    "RelativeVelocityResidual",
    "StructureEdge",
    "aggregate_object_dynamic_evidence",
    "assemble_object_point_tracks_3d",
    "bind_point_tracks_to_objects",
    "build_object_structure_graph",
    "compute_direction_consistency_residuals",
    "compute_relative_velocity_residuals",
    "compute_structure_temporal_residuals",
    "select_stable_object_point_tracks",
    "PersonKeypointBindingResult",
    "PersonKeypointCoverage",
    "bind_person_keypoints_to_shared_3d",
    "BaseInstanceMaskProvider",
    "BoundaryOcclusionResidual",
    "ExistingDetectionMaskAdapter",
    "InstanceMaskCandidate",
    "InstanceSegmentationModelMetadata",
    "InstanceMaskObservation",
    "MaskBoundaryObservation",
    "MaskTracker",
    "MaskAssociationDiagnostic",
    "MaskObjectAssociationResult",
    "MockInstanceMaskProvider",
    "ObjectVisibilityObservation",
    "OcclusionDepthOrderResidual",
    "OcclusionGraph",
    "OcclusionRelation",
    "PredictedObjectSupport",
    "RealInstanceMaskProvider",
    "ReappearanceConsistencyResidual",
    "ReappearanceObservation",
    "SyntheticInstanceMaskProvider",
    "TrackedMaskObservation",
    "TrackedMaskProvider",
    "VisibilityExplanation",
    "VisibilityExplanationResidual",
    "VisibilityState",
    "SceneCutStatistics",
    "OcclusionEventInputs",
    "OcclusionEventValidation",
    "BranchCoverageStatus",
    "CoverageReadiness",
    "EvidenceApplicability",
    "AggregationEvidence",
    "EvidenceBranchSpec",
    "EvidenceFormality",
    "EvidenceLevel",
    "PointEvidenceAggregate",
    "EdgeEvidenceAggregate",
    "ObjectEvidenceAggregate",
    "FrameEvidenceAggregate",
    "ClipEvidenceAggregate",
    "MultilevelAggregationResult",
    "associate_instance_masks",
    "build_occlusion_graph",
    "compute_boundary_occlusion_residual",
    "compute_occlusion_depth_order_residual",
    "compute_visibility_explanation_residual",
    "evaluate_reappearance",
    "infer_visibility_state",
    "predict_object_support",
    "compute_scene_cut_statistics",
    "validate_occlusion_event",
    "adaptive_erosion_pixels",
    "eroded_mask_interior",
    "group_mask_points_by_track",
    "select_formal_mask_internal_points",
    "track_formal_mask_internal_points",
    "evaluate_coverage_readiness",
    "aggregate_evidence_v2",
    "aggregate_multilevel_evidence",
    "get_evidence_registry",
    "localize_temporal_intervals",
]


def __getattr__(name: str) -> object:
    """Lazily expose public APIs without importing every submodule at startup."""

    if name in {
        "BaseDepthProvider",
        "DepthObservation",
        "DepthRepresentation",
        "DepthScaleStatus",
        "FrameMockDepthProvider",
        "LargerValueMeans",
        "LegacyDepthProviderAdapter",
        "RealDepthProvider",
    }:
        from . import depth_provider

        return getattr(depth_provider, name)

    if name in {
        "CameraObservation",
        "CoordinateConvention",
        "DepthDefinition",
        "PixelCenterConvention",
        "TransformConvention",
    }:
        from .geometry import camera

        return getattr(camera, name)

    if name in {
        "GeometryScaleStatus",
        "GeometryScaleUnit",
        "Object3DObservation",
        "Point2DObservation",
        "Point3DObservation",
        "ReconstructionFrame",
        "Shared3DFrameObservation",
        "VisibilityStatus",
    }:
        from . import shared_3d_observation

        return getattr(shared_3d_observation, name)

    if name in {"backproject_pixel", "backproject_points"}:
        from .geometry import backprojection

        return getattr(backprojection, name)

    if name in {"project_point", "project_points"}:
        from .geometry import projection

        return getattr(projection, name)

    if name in {
        "camera_center_world",
        "camera_to_world",
        "transform_points",
        "world_to_camera",
    }:
        from .geometry import transforms

        return getattr(transforms, name)

    if name in {
        "DepthSample",
        "DepthSamplingMethod",
        "Object3DReconstructor",
        "Object3DReconstructorConfig",
        "Shared3DFrameBuilder",
        "build_shared_3d_frame_observation",
        "sample_depth",
    }:
        from . import reconstruction

        return getattr(reconstruction, name)

    if name in {
        "MissingReason",
        "ResidualEvidence",
        "aggregate_residual_evidence",
    }:
        from . import validity

        return getattr(validity, name)

    if name in {
        "BoundaryDepth3DResidual",
        "DepthOrder3DResidual",
        "EvidenceRole",
        "ReconstructionQualityEvidence",
        "SemanticSize3DResidual",
        "SpatialIntersection3DResidual",
        "Static3DContext",
        "reprojection_cycle_evidence",
    }:
        from . import static_3d

        return getattr(static_3d, name)

    if name in {
        "BaseSceneCutDetector",
        "BaseSequenceGeometryProvider",
        "DepthAlignmentMode",
        "DepthAlignmentObservation",
        "DepthAlignmentModelSelection",
        "Dynamic3DFrameValidity",
        "ForegroundMaskObservation",
        "HistogramFeatureSceneCutDetector",
        "LegacyDepthPoseSequenceAdapter",
        "MockSequenceGeometryProvider",
        "RelativePoseObservation",
        "CameraMotionRegime",
        "MotionRegimeObservation",
        "MotionRegimeThresholds",
        "PoseEstimateCandidate",
        "PoseGraphResult",
        "PoseModelType",
        "TranslationScaleStatus",
        "SceneCutDecision",
        "SequenceGeometryObservation",
        "SequenceGeometryQuality",
        "SequenceGeometryStabilizationResult",
        "SequenceDepthAlignmentResult",
        "SequenceScaleStatus",
        "Shared3DClipObservation",
        "SyntheticSequenceGeometryProvider",
        "UnifiedSequenceGeometryProvider",
        "apply_depth_alignment",
        "build_foreground_mask",
        "build_pose_graph",
        "classify_motion_regime",
        "estimate_adaptive_pose_candidates",
        "estimate_depth_alignment",
        "estimate_depth_alignment_from_correspondences",
        "estimate_depth_alignment_with_holdout",
        "select_depth_alignment_model",
        "select_depth_alignment_from_correspondences",
        "align_depth_sequence_to_reference",
        "stabilize_sequence_geometry",
    }:
        from . import sequence_geometry

        return getattr(sequence_geometry, name)

    if name in {
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
        "SyntheticPointTracker",
        "Track3DContinuityResidual",
        "assess_dynamic_3d_readiness",
        "compute_dynamic_reprojection_residual",
        "compute_track_3d_continuity_residuals",
        "reconstruct_point_tracks_3d",
        "relative_improvement",
        "summarize_point_track_coverage",
        "BaseObjectMotionModel",
        "DirectionConsistencyResidual",
        "EdgeTemporalResidual",
        "ObjectBindingResult",
        "ObjectDynamicAggregate",
        "ObjectDynamicObservation",
        "ObjectMedianTranslationModel",
        "ObjectMotionPrediction",
        "ObjectPointBinding",
        "ObjectPointTrack3D",
        "ObjectStructureGraph",
        "ObjectStructureTemporalResidual",
        "PointConstantVelocityModel",
        "PointRole",
        "RelativeVelocityResidual",
        "StructureEdge",
        "aggregate_object_dynamic_evidence",
        "assemble_object_point_tracks_3d",
        "bind_point_tracks_to_objects",
        "build_object_structure_graph",
        "compute_direction_consistency_residuals",
        "compute_relative_velocity_residuals",
        "compute_structure_temporal_residuals",
        "select_stable_object_point_tracks",
        "PersonKeypointBindingResult",
        "PersonKeypointCoverage",
        "bind_person_keypoints_to_shared_3d",
    }:
        from . import dynamic_3d

        return getattr(dynamic_3d, name)

    if name in {
        "BaseInstanceMaskProvider",
        "BoundaryOcclusionResidual",
        "ExistingDetectionMaskAdapter",
        "InstanceMaskCandidate",
        "InstanceSegmentationModelMetadata",
        "InstanceMaskObservation",
        "MaskBoundaryObservation",
        "MaskTracker",
        "MaskAssociationDiagnostic",
        "MaskObjectAssociationResult",
        "MockInstanceMaskProvider",
        "ObjectVisibilityObservation",
        "OcclusionDepthOrderResidual",
        "OcclusionGraph",
        "OcclusionRelation",
        "PredictedObjectSupport",
        "RealInstanceMaskProvider",
        "ReappearanceConsistencyResidual",
        "ReappearanceObservation",
        "SyntheticInstanceMaskProvider",
        "TrackedMaskObservation",
        "TrackedMaskProvider",
        "VisibilityExplanation",
        "VisibilityExplanationResidual",
        "VisibilityState",
        "SceneCutStatistics",
        "OcclusionEventInputs",
        "OcclusionEventValidation",
        "associate_instance_masks",
        "build_occlusion_graph",
        "compute_boundary_occlusion_residual",
        "compute_occlusion_depth_order_residual",
        "compute_visibility_explanation_residual",
        "evaluate_reappearance",
        "infer_visibility_state",
        "predict_object_support",
        "compute_scene_cut_statistics",
        "validate_occlusion_event",
        "adaptive_erosion_pixels",
        "eroded_mask_interior",
        "group_mask_points_by_track",
        "select_formal_mask_internal_points",
        "track_formal_mask_internal_points",
    }:
        from . import occlusion

        return getattr(occlusion, name)

    if name in {
        "BranchCoverageStatus",
        "CoverageReadiness",
        "evaluate_coverage_readiness",
    }:
        from . import coverage_readiness

        return getattr(coverage_readiness, name)

    if name in {
        "EvidenceApplicability",
        "AggregationEvidence",
        "EvidenceBranchSpec",
        "EvidenceFormality",
        "EvidenceLevel",
        "PointEvidenceAggregate",
        "EdgeEvidenceAggregate",
        "ObjectEvidenceAggregate",
        "FrameEvidenceAggregate",
        "ClipEvidenceAggregate",
        "MultilevelAggregationResult",
        "aggregate_evidence_v2",
        "aggregate_multilevel_evidence",
        "get_evidence_registry",
        "localize_temporal_intervals",
    }:
        from . import aggregation_v2

        return getattr(aggregation_v2, name)

    if name in {
        "BaseKeypointProvider",
        "Keypoint2D",
        "KeypointPrediction",
        "MockKeypointProvider",
        "RealHumanKeypointProvider",
    }:
        from . import keypoint_provider

        return getattr(keypoint_provider, name)

    if name in {
        "ApplicabilityGateResult",
        "CupHeightApplicabilityConfig",
        "CupHeightApplicabilityGate",
        "PersonHeightApplicabilityConfig",
        "PersonHeightApplicabilityGate",
        "applicability_skip_reason",
    }:
        from . import pose_applicability

        return getattr(pose_applicability, name)

    if name in {
        "DepthStatistic",
        "compute_depth_strategy",
        "make_frame_pair_id",
        "make_track_pair_id",
        "recompute_scale_depth_formula",
        "safe_bbox_bounds",
        "swapped_log_residual_consistent",
    }:
        from . import rsd_v2_error_audit

        return getattr(rsd_v2_error_audit, name)

    if name in {
        "aggregate_map_by_mask",
        "aggregate_points_by_mask",
        "aggregate_values",
        "topk_mean",
    }:
        from . import aggregation

        return getattr(aggregation, name)

    if name in {
        "DepthTemporalResidualResult",
        "aggregate_clip_depth_residuals",
        "aggregate_track_depth_residuals",
        "compute_depth_temporal_residual",
        "compute_frame_depth_reference",
        "aggregate_depth_transition_evidence",
        "depth_transition_evidence",
        "r_depth_cons_2p5d",
        "depth_consistency_plot_series_from_csv",
        "save_depth_consistency_tracks_plot_from_csv",
        "save_raw_and_thresholded_residual_plots_from_csv",
        "save_depth_consistency_tracks_plot",
    }:
        from . import depth_temporal_consistency

        return getattr(depth_temporal_consistency, name)

    if name in {
        "ObjectAssociator",
        "AssociationDiagnostics",
        "bbox_area_ratio",
        "bbox_iou",
        "deduplicate_frames_by_index",
        "normalize_object_label",
        "normalized_center_distance",
    }:
        from . import object_association

        return getattr(object_association, name)

    if name in {
        "ClipResidualSummary",
        "ObjectLevelResidual",
        "ObjectLevelResidualEvidence",
        "ObjectMaskObservation",
        "ObjectPairResidual",
        "build_object_level_residuals",
        "build_object_level_residuals_with_details",
        "build_object_level_residual_evidence",
        "build_object_pair_residuals",
        "summarize_clip_residuals",
    }:
        from . import multilevel_residuals

        return getattr(multilevel_residuals, name)

    if name in {
        "ObjectObservation",
        "ScalePrior",
        "compute_scale_depth_interval",
        "pairwise_scale_depth_residuals",
        "scale_depth_residual",
        "scale_depth_residual_log",
        "rsd_2d_coarse",
        "rsd_2d_coarse_log",
    }:
        from . import scale_depth

        return getattr(scale_depth, name)

    if name in {
        "ResolvedScalePrior",
        "ScalePriorRecord",
        "ScalePriorResolver",
        "default_scale_prior_resolver",
        "load_scale_prior_resolver",
    }:
        from . import scale_prior

        return getattr(scale_prior, name)

    if name in {
        "StrictPhysicalPriorEntry",
        "StrictPhysicalScalePriorResolver",
        "StrictResolvedPhysicalPrior",
        "load_strict_physical_prior_resolver",
    }:
        from . import strict_scale_prior

        return getattr(strict_scale_prior, name)

    if name in {
        "DimensionAlignedPriorEntry",
        "DimensionAlignedPriorResolver",
        "compute_dimension_aligned_rsd",
        "load_dimension_aligned_prior_resolver",
        "rsd_2d_dimension_aligned",
    }:
        from . import dimension_aligned_scale_depth

        return getattr(dimension_aligned_scale_depth, name)

    if name in {
        "ProjectedMeasurementResult",
        "ProjectedMeasurementRule",
        "ProjectedMeasurementRules",
        "compute_projected_measurement",
        "load_projected_measurement_rules",
    }:
        from . import projected_measurement

        return getattr(projected_measurement, name)

    if name in {
    "PhysicalPriorGateResult",
    "PersonHeightApplicabilityConfig",
    "PersonHeightApplicabilityGate",
        "evaluate_physical_prior_gate",
    }:
        from . import physical_prior_gate

        return getattr(physical_prior_gate, name)

    if name in {
        "EmpiricalPairPrior",
        "EmpiricalPairPriorMatch",
        "EmpiricalPairPriorResolver",
    }:
        from . import empirical_pair_prior

        return getattr(empirical_pair_prior, name)

    if name in {
        "ClipObservation",
        "FrameObservation",
        "ObjectTrack",
        "ResidualReport",
    }:
        from . import data_structures

        return getattr(data_structures, name)

    if name in {
        "ClipObservationJSON",
        "ClipResidualResultJSON",
        "FrameObservationJSON",
        "ObjectObservationJSON",
    }:
        from . import observations

        return getattr(observations, name)

    if name in {
        "BaseObjectProvider",
        "MockDepthProvider",
        "MockObjectProvider",
    }:
        from . import providers

        return getattr(providers, name)

    if name in {
        "RealObjectProvider",
        "bbox_area_to_mask_area",
        "normalize_label",
    }:
        from . import real_object_provider

        return getattr(real_object_provider, name)

    if name in {
        "get_object_provider",
    }:
        from . import provider_registry

        return getattr(provider_registry, name)

    if name in {
        "build_clip_observation",
        "build_frame_observation",
    }:
        from . import build_observations

        return getattr(build_observations, name)

    if name in {
        "ClipWindow",
        "build_clips",
        "extract_frames",
    }:
        from . import video_preprocess

        return getattr(video_preprocess, name)

    if name in {
        "load_clip_observation",
        "load_clip_residual_result",
        "load_frame_observation",
        "save_clip_observation",
        "save_clip_residual_result",
        "save_frame_observation",
    }:
        from . import io

        return getattr(io, name)

    if name in {
        "DepthProvider",
        "FlowProvider",
        "SegmentationProvider",
        "TrackerProvider",
    }:
        from . import interfaces

        return getattr(interfaces, name)

    if name in {
        "draw_object_summary",
        "draw_object_residual_map",
        "draw_pairwise_residual_graph",
        "draw_residual_heatmap_from_masks",
        "draw_multilevel_summary",
    }:
        from . import visualization

        return getattr(visualization, name)

    if name in {
        "ResidualValues",
        "ResidualWeights",
        "RealHumanKeypointProvider",
        "fuse_residual_evidence",
        "fuse_residuals",
        "normalize_residuals",
        "residual_values_to_dict",
        "video_risk_score",
    }:
        from . import residual_fusion

        return getattr(residual_fusion, name)

    raise AttributeError(f"module 'semantic3d' has no attribute {name!r}")
