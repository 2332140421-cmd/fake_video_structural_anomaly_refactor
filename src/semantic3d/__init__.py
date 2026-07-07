"""Semantic 3D structural residuals for forged video anomaly prototypes."""

from __future__ import annotations

__all__ = [
    "BaseObjectProvider",
    "ClipWindow",
    "ClipObservation",
    "ClipObservationJSON",
    "ClipResidualResultJSON",
    "ClipResidualSummary",
    "DepthProvider",
    "FlowProvider",
    "FrameObservation",
    "FrameObservationJSON",
    "MockDepthProvider",
    "MockObjectProvider",
    "ObjectLevelResidual",
    "ObjectMaskObservation",
    "ObjectObservation",
    "ObjectObservationJSON",
    "ObjectPairResidual",
    "ObjectTrack",
    "ResidualReport",
    "ResidualValues",
    "ResidualWeights",
    "ScalePrior",
    "SegmentationProvider",
    "TrackerProvider",
    "aggregate_map_by_mask",
    "aggregate_points_by_mask",
    "aggregate_values",
    "build_clip_observation",
    "build_clips",
    "build_frame_observation",
    "bbox_area_to_mask_area",
    "build_object_level_residuals",
    "build_object_level_residuals_with_details",
    "build_object_pair_residuals",
    "compute_scale_depth_interval",
    "draw_multilevel_summary",
    "draw_object_residual_map",
    "draw_object_summary",
    "draw_pairwise_residual_graph",
    "draw_residual_heatmap_from_masks",
    "extract_frames",
    "fuse_residuals",
    "load_clip_observation",
    "load_clip_residual_result",
    "load_frame_observation",
    "normalize_residuals",
    "pairwise_scale_depth_residuals",
    "residual_values_to_dict",
    "save_clip_observation",
    "save_clip_residual_result",
    "save_frame_observation",
    "scale_depth_residual",
    "scale_depth_residual_log",
    "summarize_clip_residuals",
    "topk_mean",
    "video_risk_score",
]


def __getattr__(name: str) -> object:
    """Lazily expose public APIs without importing every submodule at startup."""

    if name in {
        "aggregate_map_by_mask",
        "aggregate_points_by_mask",
        "aggregate_values",
        "topk_mean",
    }:
        from . import aggregation

        return getattr(aggregation, name)

    if name in {
        "ClipResidualSummary",
        "ObjectLevelResidual",
        "ObjectMaskObservation",
        "ObjectPairResidual",
        "build_object_level_residuals",
        "build_object_level_residuals_with_details",
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
    }:
        from . import scale_depth

        return getattr(scale_depth, name)

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
        "fuse_residuals",
        "normalize_residuals",
        "residual_values_to_dict",
        "video_risk_score",
    }:
        from . import residual_fusion

        return getattr(residual_fusion, name)

    raise AttributeError(f"module 'semantic3d' has no attribute {name!r}")
    "RealObjectProvider",
    "get_object_provider",
    "normalize_label",
