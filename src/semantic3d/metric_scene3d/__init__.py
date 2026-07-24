"""Single-frame camera-coordinate metric visible-surface 2.5D reconstruction."""

from .boundary import reconstruct_boundary_points, sample_mask_boundary
from .contracts import (
    BoundaryMetricPoint,
    MetricPointType,
    MetricSurfacePoint,
    ObjectSurfacePointCloud,
    SingleFrameStructureGraph,
    StructureEdge3D,
)
from .image_geometry import ImageCoordinateTransform, align_binary_mask
from .reconstruction import (
    backproject_metric_arrays,
    build_object_surface_pointcloud,
    build_scene_surface_points,
    canonicalize_z_depth,
    project_metric_arrays,
    to_shared_object_observation,
)
from .structure_graph import build_single_frame_structure_graph
from .structure_points import select_geometric_track_points

__all__ = [
    "BoundaryMetricPoint",
    "ImageCoordinateTransform",
    "MetricPointType",
    "MetricSurfacePoint",
    "ObjectSurfacePointCloud",
    "SingleFrameStructureGraph",
    "StructureEdge3D",
    "align_binary_mask",
    "backproject_metric_arrays",
    "build_object_surface_pointcloud",
    "build_scene_surface_points",
    "build_single_frame_structure_graph",
    "canonicalize_z_depth",
    "project_metric_arrays",
    "reconstruct_boundary_points",
    "sample_mask_boundary",
    "select_geometric_track_points",
    "to_shared_object_observation",
]
