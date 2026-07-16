"""Joint P3-0.5 pose/depth graph stabilization and validity gating."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ..depth_provider import DepthObservation
from .depth_alignment import (
    DepthAlignmentModelSelection,
    SequenceDepthAlignmentResult,
    align_depth_sequence_to_reference,
    select_depth_alignment_from_correspondences,
)
from .observation import SequenceScaleStatus
from .pose_estimation import (
    LayeredPoseEstimator,
    PosePairEstimation,
    estimate_adaptive_pose_candidates,
)
from .pose_graph import PoseGraphResult, build_pose_graph


@dataclass(frozen=True)
class Dynamic3DFrameValidity:
    """Joint pose/depth eligibility for future dynamic 3D, not an anomaly."""

    frame_index: int
    dynamic_3d_valid: bool
    value: float
    quality: float
    pose_graph_connected: bool
    depth_graph_connected: bool
    pose_scale_compatible_with_depth: bool
    coordinate_conventions_consistent: bool
    scene_cut: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = float(self.value)
        quality = float(self.quality)
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("Dynamic3DFrameValidity quality must be in [0, 1].")
        if self.dynamic_3d_valid:
            if not math.isfinite(value) or self.missing_reason:
                raise ValueError("Valid dynamic 3D frame requires finite value and no reason.")
        else:
            if not math.isnan(value) or not self.missing_reason:
                raise ValueError("Invalid dynamic 3D frame requires NaN value and reason.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SequenceGeometryStabilizationResult:
    """Pose graph, depth graph, and evidence-aware frame gates."""

    frame_indices: tuple[int, ...]
    reference_frame: int
    pose_pairs: tuple[PosePairEstimation, ...]
    pose_graph: PoseGraphResult
    depth_selections: tuple[DepthAlignmentModelSelection, ...]
    depth_alignment: SequenceDepthAlignmentResult
    frame_validity: Mapping[int, Dynamic3DFrameValidity]
    sequence_scale_status: SequenceScaleStatus | str
    dynamic_3d_valid: bool
    quality: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        status = SequenceScaleStatus(self.sequence_scale_status)
        if not math.isfinite(self.quality) or not 0.0 <= self.quality <= 1.0:
            raise ValueError("Stabilization quality must be in [0, 1].")
        if self.valid and self.missing_reason:
            raise ValueError("Valid stabilization cannot have missing_reason.")
        if not self.valid and not self.missing_reason:
            raise ValueError("Invalid stabilization requires missing_reason.")
        if self.dynamic_3d_valid and not all(
            item.dynamic_3d_valid for item in self.frame_validity.values()
        ):
            raise ValueError("Sequence dynamic_3d_valid requires every frame gate.")
        object.__setattr__(self, "frame_indices", tuple(self.frame_indices))
        object.__setattr__(self, "pose_pairs", tuple(self.pose_pairs))
        object.__setattr__(self, "depth_selections", tuple(self.depth_selections))
        object.__setattr__(self, "frame_validity", dict(self.frame_validity))
        object.__setattr__(self, "sequence_scale_status", status)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        """Return a concise machine-readable stabilization summary."""

        return {
            "frame_indices": list(self.frame_indices),
            "reference_frame": self.reference_frame,
            "pose_graph": self.pose_graph.to_dict(),
            "depth_alignment": self.depth_alignment.to_dict(),
            "frame_validity": {
                str(index): asdict(item) for index, item in self.frame_validity.items()
            },
            "sequence_scale_status": self.sequence_scale_status.value,
            "dynamic_3d_valid": self.dynamic_3d_valid,
            "quality": self.quality,
            "valid": self.valid,
            "missing_reason": self.missing_reason,
            "metadata": dict(self.metadata),
        }


def stabilize_sequence_geometry(
    images: Mapping[int, np.ndarray],
    depths: Mapping[int, DepthObservation],
    K: np.ndarray,
    *,
    frame_indices: Sequence[int],
    foreground_masks: Mapping[int, np.ndarray] | None = None,
    scene_cut_flags: Mapping[int, bool] | None = None,
    temporal_strides: Sequence[int] = (1, 2, 4),
    coordinate_conventions_consistent: bool = True,
    minimum_pose_quality: float = 0.15,
    minimum_depth_quality: float = 0.20,
) -> SequenceGeometryStabilizationResult:
    """Run adaptive pose/depth graphs and apply the joint dynamic-3D gate."""

    indices = tuple(int(index) for index in frame_indices)
    reference = indices[0]
    pose_pairs = estimate_adaptive_pose_candidates(
        images,
        K,
        frame_indices=indices,
        foreground_masks=foreground_masks,
        depths=depths,
        scene_cut_flags=scene_cut_flags,
        temporal_strides=temporal_strides,
        estimator=LayeredPoseEstimator(minimum_quality=minimum_pose_quality),
    )
    pose_candidates = tuple(pair.selected for pair in pose_pairs)
    pose_graph = build_pose_graph(
        indices,
        pose_candidates,
        reference_frame=reference,
        minimum_edge_quality=minimum_pose_quality,
    )
    depth_selections: list[DepthAlignmentModelSelection] = []
    for pair in pose_pairs:
        tracks = pair.tracks
        if tracks.match_count == 0:
            continue
        depth_selections.append(
            select_depth_alignment_from_correspondences(
                depths[tracks.source_frame_index],
                depths[tracks.target_frame_index],
                tracks.source_points,
                tracks.target_points,
                source_frame=tracks.source_frame_index,
                target_frame=tracks.target_frame_index,
                source_foreground_mask=(
                    None
                    if foreground_masks is None
                    else foreground_masks.get(tracks.source_frame_index)
                ),
                target_foreground_mask=(
                    None
                    if foreground_masks is None
                    else foreground_masks.get(tracks.target_frame_index)
                ),
                metadata={
                    "pose_candidate_valid": pair.selected.valid,
                    "motion_regime": pair.motion_regime.regime.value,
                    "semantic_scale_prior_used": False,
                },
            )
        )
    depth_alignment = align_depth_sequence_to_reference(
        indices,
        depth_selections,
        reference_frame=reference,
        minimum_edge_quality=minimum_depth_quality,
    )
    scale_status = (
        SequenceScaleStatus.RELATIVE_ALIGNED_SEQUENCE
        if depth_alignment.valid
        else SequenceScaleStatus.RELATIVE_PER_FRAME
    )
    selected_edge_for_frame = {
        edge.target_frame_index: edge for edge in pose_graph.selected_edges
    }
    selected_edge_for_frame.update(
        {
            edge.source_frame_index: edge
            for edge in pose_graph.selected_edges
            if edge.source_frame_index != reference
            and edge.source_frame_index not in selected_edge_for_frame
        }
    )
    cuts = dict(scene_cut_flags or {})
    gates: dict[int, Dynamic3DFrameValidity] = {}
    for index in indices:
        pose_connected = pose_graph.T_world_from_camera_by_frame[index] is not None
        depth_frame = depth_alignment.per_frame[index]
        depth_connected = depth_frame.valid
        edge = selected_edge_for_frame.get(index)
        pose_scale_compatible = (
            True if index == reference else bool(edge and edge.pose_scale_compatible_with_depth)
        )
        cut = bool(cuts.get(index, False))
        reason = ""
        if cut:
            reason = "scene_cut_boundary"
        elif not coordinate_conventions_consistent:
            reason = "camera_depth_coordinate_convention_mismatch"
        elif not pose_connected:
            reason = "pose_graph_disconnected"
        elif not depth_connected:
            reason = "depth_alignment_graph_disconnected"
        elif not pose_scale_compatible:
            reason = "pose_translation_scale_unavailable"
        elif edge is not None and edge.quality < minimum_pose_quality:
            reason = "pose_quality_below_threshold"
        elif depth_frame.quality < minimum_depth_quality and index != reference:
            reason = "depth_alignment_holdout_quality_below_threshold"
        dynamic_valid = not reason
        local_quality = float(
            min(
                1.0 if index == reference else (edge.quality if edge else 0.0),
                depth_frame.quality,
            )
        )
        gates[index] = Dynamic3DFrameValidity(
            frame_index=index,
            dynamic_3d_valid=dynamic_valid,
            value=local_quality if dynamic_valid else float("nan"),
            quality=local_quality,
            pose_graph_connected=pose_connected,
            depth_graph_connected=depth_connected,
            pose_scale_compatible_with_depth=pose_scale_compatible,
            coordinate_conventions_consistent=coordinate_conventions_consistent,
            scene_cut=cut,
            missing_reason=reason,
            metadata={
                "quality_is_probability": False,
                "anomaly_score": False,
            },
        )
    all_dynamic = all(item.dynamic_3d_valid for item in gates.values())
    quality = float(
        np.mean(
            [
                pose_graph.pose_graph_quality,
                depth_alignment.quality,
                np.mean([item.quality for item in gates.values()]),
            ]
        )
    )
    pipeline_valid = bool(pose_graph.connected_frame_ratio > 0.0 and depths)
    return SequenceGeometryStabilizationResult(
        frame_indices=indices,
        reference_frame=reference,
        pose_pairs=pose_pairs,
        pose_graph=pose_graph,
        depth_selections=tuple(depth_selections),
        depth_alignment=depth_alignment,
        frame_validity=gates,
        sequence_scale_status=scale_status,
        dynamic_3d_valid=all_dynamic,
        quality=quality,
        valid=pipeline_valid,
        missing_reason="" if pipeline_valid else "no_connected_sequence_geometry",
        metadata={
            "formal_dynamic_anomaly_residuals_computed": False,
            "semantic_scale_prior_used": False,
            "scale_prior_config_read": False,
            "geometry_failure_is_forgery": False,
        },
    )

