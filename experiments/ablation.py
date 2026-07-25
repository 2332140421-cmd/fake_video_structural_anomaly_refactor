"""The six paper-specific residual-family ablations."""

from __future__ import annotations

from typing import Iterable

from data.schemas import ResidualEvidence

ABLATIONS = {
    "without_3d_structure": set(),
    "motion_tracks_only": {
        "dynamic_reprojection",
        "track_3d_continuity",
        "direction_consistency",
        "relative_velocity",
    },
    "with_metric_object_semantic": {
        "dynamic_reprojection",
        "track_3d_continuity",
        "semantic_metric_prior",
        "semantic_metric_temporal",
    },
    "with_d2_reprojection": {
        "dynamic_reprojection",
        "track_3d_continuity",
        "semantic_metric_prior",
        "semantic_metric_temporal",
        "point_reprojection",
        "boundary_reprojection",
        "depth_reprojection",
    },
    "with_d3_relations": {
        "dynamic_reprojection",
        "track_3d_continuity",
        "direction_consistency",
        "relative_velocity",
        "semantic_metric_prior",
        "semantic_metric_temporal",
        "point_reprojection",
        "boundary_reprojection",
        "depth_reprojection",
        "relation",
        "occlusion",
        "reappearance",
    },
    "full_fusion": None,
}


def select_ablation(
    residuals: Iterable[ResidualEvidence],
    name: str,
) -> list[ResidualEvidence]:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown paper ablation: {name}.")
    selected = ABLATIONS[name]
    return list(residuals) if selected is None else [row for row in residuals if row.name in selected]


__all__ = ["ABLATIONS", "select_ablation"]
