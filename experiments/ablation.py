"""The six paper-specific residual-family ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from data.residual_dataset import RESIDUAL_NAMES, ResidualSequence
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

_CHANNEL_GROUPS = {
    "use_semantic_metric_prior": ("semantic_metric_prior",),
    "use_semantic_metric_temporal": ("semantic_metric_temporal",),
    "use_d1": (
        "dynamic_reprojection",
        "track_3d_continuity",
        "direction_consistency",
        "relative_velocity",
    ),
    "use_d2": (
        "point_reprojection",
        "boundary_reprojection",
        "depth_reprojection",
    ),
    "use_d3": ("relation", "occlusion", "reappearance"),
}
_DEFAULT_ABLATION = {
    **{key: True for key in _CHANNEL_GROUPS},
    "use_confidence": True,
    "use_availability_mask": True,
    "temporal_aggregation": "learned_head",
}


@dataclass(frozen=True)
class AblationConfig:
    values: Mapping[str, Any]
    changed_fields: tuple[str, ...]


def parse_ablation_config(config: Mapping[str, Any] | None = None) -> AblationConfig:
    supplied = dict(config or {})
    unknown = sorted(set(supplied) - set(_DEFAULT_ABLATION))
    if unknown:
        raise ValueError(f"Unknown ablation fields: {unknown}.")
    values = {**_DEFAULT_ABLATION, **supplied}
    for key in (*_CHANNEL_GROUPS, "use_confidence", "use_availability_mask"):
        if not isinstance(values[key], bool):
            raise ValueError(f"{key} must be boolean.")
    if values["temporal_aggregation"] not in {
        "learned_head",
        "mean_pool",
        "max_pool",
    }:
        raise ValueError("Unsupported temporal_aggregation.")
    changed = tuple(
        key for key, default in _DEFAULT_ABLATION.items() if values[key] != default
    )
    return AblationConfig(values=values, changed_fields=changed)


def apply_input_ablation(
    sample: ResidualSequence,
    config: Mapping[str, Any] | AblationConfig,
) -> tuple[ResidualSequence, tuple[str, ...]]:
    parsed = config if isinstance(config, AblationConfig) else parse_ablation_config(config)
    values = np.array(sample.residuals, copy=True)
    availability = np.array(sample.availability, copy=True)
    confidence = np.array(sample.confidence, copy=True)
    for flag, names in _CHANNEL_GROUPS.items():
        if parsed.values[flag]:
            continue
        for name in names:
            index = RESIDUAL_NAMES.index(name)
            values[:, index] = np.nan
            availability[:, index] = False
            confidence[:, index] = 0.0
    if not parsed.values["use_confidence"]:
        confidence = availability.astype(np.float32)
    if not parsed.values["use_availability_mask"]:
        values = np.nan_to_num(values, nan=0.0)
        availability = np.ones_like(availability, dtype=bool)
        confidence = np.where(availability, confidence, 0.0).astype(np.float32)
    transformed = ResidualSequence(
        residuals=values,
        availability=availability,
        confidence=confidence,
        label=sample.label,
        sample_id=sample.sample_id,
        clip_ids=sample.clip_ids,
        dataset_name=sample.dataset_name,
        source_video_id=sample.source_video_id,
        group_id=sample.group_id,
        residual_sequence_path=sample.residual_sequence_path,
    )
    return transformed, parsed.changed_fields


def select_ablation(
    residuals: Iterable[ResidualEvidence],
    name: str,
) -> list[ResidualEvidence]:
    if name not in ABLATIONS:
        raise ValueError(f"Unknown paper ablation: {name}.")
    selected = ABLATIONS[name]
    return list(residuals) if selected is None else [row for row in residuals if row.name in selected]


__all__ = [
    "ABLATIONS",
    "AblationConfig",
    "apply_input_ablation",
    "parse_ablation_config",
    "select_ablation",
]
