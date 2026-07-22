"""Versioned offline structural-enhancement dataset construction."""

from .cache import CacheLookup, StageCache, build_cache_key
from .clip_alignment import ClipAlignmentObservation
from .ids import StableIdFactory, stable_id
from .manifest import build_clip_manifests, inspect_video, split_scene_segments
from .schema import (
    P4B5_PIPELINE_VERSION,
    SCHEMA_VERSION,
    Applicability,
    DatasetManifest,
    EvidenceRecord,
)
from .p4b5_contracts import (
    ClipGeometryDecision,
    ClipTrackHandoffObservation,
    CoverageMetric,
    SynchronizedDepthOrder,
    adaptive_structure_point_target,
    build_fixed_structure_edges,
    classify_clip_geometry,
    synchronized_depth_order,
)
from .validation import DatasetValidationReport, validate_dataset
from .writer import DatasetWriter

__all__ = [
    "SCHEMA_VERSION",
    "P4B5_PIPELINE_VERSION",
    "Applicability",
    "CacheLookup",
    "ClipGeometryDecision",
    "ClipAlignmentObservation",
    "ClipTrackHandoffObservation",
    "CoverageMetric",
    "DatasetManifest",
    "DatasetValidationReport",
    "DatasetWriter",
    "EvidenceRecord",
    "StableIdFactory",
    "SynchronizedDepthOrder",
    "StageCache",
    "build_cache_key",
    "adaptive_structure_point_target",
    "build_fixed_structure_edges",
    "build_clip_manifests",
    "inspect_video",
    "split_scene_segments",
    "classify_clip_geometry",
    "stable_id",
    "validate_dataset",
    "synchronized_depth_order",
]
