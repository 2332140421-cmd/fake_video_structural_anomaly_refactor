"""Formal branch registry for quality-aware structural evidence aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Mapping


class EvidenceLevel(str, Enum):
    POINT = "point"
    EDGE = "edge"
    OBJECT = "object"
    PAIR = "pair"
    FRAME = "frame"


class EvidenceFormality(str, Enum):
    FORMAL = "formal"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class EvidenceBranchSpec:
    """Static configuration for one residual branch, never learned from labels."""

    name: str
    supported_geometry_modes: tuple[str, ...]
    evidence_level: EvidenceLevel | str
    event_conditioned: bool
    default_quality_floor: float
    normalization_type: str
    localization_target: str
    formal_or_diagnostic: EvidenceFormality | str = EvidenceFormality.FORMAL

    def __post_init__(self) -> None:
        if not self.name or not self.supported_geometry_modes:
            raise ValueError("Evidence branch name and geometry modes are required.")
        if not 0.0 <= float(self.default_quality_floor) <= 1.0:
            raise ValueError("default_quality_floor must be in [0, 1].")
        object.__setattr__(self, "evidence_level", EvidenceLevel(self.evidence_level))
        object.__setattr__(self, "formal_or_diagnostic", EvidenceFormality(self.formal_or_diagnostic))
        object.__setattr__(self, "supported_geometry_modes", tuple(self.supported_geometry_modes))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready registry row."""

        result = asdict(self)
        result["evidence_level"] = self.evidence_level.value
        result["formal_or_diagnostic"] = self.formal_or_diagnostic.value
        return result


_STATIC_MODES = ("static_camera_3d", "full_se3_3d")
_ROTATION_MODES = ("static_camera_3d", "rotation_compensated", "full_se3_3d")


DEFAULT_EVIDENCE_REGISTRY: Mapping[str, EvidenceBranchSpec] = {
    "semantic_size_3d": EvidenceBranchSpec("semantic_size_3d", _STATIC_MODES, "object", False, 0.50, "prior_interval_distance", "object_mask"),
    "depth_order_3d": EvidenceBranchSpec("depth_order_3d", _STATIC_MODES, "pair", True, 0.45, "relative_depth_order", "object_pair"),
    "boundary_depth_3d": EvidenceBranchSpec("boundary_depth_3d", _STATIC_MODES, "edge", False, 0.45, "robust_depth_jump", "object_boundary"),
    "spatial_intersection_3d": EvidenceBranchSpec("spatial_intersection_3d", _STATIC_MODES, "pair", False, 0.50, "intersection_ratio", "object_pair"),
    "track_3d_continuity": EvidenceBranchSpec("track_3d_continuity", _STATIC_MODES, "point", False, 0.40, "object_scale_normalized", "point_track"),
    "direction_consistency": EvidenceBranchSpec("direction_consistency", _ROTATION_MODES, "point", False, 0.40, "one_minus_cosine", "point_track"),
    "relative_velocity_change": EvidenceBranchSpec("relative_velocity_change", _STATIC_MODES, "point", False, 0.40, "object_scale_per_frame", "point_track"),
    "dynamic_reprojection": EvidenceBranchSpec("dynamic_reprojection", _ROTATION_MODES, "point", False, 0.40, "image_diagonal_normalized", "image_point"),
    "structure_temporal": EvidenceBranchSpec("structure_temporal", _STATIC_MODES, "edge", False, 0.45, "object_scale_normalized", "structure_edge"),
    "occlusion_depth_order": EvidenceBranchSpec("occlusion_depth_order", _STATIC_MODES, "pair", True, 0.50, "relative_depth_order", "occlusion_pair"),
    "visibility_explanation": EvidenceBranchSpec("visibility_explanation", _ROTATION_MODES, "object", True, 0.50, "visible_support_change", "object_mask"),
    "boundary_occlusion": EvidenceBranchSpec("boundary_occlusion", _ROTATION_MODES, "edge", True, 0.50, "image_diagonal_normalized", "mask_boundary"),
    "reappearance_consistency": EvidenceBranchSpec("reappearance_consistency", _ROTATION_MODES, "object", True, 0.50, "identity_and_structure_distance", "object_mask"),
}


def get_evidence_registry(*, formal_only: bool = False) -> dict[str, EvidenceBranchSpec]:
    """Return an independent registry copy, optionally excluding diagnostics."""

    return {
        name: spec for name, spec in DEFAULT_EVIDENCE_REGISTRY.items()
        if not formal_only or spec.formal_or_diagnostic == EvidenceFormality.FORMAL
    }


def register_evidence_branch(
    registry: Mapping[str, EvidenceBranchSpec],
    spec: EvidenceBranchSpec,
    *,
    allow_diagnostic: bool = False,
) -> dict[str, EvidenceBranchSpec]:
    """Return a registry with one explicit branch added.

    Diagnostic branches require an explicit opt-in, preventing accidental use
    as formal anomaly evidence.
    """

    if spec.formal_or_diagnostic == EvidenceFormality.DIAGNOSTIC and not allow_diagnostic:
        raise ValueError("Diagnostic evidence cannot be auto-registered as a formal branch.")
    output = dict(registry)
    if spec.name in output:
        raise ValueError(f"Evidence branch already registered: {spec.name}")
    output[spec.name] = spec
    return output
