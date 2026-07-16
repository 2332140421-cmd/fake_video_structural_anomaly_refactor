"""Validity-aware evidence contracts for structural residuals.

This module is the canonical validity boundary for new 3D code. Legacy
residual APIs remain available, but new geometry and residual modules should
exchange :class:`ResidualEvidence` instead of using zero for missing evidence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class MissingReason(str, Enum):
    """Stable missing-evidence reasons shared by new structural modules."""

    NOT_OBSERVED = "not_observed"
    MISSING_DEPTH = "missing_depth"
    INVALID_DEPTH = "invalid_depth"
    MISSING_INTRINSICS = "missing_intrinsics"
    MISSING_POSE = "missing_pose"
    MISSING_CAMERA = "missing_camera"
    MISSING_3D_RECONSTRUCTION = "missing_3d_reconstruction"
    MISSING_RESIDUAL_SOURCE = "missing_residual_source"
    UNSUPPORTED_REPRESENTATION = "unsupported_representation"
    LEGACY_NORMALIZED_DEPTH = "legacy_normalized_depth_not_geometry"
    NO_VALID_EVIDENCE = "no_valid_evidence"
    UNKNOWN = "unknown"


def _reason_value(reason: MissingReason | str) -> str:
    """Return a stable string for an enum or caller-defined reason."""

    return reason.value if isinstance(reason, MissingReason) else str(reason)


@dataclass(frozen=True)
class ResidualEvidence:
    """One residual value with explicit validity, quality, and provenance.

    ``quality`` is an engineering quality score, not a calibrated probability.
    A valid normal observation may have ``value=0``. An invalid or missing
    observation must have ``value=NaN`` so absence cannot be interpreted as a
    low anomaly score.
    """

    name: str
    value: float
    valid: bool
    quality: float
    missing_reason: str = ""
    source_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = float(self.value)
        quality = float(self.quality)
        if not self.name.strip():
            raise ValueError("ResidualEvidence.name must not be empty.")
        if not math.isfinite(quality) or not 0.0 <= quality <= 1.0:
            raise ValueError("ResidualEvidence.quality must be finite and in [0, 1].")
        if self.valid:
            if not math.isfinite(value):
                raise ValueError("Valid ResidualEvidence requires a finite value.")
            if self.missing_reason:
                raise ValueError("Valid ResidualEvidence cannot have missing_reason.")
        else:
            if not math.isnan(value):
                raise ValueError("Invalid ResidualEvidence must use value=NaN.")
            if not self.missing_reason:
                raise ValueError("Invalid ResidualEvidence requires missing_reason.")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "quality", quality)
        object.__setattr__(self, "source_ids", tuple(str(item) for item in self.source_ids))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def observed(
        cls,
        name: str,
        value: float,
        quality: float = 1.0,
        source_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResidualEvidence":
        """Create valid observed evidence; zero is accepted as a normal value."""

        return cls(
            name=name,
            value=float(value),
            valid=True,
            quality=float(quality),
            source_ids=tuple(source_ids),
            metadata=dict(metadata or {}),
        )

    @classmethod
    def missing(
        cls,
        name: str,
        reason: MissingReason | str,
        source_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "ResidualEvidence":
        """Create invalid evidence whose value is explicitly NaN."""

        return cls(
            name=name,
            value=float("nan"),
            valid=False,
            quality=0.0,
            missing_reason=_reason_value(reason),
            source_ids=tuple(source_ids),
            metadata=dict(metadata or {}),
        )


def aggregate_residual_evidence(
    name: str,
    evidences: Iterable[ResidualEvidence],
    method: str = "mean",
    require_all_valid: bool = True,
) -> ResidualEvidence:
    """Aggregate evidence without ever substituting zero for missing input.

    With the default strict policy, any invalid component makes the aggregate
    invalid. Setting ``require_all_valid=False`` explicitly aggregates only
    valid components and records the number of skipped components in metadata.
    """

    items = list(evidences)
    if not items:
        return ResidualEvidence.missing(name, MissingReason.NO_VALID_EVIDENCE)
    invalid = [item for item in items if not item.valid]
    if invalid and require_all_valid:
        return ResidualEvidence.missing(
            name,
            invalid[0].missing_reason,
            source_ids=tuple(source for item in items for source in item.source_ids),
            metadata={"invalid_components": len(invalid), "total_components": len(items)},
        )
    valid = [item for item in items if item.valid]
    if not valid:
        return ResidualEvidence.missing(name, MissingReason.NO_VALID_EVIDENCE)
    values = np.asarray([item.value for item in valid], dtype=float)
    if method == "mean":
        value = float(values.mean())
    elif method == "max":
        value = float(values.max())
    elif method == "median":
        value = float(np.median(values))
    else:
        raise ValueError("method must be 'mean', 'max', or 'median'.")
    return ResidualEvidence.observed(
        name,
        value,
        quality=float(np.mean([item.quality for item in valid])),
        source_ids=tuple(source for item in valid for source in item.source_ids),
        metadata={"valid_components": len(valid), "invalid_components": len(invalid)},
    )


def evidence_from_legacy_value(
    name: str,
    value: float,
    *,
    observed: bool,
    quality: float = 1.0,
    missing_reason: MissingReason | str = MissingReason.MISSING_RESIDUAL_SOURCE,
    source_ids: Sequence[str] = (),
) -> ResidualEvidence:
    """Adapt a legacy scalar only when observation presence is explicit."""

    if not observed:
        return ResidualEvidence.missing(
            name,
            missing_reason,
            source_ids=source_ids,
            metadata={"adapter": "legacy_scalar", "legacy_value_ignored": float(value)},
        )
    return ResidualEvidence.observed(
        name,
        value,
        quality=quality,
        source_ids=source_ids,
        metadata={"adapter": "legacy_scalar"},
    )
