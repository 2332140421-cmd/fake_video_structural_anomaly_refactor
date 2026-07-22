"""Explicit transforms between otherwise isolated clip-local 3D systems."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class ClipAlignmentObservation:
    """Validated similarity alignment inferred only from overlapping frames."""

    source_clip_id: str
    target_clip_id: str
    transform: Optional[np.ndarray]
    scale: float
    shift: Optional[np.ndarray]
    overlap_frame_count: int
    background_support_count: int
    alignment_error: float
    holdout_error: float
    valid: bool
    missing_reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_clip_id == self.target_clip_id:
            raise ValueError("Clip alignment must connect distinct clips")
        if self.valid:
            transform = np.asarray(self.transform, dtype=float)
            shift = np.asarray(self.shift, dtype=float)
            if transform.shape != (4, 4) or shift.shape != (3,):
                raise ValueError("Valid alignment requires 4x4 transform and 3-vector shift")
            if not math.isfinite(self.scale) or self.scale <= 0.0:
                raise ValueError("Valid alignment scale must be positive and finite")
            if self.overlap_frame_count <= 0 or self.background_support_count <= 0:
                raise ValueError("Valid alignment requires overlap and background support")
            if self.missing_reason:
                raise ValueError("Valid alignment cannot have missing_reason")
            object.__setattr__(self, "transform", transform)
            object.__setattr__(self, "shift", shift)
        else:
            if not self.missing_reason:
                raise ValueError("Invalid alignment requires missing_reason")
            if self.transform is not None:
                object.__setattr__(self, "transform", np.asarray(self.transform, dtype=float))
            if self.shift is not None:
                object.__setattr__(self, "shift", np.asarray(self.shift, dtype=float))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points only after an overlap-supported alignment is valid."""

        if not self.valid or self.transform is None:
            raise ValueError("Cannot transform points across unaligned clip coordinates")
        values = np.asarray(points, dtype=float)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError("points must have shape Nx3")
        homogeneous = np.column_stack([values * self.scale, np.ones(len(values))])
        transformed = (self.transform @ homogeneous.T).T[:, :3]
        if self.shift is not None:
            transformed += self.shift
        return transformed
