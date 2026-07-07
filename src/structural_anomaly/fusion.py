"""Residual fusion interface reserved for the full prototype.

The current implementation focuses on R_sd. This module keeps a simple,
testable interface that can later absorb R_flow, R_track, R_depth_cons, R_occ,
and R_corr without changing call sites.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional


@dataclass(frozen=True)
class ResidualBundle:
    """Container for object-pair structural residuals.

    Attributes:
        r_sd: Scale-depth consistency residual.
        r_flow: Placeholder for optical-flow consistency residual.
        r_track: Placeholder for point-trajectory consistency residual.
        r_depth_cons: Placeholder for depth consistency residual.
        r_occ: Placeholder for occlusion-order residual.
        r_corr: Placeholder for spatial correspondence residual.
    """

    r_sd: float
    r_flow: Optional[float] = None
    r_track: Optional[float] = None
    r_depth_cons: Optional[float] = None
    r_occ: Optional[float] = None
    r_corr: Optional[float] = None


def fuse_residuals(
    residuals: ResidualBundle, weights: Optional[Mapping[str, float]] = None
) -> float:
    """Compute a weighted sum over available residual terms.

    Missing residuals are skipped, so future terms can be added incrementally.
    By default, R_sd has weight 1.0 and all future terms have weight 0.0.
    """

    default_weights = {
        "r_sd": 1.0,
        "r_flow": 0.0,
        "r_track": 0.0,
        "r_depth_cons": 0.0,
        "r_occ": 0.0,
        "r_corr": 0.0,
    }
    if weights is not None:
        default_weights.update(weights)

    total = 0.0
    for name, weight in default_weights.items():
        value = getattr(residuals, name)
        if value is not None:
            total += float(weight) * float(value)
    return total
