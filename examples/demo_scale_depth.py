"""Demo for scale-depth residuals using synthetic football and elephant data."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from structural_anomaly import (
    ObjectObservation,
    compute_scale_depth_residual,
    compute_scale_depth_residual_log,
)


def _to_float(value: object) -> float:
    """Convert a scalar numpy value or Python number to float for printing."""

    return float(value.item()) if hasattr(value, "item") else float(value)


def main() -> None:
    frame_area = 1920 * 1080
    scale_prior = {
        # Approximate physical height/diameter intervals in meters.
        "football": (0.20, 0.24),
        "elephant": (2.40, 3.40),
    }

    football = ObjectObservation(
        class_name="football",
        mask_area=4_800,
        depth=12.0,
    )
    elephant_consistent = ObjectObservation(
        class_name="elephant",
        mask_area=120_000,
        depth=36.0,
    )
    elephant_suspicious = ObjectObservation(
        class_name="elephant",
        mask_area=120_000,
        depth=8.0,
    )

    for name, elephant in [
        ("consistent pair", elephant_consistent),
        ("suspicious pair", elephant_suspicious),
    ]:
        ratio_result = compute_scale_depth_residual(
            football, elephant, frame_area, scale_prior
        )
        log_result = compute_scale_depth_residual_log(
            football, elephant, frame_area, scale_prior
        )

        print(f"\n{name}")
        print(f"p_football = {_to_float(ratio_result.projection_a):.6f}")
        print(f"p_elephant = {_to_float(ratio_result.projection_b):.6f}")
        print(
            "ratio-space: "
            f"Z_A/Z_B={_to_float(ratio_result.ratio):.6f}, "
            f"interval=[{_to_float(ratio_result.lower):.6f}, "
            f"{_to_float(ratio_result.upper):.6f}], "
            f"R_sd={_to_float(ratio_result.residual):.6f}"
        )
        print(
            "log-space:   "
            f"log_ratio={_to_float(log_result.ratio):.6f}, "
            f"interval=[{_to_float(log_result.lower):.6f}, "
            f"{_to_float(log_result.upper):.6f}], "
            f"R_sd_log={_to_float(log_result.residual):.6f}"
        )


if __name__ == "__main__":
    main()
