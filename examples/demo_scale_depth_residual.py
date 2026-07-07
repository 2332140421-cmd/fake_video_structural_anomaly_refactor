"""Demo for object-level scale-depth residual R_sd."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.scale_depth import (  # noqa: E402
    ObjectObservation,
    ScalePrior,
    compute_scale_depth_interval,
    scale_depth_residual,
    scale_depth_residual_log,
)


def main() -> None:
    frame_area = 1920.0 * 1080.0
    scale_priors = {
        "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
        "elephant": ScalePrior(min_size=2.40, max_size=3.40),
    }

    soccer_ball = ObjectObservation(
        object_id="soccer_ball",
        label="soccer_ball",
        mask_area=4_800.0,
        frame_area=frame_area,
        depth=12.0,
        confidence=0.98,
    )
    elephant = ObjectObservation(
        object_id="elephant",
        label="elephant",
        mask_area=120_000.0,
        frame_area=frame_area,
        depth=36.0,
        confidence=0.96,
    )
    anomalous_elephant = ObjectObservation(
        object_id="tiny_elephant",
        label="elephant",
        mask_area=1_200.0,
        frame_area=frame_area,
        depth=10.0,
        confidence=0.96,
    )

    cases = [
        ("reasonable case", soccer_ball, elephant),
        ("anomalous case: tiny elephant at similar depth", soccer_ball, anomalous_elephant),
    ]

    for title, obj_a, obj_b in cases:
        lower, upper = compute_scale_depth_interval(obj_a, obj_b, scale_priors)
        residual, details = scale_depth_residual(obj_a, obj_b, scale_priors)
        residual_log, log_details = scale_depth_residual_log(
            obj_a, obj_b, scale_priors
        )

        print(f"\n{title}")
        print("Object A:", obj_a.object_id)
        print(f"  label = {obj_a.label}")
        print(f"  depth = {obj_a.depth:.2f}")
        print(f"  projection_ratio = {obj_a.projection_ratio:.8f}")
        print(
            "  equivalent_projection_scale = "
            f"{obj_a.equivalent_projection_scale:.8f}"
        )
        print("Object B:", obj_b.object_id)
        print(f"  label = {obj_b.label}")
        print(f"  depth = {obj_b.depth:.2f}")
        print(f"  projection_ratio = {obj_b.projection_ratio:.8f}")
        print(
            "  equivalent_projection_scale = "
            f"{obj_b.equivalent_projection_scale:.8f}"
        )
        print(f"depth_ratio Z_A/Z_B = {details['depth_ratio']:.8f}")
        print(f"reasonable interval = [{lower:.8f}, {upper:.8f}]")
        print(f"R_sd ratio-space = {residual:.8f}")
        print(
            "R_sd log-space = "
            f"{residual_log:.8f} "
            f"(log interval [{log_details['log_lower']:.8f}, "
            f"{log_details['log_upper']:.8f}])"
        )


if __name__ == "__main__":
    main()
