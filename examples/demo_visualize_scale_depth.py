"""Generate explanatory scale-depth residual visualizations with synthetic masks."""

from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.scale_depth import ObjectObservation, ScalePrior, scale_depth_residual  # noqa: E402
from semantic3d.visualization import (  # noqa: E402
    draw_object_summary,
    draw_pairwise_residual_graph,
    draw_residual_heatmap_from_masks,
)


def _ellipse_mask(
    height: int,
    width: int,
    center_x: float,
    center_y: float,
    radius_x: float,
    radius_y: float,
) -> np.ndarray:
    """Create a binary ellipse mask for synthetic objects."""

    y_grid, x_grid = np.ogrid[:height, :width]
    normalized = ((x_grid - center_x) / radius_x) ** 2 + (
        (y_grid - center_y) / radius_y
    ) ** 2
    return normalized <= 1.0


def main() -> None:
    width, height = 640, 360
    frame_area = float(width * height)
    output_dir = PROJECT_ROOT / "outputs"

    masks = {
        "soccer_ball": _ellipse_mask(height, width, 175, 220, 26, 26),
        # The elephant is deliberately too small and at a similar depth, so the
        # scale-depth residual becomes visually obvious.
        "elephant": _ellipse_mask(height, width, 440, 185, 32, 24),
    }

    soccer_ball = ObjectObservation(
        object_id="soccer_ball",
        label="soccer_ball",
        mask_area=float(masks["soccer_ball"].sum()),
        frame_area=frame_area,
        depth=10.0,
    )
    elephant = ObjectObservation(
        object_id="elephant",
        label="elephant",
        mask_area=float(masks["elephant"].sum()),
        frame_area=frame_area,
        depth=10.0,
    )
    objects = [soccer_ball, elephant]
    scale_priors = {
        "soccer_ball": ScalePrior(min_size=0.20, max_size=0.24),
        "elephant": ScalePrior(min_size=2.40, max_size=3.40),
    }

    residual, details = scale_depth_residual(soccer_ball, elephant, scale_priors)
    pair_residuals = {
        ("soccer_ball", "elephant"): residual,
        ("elephant", "soccer_ball"): residual,
    }

    draw_object_summary(
        width,
        height,
        objects,
        masks=masks,
        output_path=output_dir / "scale_depth_object_summary.png",
    )
    draw_pairwise_residual_graph(
        width,
        height,
        objects,
        pair_residuals,
        masks=masks,
        residual_threshold=0.01,
        output_path=output_dir / "scale_depth_pair_graph.png",
    )
    draw_residual_heatmap_from_masks(
        width,
        height,
        masks,
        pair_residuals,
        output_path=output_dir / "scale_depth_heatmap.png",
    )
    plt.close("all")

    print("Scale-depth visualization demo")
    print(f"depth_ratio = {details['depth_ratio']:.6f}")
    print(f"reasonable interval = [{details['lower']:.6f}, {details['upper']:.6f}]")
    print(f"R_sd = {residual:.6f}")
    print(f"PNG outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
