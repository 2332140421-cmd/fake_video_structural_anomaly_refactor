"""Demo for multi-level structural residual visualizations."""

from __future__ import annotations

import sys

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from semantic3d.multilevel_residuals import (  # noqa: E402
    ObjectMaskObservation,
    ObjectLevelResidual,
    ObjectPairResidual,
    summarize_clip_residuals,
)
from semantic3d.visualization import (  # noqa: E402
    draw_multilevel_summary,
    draw_object_residual_map,
    draw_pairwise_residual_graph,
)


def _rectangle_mask(
    height: int, width: int, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """Create a rectangular object mask."""

    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def main() -> None:
    image_shape = (128, 128)
    height, width = image_shape
    output_dir = PROJECT_ROOT / "outputs" / "visualizations"

    soccer_ball = ObjectMaskObservation(
        object_id="soccer_ball",
        label="soccer_ball",
        mask=_rectangle_mask(height, width, 16, 76, 34, 94),
    )
    elephant = ObjectMaskObservation(
        object_id="elephant",
        label="elephant",
        mask=_rectangle_mask(height, width, 72, 24, 120, 94),
    )
    objects = [soccer_ball, elephant]

    object_residuals = [
        ObjectLevelResidual(
            object_id="soccer_ball",
            label="soccer_ball",
            flow=0.03,
            track=0.05,
            depth_cons=0.02,
            corr=0.04,
        ),
        ObjectLevelResidual(
            object_id="elephant",
            label="elephant",
            flow=0.16,
            track=0.18,
            depth_cons=0.38,
            corr=0.32,
        ),
    ]
    pair_residuals = [
        ObjectPairResidual(
            object_id_a="soccer_ball",
            object_id_b="elephant",
            label_a="soccer_ball",
            label_b="elephant",
            scale_depth=2.20,
            occ=0.08,
            relative_motion=0.12,
        )
    ]

    summary = summarize_clip_residuals(object_residuals, pair_residuals)

    draw_object_residual_map(
        image_shape,
        objects,
        object_residuals,
        output_path=output_dir / "object_residual_map.png",
    )
    draw_pairwise_residual_graph(
        objects,
        pair_residuals,
        output_path=output_dir / "pairwise_residual_graph.png",
        threshold=0.1,
    )
    draw_multilevel_summary(
        objects,
        object_residuals,
        pair_residuals,
        summary.clip_score,
        output_path=output_dir / "multilevel_summary.png",
    )

    print("Multi-level visualization demo")
    print(f"clip_score = {summary.clip_score:.3f}")
    print(f"PNG outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
