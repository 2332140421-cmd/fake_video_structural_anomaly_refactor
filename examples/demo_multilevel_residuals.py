"""Demo for multi-granularity 3D structural residual aggregation."""

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
    build_object_level_residuals,
    build_object_pair_residuals,
    summarize_clip_residuals,
)


def _rectangle_mask(
    height: int, width: int, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    """Create a rectangular boolean mask."""

    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def main() -> None:
    height, width = 128, 128
    soccer_mask = _rectangle_mask(height, width, 22, 48, 44, 70)
    elephant_mask = _rectangle_mask(height, width, 66, 34, 118, 96)
    objects = [
        ObjectMaskObservation("soccer_ball", "soccer_ball", soccer_mask),
        ObjectMaskObservation("elephant", "elephant", elephant_mask),
    ]

    flow_residual_map = np.zeros((height, width), dtype=float)
    depth_residual_map = np.zeros((height, width), dtype=float)
    corr_residual_map = np.zeros((height, width), dtype=float)
    flow_residual_map[elephant_mask] = 0.25
    depth_residual_map[elephant_mask] = 0.40
    corr_residual_map[soccer_mask] = 0.10
    corr_residual_map[elephant_mask] = 0.30

    track_points_xy = np.array(
        [[30, 56], [36, 64], [80, 48], [102, 80], [10, 10]],
        dtype=float,
    )
    track_residuals = np.array([0.05, 0.08, 0.60, 0.75, 1.20], dtype=float)

    object_residuals = build_object_level_residuals(
        objects,
        flow_residual_map=flow_residual_map,
        depth_residual_map=depth_residual_map,
        corr_residual_map=corr_residual_map,
        track_points_xy=track_points_xy,
        track_residuals=track_residuals,
    )

    scale_depth_matrix = np.array([[0.0, 1.4], [1.2, 0.0]], dtype=float)
    pair_residuals = build_object_pair_residuals(
        objects,
        scale_depth_matrix=scale_depth_matrix,
    )
    summary = summarize_clip_residuals(object_residuals, pair_residuals)

    print("Object-level residuals")
    for residual in object_residuals:
        print(
            f"  {residual.object_id}: flow={residual.flow:.3f}, "
            f"track={residual.track:.3f}, depth_cons={residual.depth_cons:.3f}, "
            f"corr={residual.corr:.3f}"
        )

    print("\nObject-pair residuals")
    for residual in pair_residuals:
        print(
            f"  {residual.object_id_a}->{residual.object_id_b}: "
            f"R_sd={residual.scale_depth:.3f}, R_occ={residual.occ:.3f}, "
            f"R_relative_motion={residual.relative_motion:.3f}"
        )

    print(f"\nclip_score = {summary.clip_score:.3f}")
    print("details:", summary.details)


if __name__ == "__main__":
    main()
