"""Demo for computing R_sd from one manually authored observation JSON."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from _bootstrap import ensure_project_environment

PROJECT_ROOT = ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from run_manual_observation_rsd import (  # noqa: E402
    DEFAULT_SCALE_PRIORS,
    load_manual_observation,
)
from semantic3d.scale_depth import (  # noqa: E402
    compute_scale_depth_interval,
    scale_depth_residual,
    scale_depth_residual_log,
)


def draw_manual_pair_graph(
    observation_path: Path,
    output_path: Path,
) -> None:
    """Draw a simple object relation graph from bbox annotations."""

    case_id, _expected_label, frame = load_manual_observation(observation_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, frame.width)
    ax.set_ylim(frame.height, 0)
    ax.set_aspect("equal")
    ax.set_title(f"Manual Object Relation Graph: {case_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)

    centers: dict[str, tuple[float, float]] = {}
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for index, obj_json in enumerate(frame.objects):
        if obj_json.bbox is None:
            continue
        x1, y1, x2, y2 = obj_json.bbox
        color = colors[index % len(colors)]
        ax.add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        centers[obj_json.object_id] = (cx, cy)
        ax.scatter([cx], [cy], color=color, s=60, zorder=3)
        ax.text(
            cx,
            y1 - 8,
            f"{obj_json.object_id}\n{obj_json.label}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )

    objects = [obj.to_scale_depth_observation() for obj in frame.objects]
    for i, obj_a in enumerate(objects):
        for j in range(i + 1, len(objects)):
            obj_b = objects[j]
            residual_log, _details_log = scale_depth_residual_log(
                obj_a, obj_b, DEFAULT_SCALE_PRIORS
            )
            x_a, y_a = centers[obj_a.object_id]
            x_b, y_b = centers[obj_b.object_id]
            linewidth = 1.5 + min(residual_log, 3.0) * 2.5
            ax.plot([x_a, x_b], [y_a, y_b], color="#E45756", linewidth=linewidth)
            ax.text(
                (x_a + x_b) / 2.0,
                (y_a + y_b) / 2.0,
                f"R_sd_log={residual_log:.3f}",
                ha="center",
                va="center",
                fontsize=9,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.9,
                    "edgecolor": "#E45756",
                    "pad": 2,
                },
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Load one manual observation, print R_sd details, and save a graph."""

    observation_path = (
        PROJECT_ROOT
        / "data"
        / "manual_observations"
        / "manual_004_soccer_elephant_anomaly.json"
    )
    output_path = (
        PROJECT_ROOT
        / "outputs"
        / "visualizations"
        / "manual_observation_pair_graph.png"
    )

    case_id, expected_label, frame = load_manual_observation(observation_path)
    print(f"Manual observation: {case_id}")
    print(f"expected_label={expected_label}, frame_id={frame.frame_id}")

    objects = [obj.to_scale_depth_observation() for obj in frame.objects]
    print("\nObjects")
    for obj in objects:
        print(
            f"  {obj.object_id}: label={obj.label}, mask_area={obj.mask_area:.1f}, "
            f"projection_scale={obj.equivalent_projection_scale:.4f}, depth={obj.depth:.2f}"
        )

    print("\nPairwise scale-depth residuals")
    for i, obj_a in enumerate(objects):
        for j in range(i + 1, len(objects)):
            obj_b = objects[j]
            lower, upper = compute_scale_depth_interval(
                obj_a, obj_b, DEFAULT_SCALE_PRIORS
            )
            residual, details = scale_depth_residual(
                obj_a, obj_b, DEFAULT_SCALE_PRIORS
            )
            residual_log, details_log = scale_depth_residual_log(
                obj_a, obj_b, DEFAULT_SCALE_PRIORS
            )
            print(
                f"  {obj_a.object_id}->{obj_b.object_id}: "
                f"depth_ratio={details['depth_ratio']:.4f}, "
                f"interval=[{lower:.4f}, {upper:.4f}], "
                f"R_sd={residual:.4f}, "
                f"log_interval=[{details_log['log_lower']:.4f}, "
                f"{details_log['log_upper']:.4f}], "
                f"R_sd_log={residual_log:.4f}"
            )

    draw_manual_pair_graph(observation_path, output_path)
    print(f"\nSaved object relation anomaly graph: {output_path}")


if __name__ == "__main__":
    main()
