"""Visualization helpers for object-level scale-depth residuals.

The functions in this module create explanatory figures from object masks or
bounding boxes. They use only matplotlib and numpy, so they can run without any
vision model dependency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle
import numpy as np

from .scale_depth import ObjectObservation
from .multilevel_residuals import (
    ObjectLevelResidual,
    ObjectMaskObservation,
    ObjectPairResidual,
)

Box = Tuple[float, float, float, float]
MaskMap = Mapping[str, np.ndarray]
BoxMap = Mapping[str, Box]
PairResidualMap = Mapping[Tuple[str, str], float]


def _ensure_rgb_canvas(
    width: int,
    height: int,
    image: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an RGB image canvas for visualization."""

    if image is None:
        return np.full((height, width, 3), 245, dtype=np.uint8)

    canvas = np.asarray(image)
    if canvas.ndim == 2:
        canvas = np.repeat(canvas[:, :, None], 3, axis=2)
    if canvas.shape[:2] != (height, width):
        raise ValueError(
            f"image shape {canvas.shape[:2]} does not match frame {(height, width)}."
        )
    return canvas


def _object_center(
    object_id: str,
    masks: Optional[MaskMap] = None,
    boxes: Optional[BoxMap] = None,
) -> Tuple[float, float]:
    """Estimate object center from a mask first, then from a bounding box."""

    if masks is not None and object_id in masks:
        ys, xs = np.nonzero(np.asarray(masks[object_id]) > 0)
        if xs.size > 0:
            return float(xs.mean()), float(ys.mean())

    if boxes is not None and object_id in boxes:
        x1, y1, x2, y2 = boxes[object_id]
        return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0

    raise KeyError(f"Missing mask or box for object '{object_id}'.")


def _mask_to_box(mask: np.ndarray) -> Box:
    """Compute a bounding box (x1, y1, x2, y2) from a binary mask."""

    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if xs.size == 0:
        raise ValueError("Cannot compute a bounding box from an empty mask.")
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _save_if_requested(fig: Figure, output_path: Optional[str | Path]) -> None:
    """Save the figure if an output path is provided."""

    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")


def draw_object_summary(
    width: int,
    height: int,
    objects: Sequence[ObjectObservation],
    masks: Optional[MaskMap] = None,
    boxes: Optional[BoxMap] = None,
    image: Optional[np.ndarray] = None,
    output_path: Optional[str | Path] = None,
) -> Tuple[Figure, Axes]:
    """Draw object regions with label, depth, and projection scale.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        objects: Object observations to annotate.
        masks: Optional binary masks keyed by object_id.
        boxes: Optional bounding boxes keyed by object_id as (x1, y1, x2, y2).
        image: Optional RGB or grayscale image. If omitted, a blank canvas is used.
        output_path: Optional PNG path.

    Returns:
        Matplotlib figure and axes.
    """

    canvas = _ensure_rgb_canvas(width, height, image)
    fig, ax = plt.subplots(figsize=(width / 120, height / 120))
    ax.imshow(canvas)
    ax.set_title("Object Summary")
    ax.axis("off")

    cmap = plt.get_cmap("tab10")
    for index, obj in enumerate(objects):
        color = cmap(index % 10)
        mask = None if masks is None else masks.get(obj.object_id)
        box = None if boxes is None else boxes.get(obj.object_id)

        if mask is not None:
            mask_array = np.asarray(mask) > 0
            overlay = np.zeros((height, width, 4), dtype=float)
            overlay[mask_array] = (*color[:3], 0.28)
            ax.imshow(overlay)
            box = _mask_to_box(mask_array)

        if box is not None:
            x1, y1, x2, y2 = box
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

        cx, cy = _object_center(obj.object_id, masks=masks, boxes=boxes)
        text = (
            f"{obj.label}\n"
            f"Z={obj.depth:.2f}\n"
            f"p={obj.equivalent_projection_scale:.3f}"
        )
        ax.text(
            cx,
            cy,
            text,
            color="white",
            ha="center",
            va="center",
            fontsize=8,
            bbox={"facecolor": color, "alpha": 0.85, "edgecolor": "none", "pad": 2},
        )

    _save_if_requested(fig, output_path)
    return fig, ax


def _draw_scale_depth_pairwise_residual_graph(
    width: int,
    height: int,
    objects: Sequence[ObjectObservation],
    pair_residuals: PairResidualMap,
    masks: Optional[MaskMap] = None,
    boxes: Optional[BoxMap] = None,
    image: Optional[np.ndarray] = None,
    residual_threshold: float = 0.0,
    output_path: Optional[str | Path] = None,
) -> Tuple[Figure, Axes]:
    """Draw objects as nodes and high-R_sd object pairs as weighted edges."""

    canvas = _ensure_rgb_canvas(width, height, image)
    fig, ax = plt.subplots(figsize=(width / 120, height / 120))
    ax.imshow(canvas)
    ax.set_title("Pairwise Scale-Depth Residual Graph")
    ax.axis("off")

    centers = {
        obj.object_id: _object_center(obj.object_id, masks=masks, boxes=boxes)
        for obj in objects
    }
    max_residual = max([abs(v) for v in pair_residuals.values()] + [1e-8])

    # Edges are drawn before nodes so labels stay readable.
    for (object_a, object_b), residual in pair_residuals.items():
        if residual <= residual_threshold:
            continue
        x_a, y_a = centers[object_a]
        x_b, y_b = centers[object_b]
        strength = residual / max_residual
        ax.plot(
            [x_a, x_b],
            [y_a, y_b],
            color=(1.0, 0.1, 0.05, 0.35 + 0.55 * min(strength, 1.0)),
            linewidth=1.0 + 6.0 * min(strength, 1.0),
        )
        ax.text(
            (x_a + x_b) / 2.0,
            (y_a + y_b) / 2.0,
            f"{residual:.3f}",
            color="black",
            fontsize=8,
            ha="center",
            va="center",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1},
        )

    for obj in objects:
        cx, cy = centers[obj.object_id]
        ax.add_patch(Circle((cx, cy), radius=9, color="#1f77b4", zorder=4))
        ax.text(
            cx,
            cy - 16,
            obj.label,
            ha="center",
            va="top",
            fontsize=8,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 1},
        )

    _save_if_requested(fig, output_path)
    return fig, ax


def _require_objects(objects: Sequence[ObjectMaskObservation]) -> None:
    """Raise a clear error for empty object lists."""

    if not objects:
        raise ValueError("objects must not be empty.")


def _object_total_score(residual: ObjectLevelResidual) -> float:
    """Return total object-level residual score."""

    return float(residual.flow + residual.track + residual.depth_cons + residual.corr)


def _object_score(residual: ObjectLevelResidual, score_field: str) -> float:
    """Select one object-level residual field or total score."""

    if score_field == "total":
        return _object_total_score(residual)
    if score_field in {"flow", "track", "depth_cons", "corr"}:
        return float(getattr(residual, score_field))
    raise ValueError(
        f"Unknown score_field '{score_field}'. "
        "Use flow, track, depth_cons, corr, or total."
    )


def _pair_score(pair: ObjectPairResidual) -> float:
    """Return total object-pair residual score."""

    return float(pair.scale_depth + pair.occ + pair.relative_motion)


def _object_residual_lookup(
    object_residuals: Sequence[ObjectLevelResidual],
) -> dict[str, ObjectLevelResidual]:
    """Build object_id -> residual lookup."""

    return {residual.object_id: residual for residual in object_residuals}


def _draw_object_residual_map_on_ax(
    ax: Axes,
    image_shape: tuple[int, int],
    objects: Sequence[ObjectMaskObservation],
    object_residuals: Sequence[ObjectLevelResidual],
    score_field: str = "total",
) -> np.ndarray:
    """Draw object-level residual map on an existing axes.

    This map only visualizes single-object residual terms. Object-pair terms
    such as R_sd are intentionally excluded and should be shown by the pairwise
    graph instead.
    """

    _require_objects(objects)
    height, width = image_shape
    canvas = _ensure_rgb_canvas(width, height)
    residual_lookup = _object_residual_lookup(object_residuals)
    heatmap = np.zeros((height, width), dtype=float)
    scores = []

    for obj in objects:
        if obj.object_id not in residual_lookup:
            raise KeyError(f"Missing object residual for object_id '{obj.object_id}'.")
        if obj.mask.shape != (height, width):
            raise ValueError(
                f"Mask for object '{obj.object_id}' has shape {obj.mask.shape}, "
                f"expected {(height, width)}."
            )
        score = _object_score(residual_lookup[obj.object_id], score_field)
        scores.append(score)
        heatmap[obj.mask] = score

    ax.imshow(canvas)
    heat = ax.imshow(heatmap, cmap="inferno", alpha=0.68, vmin=0.0)
    max_score = max(scores + [1e-8])
    for index, obj in enumerate(objects):
        residual = residual_lookup[obj.object_id]
        score = _object_score(residual, score_field)
        x1, y1, x2, y2 = obj.bbox
        cx, cy = obj.center
        edge_color = "#00a6d6" if index % 2 == 0 else "#ff9f1c"
        ax.contour(obj.mask.astype(float), levels=[0.5], colors=[edge_color], linewidths=2.0)
        ax.add_patch(
            Rectangle(
                (x1, y1),
                max(x2 - x1, 1),
                max(y2 - y1, 1),
                fill=False,
                edgecolor=edge_color,
                linewidth=1.6,
            )
        )

        label_y = max(y1 - 8, 3)
        score_y = min(y2 + 9, height - 4)
        ax.text(
            cx,
            label_y,
            f"{obj.object_id} | {obj.label}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": edge_color, "pad": 2},
            clip_on=False,
        )
        ax.text(
            cx,
            score_y if y2 - y1 < 26 else cy,
            f"{score_field}={score:.3f}",
            ha="center",
            va="top" if y2 - y1 < 26 else "center",
            fontsize=8,
            color="white" if score / max_score > 0.25 else "black",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 2},
            clip_on=False,
        )
    ax.set_title(f"Object Residual Map ({score_field})")
    ax.set_xlim(-6, width + 6)
    ax.set_ylim(height + 12, -18)
    ax.axis("off")
    return heat


def draw_object_residual_map(
    image_shape: tuple[int, int],
    objects: Sequence[ObjectMaskObservation],
    object_residuals: Sequence[ObjectLevelResidual],
    score_field: str = "total",
    output_path: Optional[str | Path] = None,
) -> tuple[Figure, Axes, np.ndarray]:
    """Draw object masks colored by object-level residual strength.

    Args:
        image_shape: Image shape as (height, width).
        objects: Object masks to visualize.
        object_residuals: Object-level residual records.
        score_field: flow, track, depth_cons, corr, or total.
        output_path: PNG path. Parent directories are created automatically.
    """

    _require_objects(objects)
    height, width = image_shape
    fig, ax = plt.subplots(figsize=(max(8.0, width / 80), max(6.0, height / 80)))
    heat = _draw_object_residual_map_on_ax(
        ax, image_shape, objects, object_residuals, score_field
    )
    fig.colorbar(heat, ax=ax, fraction=0.035, pad=0.025, label=score_field)
    fig.tight_layout()
    _save_if_requested(fig, output_path)
    return fig, ax, heat.get_array()


def _draw_multilevel_pairwise_residual_graph_on_ax(
    ax: Axes,
    objects: Sequence[ObjectMaskObservation],
    pair_residuals: Sequence[ObjectPairResidual],
    image_shape: tuple[int, int],
    threshold: float = 0.1,
) -> None:
    """Draw object-pair residual graph on an existing axes."""

    _require_objects(objects)
    height, width = image_shape
    ax.imshow(_ensure_rgb_canvas(width, height))
    ax.set_title("Object-Pair Residual Graph")
    ax.axis("off")

    centers = {obj.object_id: obj.center for obj in objects}
    max_score = max([_pair_score(pair) for pair in pair_residuals] + [1e-8])

    for pair in pair_residuals:
        score = _pair_score(pair)
        if max(pair.scale_depth, score) <= threshold:
            continue
        x_a, y_a = centers[pair.object_id_a]
        x_b, y_b = centers[pair.object_id_b]
        strength = min(score / max_score, 1.0)
        ax.plot(
            [x_a, x_b],
            [y_a, y_b],
            color=(1.0, 0.08, 0.05, 0.35 + 0.55 * strength),
            linewidth=1.5 + 7.0 * strength,
            solid_capstyle="round",
            zorder=2,
        )
        mid_x = (x_a + x_b) / 2.0
        mid_y = (y_a + y_b) / 2.0
        ax.text(
            mid_x,
            mid_y - 6,
            f"R_sd={pair.scale_depth:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d62728", "pad": 2},
            zorder=5,
        )

    for index, obj in enumerate(objects):
        x, y = obj.center
        node_color = "#1f77b4" if index % 2 == 0 else "#2ca02c"
        ax.add_patch(Circle((x, y), radius=6, color=node_color, zorder=4))
        vertical_offset = 14 + (index % 2) * 8
        ax.text(
            x,
            min(y + vertical_offset, height + 12),
            f"{obj.object_id}\n{obj.label}",
            ha="center",
            va="top",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": node_color, "pad": 2},
            zorder=6,
            clip_on=False,
        )
    ax.set_xlim(-10, width + 10)
    ax.set_ylim(height + 22, -10)


def _draw_multilevel_pairwise_residual_graph(
    objects: Sequence[ObjectMaskObservation],
    pair_residuals: Sequence[ObjectPairResidual],
    output_path: Optional[str | Path] = None,
    threshold: float = 0.1,
) -> tuple[Figure, Axes]:
    """Draw object centers as nodes and high object-pair residuals as edges."""

    _require_objects(objects)
    height, width = objects[0].mask.shape
    fig, ax = plt.subplots(figsize=(max(8.0, width / 80), max(6.0, height / 80)))
    _draw_multilevel_pairwise_residual_graph_on_ax(
        ax, objects, pair_residuals, (height, width), threshold
    )
    _save_if_requested(fig, output_path)
    return fig, ax


def draw_pairwise_residual_graph(*args: Any, **kwargs: Any) -> tuple[Figure, Axes]:
    """Draw pairwise residual graph.

    Supports two calling conventions:
        1. Legacy scale-depth visualization:
           draw_pairwise_residual_graph(width, height, objects, pair_residuals, ...)
        2. Multi-level residual visualization:
           draw_pairwise_residual_graph(objects, pair_residuals, output_path=..., ...)
    """

    if args and isinstance(args[0], int):
        return _draw_scale_depth_pairwise_residual_graph(*args, **kwargs)
    return _draw_multilevel_pairwise_residual_graph(*args, **kwargs)


def _bar_values(
    object_residuals: Sequence[ObjectLevelResidual],
    pair_residuals: Sequence[ObjectPairResidual],
) -> tuple[list[str], list[float], list[str]]:
    """Create labels, values, and colors for the summary bar chart."""

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for residual in object_residuals:
        short_id = residual.object_id.replace("_", " ")
        labels.extend(
            [
                f"{short_id} flow",
                f"{short_id} track",
                f"{short_id} depth",
                f"{short_id} corr",
            ]
        )
        values.extend([residual.flow, residual.track, residual.depth_cons, residual.corr])
        colors.extend(["#4c78a8", "#72b7b2", "#54a24b", "#b279a2"])
    for residual in pair_residuals:
        pair_name = (
            f"{residual.object_id_a.replace('_', ' ')} -> "
            f"{residual.object_id_b.replace('_', ' ')}"
        )
        labels.extend([f"{pair_name} R_sd", f"{pair_name} R_occ", f"{pair_name} motion"])
        values.extend([residual.scale_depth, residual.occ, residual.relative_motion])
        colors.extend(["#e45756", "#f58518", "#ff9da6"])
    return labels, [float(value) for value in values], colors


def draw_multilevel_summary(
    objects: Sequence[ObjectMaskObservation],
    object_residuals: Sequence[ObjectLevelResidual],
    pair_residuals: Sequence[ObjectPairResidual],
    clip_score: float,
    output_path: Optional[str | Path] = None,
) -> tuple[Figure, np.ndarray]:
    """Draw residual heatmap, object-pair graph, and residual bar chart."""

    _require_objects(objects)
    height, width = objects[0].mask.shape
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1.05, 1.05, 1.35]},
    )
    _draw_object_residual_map_on_ax(axes[0], (height, width), objects, object_residuals)
    _draw_multilevel_pairwise_residual_graph_on_ax(
        axes[1], objects, pair_residuals, (height, width), threshold=0.1
    )

    labels, values, colors = _bar_values(object_residuals, pair_residuals)
    y_pos = np.arange(len(values))
    axes[2].barh(y_pos, values, color=colors)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Residual value")
    axes[2].set_title("Residual Components")
    axes[2].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle(f"Multi-Level Structural Residual Summary | clip_score={clip_score:.3f}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_if_requested(fig, output_path)
    return fig, axes


def draw_residual_heatmap_from_masks(
    width: int,
    height: int,
    object_masks: MaskMap,
    pair_residuals: PairResidualMap,
    image: Optional[np.ndarray] = None,
    output_path: Optional[str | Path] = None,
) -> Tuple[Figure, Axes, np.ndarray]:
    """Map object-pair R_sd values back to object masks as a heatmap.

    Each object's heat value is the maximum residual among pairs that include
    that object. This is a simple object-level localization signal.
    """

    canvas = _ensure_rgb_canvas(width, height, image)
    heatmap = np.zeros((height, width), dtype=float)
    object_scores = {object_id: 0.0 for object_id in object_masks}

    for (object_a, object_b), residual in pair_residuals.items():
        if object_a in object_scores:
            object_scores[object_a] = max(object_scores[object_a], float(residual))
        if object_b in object_scores:
            object_scores[object_b] = max(object_scores[object_b], float(residual))

    for object_id, mask in object_masks.items():
        mask_array = np.asarray(mask) > 0
        if mask_array.shape != (height, width):
            raise ValueError(
                f"Mask for object '{object_id}' has shape {mask_array.shape}, "
                f"expected {(height, width)}."
            )
        heatmap[mask_array] = object_scores[object_id]

    fig, ax = plt.subplots(figsize=(width / 120, height / 120))
    ax.imshow(canvas)
    heat = ax.imshow(heatmap, cmap="inferno", alpha=0.65, vmin=0.0)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="object-level R_sd")
    ax.set_title("Scale-Depth Residual Heatmap")
    ax.axis("off")

    _save_if_requested(fig, output_path)
    return fig, ax, heatmap
