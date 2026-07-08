#!/usr/bin/env python3
"""Visualize no-invert and invert depth maps with object annotations."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.io import load_clip_observation  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create debug figures for depth direction comparison."
    )
    parser.add_argument("--no_invert_observation_dir", required=True)
    parser.add_argument("--invert_observation_dir", required=True)
    parser.add_argument("--no_invert_depth_dir", required=True)
    parser.add_argument("--invert_depth_dir", required=True)
    parser.add_argument("--no_invert_rsd_csv", required=True)
    parser.add_argument("--invert_rsd_csv", required=True)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "visualizations" / "depth_direction_debug"),
    )
    parser.add_argument("--max_frames", type=int, default=10)
    return parser.parse_args()


def _load_frames(observation_dir: Path) -> list[object]:
    """Load all frames from clip observation JSON files."""

    frames = []
    for path in sorted(observation_dir.glob("*.json")):
        clip = load_clip_observation(path)
        frames.extend(clip.frames)
    unique: dict[int, object] = {}
    for frame in frames:
        unique.setdefault(int(frame.frame_index), frame)
    return [unique[index] for index in sorted(unique)]


def _read_rsd_by_frame(csv_path: Path) -> dict[int, list[dict[str, str]]]:
    """Group R_sd CSV rows by frame index."""

    if not csv_path.exists():
        return {}
    grouped: dict[int, list[dict[str, str]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            grouped.setdefault(int(row["frame_index"]), []).append(row)
    return grouped


def _load_depth(frame: object, depth_dir: Path) -> np.ndarray | None:
    """Load the depth map for a frame, preferring JSON path then depth_dir."""

    depth_map_path = getattr(frame, "depth_map_path", None)
    candidates: list[Path] = []
    if depth_map_path:
        candidates.append(Path(depth_map_path))
    image_path = Path(str(getattr(frame, "image_path", "")))
    if image_path.name:
        candidates.append(depth_dir / f"{image_path.stem}_depth.npy")

    for candidate in candidates:
        if candidate.exists():
            return np.load(candidate)
    return None


def _read_image(frame: object) -> np.ndarray:
    """Read a frame image as RGB or create a blank fallback."""

    image_path = Path(str(getattr(frame, "image_path", "")))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR) if image_path.exists() else None
    if image is None:
        height = int(getattr(frame, "height", 256))
        width = int(getattr(frame, "width", 256))
        return np.full((height, width, 3), 245, dtype=np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _draw_objects(ax: plt.Axes, frame: object, title: str) -> None:
    """Draw bboxes and object labels on an axis."""

    ax.set_title(title, fontsize=11)
    for obj in getattr(frame, "objects", []):
        bbox = getattr(obj, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=1.8,
            edgecolor="lime",
        )
        ax.add_patch(rect)
        label = getattr(obj, "label", "object")
        confidence = float(getattr(obj, "confidence", 0.0))
        depth = float(getattr(obj, "depth", 0.0))
        ax.text(
            x1,
            max(0.0, y1 - 4.0),
            f"{label} {confidence:.2f}\nd={depth:.2f}",
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
    ax.axis("off")


def _draw_rsd_lines(
    ax: plt.Axes,
    frame: object,
    rows: list[dict[str, str]],
    threshold: float = 0.0,
) -> None:
    """Draw pairwise R_sd lines if residual rows are available."""

    centers = {}
    for obj in getattr(frame, "objects", []):
        bbox = getattr(obj, "bbox", None)
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        centers[getattr(obj, "object_id")] = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    for row in rows:
        score = float(row.get("R_sd_log", 0.0))
        if score <= threshold:
            continue
        left, right = row["object_pair"].split("->", maxsplit=1)
        if left not in centers or right not in centers:
            continue
        x1, y1 = centers[left]
        x2, y2 = centers[right]
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.5 + score)
        ax.text(
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
            f"R_sd={score:.2f}",
            color="red",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 1},
        )


def _show_depth(ax: plt.Axes, depth: np.ndarray | None, title: str) -> None:
    """Draw a depth heatmap or a missing-data message."""

    ax.set_title(title, fontsize=11)
    if depth is None:
        ax.text(0.5, 0.5, "No depth map", ha="center", va="center")
        ax.axis("off")
        return
    im = ax.imshow(depth, cmap="inferno")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def create_debug_figures(
    no_invert_observation_dir: Path,
    invert_observation_dir: Path,
    no_invert_depth_dir: Path,
    invert_depth_dir: Path,
    no_invert_rsd_csv: Path,
    invert_rsd_csv: Path,
    output_dir: Path,
    max_frames: int = 10,
) -> list[Path]:
    """Create depth direction debug PNGs and return their paths."""

    no_frames = _load_frames(no_invert_observation_dir)
    invert_frames = {frame.frame_index: frame for frame in _load_frames(invert_observation_dir)}
    no_rsd = _read_rsd_by_frame(no_invert_rsd_csv)
    invert_rsd = _read_rsd_by_frame(invert_rsd_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for frame in no_frames[:max_frames]:
        frame_index = int(frame.frame_index)
        invert_frame = invert_frames.get(frame_index, frame)
        image = _read_image(frame)
        no_depth = _load_depth(frame, no_invert_depth_dir)
        inverted_depth = _load_depth(invert_frame, invert_depth_dir)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image)
        _draw_objects(axes[0], frame, f"Frame {frame_index}: objects + no-invert depth")
        _draw_rsd_lines(axes[0], frame, no_rsd.get(frame_index, []))

        _show_depth(axes[1], no_depth, "No invert depth heatmap")
        _show_depth(axes[2], inverted_depth, "Invert depth heatmap")
        _draw_rsd_lines(axes[2], invert_frame, invert_rsd.get(frame_index, []))

        fig.suptitle(
            "Depth Direction Debug: larger depth should mean farther",
            fontsize=13,
        )
        fig.tight_layout()
        output_path = output_dir / f"depth_direction_frame_{frame_index:06d}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        saved.append(output_path)

    if not saved:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No frames available for visualization", ha="center")
        ax.axis("off")
        output_path = output_dir / "depth_direction_empty.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        saved.append(output_path)
    return saved


def main() -> None:
    """Run the debug visualization script."""

    args = parse_args()
    saved = create_debug_figures(
        no_invert_observation_dir=Path(args.no_invert_observation_dir),
        invert_observation_dir=Path(args.invert_observation_dir),
        no_invert_depth_dir=Path(args.no_invert_depth_dir),
        invert_depth_dir=Path(args.invert_depth_dir),
        no_invert_rsd_csv=Path(args.no_invert_rsd_csv),
        invert_rsd_csv=Path(args.invert_rsd_csv),
        output_dir=Path(args.output_dir),
        max_frames=args.max_frames,
    )
    print(f"Saved {len(saved)} depth direction debug PNG file(s):")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
