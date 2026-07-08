#!/usr/bin/env python3
"""Run a smoke test for the real depth-estimation provider."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from semantic3d.depth_provider import (  # noqa: E402
    RealDepthProvider,
    save_depth_visualization,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate depth maps for one image or the first frames of a video."
    )
    parser.add_argument("--image_path", default=None, help="Optional image path.")
    parser.add_argument("--video_path", default=None, help="Optional video path.")
    parser.add_argument(
        "--model_name",
        default="depth-anything/Depth-Anything-V2-Small",
        help="Hugging Face model id or local path.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "depth_smoke_test"),
        help="Output directory for .npy and PNG depth maps.",
    )
    parser.add_argument("--max_frames", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--invert_depth",
        action="store_true",
        help="Invert depth output when larger raw values mean closer objects.",
    )
    return parser.parse_args()


def _extract_video_frames(video_path: Path, output_dir: Path, max_frames: int) -> list[Path]:
    """Extract the first max_frames frames from a video for smoke testing."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_paths: list[Path] = []
    frame_index = 0
    while len(frame_paths) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame_path = frame_dir / f"{video_path.stem}_frame_{frame_index:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        frame_index += 1
    cap.release()

    if not frame_paths:
        raise ValueError(f"No frames could be extracted from video: {video_path}")
    return frame_paths


def _input_frames(args: argparse.Namespace, output_dir: Path) -> Iterable[Path]:
    """Resolve image/video inputs to frame image paths."""

    if args.image_path:
        path = Path(args.image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")
        return [path]
    if args.video_path:
        path = Path(args.video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video does not exist: {path}")
        return _extract_video_frames(path, output_dir, args.max_frames)
    raise ValueError("Provide --image_path or --video_path.")


def _print_stats(path: Path, depth_map: np.ndarray) -> None:
    """Print depth map statistics."""

    finite = np.isfinite(depth_map)
    print(f"frame: {path}")
    print(f"  shape: {depth_map.shape}")
    print(f"  min: {float(np.min(depth_map)):.6f}")
    print(f"  max: {float(np.max(depth_map)):.6f}")
    print(f"  mean: {float(np.mean(depth_map)):.6f}")
    print(f"  has_nan_or_inf: {not bool(finite.all())}")


def main() -> None:
    """Run the smoke test."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        provider = RealDepthProvider(
            model_name=args.model_name,
            device=args.device,
            invert_depth=args.invert_depth,
        )
    except Exception as exc:
        print(
            "ERROR: could not load real depth model. "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        print("mock_depth remains available and is not affected.", file=sys.stderr)
        sys.exit(2)

    try:
        frame_paths = list(_input_frames(args, output_dir))
        for frame_path in frame_paths:
            depth_map = provider.predict_depth(frame_path)
            npy_path = output_dir / f"{frame_path.stem}_depth.npy"
            png_path = output_dir / f"{frame_path.stem}_depth.png"
            np.save(npy_path, depth_map.astype(np.float32))
            save_depth_visualization(depth_map, png_path)
            _print_stats(frame_path, depth_map)
            print(f"  saved npy: {npy_path}")
            print(f"  saved png: {png_path}")
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
