#!/usr/bin/env python3
"""Build clip observation JSON files from a local video using mock providers."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _ensure_project_environment() -> Path:
    """Re-execute this script with the project .venv Python when available."""

    project_root = Path(__file__).resolve().parents[1]
    project_python = project_root / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])
    return project_root


PROJECT_ROOT = _ensure_project_environment()
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.build_observations import (  # noqa: E402
    build_clip_observation,
    build_frame_observation,
)
from semantic3d.io import save_clip_observation  # noqa: E402
from semantic3d.providers import MockObjectProvider  # noqa: E402
from semantic3d.video_preprocess import build_clips, extract_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract video frames and build mock observation JSON clips."
    )
    parser.add_argument("--video_path", required=True, help="Path to a local video.")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "observations"),
        help="Directory where clip observation JSON files are saved.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Optional frame FPS.")
    parser.add_argument(
        "--max_frames", type=int, default=None, help="Optional maximum frames to save."
    )
    parser.add_argument("--clip_len", type=int, default=8, help="Frames per clip.")
    parser.add_argument("--stride", type=int, default=4, help="Sliding-window stride.")
    parser.add_argument(
        "--mock_mode",
        choices=["reasonable", "anomaly"],
        default="reasonable",
        help="Mock geometry mode used by the object/depth providers.",
    )
    return parser.parse_args()


def build_observations_from_video(
    video_path: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    clip_len: int = 8,
    stride: int = 4,
    mock_mode: str = "reasonable",
) -> list[Path]:
    """Extract frames, build clip observations, and save JSON files."""

    video_id = video_path.stem
    frame_output_dir = PROJECT_ROOT / "outputs" / "frames" / f"{video_id}_{mock_mode}"
    frame_paths, _frame_indices = extract_frames(
        video_path,
        frame_output_dir,
        fps=fps,
        max_frames=max_frames,
    )
    clip_windows = build_clips(frame_paths, clip_len=clip_len, stride=stride)
    if not clip_windows:
        raise ValueError(f"No frames were extracted from video: {video_path}")

    provider = MockObjectProvider(mode=mock_mode)  # type: ignore[arg-type]
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for clip_window in clip_windows:
        frame_observations = [
            build_frame_observation(frame_path, frame_index, provider)
            for frame_path, frame_index in zip(
                clip_window.frame_paths, clip_window.frame_indices
            )
        ]
        full_clip_id = f"{video_id}_{mock_mode}_{clip_window.clip_id}"
        clip_obs = build_clip_observation(
            video_id=video_id,
            clip_id=full_clip_id,
            frames=frame_observations,
            metadata={
                "video_path": str(video_path),
                "mock_mode": mock_mode,
                "expected_mode": mock_mode,
                "clip_len": clip_len,
                "stride": stride,
                "provider": "MockObjectProvider",
            },
        )
        output_path = output_dir / f"{full_clip_id}.json"
        save_clip_observation(clip_obs, output_path)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    """Run the video-to-observation builder from the command line."""

    args = parse_args()
    saved_paths = build_observations_from_video(
        video_path=Path(args.video_path),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        max_frames=args.max_frames,
        clip_len=args.clip_len,
        stride=args.stride,
        mock_mode=args.mock_mode,
    )
    print(f"Saved {len(saved_paths)} clip observation JSON file(s):")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
