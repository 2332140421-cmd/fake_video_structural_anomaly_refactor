#!/usr/bin/env python3
"""Build observation JSON from video frames using mock or real object providers."""

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
from semantic3d.provider_registry import get_object_provider  # noqa: E402
from semantic3d.video_preprocess import build_clips, extract_frames  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Extract video frames and build real-object observation JSON."
    )
    parser.add_argument("--video_path", required=True, help="Path to a local video.")
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "real_observations"),
        help="Directory where clip observation JSON files are saved.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Optional frame FPS.")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument(
        "--object_provider",
        choices=["mock", "real_detector"],
        default="real_detector",
        help="Object provider used to create observations.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Minimum confidence for real detector observations.",
    )
    parser.add_argument(
        "--default_depth",
        type=float,
        default=None,
        help="Optional fixed temporary depth for all detected objects.",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        help="Local real detector weights path. No model is downloaded.",
    )
    parser.add_argument(
        "--mock_mode",
        choices=["reasonable", "anomaly"],
        default="reasonable",
        help="Mock mode used only when --object_provider mock.",
    )
    return parser.parse_args()


def build_real_object_observations_from_video(
    video_path: Path,
    output_dir: Path,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    clip_len: int = 8,
    stride: int = 4,
    object_provider: str = "real_detector",
    confidence_threshold: float = 0.3,
    default_depth: Optional[float] = None,
    model_path: Optional[str] = None,
    mock_mode: str = "reasonable",
) -> list[Path]:
    """Extract frames, run provider, and save clip observation JSON."""

    provider_kwargs: dict[str, object] = {}
    if object_provider == "mock":
        provider_kwargs["mock_mode"] = mock_mode
    else:
        provider_kwargs.update(
            {
                "confidence_threshold": confidence_threshold,
                "default_depth": default_depth,
                "model_path": model_path,
            }
        )
    provider = get_object_provider(object_provider, **provider_kwargs)

    video_id = video_path.stem
    frame_output_dir = (
        PROJECT_ROOT / "outputs" / "frames" / f"{video_id}_{object_provider}"
    )
    frame_paths, _frame_indices = extract_frames(
        video_path,
        frame_output_dir,
        fps=fps,
        max_frames=max_frames,
    )
    clip_windows = build_clips(frame_paths, clip_len=clip_len, stride=stride)
    if not clip_windows:
        raise ValueError(f"No frames were extracted from video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for clip_window in clip_windows:
        frame_observations = [
            build_frame_observation(frame_path, frame_index, provider)
            for frame_path, frame_index in zip(
                clip_window.frame_paths, clip_window.frame_indices
            )
        ]
        full_clip_id = f"{video_id}_{object_provider}_{clip_window.clip_id}"
        clip_obs = build_clip_observation(
            video_id=video_id,
            clip_id=full_clip_id,
            frames=frame_observations,
            metadata={
                "video_path": str(video_path),
                "provider": object_provider,
                "object_provider": object_provider,
                "mock_mode": mock_mode if object_provider == "mock" else "",
                "expected_mode": object_provider,
                "confidence_threshold": confidence_threshold,
                "default_depth": default_depth,
                "model_path": model_path,
                "clip_len": clip_len,
                "stride": stride,
                "mask_area_source": "bbox_area",
            },
        )
        output_path = output_dir / f"{full_clip_id}.json"
        save_clip_observation(clip_obs, output_path)
        saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    """Run the real-object observation builder from the command line."""

    args = parse_args()
    try:
        saved_paths = build_real_object_observations_from_video(
            video_path=Path(args.video_path),
            output_dir=Path(args.output_dir),
            fps=args.fps,
            max_frames=args.max_frames,
            clip_len=args.clip_len,
            stride=args.stride,
            object_provider=args.object_provider,
            confidence_threshold=args.confidence_threshold,
            default_depth=args.default_depth,
            model_path=args.model_path,
            mock_mode=args.mock_mode,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)

    print(f"Saved {len(saved_paths)} real-object observation JSON file(s):")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
