#!/usr/bin/env python3
"""Batch compare no-invert and inverted monocular depth directions for R_sd."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
BATCH_FIELDS = [
    "video_id",
    "video_path",
    "mode",
    "num_clips",
    "num_frames",
    "num_objects",
    "unique_depth_count",
    "depth_min",
    "depth_max",
    "depth_mean",
    "depth_median",
    "rsd_rows",
    "rsd_log_min",
    "rsd_log_max",
    "rsd_log_mean",
    "clip_score_min",
    "clip_score_max",
    "clip_score_mean",
]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run depth-direction R_sd comparison for every video in a directory. "
            "The project convention is larger depth = farther. Monocular depth "
            "is relative depth, not metric distance in meters."
        )
    )
    parser.add_argument(
        "--video_dir",
        default=str(PROJECT_ROOT / "data" / "videos" / "depth_check"),
        help="Directory containing real videos to compare.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "depth_direction_batch"),
        help="Working directory for per-video observations, depth maps, and CSVs.",
    )
    parser.add_argument(
        "--summary_csv",
        default=str(
            PROJECT_ROOT / "outputs" / "results" / "depth_direction_batch_summary.csv"
        ),
        help="Batch summary CSV path.",
    )
    parser.add_argument(
        "--visualization_dir",
        default=str(
            PROJECT_ROOT / "outputs" / "visualizations" / "depth_direction_batch"
        ),
        help="Directory for per-video depth direction debug PNG files.",
    )
    parser.add_argument(
        "--scale_prior_report_csv",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "results"
            / "depth_direction_scale_prior_coverage.csv"
        ),
    )
    parser.add_argument(
        "--scale_prior_candidates_yaml",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "results"
            / "depth_direction_scale_prior_candidates.yaml"
        ),
    )
    parser.add_argument(
        "--scale_prior_path",
        default=str(PROJECT_ROOT / "configs" / "scale_priors.yaml"),
    )
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "checkpoints" / "yolov8n.pt"),
        help="YOLO model path for real_detector.",
    )
    parser.add_argument(
        "--depth_model_name",
        default="depth-anything/Depth-Anything-V2-Small",
        help="Depth model id or local path for real_depth.",
    )
    parser.add_argument("--max_frames", type=int, default=20)
    parser.add_argument("--clip_len", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--default_depth", type=float, default=5.0)
    parser.add_argument(
        "--object_provider",
        choices=["real_detector", "mock"],
        default="real_detector",
        help="Use mock only for tests; real experiments should use real_detector.",
    )
    parser.add_argument(
        "--mock_mode",
        choices=["reasonable", "anomaly"],
        default="reasonable",
        help="Used only when --object_provider mock.",
    )
    parser.add_argument(
        "--real_depth_backend",
        choices=["real_depth", "mock_depth"],
        default="real_depth",
        help="Use mock_depth in tests when the real depth model is unavailable.",
    )
    parser.add_argument(
        "--max_debug_frames",
        type=int,
        default=5,
        help="Maximum debug PNG frames to save for each video.",
    )
    parser.add_argument(
        "--include_no_depth_in_summary",
        action="store_true",
        help="Also include the no_depth baseline rows in the batch summary.",
    )
    return parser.parse_args()


def find_videos(video_dir: Path) -> list[Path]:
    """Return supported video files from a directory."""

    if not video_dir.exists():
        raise FileNotFoundError(
            f"Video directory does not exist: {video_dir}. "
            "Create it and place videos there, or pass --video_dir."
        )
    videos = [
        path
        for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(
            f"No video files found in {video_dir}. Supported extensions: "
            f"{', '.join(sorted(VIDEO_EXTENSIONS))}."
        )
    return videos


def run_video_comparison(args: argparse.Namespace, video_path: Path) -> Path:
    """Run the existing single-video comparison script for one video."""

    output_dir = Path(args.output_dir) / video_path.stem
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "compare_depth_direction_rsd.py"),
        "--video_path",
        str(video_path),
        "--output_dir",
        str(output_dir),
        "--model_path",
        str(args.model_path),
        "--depth_model_name",
        str(args.depth_model_name),
        "--max_frames",
        str(args.max_frames),
        "--clip_len",
        str(args.clip_len),
        "--stride",
        str(args.stride),
        "--confidence_threshold",
        str(args.confidence_threshold),
        "--device",
        str(args.device),
        "--default_depth",
        str(args.default_depth),
        "--object_provider",
        str(args.object_provider),
        "--mock_mode",
        str(args.mock_mode),
        "--real_depth_backend",
        str(args.real_depth_backend),
    ]
    print(f"\n=== Batch video: {video_path.name} ===")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return output_dir


def run_debug_visualization(
    video_stem: str,
    comparison_dir: Path,
    visualization_root: Path,
    max_frames: int,
) -> Path:
    """Create no-invert vs invert debug visualizations for one video."""

    output_dir = visualization_root / video_stem
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "visualize_depth_direction_debug.py"),
        "--no_invert_observation_dir",
        str(comparison_dir / "real_depth_no_invert_observations"),
        "--invert_observation_dir",
        str(comparison_dir / "real_depth_invert_observations"),
        "--no_invert_depth_dir",
        str(comparison_dir / "real_depth_no_invert_depth_maps"),
        "--invert_depth_dir",
        str(comparison_dir / "real_depth_invert_depth_maps"),
        "--no_invert_rsd_csv",
        str(comparison_dir / "real_depth_no_invert_rsd_results.csv"),
        "--invert_rsd_csv",
        str(comparison_dir / "real_depth_invert_rsd_results.csv"),
        "--output_dir",
        str(output_dir),
        "--max_frames",
        str(max_frames),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return output_dir


def load_video_summary(
    video_path: Path,
    comparison_dir: Path,
    include_no_depth: bool = False,
) -> list[dict[str, str]]:
    """Load one per-video summary and attach video id/path columns."""

    summary_path = comparison_dir / "depth_direction_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing per-video summary CSV: {summary_path}")

    allowed_modes = {"real_depth_no_invert", "real_depth_invert"}
    if include_no_depth:
        allowed_modes.add("no_depth")

    rows: list[dict[str, str]] = []
    with summary_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if str(row["mode"]) not in allowed_modes:
                continue
            rows.append(
                {
                    "video_id": video_path.stem,
                    "video_path": str(video_path),
                    **row,
                }
            )
    return rows


def save_batch_summary(rows: list[dict[str, str]], output_csv: Path) -> None:
    """Save batch summary CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=BATCH_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in BATCH_FIELDS})


def run_scale_prior_reports(args: argparse.Namespace) -> None:
    """Generate exact/alias/missing/unreliable coverage report and candidates."""

    report_csv = Path(args.scale_prior_report_csv)
    candidates_yaml = Path(args.scale_prior_candidates_yaml)
    report_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "report_missing_scale_priors.py"),
        "--observation_dir",
        str(args.output_dir),
        "--scale_prior_path",
        str(args.scale_prior_path),
        "--output_csv",
        str(report_csv),
        "--min_count",
        "1",
    ]
    candidate_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_scale_prior_candidates.py"),
        "--missing_report_csv",
        str(report_csv),
        "--output_yaml",
        str(candidates_yaml),
        "--min_count",
        "2",
    ]
    subprocess.run(report_command, cwd=PROJECT_ROOT, check=True)
    subprocess.run(candidate_command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    """Run batch depth-direction comparison and coverage reporting."""

    args = parse_args()
    videos = find_videos(Path(args.video_dir))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.visualization_dir).mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []
    for video_path in videos:
        comparison_dir = run_video_comparison(args, video_path)
        run_debug_visualization(
            video_stem=video_path.stem,
            comparison_dir=comparison_dir,
            visualization_root=Path(args.visualization_dir),
            max_frames=args.max_debug_frames,
        )
        all_rows.extend(
            load_video_summary(
                video_path=video_path,
                comparison_dir=comparison_dir,
                include_no_depth=args.include_no_depth_in_summary,
            )
        )

    save_batch_summary(all_rows, Path(args.summary_csv))
    run_scale_prior_reports(args)

    print(f"\nSaved batch summary CSV: {args.summary_csv}")
    print(f"Saved debug visualizations under: {args.visualization_dir}")
    print(f"Saved scale-prior coverage report: {args.scale_prior_report_csv}")
    print(f"Saved scale-prior candidate YAML: {args.scale_prior_candidates_yaml}")
    print("\nDepth convention: larger depth values mean farther objects.")
    print("Monocular model depth is relative depth, not metric distance in meters.")


if __name__ == "__main__":
    main()
