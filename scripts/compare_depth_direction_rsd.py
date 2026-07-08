#!/usr/bin/env python3
"""Compare no-depth, real-depth, and inverted-depth R_sd results."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402

from semantic3d.io import load_clip_observation, save_clip_observation  # noqa: E402
from scripts.build_real_object_observations_from_video import (  # noqa: E402
    build_real_object_observations_from_video,
)
from scripts.run_observation_rsd_pipeline import (  # noqa: E402
    compute_rows,
    save_clip_score_plot,
    save_rows_csv,
)


SUMMARY_FIELDS = [
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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare R_sd under no depth, real depth, and inverted depth."
    )
    parser.add_argument("--video_path", required=True)
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
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "depth_direction_comparison"),
    )
    parser.add_argument(
        "--object_provider",
        choices=["real_detector", "mock"],
        default="real_detector",
        help="Mostly for tests; real experiments should use real_detector.",
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
        help="Use mock_depth in tests when real depth model is unavailable.",
    )
    return parser.parse_args()


def _mode_configs(args: argparse.Namespace) -> list[dict[str, object]]:
    """Return the three comparison modes."""

    return [
        {
            "mode": "no_depth",
            "depth_provider": "none",
            "invert_depth": False,
            "observation_dir": Path(args.output_dir) / "no_depth_observations",
            "depth_dir": Path(args.output_dir) / "no_depth_depth_maps",
            "csv_path": Path(args.output_dir) / "no_depth_rsd_results.csv",
            "plot_path": Path(args.output_dir) / "no_depth_rsd_scores.png",
        },
        {
            "mode": "real_depth_no_invert",
            "depth_provider": args.real_depth_backend,
            "invert_depth": False,
            "observation_dir": Path(args.output_dir)
            / "real_depth_no_invert_observations",
            "depth_dir": Path(args.output_dir) / "real_depth_no_invert_depth_maps",
            "csv_path": Path(args.output_dir) / "real_depth_no_invert_rsd_results.csv",
            "plot_path": Path(args.output_dir) / "real_depth_no_invert_rsd_scores.png",
        },
        {
            "mode": "real_depth_invert",
            "depth_provider": args.real_depth_backend,
            "invert_depth": True,
            "observation_dir": Path(args.output_dir)
            / "real_depth_invert_observations",
            "depth_dir": Path(args.output_dir) / "real_depth_invert_depth_maps",
            "csv_path": Path(args.output_dir) / "real_depth_invert_rsd_results.csv",
            "plot_path": Path(args.output_dir) / "real_depth_invert_rsd_scores.png",
        },
    ]


def run_mode(args: argparse.Namespace, config: dict[str, object]) -> dict[str, object]:
    """Run one depth comparison mode and return summary statistics."""

    mode = str(config["mode"])
    observation_dir = Path(config["observation_dir"])
    depth_dir = Path(config["depth_dir"])
    csv_path = Path(config["csv_path"])
    plot_path = Path(config["plot_path"])
    depth_provider = str(config["depth_provider"])
    invert_depth = bool(config["invert_depth"])

    print(f"\n=== Running mode: {mode} ===")
    try:
        build_real_object_observations_from_video(
            video_path=Path(args.video_path),
            output_dir=observation_dir,
            max_frames=args.max_frames,
            clip_len=args.clip_len,
            stride=args.stride,
            object_provider=args.object_provider,
            confidence_threshold=args.confidence_threshold,
            default_depth=args.default_depth,
            model_path=args.model_path,
            device=args.device,
            skip_unknown_scale_prior=False,
            mock_mode=args.mock_mode,
            depth_provider=depth_provider,
            depth_model_name=args.depth_model_name,
            invert_depth=invert_depth,
            save_depth_maps=depth_provider != "none",
            depth_output_dir=str(depth_dir),
        )
        if mode == "no_depth":
            _force_default_depth(observation_dir, args.default_depth)

        rows = compute_rows(observation_dir)
        save_rows_csv(rows, csv_path)
        save_clip_score_plot(rows, plot_path)
        if not rows:
            print(
                "No R_sd rows were produced. Possible reasons: fewer than two "
                "detected objects per frame, missing scale priors, or no valid "
                "object pairs."
            )
        print(f"Saved mode CSV: {csv_path}")
    except Exception as exc:
        if depth_provider == "real_depth":
            print(
                f"WARNING: mode {mode} failed but no_depth can still be used: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            rows = []
        else:
            raise

    summary = summarize_mode(mode, observation_dir, csv_path)
    return summary


def _force_default_depth(observation_dir: Path, default_depth: float) -> None:
    """Rewrite no-depth observations so object.depth equals the default depth."""

    for json_path in observation_dir.glob("*.json"):
        clip = load_clip_observation(json_path)
        frames = []
        for frame in clip.frames:
            objects = [replace(obj, depth=float(default_depth)) for obj in frame.objects]
            frames.append(replace(frame, objects=objects, depth_map_path=None))
        save_clip_observation(replace(clip, frames=frames), json_path)


def summarize_mode(mode: str, observation_dir: Path, csv_path: Path) -> dict[str, object]:
    """Summarize object depths and R_sd rows for one mode."""

    clips = [load_clip_observation(path) for path in sorted(observation_dir.glob("*.json"))]
    frames = [frame for clip in clips for frame in clip.frames]
    objects = [obj for frame in frames for obj in frame.objects]
    depths = np.asarray([float(obj.depth) for obj in objects], dtype=float)

    rows = _read_csv_rows(csv_path)
    rsd_logs = np.asarray([float(row["R_sd_log"]) for row in rows], dtype=float)
    clip_scores_by_clip: dict[str, float] = {}
    for row in rows:
        clip_scores_by_clip[str(row["clip_id"])] = float(row["clip_score"])
    clip_scores = np.asarray(list(clip_scores_by_clip.values()), dtype=float)

    return {
        "mode": mode,
        "num_clips": len(clips),
        "num_frames": len(frames),
        "num_objects": len(objects),
        "unique_depth_count": _unique_count(depths),
        "depth_min": _stat(depths, "min"),
        "depth_max": _stat(depths, "max"),
        "depth_mean": _stat(depths, "mean"),
        "depth_median": _stat(depths, "median"),
        "rsd_rows": len(rows),
        "rsd_log_min": _stat(rsd_logs, "min"),
        "rsd_log_max": _stat(rsd_logs, "max"),
        "rsd_log_mean": _stat(rsd_logs, "mean"),
        "clip_score_min": _stat(clip_scores, "min"),
        "clip_score_max": _stat(clip_scores, "max"),
        "clip_score_mean": _stat(clip_scores, "mean"),
    }


def _read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read residual CSV rows, handling missing/empty files."""

    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _unique_count(values: np.ndarray) -> int:
    """Return rounded unique count for numeric arrays."""

    if values.size == 0:
        return 0
    return int(len(set(round(float(value), 4) for value in values)))


def _stat(values: np.ndarray, name: str) -> float:
    """Compute a scalar statistic or return 0 for empty arrays."""

    if values.size == 0:
        return 0.0
    if name == "min":
        return float(np.min(values))
    if name == "max":
        return float(np.max(values))
    if name == "mean":
        return float(np.mean(values))
    if name == "median":
        return float(np.median(values))
    raise ValueError(f"Unknown stat: {name}")


def save_summary(summary_rows: Iterable[dict[str, object]], output_path: Path) -> None:
    """Save summary statistics CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row[field] for field in SUMMARY_FIELDS})


def print_human_check_prompt() -> None:
    """Print manual inspection guidance for depth direction."""

    print("\nDepth direction inspection guide:")
    print("  Project convention: larger depth values mean farther objects.")
    print("  Please inspect the debug visualizations manually:")
    print("  - Nearby objects should have smaller depth values.")
    print("  - Farther objects should have larger depth values.")
    print("  - If the relationship is reversed, prefer --invert_depth.")
    print("  - Monocular depth is relative depth, not metric distance in meters.")
    print("  - Interpret R_sd together with visual evidence and ablation results.")


def main() -> None:
    """Run all comparison modes and save a summary CSV."""

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [run_mode(args, config) for config in _mode_configs(args)]
    summary_path = output_dir / "depth_direction_summary.csv"
    save_summary(summaries, summary_path)
    print(f"\nSaved summary CSV: {summary_path}")
    print_human_check_prompt()


if __name__ == "__main__":
    main()
