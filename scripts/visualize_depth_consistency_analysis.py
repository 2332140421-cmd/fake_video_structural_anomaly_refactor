#!/usr/bin/env python3
"""Create raw and thresholded R_depth_cons analysis plots from pair CSV."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.depth_temporal_consistency import (  # noqa: E402
    depth_consistency_plot_series_from_csv,
    save_depth_consistency_tracks_plot_from_csv,
    save_raw_and_thresholded_residual_plots_from_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize R_depth_cons analysis from CSV.")
    parser.add_argument(
        "--pair_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "depth_consistency_pairs.csv"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "visualizations"),
    )
    parser.add_argument("--tolerance", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path, thresholded_path = save_raw_and_thresholded_residual_plots_from_csv(
        args.pair_csv,
        output_dir / "depth_consistency_raw_residual.png",
        output_dir / "depth_consistency_thresholded_residual.png",
        tolerance=args.tolerance,
    )
    combined_path = save_depth_consistency_tracks_plot_from_csv(
        args.pair_csv,
        output_dir / "depth_consistency_combined_analysis.png",
    )
    series = depth_consistency_plot_series_from_csv(args.pair_csv)
    point_count = sum(len(value["residual"]) for value in series.values())
    print(f"Saved raw residual plot: {raw_path}")
    print(f"Saved thresholded residual plot: {thresholded_path}")
    print(f"Saved combined analysis plot: {combined_path}")
    print(f"valid transition/residual point count: {point_count}")


if __name__ == "__main__":
    main()
