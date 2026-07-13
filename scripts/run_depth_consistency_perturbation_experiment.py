#!/usr/bin/env python3
"""Run controlled perturbation experiments for R_depth_cons."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.generate_depth_consistency_perturbations import (  # noqa: E402
    auto_select_target,
    load_associated_frames,
)


FIELDS = [
    "case_id",
    "perturbation_type",
    "factor",
    "video_id",
    "track_id",
    "frame_index",
    "original_depth",
    "perturbed_depth",
    "original_mask_area",
    "perturbed_mask_area",
    "previous_relative_depth",
    "current_relative_depth",
    "previous_projection_scale",
    "current_projection_scale",
    "raw_residual",
    "tolerance",
    "R_depth_cons",
    "weighted_residual",
    "is_target_transition",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R_depth_cons perturbation experiment.")
    parser.add_argument("--input_observation_dir", required=True)
    parser.add_argument("--depth_map_dir", required=True)
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "depth_consistency_perturbation_experiment"),
    )
    parser.add_argument(
        "--depth_mode",
        choices=["auto", "no_depth", "real_depth_no_invert", "real_depth_invert"],
        default="real_depth_invert",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    return parser.parse_args()


def run_pipeline_case(
    observation_dir: Path,
    depth_map_dir: Path,
    output_dir: Path,
    depth_mode: str,
    tolerance: float,
    case_id: str,
) -> Path:
    """Run pipeline for one case and return pair CSV."""

    case_dir = output_dir / "runs" / case_id
    pair_csv = case_dir / "depth_consistency_pairs.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(depth_map_dir),
        "--depth_mode",
        str(depth_mode),
        "--output_pair_csv",
        str(pair_csv),
        "--output_track_csv",
        str(case_dir / "depth_consistency_tracks.csv"),
        "--output_clip_csv",
        str(case_dir / "depth_consistency_clips.csv"),
        "--associated_observation_dir",
        str(case_dir / "observations_with_tracks"),
        "--raw_residual_visualization_path",
        str(case_dir / "raw_residual.png"),
        "--thresholded_residual_visualization_path",
        str(case_dir / "thresholded_residual.png"),
        "--combined_visualization_path",
        str(case_dir / "combined_analysis.png"),
        "--visualization_path",
        str(case_dir / "legacy_tracks.png"),
        "--tolerance",
        str(tolerance),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return pair_csv


def generate_case(
    input_observation_dir: Path,
    output_case_dir: Path,
    depth_mode: str,
    perturbation_type: str,
    factor: float,
    video_id: str,
    track_id: str,
    frame_index: int,
) -> dict[str, Any]:
    """Generate one perturbed observation case and return metadata."""

    metadata_path = output_case_dir / "perturbation_metadata.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_depth_consistency_perturbations.py"),
        "--input_observation_dir",
        str(input_observation_dir),
        "--output_observation_dir",
        str(output_case_dir),
        "--depth_mode",
        str(depth_mode),
        "--video_id",
        str(video_id),
        "--track_id",
        str(track_id),
        "--frame_index",
        str(frame_index),
        "--perturbation_type",
        perturbation_type,
        "--factor",
        str(factor),
        "--metadata_output",
        str(metadata_path),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    with metadata_path.open("r", encoding="utf-8") as file:
        return json.load(file)["perturbation"]


def extract_target_row(
    pair_csv: Path,
    metadata: dict[str, Any],
    case_id: str,
    perturbation_type: str,
    factor: float,
) -> dict[str, Any]:
    """Extract target transition response row from a pair CSV."""

    video_id = str(metadata["target_video_id"])
    track_id = str(metadata["target_track_id"])
    frame_index = int(metadata["target_frame_index"])
    rows = _read_rows(pair_csv)
    candidates = [
        row
        for row in rows
        if row.get("video_id") == video_id
        and row.get("track_id") == track_id
        and int(float(row.get("current_frame_index", -1))) == frame_index
    ]
    if not candidates:
        candidates = [
            row
            for row in rows
            if row.get("video_id") == video_id
            and row.get("track_id") == track_id
            and int(float(row.get("previous_frame_index", -1))) == frame_index
        ]
    row = candidates[0] if candidates else {}
    return {
        "case_id": case_id,
        "perturbation_type": perturbation_type,
        "factor": factor,
        "video_id": video_id,
        "track_id": track_id,
        "frame_index": frame_index,
        "original_depth": metadata.get("original_depth", ""),
        "perturbed_depth": metadata.get("perturbed_depth", metadata.get("original_depth", "")),
        "original_mask_area": metadata.get("original_mask_area", ""),
        "perturbed_mask_area": metadata.get("perturbed_mask_area", metadata.get("original_mask_area", "")),
        "previous_relative_depth": row.get("previous_relative_depth", ""),
        "current_relative_depth": row.get("current_relative_depth", ""),
        "previous_projection_scale": row.get("previous_projection_scale", ""),
        "current_projection_scale": row.get("current_projection_scale", ""),
        "raw_residual": row.get("raw_residual", "0"),
        "tolerance": row.get("tolerance", ""),
        "R_depth_cons": row.get("residual", "0"),
        "weighted_residual": row.get("weighted_residual", "0"),
        "is_target_transition": bool(row),
    }


def save_results(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})


def save_response_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(row["case_id"]) for row in rows]
    values = [float(row["raw_residual"]) for row in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_ylabel("target transition raw_residual")
    ax.set_title("R_depth_cons Controlled Perturbation Response")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _case_specs() -> list[tuple[str, str, float]]:
    return [
        ("original", "none", 1.0),
        ("depth_scale_1_2", "depth_scale", 1.2),
        ("depth_scale_1_5", "depth_scale", 1.5),
        ("depth_scale_2_0", "depth_scale", 2.0),
        ("mask_area_scale_1_2", "mask_area_scale", 1.2),
        ("mask_area_scale_1_5", "mask_area_scale", 1.5),
        ("mask_area_scale_2_0", "mask_area_scale", 2.0),
        ("combined_inconsistent_1_5", "combined_inconsistent", 1.5),
        ("combined_inconsistent_2_0", "combined_inconsistent", 2.0),
    ]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_mode, frames_by_video = load_associated_frames(
        Path(args.input_observation_dir),
        args.depth_mode,
    )
    video_id, track_id, frame_index = auto_select_target(frames_by_video)
    print(f"Selected target: video={video_id}, track={track_id}, frame={frame_index}")

    rows: list[dict[str, Any]] = []
    original_pair_csv = run_pipeline_case(
        Path(args.input_observation_dir),
        Path(args.depth_map_dir),
        output_dir,
        depth_mode,
        args.tolerance,
        "original",
    )
    original_metadata = {
        "target_video_id": video_id,
        "target_track_id": track_id,
        "target_frame_index": frame_index,
        "original_depth": "",
        "perturbed_depth": "",
        "original_mask_area": "",
        "perturbed_mask_area": "",
    }
    rows.append(
        extract_target_row(
            original_pair_csv,
            original_metadata,
            "original",
            "none",
            1.0,
        )
    )

    for case_id, perturbation_type, factor in _case_specs()[1:]:
        case_observation_dir = output_dir / "perturbed_observations" / case_id
        metadata = generate_case(
            Path(args.input_observation_dir),
            case_observation_dir,
            depth_mode,
            perturbation_type,
            factor,
            video_id,
            track_id,
            frame_index,
        )
        pair_csv = run_pipeline_case(
            case_observation_dir,
            Path(args.depth_map_dir),
            output_dir,
            depth_mode,
            args.tolerance,
            case_id,
        )
        rows.append(
            extract_target_row(
                pair_csv,
                metadata,
                case_id,
                perturbation_type,
                factor,
            )
        )

    output_csv = output_dir / "depth_consistency_perturbation_results.csv"
    output_png = output_dir / "depth_consistency_perturbation_response.png"
    root_csv = PROJECT_ROOT / "outputs" / "results" / "depth_consistency_perturbation_results.csv"
    root_png = PROJECT_ROOT / "outputs" / "visualizations" / "depth_consistency_perturbation_response.png"
    save_results(rows, output_csv)
    save_results(rows, root_csv)
    save_response_plot(rows, output_png)
    save_response_plot(rows, root_png)
    print(f"Saved perturbation results: {output_csv}")
    print(f"Saved root perturbation results: {root_csv}")
    print(f"Saved perturbation plot: {output_png}")
    print(f"Saved root perturbation plot: {root_png}")
    for row in rows:
        print(
            f"{row['case_id']}: raw={float(row['raw_residual']):.6f}, "
            f"R={float(row['R_depth_cons']):.6f}"
        )


if __name__ == "__main__":
    main()
