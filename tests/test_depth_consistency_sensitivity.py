from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from semantic3d.depth_temporal_consistency import (
    depth_consistency_plot_series_from_csv,
    save_raw_and_thresholded_residual_plots_from_csv,
)
from semantic3d.io import save_clip_observation
from semantic3d.observations import (
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_AREA = 640.0 * 480.0


def _object(frame_index: int, depth: float = 5.0) -> ObjectObservationJSON:
    return ObjectObservationJSON(
        object_id=f"person_f{frame_index}",
        label="person",
        mask_area=0.2**2 * FRAME_AREA,
        frame_area=FRAME_AREA,
        depth=depth,
        confidence=1.0,
        bbox=[100 + frame_index, 100, 180 + frame_index, 260],
    )


def _frame(index: int, depth: float = 5.0) -> FrameObservationJSON:
    return FrameObservationJSON(
        frame_index=index,
        frame_id=f"frame_{index:06d}",
        width=640,
        height=480,
        objects=[_object(index, depth=depth)],
    )


def _write_clip(path: Path, depth_mode: str, depths: list[float]) -> None:
    if depth_mode == "no_depth":
        depth_provider = "none"
        invert_depth = False
    elif depth_mode == "real_depth_no_invert":
        depth_provider = "real_depth"
        invert_depth = False
    else:
        depth_provider = "real_depth"
        invert_depth = True
    clip = ClipObservationJSON(
        clip_id=f"clip_{depth_mode}",
        video_id="video_mode",
        frame_indices=list(range(len(depths))),
        frames=[_frame(index, depth) for index, depth in enumerate(depths)],
        metadata={
            "depth_mode": depth_mode,
            "depth_provider": depth_provider,
            "invert_depth": invert_depth,
        },
    )
    save_clip_observation(clip, path)


def test_mixed_depth_modes_require_explicit_selection(tmp_path: Path) -> None:
    observation_dir = tmp_path / "observations"
    _write_clip(
        observation_dir / "real_depth_invert_observations" / "clip.json",
        "real_depth_invert",
        [4.0, 5.0],
    )
    _write_clip(
        observation_dir / "real_depth_no_invert_observations" / "clip.json",
        "real_depth_no_invert",
        [4.0, 5.0],
    )

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(tmp_path / "depth_maps"),
        "--output_pair_csv",
        str(tmp_path / "pairs.csv"),
        "--output_track_csv",
        str(tmp_path / "tracks.csv"),
        "--output_clip_csv",
        str(tmp_path / "clips.csv"),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "multiple depth modes" in completed.stderr


def test_explicit_depth_mode_filters_input(tmp_path: Path) -> None:
    observation_dir = tmp_path / "observations"
    _write_clip(
        observation_dir / "real_depth_invert_observations" / "clip.json",
        "real_depth_invert",
        [4.0, 9.0],
    )
    _write_clip(
        observation_dir / "real_depth_no_invert_observations" / "clip.json",
        "real_depth_no_invert",
        [4.0, 4.0],
    )
    pair_csv = tmp_path / "pairs.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(tmp_path / "depth_maps"),
        "--depth_mode",
        "real_depth_invert",
        "--output_pair_csv",
        str(pair_csv),
        "--output_track_csv",
        str(tmp_path / "tracks.csv"),
        "--output_clip_csv",
        str(tmp_path / "clips.csv"),
        "--tolerance",
        "0",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    with pair_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows
    assert {row["depth_mode"] for row in rows} == {"real_depth_invert"}


def test_raw_plot_points_equal_valid_transition_count(tmp_path: Path) -> None:
    pair_csv = tmp_path / "pairs.csv"
    pair_csv.write_text(
        "video_id,depth_mode,track_id,label,previous_frame_index,current_frame_index,"
        "previous_depth,current_depth,previous_depth_reference,current_depth_reference,"
        "previous_relative_depth,current_relative_depth,previous_projection_scale,"
        "current_projection_scale,previous_geometry_state,current_geometry_state,"
        "raw_residual,tolerance,residual,confidence_weight,weighted_residual,valid,skip_reason\n"
        "video,real_depth_invert,trk,person,0,1,4,5,10,10,0.4,0.5,0.2,0.2,"
        "0,0,0.02,0.01,0.01,1,0.01,True,\n"
        "video,real_depth_invert,trk,person,1,2,5,0,10,10,0,0,0,0,"
        "0,0,0,0.01,0,1,0,False,invalid_current_depth\n",
        encoding="utf-8",
    )

    series = depth_consistency_plot_series_from_csv(pair_csv)
    raw_count = sum(len(value["raw_residual"]) for value in series.values())
    residual_count = sum(len(value["residual"]) for value in series.values())
    save_raw_and_thresholded_residual_plots_from_csv(
        pair_csv,
        tmp_path / "raw.png",
        tmp_path / "thresholded.png",
        tolerance=0.01,
    )

    assert raw_count == 1
    assert residual_count == 1
    assert (tmp_path / "raw.png").exists()
    assert (tmp_path / "thresholded.png").exists()


def test_sensitivity_monotonic_nonzero_count(tmp_path: Path) -> None:
    observation_dir = tmp_path / "observations"
    _write_clip(
        observation_dir / "real_depth_invert_observations" / "clip.json",
        "real_depth_invert",
        [4.0, 9.0, 9.2],
    )
    output_dir = tmp_path / "sensitivity"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_sensitivity.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(tmp_path / "depth_maps"),
        "--depth_mode",
        "real_depth_invert",
        "--tolerances",
        "0",
        "0.01",
        "0.1",
        "--output_dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    with (output_dir / "depth_consistency_sensitivity.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    counts = [int(row["nonzero_residual_count"]) for row in rows]
    assert counts == sorted(counts, reverse=True)
    zero_row = rows[0]
    assert float(zero_row["mean_R_depth_cons"]) == pytest.approx(
        float(zero_row["mean_raw_residual"])
    )
    assert (output_dir / "depth_consistency_threshold_recommendations.json").exists()
