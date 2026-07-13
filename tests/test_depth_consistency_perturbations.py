from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from semantic3d.io import load_clip_observation, save_clip_observation
from semantic3d.observations import (
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRAME_AREA = 640.0 * 480.0


def _make_input_observations(tmp_path: Path) -> tuple[Path, Path]:
    observation_dir = tmp_path / "observations" / "real_depth_invert_observations"
    depth_dir = tmp_path / "depth_maps"
    depth_dir.mkdir(parents=True)
    frames = []
    for index, depth in enumerate([4.0, 5.0, 6.0]):
        depth_path = depth_dir / f"frame_{index:06d}_depth.npy"
        np.save(depth_path, np.full((16, 16), 10.0, dtype=np.float32))
        frames.append(
            FrameObservationJSON(
                frame_index=index,
                frame_id=f"frame_{index:06d}",
                width=640,
                height=480,
                depth_map_path=str(depth_path),
                objects=[
                    ObjectObservationJSON(
                        object_id=f"person_f{index}",
                        label="person",
                        mask_area=0.2**2 * FRAME_AREA,
                        frame_area=FRAME_AREA,
                        depth=depth,
                        confidence=1.0,
                        bbox=[100 + index, 100, 180 + index, 260],
                    )
                ],
            )
        )
    clip = ClipObservationJSON(
        clip_id="clip_real_depth_invert",
        video_id="video_perturb",
        frame_indices=[0, 1, 2],
        frames=frames,
        metadata={
            "depth_mode": "real_depth_invert",
            "depth_provider": "real_depth",
            "invert_depth": True,
        },
    )
    save_clip_observation(clip, observation_dir / "clip.json")
    return observation_dir.parent, depth_dir


def _run_pipeline(observation_dir: Path, depth_dir: Path, output_dir: Path) -> Path:
    pair_csv = output_dir / "pairs.csv"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_pipeline.py"),
        "--observation_dir",
        str(observation_dir),
        "--depth_map_dir",
        str(depth_dir),
        "--depth_mode",
        "real_depth_invert",
        "--output_pair_csv",
        str(pair_csv),
        "--output_track_csv",
        str(output_dir / "tracks.csv"),
        "--output_clip_csv",
        str(output_dir / "clips.csv"),
        "--associated_observation_dir",
        str(output_dir / "tracked"),
        "--raw_residual_visualization_path",
        str(output_dir / "raw.png"),
        "--thresholded_residual_visualization_path",
        str(output_dir / "thresholded.png"),
        "--combined_visualization_path",
        str(output_dir / "combined.png"),
        "--visualization_path",
        str(output_dir / "legacy.png"),
        "--tolerance",
        "0",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return pair_csv


def _target_raw(pair_csv: Path) -> float:
    with pair_csv.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    target = [row for row in rows if int(float(row["current_frame_index"])) == 1]
    assert target
    return float(target[0]["raw_residual"])


def _generate(
    input_dir: Path,
    output_dir: Path,
    perturbation_type: str,
    factor: float,
) -> dict:
    metadata_path = output_dir / "metadata.json"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "generate_depth_consistency_perturbations.py"),
        "--input_observation_dir",
        str(input_dir),
        "--output_observation_dir",
        str(output_dir),
        "--depth_mode",
        "real_depth_invert",
        "--auto_select_track",
        "--perturbation_type",
        perturbation_type,
        "--factor",
        str(factor),
        "--metadata_output",
        str(metadata_path),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)
    return json.loads(metadata_path.read_text(encoding="utf-8"))["perturbation"]


def test_depth_scale_perturbation_changes_raw_residual(tmp_path: Path) -> None:
    input_dir, depth_dir = _make_input_observations(tmp_path)
    original_text = (input_dir / "real_depth_invert_observations" / "clip.json").read_text(
        encoding="utf-8"
    )
    original_raw = _target_raw(_run_pipeline(input_dir, depth_dir, tmp_path / "original"))
    perturbed_dir = tmp_path / "perturbed_depth"

    metadata = _generate(input_dir, perturbed_dir, "depth_scale", 1.5)
    perturbed_raw = _target_raw(
        _run_pipeline(perturbed_dir, depth_dir, tmp_path / "perturbed_result")
    )

    assert perturbed_raw != original_raw
    assert metadata["perturbed_depth"] > metadata["original_depth"]
    assert (
        input_dir / "real_depth_invert_observations" / "clip.json"
    ).read_text(encoding="utf-8") == original_text


def test_mask_area_perturbation_changes_raw_residual(tmp_path: Path) -> None:
    input_dir, depth_dir = _make_input_observations(tmp_path)
    original_raw = _target_raw(_run_pipeline(input_dir, depth_dir, tmp_path / "original"))
    perturbed_dir = tmp_path / "perturbed_area"

    metadata = _generate(input_dir, perturbed_dir, "mask_area_scale", 1.5)
    perturbed_raw = _target_raw(
        _run_pipeline(perturbed_dir, depth_dir, tmp_path / "perturbed_result")
    )

    assert perturbed_raw != original_raw
    assert metadata["perturbed_mask_area"] > metadata["original_mask_area"]


def test_combined_inconsistent_and_factor_trend(tmp_path: Path) -> None:
    input_dir, depth_dir = _make_input_observations(tmp_path)
    values = []
    for factor in [1.2, 1.5, 2.0]:
        perturbed_dir = tmp_path / f"combined_{factor}"
        _generate(input_dir, perturbed_dir, "combined_inconsistent", factor)
        values.append(
            _target_raw(_run_pipeline(perturbed_dir, depth_dir, tmp_path / f"result_{factor}"))
        )

    assert values == sorted(values)
    assert values[-1] > values[0]


def test_non_target_track_not_modified(tmp_path: Path) -> None:
    input_dir, _ = _make_input_observations(tmp_path)
    perturbed_dir = tmp_path / "perturbed"
    _generate(input_dir, perturbed_dir, "depth_scale", 1.5)

    clip_path = next(
        path for path in perturbed_dir.rglob("*.json") if path.name != "metadata.json"
    )
    clip = load_clip_observation(clip_path)
    untouched = [
        obj
        for frame in clip.frames
        for obj in frame.objects
        if int(frame.frame_index) != 1
    ]
    assert untouched
    assert {round(obj.depth, 3) for obj in untouched} == {4.0, 6.0}


def test_perturbation_experiment_outputs_csv_and_plot(tmp_path: Path) -> None:
    input_dir, depth_dir = _make_input_observations(tmp_path)
    output_dir = tmp_path / "experiment"
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_depth_consistency_perturbation_experiment.py"),
        "--input_observation_dir",
        str(input_dir),
        "--depth_map_dir",
        str(depth_dir),
        "--depth_mode",
        "real_depth_invert",
        "--output_dir",
        str(output_dir),
        "--tolerance",
        "0.02",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True)

    csv_path = output_dir / "depth_consistency_perturbation_results.csv"
    png_path = output_dir / "depth_consistency_perturbation_response.png"
    assert csv_path.exists()
    assert png_path.exists()
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 9
    combined = [row for row in rows if row["perturbation_type"] == "combined_inconsistent"]
    assert combined
    assert max(float(row["raw_residual"]) for row in combined) > float(rows[0]["raw_residual"])
