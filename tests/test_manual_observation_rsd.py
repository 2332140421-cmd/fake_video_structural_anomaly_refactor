from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_manual_observation_rsd import (  # noqa: E402
    DEFAULT_SCALE_PRIORS,
    evaluate_manual_observation,
    load_manual_observation,
    run_manual_batch,
)
from semantic3d.scale_depth import scale_depth_residual_log  # noqa: E402


MANUAL_DIR = PROJECT_ROOT / "data" / "manual_observations"


def test_manual_observation_can_be_loaded() -> None:
    path = MANUAL_DIR / "manual_001_person_car_reasonable.json"

    case_id, expected_label, frame = load_manual_observation(path)

    assert case_id == "manual_001_person_car_reasonable"
    assert expected_label == 0
    assert frame.frame_id == "manual_person_car_reasonable"
    assert len(frame.objects) == 2
    assert frame.objects[0].label == "person"


def test_manual_observation_rsd_can_be_computed() -> None:
    path = MANUAL_DIR / "manual_003_soccer_elephant_reasonable.json"
    _case_id, _expected_label, frame = load_manual_observation(path)
    soccer_ball = frame.objects[0].to_scale_depth_observation()
    elephant = frame.objects[1].to_scale_depth_observation()

    residual_log, details = scale_depth_residual_log(
        soccer_ball, elephant, DEFAULT_SCALE_PRIORS
    )

    assert residual_log == pytest.approx(0.0, abs=1e-6)
    assert details["log_lower"] <= details["log_depth_ratio"] <= details["log_upper"]


def test_reasonable_scores_are_lower_than_anomaly_scores() -> None:
    reasonable_paths = sorted(MANUAL_DIR.glob("*reasonable.json"))
    anomaly_paths = sorted(MANUAL_DIR.glob("*anomaly.json"))

    reasonable_scores = [
        float(evaluate_manual_observation(path)["max_R_sd_log"])
        for path in reasonable_paths
    ]
    anomaly_scores = [
        float(evaluate_manual_observation(path)["max_R_sd_log"])
        for path in anomaly_paths
    ]

    assert reasonable_scores
    assert anomaly_scores
    assert max(reasonable_scores) < 0.1
    assert min(anomaly_scores) > 1.0


def test_run_manual_batch_generates_csv_and_png(tmp_path: Path) -> None:
    output_csv = tmp_path / "results" / "manual_observation_rsd_results.csv"
    output_png = tmp_path / "visualizations" / "manual_observation_rsd_scores.png"

    rows = run_manual_batch(MANUAL_DIR, output_csv, output_png)

    assert output_csv.exists()
    assert output_png.exists()
    assert len(rows) >= 5

    with output_csv.open("r", encoding="utf-8", newline="") as file:
        csv_rows = list(csv.DictReader(file))
    assert len(csv_rows) == len(rows)
    assert {
        "case_id",
        "expected_label",
        "max_R_sd",
        "mean_R_sd",
        "topk_mean_R_sd",
        "max_R_sd_log",
        "mean_R_sd_log",
        "topk_mean_R_sd_log",
    }.issubset(csv_rows[0].keys())


def test_manual_scripts_run() -> None:
    batch_script = PROJECT_ROOT / "scripts" / "run_manual_observation_rsd.py"
    demo_script = PROJECT_ROOT / "examples" / "demo_manual_observation_rsd.py"

    subprocess.run(
        [sys.executable, str(batch_script)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [sys.executable, str(demo_script)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        PROJECT_ROOT / "outputs" / "results" / "manual_observation_rsd_results.csv"
    ).exists()
    assert (
        PROJECT_ROOT / "outputs" / "visualizations" / "manual_observation_rsd_scores.png"
    ).exists()
    assert (
        PROJECT_ROOT / "outputs" / "visualizations" / "manual_observation_pair_graph.png"
    ).exists()
