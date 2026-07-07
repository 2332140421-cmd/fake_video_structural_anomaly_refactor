from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_scale_depth_experiment.py"
DATA_DIR = PROJECT_ROOT / "data" / "synthetic_observations"
RESULT_CSV = PROJECT_ROOT / "outputs" / "results" / "scale_depth_results.csv"
RESULT_METRICS = PROJECT_ROOT / "outputs" / "results" / "scale_depth_metrics.json"
RESULT_PNG = PROJECT_ROOT / "outputs" / "visualizations" / "scale_depth_scores.png"


def _run_experiment(threshold: float = 0.1) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--threshold", str(threshold)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _read_result_rows() -> list[dict[str, str]]:
    with RESULT_CSV.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def test_synthetic_observation_dataset_has_at_least_thirty_json_samples() -> None:
    sample_paths = sorted(DATA_DIR.glob("*.json"))
    assert len(sample_paths) >= 30

    expected_labels = []
    invalid_cases = []
    for path in sample_paths:
        sample = json.loads(path.read_text(encoding="utf-8"))
        expected_labels.append(sample["expected_label"])
        if sample["case_id"].startswith("case_0") and "invalid" in sample["case_id"]:
            invalid_cases.append(sample["case_id"])
        assert {
            "case_id",
            "expected_label",
            "description",
            "frame_width",
            "frame_height",
            "objects",
        }.issubset(sample)
        assert sample["expected_label"] in {0, 1}
        assert len(sample["objects"]) >= 2
        for obj in sample["objects"]:
            assert {
                "object_id",
                "label",
                "mask_area",
                "frame_area",
                "depth",
                "confidence",
            }.issubset(obj)

    assert 0 in expected_labels
    assert 1 in expected_labels
    assert len(invalid_cases) >= 4


def test_scale_depth_experiment_script_generates_csv_metrics_and_plot() -> None:
    result = _run_experiment()

    assert "Processed 30 synthetic samples" in result.stdout
    assert RESULT_CSV.exists()
    assert RESULT_METRICS.exists()
    assert RESULT_PNG.exists()
    assert RESULT_CSV.stat().st_size > 0
    assert RESULT_METRICS.stat().st_size > 0
    assert RESULT_PNG.stat().st_size > 0


def test_csv_contains_required_fields() -> None:
    _run_experiment()
    rows = _read_result_rows()
    assert len(rows) >= 30
    assert {
        "case_id",
        "expected_label",
        "max_R_sd",
        "mean_R_sd",
        "topk_mean_R_sd",
        "max_R_sd_log",
        "mean_R_sd_log",
        "topk_mean_R_sd_log",
        "predicted_label",
        "is_correct",
        "error_message",
    }.issubset(rows[0].keys())


def test_reasonable_samples_have_lower_average_residual_than_anomalies() -> None:
    _run_experiment()
    rows = _read_result_rows()
    anomaly_scores = [
        float(row["max_R_sd_log"])
        for row in rows
        if row["status"] == "ok" and row["expected_label"] == "1"
    ]
    reasonable_scores = [
        float(row["max_R_sd_log"])
        for row in rows
        if row["status"] == "ok" and row["expected_label"] == "0"
    ]

    assert anomaly_scores
    assert reasonable_scores
    assert sum(anomaly_scores) / len(anomaly_scores) > sum(reasonable_scores) / len(
        reasonable_scores
    )


def test_simple_threshold_basically_separates_valid_cases() -> None:
    _run_experiment(threshold=0.1)
    metrics = json.loads(RESULT_METRICS.read_text(encoding="utf-8"))

    assert metrics["num_valid"] >= 20
    assert metrics["accuracy"] >= 0.9
    assert metrics["precision"] >= 0.9
    assert metrics["recall"] >= 0.9
    assert metrics["f1"] >= 0.9


def test_invalid_inputs_do_not_crash_and_have_clear_errors() -> None:
    _run_experiment()
    rows = _read_result_rows()
    invalid_rows = [row for row in rows if row["status"] == "invalid"]

    assert len(invalid_rows) >= 4
    assert all(row["error_message"] for row in invalid_rows)
    assert any("mask_area must be > 0" in row["error_message"] for row in invalid_rows)
    assert any("depth must be > 0" in row["error_message"] for row in invalid_rows)
    assert any("Missing scale prior" in row["error_message"] for row in invalid_rows)
    assert any("confidence must be >=" in row["error_message"] for row in invalid_rows)
