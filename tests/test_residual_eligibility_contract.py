"""Direct assertions for collect-all eligibility and split-decoupled loading."""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from data.eligibility import inspect_residual_eligibility
from data.residual_dataset import RESIDUAL_NAMES, build_manifest_samples


def _write_result(path: Path, *, valid_rows: int) -> None:
    residuals = []
    for index, name in enumerate(RESIDUAL_NAMES):
        valid = index < valid_rows
        residuals.append(
            {
                "name": name,
                "normalized_value": 0.1 if valid else None,
                "availability": "observed" if valid else "blocked_by_input",
                "valid_mask": valid,
                "confidence": 0.9 if valid else 0.0,
            }
        )
    path.write_text(
        json.dumps(
            {
                "clips": [
                    {
                        "clip_id": "local_clip_0000",
                        "start_frame": 0,
                        "residuals": residuals,
                    }
                ],
                "metadata": {
                    "authenticity_label_used": False,
                    "historical_csv_read": False,
                    "m6_to_a2_bridge_called": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _manifest(tmp_path: Path, *, no_valid_test_count: int = 1) -> Path:
    specifications = [
        ("train-real", "train", 0, 1),
        ("train-fake", "train", 1, 1),
        ("validation-real", "validation", 0, 1),
        ("validation-fake", "validation", 1, 1),
    ]
    specifications.extend(
        (f"test-empty-{index}", "test", index % 2, 0)
        for index in range(no_valid_test_count)
    )
    rows = []
    for sample_id, split, label, valid_rows in specifications:
        result = tmp_path / f"{sample_id}.json"
        _write_result(result, valid_rows=valid_rows)
        rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "fixture",
                "source_video_id": f"source-{sample_id}",
                "group_id": f"group-{sample_id}",
                "split": split,
                "label": label,
                "residual_sequence_path": str(result),
                "source_commit": "commit",
                "source_config_sha256": "config",
            }
        )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_partially_available_residual_is_model_eligible() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        result = Path(temporary) / "partial.json"
        _write_result(result, valid_rows=1)
        inspection = inspect_residual_eligibility(
            sample_id="sample",
            split="train",
            label=0,
            residual_path=result,
            channel_names=RESIDUAL_NAMES,
        )
        assert inspection.report.eligibility_status == "partial_missing"
        assert inspection.report.model_eligible
        assert inspection.report.valid_count == 1


def test_all_unavailable_residual_is_no_valid_residual() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        result = Path(temporary) / "empty.json"
        _write_result(result, valid_rows=0)
        inspection = inspect_residual_eligibility(
            sample_id="sample",
            split="test",
            label=0,
            residual_path=result,
            channel_names=RESIDUAL_NAMES,
        )
        assert inspection.report.eligibility_status == "no_valid_residual"
        assert not inspection.report.model_eligible


def test_exclude_initializes_train_validation_and_reports_test_exclusion() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        manifest = _manifest(Path(temporary))
        bundle = build_manifest_samples(
            manifest,
            load_splits=("train", "validation"),
            no_valid_residual_policy="exclude",
            leakage_check={"enabled": True, "mode": "strict", "scope": "source"},
        )
        assert len(bundle.samples["train"]) == 2
        assert len(bundle.samples["validation"]) == 2
        assert bundle.samples["test"] == ()
        assert bundle.eligibility_summary["splits"]["test"] == {
            "original_count": 1,
            "eligible_count": 0,
            "excluded_count": 1,
            "coverage": 0.0,
            "status_counts": {
                "eligible": 0,
                "partial_missing": 0,
                "no_valid_residual": 1,
                "malformed": 0,
                "missing_file": 0,
                "schema_error": 0,
            },
        }


def test_error_policy_raises_clear_aggregate() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        manifest = _manifest(Path(temporary))
        try:
            build_manifest_samples(
                manifest,
                load_splits=("train", "validation"),
                no_valid_residual_policy="error",
            )
        except ValueError as error:
            assert "no_valid_residual" in str(error)
            assert "test-empty-0" in str(error)
        else:
            raise AssertionError("error policy did not reject no_valid_residual")


def test_collect_all_does_not_stop_at_first_no_valid_residual() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        manifest = _manifest(Path(temporary), no_valid_test_count=2)
        bundle = build_manifest_samples(
            manifest,
            load_splits=("train", "validation"),
            no_valid_residual_policy="exclude",
        )
        reports = [
            report
            for report in bundle.eligibility_reports
            if report.eligibility_status == "no_valid_residual"
        ]
        assert [report.sample_id for report in reports] == [
            "test-empty-0",
            "test-empty-1",
        ]


def test_strict_source_leakage_still_raises() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        tmp_path = Path(temporary)
        manifest = _manifest(tmp_path, no_valid_test_count=0)
        with manifest.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[2]["source_video_id"] = rows[0]["source_video_id"]
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        try:
            build_manifest_samples(
                manifest,
                load_splits=("train", "validation"),
                no_valid_residual_policy="exclude",
                leakage_check={
                    "enabled": True,
                    "mode": "strict",
                    "scope": "source",
                },
            )
        except ValueError as error:
            assert "Cross-split leakage" in str(error)
        else:
            raise AssertionError("strict source leakage was not rejected")


if __name__ == "__main__":
    tests = (
        test_partially_available_residual_is_model_eligible,
        test_all_unavailable_residual_is_no_valid_residual,
        test_exclude_initializes_train_validation_and_reports_test_exclusion,
        test_error_policy_raises_clear_aggregate,
        test_collect_all_does_not_stop_at_first_no_valid_residual,
        test_strict_source_leakage_still_raises,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
