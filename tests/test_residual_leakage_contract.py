"""Clip identity, leakage policy, and runtime path relocation contracts."""

from __future__ import annotations

import csv
import json
import tempfile
import warnings
from pathlib import Path

from data.leakage import audit_split_leakage, build_clip_uid
from data.residual_dataset import RESIDUAL_NAMES, build_manifest_samples


def _record(
    sample_id: str,
    clip_id: str,
    split: str,
    *,
    source: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "clip_ids": (clip_id,),
        "split": split,
        "dataset_name": "fixture",
        "source_video_id": source,
        "group_id": f"group-{source}",
        "source_video_sha256": f"sha-{source}",
        "source_video_path": f"/videos/{source}.mp4",
        "residual_sequence_path": f"/residuals/{sample_id}.json",
    }


def test_local_clip_id_reuse_across_samples_is_not_leakage() -> None:
    result = audit_split_leakage(
        (
            _record("sample-a", "real_clip_0000", "train", source="source-a"),
            _record(
                "sample-b",
                "real_clip_0000",
                "validation",
                source="source-b",
            ),
        ),
        {"mode": "strict", "scope": "source"},
    )
    assert result["status"] == "PASS"
    assert build_clip_uid("sample-a", "real_clip_0000") != build_clip_uid(
        "sample-b", "real_clip_0000"
    )


def test_same_sample_and_clip_across_splits_is_leakage() -> None:
    records = (
        _record("sample-a", "clip-0", "train", source="source-a"),
        _record("sample-a", "clip-0", "validation", source="source-a"),
    )
    try:
        audit_split_leakage(records, {"mode": "strict", "scope": "clip"})
    except ValueError as error:
        assert "Cross-split leakage" in str(error)
    else:
        raise AssertionError("strict clip leakage did not raise")


def test_same_source_different_clips_is_leakage_in_source_scope() -> None:
    records = (
        _record("sample-a", "clip-0", "train", source="shared-source"),
        _record("sample-b", "clip-1", "validation", source="shared-source"),
    )
    try:
        audit_split_leakage(records, {"mode": "strict", "scope": "source"})
    except ValueError as error:
        assert "Cross-split leakage" in str(error)
    else:
        raise AssertionError("strict source leakage did not raise")


def test_different_sources_and_clip_uids_pass() -> None:
    result = audit_split_leakage(
        (
            _record("sample-a", "clip-0", "train", source="source-a"),
            _record("sample-b", "clip-1", "validation", source="source-b"),
        ),
        {"mode": "strict", "scope": "source"},
    )
    assert result["finding_count"] == 0


def test_warn_mode_returns_findings_without_raising() -> None:
    records = (
        _record("sample-a", "clip-0", "train", source="shared-source"),
        _record("sample-b", "clip-1", "validation", source="shared-source"),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = audit_split_leakage(
            records,
            {"enabled": True, "mode": "warn", "scope": "source"},
        )
    assert any("Cross-split leakage" in str(item.message) for item in caught)
    assert result["status"] == "FAIL"
    assert result["finding_count"] > 0


def _write_result(path: Path, payload_video_path: str, clip_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "video_path": payload_video_path,
                "clips": [
                    {
                        "clip_id": clip_id,
                        "start_frame": 0,
                        "residuals": [
                            {
                                "name": name,
                                "normalized_value": 0.1,
                                "availability": "observed",
                                "valid_mask": True,
                                "confidence": 0.9,
                            }
                            for name in RESIDUAL_NAMES
                        ],
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


def test_warn_mode_allows_dataset_and_runtime_paths_are_read_only() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        _check_warn_dataset_and_runtime_paths(Path(temporary))


def _check_warn_dataset_and_runtime_paths(tmp_path: Path) -> None:
    manifest_rows = []
    runtime_rows = []
    for index, (sample_id, split) in enumerate(
        (("train-sample", "train"), ("validation-sample", "validation"))
    ):
        local_video = tmp_path / f"{sample_id}.mp4"
        local_video.write_bytes(b"video")
        local_result = tmp_path / f"{sample_id}.json"
        old_video = f"/root/autodl-tmp/videos/{sample_id}.mp4"
        _write_result(local_result, old_video, "local_clip_0000")
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "dataset_name": "fixture",
                "source_video_id": "shared-source",
                "group_id": "shared-group",
                "split": split,
                "label": index,
                "residual_sequence_path": (
                    f"/root/autodl-tmp/residuals/{sample_id}.json"
                ),
                "source_video_path": old_video,
                "source_commit": "commit",
                "source_config_sha256": "config",
            }
        )
        runtime_rows.append(
            {
                "sample_id": sample_id,
                "local_result_json": str(local_result),
                "local_video_path": str(local_video),
                "source_video_sha256": f"sha-{sample_id}",
            }
        )

    manifest = tmp_path / "manifest.csv"
    runtime = tmp_path / "runtime.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    with runtime.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runtime_rows[0]))
        writer.writeheader()
        writer.writerows(runtime_rows)
    frozen_before = manifest.read_bytes()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = build_manifest_samples(
            manifest,
            runtime_path_manifest=runtime,
            leakage_check={"enabled": True, "mode": "warn", "scope": "source"},
        )
    assert any("Cross-split leakage" in str(item.message) for item in caught)

    assert len(bundle.samples["train"]) == 1
    assert len(bundle.samples["validation"]) == 1
    assert bundle.leakage_audit["status"] == "FAIL"
    assert bundle.runtime_path_manifest == str(runtime.resolve())
    assert manifest.read_bytes() == frozen_before
    assert all(
        not row["residual_sequence_path"].startswith("/root/autodl-tmp")
        for row in bundle.manifest_rows
    )


if __name__ == "__main__":
    tests = (
        test_local_clip_id_reuse_across_samples_is_not_leakage,
        test_same_sample_and_clip_across_splits_is_leakage,
        test_same_source_different_clips_is_leakage_in_source_scope,
        test_different_sources_and_clip_uids_pass,
        test_warn_mode_returns_findings_without_raising,
        test_warn_mode_allows_dataset_and_runtime_paths_are_read_only,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
