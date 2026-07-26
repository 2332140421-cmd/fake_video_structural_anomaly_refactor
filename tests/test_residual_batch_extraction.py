import io
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from data.residual_dataset import RESIDUAL_NAMES
from inference.pipeline import ForgeryAnalysisPipeline
from scripts import extract_aigvdbench_media as media
from scripts import extract_residual_dataset as residual


def _zip_fixture(path: Path, content: bytes) -> tuple[bytes, dict]:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("videos/exact.mp4", content)
    blob = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("videos/exact.mp4")
    return blob, {
        "member_path": info.filename,
        "compressed_size": info.compress_size,
        "uncompressed_size": info.file_size,
        "crc": f"{info.CRC:08x}",
        "compression": info.compress_type,
        "local_header_offset": info.header_offset,
    }


def test_exact_range_member_extraction_crc_and_part_boundary(tmp_path, monkeypatch):
    content = b"video-payload" * 100
    blob, member = _zip_fixture(tmp_path / "fixture.zip", content)

    monkeypatch.setattr(
        media,
        "_read_range",
        lambda _url, start, size: blob[start : start + size],
    )

    def open_range(_url, start, size):
        return io.BytesIO(blob[start : start + size]), {
            "content-range": f"bytes {start}-{start + size - 1}/{len(blob)}"
        }

    monkeypatch.setattr(media, "_open_range", open_range)
    part = tmp_path / "sample.mp4.part"
    result = media._extract_member(
        url="https://official.example/archive.zip",
        member=member,
        expected_member_path="videos/exact.mp4",
        part_path=part,
    )
    assert part.read_bytes() == content
    assert result["actual_file_size"] == len(content)
    assert result["zip_crc"] == member["crc"]
    assert not (tmp_path / "sample.mp4").exists()

    bad = {**member, "crc": "00000000"}
    with pytest.raises(ValueError, match="CRC mismatch"):
        media._extract_member(
            url="https://official.example/archive.zip",
            member=bad,
            expected_member_path="videos/exact.mp4",
            part_path=tmp_path / "bad.mp4.part",
        )


def test_safe_group_encoding_is_bijective_and_only_official_url_is_used():
    assert media.safe_identity("aigvdbench:a/b") != media.safe_identity(
        "aigvdbench:a_b"
    )
    url = media.archive_url(media.REAL_ARCHIVE)
    assert "huggingface.co/datasets/AIGVDBench/AIGVDBench/" in url
    assert media.ARCHIVE_REVISION in url


def _payload(delta=0.0):
    clips = []
    for clip_index in range(4):
        rows = []
        for index, name in enumerate(RESIDUAL_NAMES):
            if index % 2:
                rows.append(
                    {
                        "name": name,
                        "normalized_value": None,
                        "availability": "blocked_by_input",
                        "valid_mask": False,
                        "confidence": 0.0,
                        "reason": "fixture_missing",
                    }
                )
            else:
                rows.append(
                    {
                        "name": name,
                        "normalized_value": 0.1 + index * 0.01 + delta,
                        "availability": "observed",
                        "valid_mask": True,
                        "confidence": 0.8,
                    }
                )
        clips.append(
            {
                "clip_id": f"clip_{clip_index}",
                "start_frame": clip_index * 8,
                "residuals": rows,
            }
        )
    return {
        "clips": clips,
        "metadata": {
            "clip_count": 4,
            "objects_total": 2,
            "object_tracks": 1,
            "point_tracks": 3,
            "branch_evidence_counts": {},
        },
    }


def test_frozen_12_channel_sequence_keeps_unavailable_missing_not_zero():
    sequence = residual._clip_sequence(_payload())
    assert sequence["channel_names"] == list(RESIDUAL_NAMES)
    assert sequence["channel_count"] == 12
    assert all(row[1] is None for row in sequence["values"])
    assert all(row[1] is False for row in sequence["availability"])
    assert all(row[1] == 0.0 for row in sequence["confidence"])
    assert sequence["authenticity_label_used"] is False


def test_batch_single_comparison_enforces_one_e_minus_six():
    assert residual._compare_payloads(_payload(), _payload())["ready"] is True
    with pytest.raises(ValueError, match="beyond"):
        residual._compare_payloads(_payload(), _payload(delta=2e-6))


def test_scale_prior_provenance_reaches_summary_and_final_manifest(tmp_path):
    payload = _payload()
    payload["metadata"].update(
        {
            "selected_frame_count": 32,
            "objects_total": 2,
            "object_tracks": 1,
            "point_tracks": 3,
            "overall_coverage": 0.5,
            "runtime_seconds": 1.0,
            "peak_gpu_memory_mb": 2.0,
            "scale_prior_schema_version": "paper_core_scale_priors_v1",
            "scale_prior_sha256": "prior-hash",
            "scale_prior_source_table_sha256": "source-hash",
            "scale_prior_entry_id": ["entry"],
            "scale_prior_confidence": {"entry": "high"},
        }
    )
    member = {
        "sample_id": "sample-real",
        "group_id": "group",
        "split": "train",
        "label": "0",
        "dataset_name": "fixture",
        "source_video_id": "video",
        "video_id": "video.mp4",
        "generator_name": "",
        "prompt_sha256": "prompt",
        "local_video_path": "/external/real.mp4",
        "license_status": "fixture",
    }
    summary = residual._summary_row(
        member,
        payload,
        source_sha256="video-hash",
        provenance_sha256="sampling-hash",
        result_sha256="result-hash",
        result_path=tmp_path / "result.json",
        source_commit="commit",
        config_sha256="config-hash",
    )
    assert summary["scale_prior_sha256"] == "prior-hash"
    assert summary["scale_prior_source_table_sha256"] == "source-hash"
    assert json.loads(summary["scale_prior_entry_id"]) == ["entry"]

    fake = {**member, "sample_id": "sample-fake", "label": "1"}
    fake_summary = {**summary, "sample_id": "sample-fake"}
    final_path = tmp_path / "final.csv"
    residual._quality_outputs(
        rows=[member, fake],
        summaries=[summary, fake_summary],
        parent=tmp_path,
        final_manifest_path=final_path,
    )
    with final_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {row["scale_prior_sha256"] for row in rows} == {"prior-hash"}
    assert {row["scale_prior_source_table_sha256"] for row in rows} == {
        "source-hash"
    }


class _Provider:
    def __init__(self):
        self.reset_count = 0

    def reset_video_state(self):
        self.reset_count += 1


class _Pose(_Provider):
    def __init__(self):
        super().__init__()
        self.pair_metadata = {(0, 1): {"video": "old"}}


def test_pipeline_resets_video_local_state_without_reinitializing_providers():
    object_provider = _Provider()
    depth_provider = _Provider()
    pose_provider = _Pose()
    track_provider = _Provider()
    pipeline = ForgeryAnalysisPipeline(
        config={},
        object_provider=object_provider,
        depth_provider=depth_provider,
        pose_provider=pose_provider,
        track_provider=track_provider,
    )
    pipeline.last_observations = ["old"]
    identities = tuple(
        id(provider)
        for provider in (object_provider, depth_provider, pose_provider, track_provider)
    )
    pipeline.reset_video_state()

    assert pipeline.last_observations == []
    assert pose_provider.pair_metadata == {}
    assert tuple(
        id(provider)
        for provider in (object_provider, depth_provider, pose_provider, track_provider)
    ) == identities
    assert [
        object_provider.reset_count,
        depth_provider.reset_count,
        pose_provider.reset_count,
        track_provider.reset_count,
    ] == [1, 1, 1, 1]
