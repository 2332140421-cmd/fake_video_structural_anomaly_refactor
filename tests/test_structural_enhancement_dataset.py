"""P4-B schema, ownership, label isolation, and Parquet tests."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import numpy as np
import pytest
import cv2
import yaml

from semantic3d.dataset_builder.clip_alignment import ClipAlignmentObservation
from semantic3d.dataset_builder.ids import StableIdFactory, stable_id
from semantic3d.dataset_builder.manifest import build_clip_manifests, split_scene_segments
from semantic3d.dataset_builder.pipeline import StructuralEnhancementDatasetBuilder
from semantic3d.dataset_builder.schema import Applicability, EvidenceRecord, SCHEMA_VERSION
from semantic3d.dataset_builder.writer import DatasetWriter, write_parquet
from semantic3d.dataset_builder.reader import DatasetReader


def test_stable_ids_are_process_order_independent() -> None:
    first = stable_id("frame", "video-a", 17)
    second = stable_id("frame", "video-a", 17)
    assert first == second
    assert first != stable_id("frame", "video-a", 18)


def test_scene_cut_and_overlap_owner_assignment() -> None:
    scenes = split_scene_segments([0.1, 0.1, 0.9, 0.9], cut_threshold=0.5)
    assert scenes == [(0, 1), (2, 3)]
    factory = StableIdFactory("dataset-test")
    clips, frames = build_clip_manifests(
        video_id="video-test",
        frame_count=20,
        scene_segments=[(0, 19)],
        id_factory=factory,
        window_size=8,
        stride=4,
        left_context=2,
        right_context=2,
        minimum_clip_length=2,
    )
    assert len(clips) > 1
    owner_rows = [row for row in frames if row["is_owned_frame"]]
    assert len(owner_rows) == 20
    assert len({row["frame_id"] for row in owner_rows}) == 20
    assert all(row["owner_clip_id"] == row["clip_id"] for row in owner_rows)
    assert all(len({row["scene_id"] for row in frames if row["clip_id"] == clip["clip_id"]}) == 1 for clip in clips)


def test_clip_coordinates_are_isolated_until_alignment_is_valid() -> None:
    alignment = ClipAlignmentObservation(
        source_clip_id="clip-a",
        target_clip_id="clip-b",
        transform=None,
        scale=math.nan,
        shift=None,
        overlap_frame_count=0,
        background_support_count=0,
        alignment_error=math.nan,
        holdout_error=math.nan,
        valid=False,
        missing_reason="insufficient_overlap",
    )
    with pytest.raises(ValueError, match="unaligned"):
        alignment.transform_points(np.ones((2, 3)))

    valid = ClipAlignmentObservation(
        source_clip_id="clip-a",
        target_clip_id="clip-b",
        transform=np.eye(4),
        scale=2.0,
        shift=np.asarray([1.0, 0.0, 0.0]),
        overlap_frame_count=4,
        background_support_count=100,
        alignment_error=0.01,
        holdout_error=0.02,
        valid=True,
    )
    transformed = valid.transform_points(np.asarray([[1.0, 2.0, 3.0]]))
    assert np.allclose(transformed, [[3.0, 4.0, 6.0]])


def test_evidence_applicability_and_nan_contract() -> None:
    missing = EvidenceRecord(
        evidence_id="e-missing",
        branch_name="occlusion_depth_order",
        evidence_level="clip",
        video_id="v",
        clip_id="c",
        raw_value=math.nan,
        valid=False,
        applicability=Applicability.NOT_APPLICABLE,
        missing_reason="no_observed_event_in_clip",
    )
    assert math.isnan(missing.raw_value)
    assert missing.applicability == Applicability.NOT_APPLICABLE

    normal = EvidenceRecord(
        evidence_id="e-valid",
        branch_name="direction_consistency",
        evidence_level="point",
        video_id="v",
        clip_id="c",
        raw_value=0.0,
        intrinsic_normalized_value=0.0,
        valid=True,
        quality=0.8,
        applicability=Applicability.APPLICABLE_VALID,
        missing_reason="",
    )
    assert normal.raw_value == 0.0
    with pytest.raises(ValueError, match="raw_value=NaN"):
        EvidenceRecord(
            evidence_id="bad",
            branch_name="x",
            evidence_level="clip",
            video_id="v",
            clip_id="c",
            raw_value=0.0,
            valid=False,
            applicability=Applicability.OBSERVATION_MISSING,
            missing_reason="missing",
        )


def test_real_parquet_roundtrip_and_array_reference(tmp_path: Path) -> None:
    writer = DatasetWriter(tmp_path)
    table = writer.parquet("manifests/videos.parquet", [{"video_id": "v1", "frame_count": 2}], ("video_id", "frame_count"))
    assert table.read_bytes()[:4] == b"PAR1"
    rows = DatasetReader(tmp_path).rows("manifests/videos.parquet")
    assert rows == [{"video_id": "v1", "frame_count": 2}]
    reference = writer.array("arrays/example.npz", points=np.eye(3, dtype=np.float32))
    assert Path(reference["path"]).exists()
    assert reference["shape"]["points"] == [3, 3]


def test_builder_api_has_no_label_parameter() -> None:
    init_parameters = inspect.signature(StructuralEnhancementDatasetBuilder.__init__).parameters
    run_parameters = inspect.signature(StructuralEnhancementDatasetBuilder.run).parameters
    assert not any("label" in name for name in (*init_parameters, *run_parameters))
    source = inspect.getsource(StructuralEnhancementDatasetBuilder._stage_11_dynamic)
    assert "label_manifest" not in source
    assert "forgery" not in source


def test_dataset_reader_refuses_implicit_label_join(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="isolated"):
        DatasetReader(tmp_path).labels()


def test_schema_version_is_explicit() -> None:
    assert SCHEMA_VERSION == "semantic3d_structural_enhancement_v1"


def _tiny_video(path: Path, frame_count: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (32, 24)
    )
    assert writer.isOpened()
    try:
        for index in range(frame_count):
            writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
    finally:
        writer.release()


def _tiny_config(root: Path, *, aggregation_token: str) -> Path:
    video = root / "data/source.avi"
    _tiny_video(video)
    config = {
        "dataset": {"name": "tiny", "output_root": "outputs/tiny"},
        "sources": {"videos": ["data/source.avi"]},
        "clip_split": {
            "window_size": 4,
            "left_context": 1,
            "right_context": 1,
            "stride": 2,
            "minimum_clip_length": 2,
            "scene_cut_threshold": 0.5,
        },
        "providers": {"weights": {}},
        "stages": {"13_multilevel_aggregation": {"token": aggregation_token}},
        "runtime": {"device": "cpu", "random_seed": 7},
    }
    path = root / "configs/tiny.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


def test_stage01_resume_is_deterministic_and_cached(tmp_path: Path) -> None:
    config = _tiny_config(tmp_path, aggregation_token="a")
    builder = StructuralEnhancementDatasetBuilder(config)
    first = builder.run(target_stage="01_video_index")
    rows_first = DatasetReader(first["output_root"]).rows("manifests/frames.parquet")
    second_builder = StructuralEnhancementDatasetBuilder(config)
    second = second_builder.run(target_stage="01_video_index", resume=True)
    rows_second = DatasetReader(second["output_root"]).rows("manifests/frames.parquet")
    assert first["dataset_id"] == second["dataset_id"]
    assert [row["frame_record_id"] for row in rows_first] == [
        row["frame_record_id"] for row in rows_second
    ]
    assert second["stages"][0]["cache_hit"] is True


def test_aggregation_config_does_not_change_dataset_identity(tmp_path: Path) -> None:
    first_config = _tiny_config(tmp_path, aggregation_token="a")
    first = StructuralEnhancementDatasetBuilder(first_config)
    payload = yaml.safe_load(first_config.read_text(encoding="utf-8"))
    payload["stages"]["13_multilevel_aggregation"]["token"] = "b"
    first_config.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    second = StructuralEnhancementDatasetBuilder(first_config)
    assert first.dataset_id == second.dataset_id
    assert first._stage_config("03_object_detection") == second._stage_config(
        "03_object_detection"
    )
    assert first._stage_config("13_multilevel_aggregation") != second._stage_config(
        "13_multilevel_aggregation"
    )
