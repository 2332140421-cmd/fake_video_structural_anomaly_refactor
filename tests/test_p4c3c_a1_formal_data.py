"""P4-C3C-A1 formal path, schema, and adapter contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from semantic3d.dataset_adapters import (
    GENVIDEO100K_SCHEMA_STATUS,
    GenVideo100KAdapter,
    GenVideo100KFieldMapping,
    UnresolvedDatasetSchemaError,
)
from semantic3d.dataset_builder.formal_schema import FORMAL_SPLITS
from semantic3d.dataset_builder.manifest import (
    VideoProbe,
    build_formal_manifest_from_directory,
    build_formal_video_sample,
    disambiguate_source_names,
    scan_video_files,
    video_manifest_row,
)
from semantic3d.dataset_builder.pipeline import StructuralEnhancementDatasetBuilder
from scripts.build_labels_manifest import build_label_rows

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _write_fixture(path: Path, payload: bytes = b"tiny-video-fixture") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _fixture_probe(_path: str | Path) -> VideoProbe:
    return VideoProbe(
        frame_count=50,
        fps=25.0,
        width=64,
        height=48,
        decode_status="ok",
    )


def test_external_data_root_recursive_duplicate_names_and_stable_ids(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "external-datasets"
    first = _write_fixture(data_root / "nested/a/same.mp4")
    second = _write_fixture(data_root / "nested/b/same.mp4")
    _write_fixture(data_root / "nested/b/ignored.txt")

    discovered = scan_video_files(data_root / "nested", recursive=True)
    assert discovered == [first.resolve(), second.resolve()]
    samples = build_formal_manifest_from_directory(
        data_root / "nested",
        data_root=data_root,
        source_dataset="fixture-dataset",
        split="train",
        probe=_fixture_probe,
    )
    repeated = build_formal_manifest_from_directory(
        data_root / "nested",
        data_root=data_root,
        source_dataset="fixture-dataset",
        split="train",
        probe=_fixture_probe,
    )

    assert len({sample.sample_id for sample in samples}) == 2
    assert [sample.sample_id for sample in samples] == [
        sample.sample_id for sample in repeated
    ]
    assert {sample.video_path for sample in samples} == {
        "nested/a/same.mp4",
        "nested/b/same.mp4",
    }
    assert all(sample.path_mode == "data_root_relative" for sample in samples)
    assert all(sample.source_path == str(path) for sample, path in zip(samples, discovered))
    assert all(sample.file_size == len(b"tiny-video-fixture") for sample in samples)
    assert all(
        sample.sha256 == hashlib.sha256(b"tiny-video-fixture").hexdigest()
        for sample in samples
    )
    legacy_rows = disambiguate_source_names(
        [
            {"source_name": "same", "video_id": samples[0].sample_id},
            {"source_name": "same", "video_id": samples[1].sample_id},
        ]
    )
    assert len({row["source_name"] for row in legacy_rows}) == 2


@pytest.mark.parametrize("split", sorted(FORMAL_SPLITS))
def test_canonical_split_and_explicit_missing_label(
    tmp_path: Path,
    split: str,
) -> None:
    data_root = tmp_path / "data"
    source = _write_fixture(data_root / f"{split}.mkv")
    lineage = {"original_source_id": "origin-1", "derivation": "fixture"}
    sample = build_formal_video_sample(
        source,
        data_root=data_root,
        source_dataset="fixture-dataset",
        split=split,
        source_lineage=lineage,
        path_mode="absolute",
        probe=_fixture_probe,
    )

    assert sample.split == split
    assert sample.label is None
    assert sample.is_real is None
    assert sample.metadata_status["label"] == "missing"
    assert sample.source_lineage == lineage
    assert sample.duration == 2.0
    assert Path(sample.video_path).is_absolute()


def test_genvideo_adapter_requires_verified_mapping_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    adapter = GenVideo100KAdapter()
    with pytest.raises(UnresolvedDatasetSchemaError, match="UNRESOLVED_SCHEMA"):
        adapter.read_samples(tmp_path / "missing.jsonl", data_root=tmp_path)
    assert GENVIDEO100K_SCHEMA_STATUS == "TODO/UNRESOLVED_SCHEMA"
    assert not adapter.official_schema_verified

    data_root = tmp_path / "genvideo"
    source = _write_fixture(data_root / "future-layout/clip.webm", b"genvideo-fixture")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = tmp_path / "fixture_metadata.jsonl"
    metadata.write_text(
        json.dumps(
            {
                "fixture_path": "future-layout/clip.webm",
                "fixture_label": "generated",
                "fixture_split": "official-val",
                "fixture_generator": "fixture-generator",
                "fixture_lineage": {"original_source_id": "fixture-origin"},
                "fixture_checksum": digest,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mapped = GenVideo100KAdapter(
        GenVideo100KFieldMapping(
            video_path="fixture_path",
            label="fixture_label",
            split="fixture_split",
            generator="fixture_generator",
            source_lineage="fixture_lineage",
            checksum="fixture_checksum",
            label_values={"generated": 1},
            split_values={"official-val": "validation"},
        )
    )
    samples = mapped.read_samples(metadata, data_root=data_root)

    assert len(samples) == 1
    sample = samples[0]
    assert sample.video_path == "future-layout/clip.webm"
    assert sample.label == 1
    assert sample.is_real is False
    assert sample.split == "validation"
    assert sample.official_split
    assert sample.generator == "fixture-generator"
    assert sample.source_lineage == {"original_source_id": "fixture-origin"}
    assert sample.sha256 == digest
    assert sample.metadata_status["spatial_annotation"] == "unresolved_schema"


def test_genvideo_adapter_rejects_unmapped_label_and_bad_checksum(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "genvideo"
    source = _write_fixture(data_root / "clip.mp4")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps([{"p": "clip.mp4", "label": "unknown", "sha": "0" * 64}]),
        encoding="utf-8",
    )
    adapter = GenVideo100KAdapter(
        GenVideo100KFieldMapping(
            video_path="p",
            label="label",
            checksum="sha",
            label_values={"real": 0},
        )
    )
    with pytest.raises(ValueError, match="Unmapped label"):
        adapter.read_samples(metadata, data_root=data_root)

    metadata.write_text(
        json.dumps([{"p": "clip.mp4", "label": "real", "sha": "0" * 64}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Checksum mismatch"):
        adapter.read_samples(metadata, data_root=data_root)
    assert source.exists()


def test_genvideo_registry_remains_an_unverified_skeleton() -> None:
    registry = yaml.safe_load(
        (
            PROJECT_ROOT
            / "configs/data_registry/genvideo_100k_adapter_skeleton_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    assert registry["adapter_status"] == "skeleton"
    assert registry["data_downloaded"] is False
    assert registry["official_schema_verified"] is False
    assert registry["schema_status"] == "TODO/UNRESOLVED_SCHEMA"


def test_label_rows_use_stable_ids_and_explicit_missing_values() -> None:
    videos = [
        {"video_id": "stable-a", "source_name": "same"},
        {"video_id": "stable-b", "source_name": "same"},
    ]
    rows = build_label_rows(
        [
            {
                "video_id": "stable-a",
                "label": "",
                "split": "test",
                "temporal_annotation": "",
                "spatial_annotation": "",
            }
        ],
        videos,
        source_manifest="/external/labels.csv",
    )
    assert rows[0]["video_id"] == "stable-a"
    assert rows[0]["label"] is None
    assert rows[0]["temporal_annotation"] is None
    assert json.loads(rows[0]["metadata_status"])["label"] == "missing"
    with pytest.raises(ValueError, match="Ambiguous source_name"):
        build_label_rows(
            [{"video_id": "same", "label": "1", "split": "train"}],
            videos,
            source_manifest="/external/labels.csv",
        )


def test_legacy_explicit_yaml_paths_and_external_absolute_manifest(
    tmp_path: Path,
) -> None:
    project = tmp_path / "legacy-project"
    legacy_source = _write_fixture(project / "data/source.avi")
    config = project / "configs/legacy.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "legacy", "output_root": "outputs/legacy"},
                "sources": {"videos": ["data/source.avi"]},
                "providers": {"weights": {}},
                "runtime": {"device": "cpu", "random_seed": 7},
            }
        ),
        encoding="utf-8",
    )
    builder = StructuralEnhancementDatasetBuilder(config)
    assert builder.source_paths == [legacy_source.resolve()]
    assert builder.source_root == project.resolve()

    outside = _write_fixture(tmp_path / "outside/source.avi")
    external_config = project / "configs/external.yaml"
    external_config.write_text(
        yaml.safe_dump(
            {
                "dataset": {"name": "external", "output_root": "outputs/external"},
                "sources": {
                    "data_root": str(outside.parent),
                    "videos": ["source.avi"],
                },
                "providers": {"weights": {}},
                "runtime": {"device": "cpu", "random_seed": 7},
            }
        ),
        encoding="utf-8",
    )
    external_builder = StructuralEnhancementDatasetBuilder(external_config)
    assert external_builder.source_root == outside.parent.resolve()
    assert external_builder.source_paths == [outside.resolve()]

    row = video_manifest_row(
        source_root=project,
        source_path=outside,
        dataset_id="fixture-dataset",
    )
    assert row["source_path_kind"] == "absolute"
    assert row["source_relative_path"] == outside.resolve().as_posix()
    assert row["source_path"] == outside.resolve().as_posix()
