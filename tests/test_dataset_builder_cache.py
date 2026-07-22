"""P4-B cache invalidation, atomic write, corruption, and resume tests."""

from __future__ import annotations

from pathlib import Path

from semantic3d.dataset_builder.cache import StageCache, build_cache_key
from semantic3d.dataset_builder.stages import descendants
from semantic3d.dataset_builder.writer import atomic_write_bytes


def _key(*, stage: str, upstream: tuple[str, ...] = (), weights: tuple[str, ...] = (), config: str = "cfg") -> str:
    return build_cache_key(
        source_video_sha256="video",
        stage_name=stage,
        stage_config_sha256=config,
        upstream_artifact_sha256=upstream,
        provider_weight_sha256=weights,
        schema_version="v1",
    )


def test_cache_hit_and_resume_contract(tmp_path: Path) -> None:
    artifact = atomic_write_bytes(tmp_path / "artifact.bin", b"stable")
    cache = StageCache(tmp_path / "cache")
    key = _key(stage="04_instance_segmentation", weights=("mask-v1",))
    assert not cache.lookup("04_instance_segmentation", key).hit
    cache.commit("04_instance_segmentation", key, [artifact])
    assert cache.lookup("04_instance_segmentation", key).hit


def test_weight_and_upstream_changes_invalidate_only_relevant_keys() -> None:
    mask_v1 = _key(stage="04_instance_segmentation", weights=("mask-v1",))
    mask_v2 = _key(stage="04_instance_segmentation", weights=("mask-v2",))
    assert mask_v1 != mask_v2
    depth_v1 = _key(stage="06_depth", config="depth-v1")
    depth_v2 = _key(stage="06_depth", config="depth-v2")
    assert depth_v1 != depth_v2
    aggregation_v1 = _key(stage="13_multilevel_aggregation", upstream=("evidence-v1",), config="agg-v1")
    aggregation_v2 = _key(stage="13_multilevel_aggregation", upstream=("evidence-v1",), config="agg-v2")
    assert aggregation_v1 != aggregation_v2
    assert "03_object_detection" not in descendants("13_multilevel_aggregation")
    assert "06_depth" not in descendants("13_multilevel_aggregation")
    assert "04_instance_segmentation" in descendants("04_instance_segmentation")
    assert "12_occlusion_evidence" in descendants("04_instance_segmentation")


def test_corrupt_cache_and_temporary_files_are_not_hits(tmp_path: Path) -> None:
    artifact = atomic_write_bytes(tmp_path / "artifact.bin", b"first")
    cache = StageCache(tmp_path / "cache")
    key = _key(stage="06_depth")
    cache.commit("06_depth", key, [artifact])
    artifact.write_bytes(b"corrupted")
    lookup = cache.lookup("06_depth", key)
    assert not lookup.hit
    assert lookup.reason == "cached_artifact_hash_mismatch"
    temporary = cache.record_path("06_depth", "temp").with_suffix(".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text("{}", encoding="utf-8")
    assert not cache.lookup("06_depth", "temp").hit


def test_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    target = atomic_write_bytes(tmp_path / "nested/result.bin", b"complete")
    assert target.read_bytes() == b"complete"
    assert not list(target.parent.glob("*.tmp"))
