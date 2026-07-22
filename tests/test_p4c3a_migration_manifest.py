"""Tests for portable migration inclusion, exclusion, and frozen hashes."""

from __future__ import annotations

import hashlib
from pathlib import Path

from semantic3d.runtime.migration_manifest import build_migration_manifest
from semantic3d.runtime.runtime_config import load_runtime_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_migration_manifest_excludes_venv_cache_and_secrets(tmp_path: Path) -> None:
    for relative in ("src", "scripts", "configs", "tests", "docs"):
        path = tmp_path / relative
        path.mkdir(parents=True)
        (path / "keep.txt").write_text(relative, encoding="utf-8")
    (tmp_path / "configs/.env").write_text("SECRET=x", encoding="utf-8")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv/package.bin").write_bytes(b"x")
    runtime_yaml = tmp_path / "runtime.yaml"
    runtime_yaml.write_text(
        f"""runtime_profile: unit
project_root: {tmp_path}
data_root: data
download_root: downloads
temporary_root: tmp
cache_root: cache
model_root: checkpoints
output_root: outputs
log_root: logs
device: cpu
num_workers: 1
storage_safety_margin: 1
batch_storage_limit: 1
required_python_dependencies: []
required_executables: []
require_cuda_for_formal_batch: true
""",
        encoding="utf-8",
    )
    manifest = build_migration_manifest(load_runtime_config(runtime_yaml))
    paths = {row["path"] for row in manifest["files"]}
    assert not any(path.startswith(".venv/") for path in paths)
    assert "configs/.env" not in paths
    assert any(row["path"] == ".venv" for row in manifest["excluded"])
    assert manifest["server_environment_rebuild_required"]


def test_migration_manifest_hash_is_repeatable() -> None:
    runtime = load_runtime_config(PROJECT_ROOT / "configs/runtime/local_wsl.yaml")
    first = build_migration_manifest(runtime)
    second = build_migration_manifest(runtime)
    assert first["migration_files_sha256"] == second["migration_files_sha256"]


def test_frozen_protocol_and_prior_hashes_remain_unchanged() -> None:
    expected = {
        "configs/scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "configs/scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
        "outputs/p4c1_experiment_manifest/experiment_manifest.jsonl": "3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == digest
    metadata = (PROJECT_ROOT / "outputs/p4c2_formal_data_readiness/build_metadata.json").read_text()
    assert "2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981" in metadata

