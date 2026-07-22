"""Tests for truthful P4-C3A server preflight readiness."""

from __future__ import annotations

from pathlib import Path

from semantic3d.runtime import preflight_report
from semantic3d.runtime.runtime_config import load_runtime_config
from semantic3d.runtime.storage_probe import probe_storage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_insufficient_disk_blocks_batch_plus_safety(tmp_path: Path) -> None:
    config = tmp_path / "runtime.yaml"
    config.write_text(
        f"""runtime_profile: unit
project_root: {tmp_path}
data_root: .
download_root: .
temporary_root: tmp
cache_root: cache
model_root: models
output_root: outputs
log_root: logs
device: cpu
num_workers: 1
storage_safety_margin: 999999999999999999
batch_storage_limit: 999999999999999999
required_python_dependencies: []
required_executables: []
require_cuda_for_formal_batch: true
""",
        encoding="utf-8",
    )
    report = probe_storage(load_runtime_config(config))
    assert not report["batch_plus_safety_available"]


def test_cuda_unavailable_blocks_formal_gpu_batch(monkeypatch, tmp_path: Path) -> None:
    runtime = load_runtime_config(PROJECT_ROOT / "configs/runtime/local_wsl.yaml")
    monkeypatch.setattr(
        preflight_report,
        "probe_dependencies",
        lambda *_: {
            "all_python_modules_importable": True,
            "all_executables_available": True,
        },
    )
    monkeypatch.setattr(
        preflight_report,
        "probe_cuda",
        lambda: {"ready_for_cuda_batch": False, "torch_version": "x", "torch_compiled_cuda_version": "x", "cuda_available": False, "device_count": 0, "minimal_matrix_operation_passed": False},
    )
    monkeypatch.setattr(
        preflight_report,
        "probe_storage",
        lambda *_: {
            "all_roots_accessible_or_creatable": True,
            "all_roots_writable": True,
            "atomic_write_and_rename": {"passed": True, "sha256_passed": True},
            "batch_plus_safety_available": True,
        },
    )
    monkeypatch.setattr(
        preflight_report,
        "probe_environment",
        lambda *_: {"all_frozen_hashes_match": True, "git_available": True, "git_worktree_clean": True},
    )
    report = preflight_report.build_server_preflight(
        runtime,
        {"required_hashes": {}},
        {"ready_for_formal_batch_build": True, "blockers": []},
    )
    assert report["ready_for_server_smoke"]
    assert not report["ready_for_formal_batch_execution"]
    assert "cuda_required_for_formal_batch_unavailable" in report["formal_batch_blocking_errors"]

