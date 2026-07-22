"""Tests for P4-C3A-G Git release and server checkout preparation."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from semantic3d.git_release.audit import build_git_release_audit, scan_absolute_paths
from semantic3d.git_release.model_registry import (
    load_model_registry,
    validate_model_registry,
)
from semantic3d.git_release.validation import (
    FROZEN_HASHES,
    P4C0_SEMANTIC_HASH,
    P4C2_SEMANTIC_HASH,
)
from semantic3d.runtime.runtime_config import canonical_logical_sha256, load_runtime_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _git(project: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)


def _small_repository(tmp_path: Path) -> Path:
    project = tmp_path / "repository"
    project.mkdir()
    _git(project, "init", "-b", "main")
    (project / ".gitignore").write_text(".env\noutputs/**\n", encoding="utf-8")
    (project / "README.md").write_text("unit\n", encoding="utf-8")
    (project / "credentials.txt").write_text("placeholder only\n", encoding="utf-8")
    _git(project, "add", ".gitignore", "README.md")
    _git(project, "add", "-f", "credentials.txt")
    subprocess.run(
        [
            "git", "-c", "user.name=Unit Test", "-c", "user.email=unit@example.invalid",
            "commit", "-m", "fixture",
        ],
        cwd=project,
        check=True,
        capture_output=True,
    )
    return project


def test_gitignore_excludes_local_assets_but_keeps_small_protocol_artifacts() -> None:
    ignored = (
        ".venv/bin/python",
        ".pytest_cache/state",
        ".env",
        "checkpoints/yolov8n.pt",
        "data/tests_videos/tests_real_videos/real_1.mp4",
        "outputs/large/run.bin",
    )
    for relative in ignored:
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=PROJECT_ROOT, check=False
        )
        assert result.returncode == 0, relative
    for relative in (
        "outputs/p4c1_experiment_manifest/experiment_manifest.jsonl",
        "outputs/p4c2_formal_data_readiness/build_metadata.json",
        "outputs/p4c3a_batch_plan/batch_plan_metadata.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=PROJECT_ROOT, check=False
        )
        assert result.returncode == 1, relative


def test_absolute_path_scanner_distinguishes_core_and_documentation(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src/module.py").write_text('ROOT = "/home/chenyh/project"\n', encoding="utf-8")
    (tmp_path / "docs/history.md").write_text("Old path: /mnt/e/archive\n", encoding="utf-8")
    rows = scan_absolute_paths(tmp_path, ("src/module.py", "docs/history.md"))
    by_path = {row["path"]: row for row in rows}
    assert by_path["src/module.py"]["blocking"]
    assert not by_path["docs/history.md"]["blocking"]


def test_sensitive_filename_is_reported(tmp_path: Path) -> None:
    project = _small_repository(tmp_path)
    audit = build_git_release_audit(project)
    assert any(row["path"] == "credentials.txt" for row in audit["sensitive"])


def test_git_release_manifest_is_deterministic(tmp_path: Path) -> None:
    project = _small_repository(tmp_path)
    first = build_git_release_audit(project)
    second = build_git_release_audit(project)
    assert first["release_content_sha256"] == second["release_content_sha256"]


def test_canonical_hash_is_checkout_path_independent(tmp_path: Path) -> None:
    configs = []
    for name in ("checkout_a", "checkout_b"):
        root = tmp_path / name
        config = tmp_path / f"{name}.yaml"
        config.write_text(
            "\n".join(
                [
                    "runtime_profile: server",
                    f"project_root: {root}",
                    "data_root: data",
                    "download_root: downloads",
                    "temporary_root: cache/tmp",
                    "cache_root: cache",
                    "model_root: models",
                    "output_root: outputs",
                    "log_root: logs",
                    "device: cpu",
                    "num_workers: 1",
                    "storage_safety_margin: 1",
                    "batch_storage_limit: 1",
                    "required_python_dependencies: []",
                    "required_executables: []",
                    "require_cuda_for_formal_batch: false",
                ]
            ),
            encoding="utf-8",
        )
        configs.append(load_runtime_config(config))
    first, second = configs
    assert canonical_logical_sha256({"path": str(first.data_root / "x")}, first) == canonical_logical_sha256(
        {"path": str(second.data_root / "x")}, second
    )


def test_model_registry_hashes_and_redistribution_gate() -> None:
    registry = load_model_registry(PROJECT_ROOT / "configs/model_registry/yolo_weights_v1.yaml")
    result = validate_model_registry(registry, local_model_root=PROJECT_ROOT / "checkpoints")
    assert result["valid"]
    assert result["server_smoke_hashes_complete"]
    assert result["all_local_files_verified"]
    assert all(not row["git_distribution_allowed"] for row in result["models"])


def test_unconfirmed_license_cannot_allow_redistribution() -> None:
    registry = load_model_registry(PROJECT_ROOT / "configs/model_registry/yolo_weights_v1.yaml")
    registry["models"][0]["redistribution_allowed"] = True
    result = validate_model_registry(registry)
    assert not result["valid"]
    assert any("unreviewed_license" in error for error in result["errors"])


def test_model_fetcher_defaults_to_dry_run(tmp_path: Path) -> None:
    import importlib.util

    script = PROJECT_ROOT / "scripts/fetch_registered_models.py"
    spec = importlib.util.spec_from_file_location("fetch_registered_models", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.fetch_registered_models(
        PROJECT_ROOT / "configs/model_registry/yolo_weights_v1.yaml", tmp_path
    )
    assert not result["execute"]
    assert not result["downloads_performed"]
    assert not list(tmp_path.iterdir())


def test_server_runtime_resolves_required_environment_variables(tmp_path: Path) -> None:
    environment = {
        "PROJECT_ROOT": str(PROJECT_ROOT),
        "DATA_ROOT": str(tmp_path / "data"),
        "DOWNLOAD_ROOT": str(tmp_path / "downloads"),
        "CACHE_ROOT": str(tmp_path / "cache"),
        "MODEL_ROOT": str(tmp_path / "models"),
        "OUTPUT_ROOT": str(tmp_path / "outputs"),
        "LOG_ROOT": str(tmp_path / "logs"),
    }
    runtime = load_runtime_config(
        PROJECT_ROOT / "configs/runtime/server_template.yaml", environment=environment
    )
    assert runtime.temporary_root == tmp_path / "cache/tmp"
    assert runtime.model_root == tmp_path / "models"


def test_bootstrap_shell_syntax() -> None:
    result = subprocess.run(
        ["bash", "-n", "scripts/bootstrap_p4_server.sh"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_frozen_hashes_remain_unchanged() -> None:
    for _, (relative, expected) in FROZEN_HASHES.items():
        assert hashlib.sha256((PROJECT_ROOT / relative).read_bytes()).hexdigest() == expected
    metadata = (PROJECT_ROOT / "outputs/p4c2_formal_data_readiness/build_metadata.json").read_text(encoding="utf-8")
    assert P4C0_SEMANTIC_HASH in metadata
    assert P4C2_SEMANTIC_HASH in metadata
