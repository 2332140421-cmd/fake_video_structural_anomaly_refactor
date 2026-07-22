"""Deterministic server migration inventory with conservative exclusions."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json, sha256_file

from .runtime_config import RuntimeConfig

_INCLUDE_ROOTS = (
    "src",
    "scripts",
    "configs",
    "tests",
    "docs",
    "data/tests_videos",
    "data/manifests",
    "data/experiment_protocol_v1",
    "outputs/p4c1_experiment_manifest",
    "outputs/p4c2_formal_data_readiness",
)
_ROOT_FILES = (
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "uv.lock",
    "poetry.lock",
    ".gitignore",
    "README.md",
    "CHANGELOG.md",
)
_EXCLUDED = (
    (".venv", "server_environment_must_be_rebuilt"),
    (".git", "server_should_clone_or_initialize_git_separately"),
    ("outputs/structural_enhancement_dataset", "large_rebuildable_structural_dataset"),
    ("outputs/runtime_tmp", "runtime_temporary_files"),
    ("outputs/p4c3a_batch_state", "machine_local_batch_state"),
    ("__pycache__", "python_bytecode_cache"),
    (".pytest_cache", "pytest_cache"),
    (".mypy_cache", "mypy_cache"),
    (".ruff_cache", "ruff_cache"),
)
_SENSITIVE_NAMES = frozenset(
    {".env", "credentials", "credentials.json", "token", "token.json", "id_rsa", "id_ed25519"}
)


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.is_dir() else 0


def _csv(rows: Iterable[dict[str, Any]], columns: tuple[str, ...]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue().encode("utf-8")


def _git(project: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=project, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def build_migration_manifest(runtime: RuntimeConfig) -> dict[str, Any]:
    """Inventory required portable files and excluded machine-local content."""

    project = runtime.project_root
    files: dict[str, dict[str, Any]] = {}
    for relative_root in _INCLUDE_ROOTS:
        root = project / relative_root
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(project).as_posix()
            if any(part == "__pycache__" for part in path.parts) or path.suffix in {".pyc", ".pyo"}:
                continue
            if path.name.lower() in _SENSITIVE_NAMES or path.suffix.lower() in {".key", ".pem"}:
                continue
            files[relative] = {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": relative_root,
                "migration_action": "copy_and_verify",
            }
    for name in _ROOT_FILES:
        path = project / name
        if path.is_file():
            files[name] = {
                "path": name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": "project_environment_definition",
                "migration_action": "copy_and_verify",
            }
    # Small local weights are portable smoke prerequisites; caches remain excluded.
    checkpoint_root = project / "checkpoints"
    if checkpoint_root.is_dir():
        for path in sorted(checkpoint_root.glob("*.pt")):
            relative = path.relative_to(project).as_posix()
            files[relative] = {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "category": "small_smoke_model_weight",
                "migration_action": "copy_and_verify_or_restore_by_matching_sha256",
            }
    included = [files[key] for key in sorted(files)]
    excluded = [
        {"path": relative, "size_bytes": _size(project / relative), "reason": reason}
        for relative, reason in _EXCLUDED
        if (project / relative).exists()
    ]
    payload = json.dumps(included, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "_project_root": str(project),
        "schema_version": "p4c3a_server_migration_manifest_v1",
        "runtime_profile": runtime.runtime_profile,
        "project_logical_root": "${PROJECT_ROOT}",
        "git_commit": _git(project, "rev-parse", "HEAD"),
        "git_tags_at_commit": sorted(_git(project, "tag", "--points-at", "HEAD").splitlines()),
        "git_worktree_clean": not bool(_git(project, "status", "--porcelain=v1")),
        "file_count": len(included),
        "total_bytes": sum(row["size_bytes"] for row in included),
        "migration_files_sha256": hashlib.sha256(payload).hexdigest(),
        "files": included,
        "excluded": excluded,
        "server_environment_rebuild_required": True,
        "copy_local_venv": False,
        "formal_data_download_performed": False,
        "formal_batch_started": False,
    }


def write_migration_artifacts(output_root: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Write migration inventory, exclusions, bootstrap steps, and validation."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_json = {
        key: value
        for key, value in manifest.items()
        if key not in {"files", "excluded"} and not key.startswith("_")
    }
    manifest_json["included_file_count"] = len(manifest["files"])
    manifest_json["excluded_entry_count"] = len(manifest["excluded"])
    atomic_write_json(root / "migration_manifest.json", manifest_json)
    atomic_write_bytes(
        root / "migration_files.csv",
        _csv(
            manifest["files"],
            ("path", "size_bytes", "sha256", "category", "migration_action"),
        ),
    )
    atomic_write_bytes(
        root / "excluded_local_files.csv",
        _csv(manifest["excluded"], ("path", "size_bytes", "reason")),
    )
    rebuild = """# Server Environment Rebuild Plan

1. Clone the recorded Git commit or copy and SHA-256 verify every migration file.
2. Do not copy `.venv`; create a fresh server-side virtual environment.
3. Install dependencies from the migrated lock/project files.
4. Restore required model weights only when their SHA-256 matches the migration manifest.
5. Configure `configs/runtime/server_template.yaml` through environment variables.
6. Run P4-C3A preflight and the complete unit-test suite before any formal registration.
7. Keep downloads, caches, outputs, and batch state on configured server storage roots.
"""
    bootstrap = """# P4 Server Bootstrap Steps

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/preflight_p4_server_environment.py \\
  --runtime-config configs/runtime/server_template.yaml
.venv/bin/python -m pytest -q -rs
.venv/bin/python scripts/plan_p4c_experiment_protocol.py
.venv/bin/python scripts/build_p4c1_experiment_manifest.py
.venv/bin/python scripts/build_p4c2_formal_data_readiness.py
```

Do not register or build formal data until all frozen hashes match and server smoke passes.
"""
    atomic_write_bytes(root / "environment_rebuild_plan.md", rebuild.encode("utf-8"))
    atomic_write_bytes(root / "server_bootstrap_steps.md", bootstrap.encode("utf-8"))
    project = Path(str(manifest["_project_root"]))
    files_valid = all(
        (project / row["path"]).is_file()
        and sha256_file(project / row["path"]) == row["sha256"]
        for row in manifest["files"]
    )
    forbidden = (".venv/", "__pycache__/", ".pytest_cache/", ".env", "id_rsa")
    no_forbidden = all(not any(token in row["path"] for token in forbidden) for row in manifest["files"])
    validation = {
        "valid": bool(manifest["files"]) and no_forbidden and files_valid,
        "included_file_count": len(manifest["files"]),
        "included_total_bytes": manifest["total_bytes"],
        "migration_files_sha256": manifest["migration_files_sha256"],
        "all_included_file_hashes_match": files_valid,
        "no_local_venv_included": all(not row["path"].startswith(".venv/") for row in manifest["files"]),
        "no_sensitive_files_included": no_forbidden,
        "server_environment_rebuild_required": True,
        "formal_batch_started": False,
    }
    atomic_write_json(root / "migration_validation_report.json", validation)
    return validation
