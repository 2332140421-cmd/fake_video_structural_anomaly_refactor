"""Compose and persist the P4-C3A server preflight report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json, sha256_file

from .cuda_probe import probe_cuda
from .dependency_probe import probe_dependencies
from .environment_probe import probe_environment
from .runtime_config import RuntimeConfig
from .storage_probe import probe_storage


def build_server_preflight(
    runtime: RuntimeConfig,
    smoke_protocol: Mapping[str, Any],
    p4c2_validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Run non-destructive environment checks and compute readiness booleans."""

    dependencies = probe_dependencies(
        runtime.required_python_dependencies, runtime.required_executables
    )
    cuda = probe_cuda()
    storage = probe_storage(runtime)
    environment = probe_environment(runtime, smoke_protocol["required_hashes"])
    errors = []
    warnings = []
    if not dependencies["all_python_modules_importable"]:
        errors.append("required_python_dependency_missing")
    if not dependencies["all_executables_available"]:
        errors.append("required_executable_missing")
    if not storage["all_roots_accessible_or_creatable"] or not storage["all_roots_writable"]:
        errors.append("runtime_root_not_accessible")
    if not storage["atomic_write_and_rename"]["passed"]:
        errors.append("atomic_write_or_rename_failed")
    if not storage["atomic_write_and_rename"]["sha256_passed"]:
        errors.append("sha256_probe_failed")
    if not environment["all_frozen_hashes_match"]:
        errors.append("frozen_hash_mismatch")
    if not environment["git_available"]:
        errors.append("git_metadata_unavailable")
    if not environment["git_worktree_clean"]:
        warnings.append("git_worktree_not_clean")
    if not cuda["ready_for_cuda_batch"]:
        warnings.append("cuda_batch_probe_failed")
    if not storage["batch_plus_safety_available"]:
        warnings.append("batch_storage_limit_plus_safety_not_available")
    if runtime.runtime_profile == "local_wsl":
        warnings.append("local_wsl_profile_is_not_server_validation")
    if not p4c2_validation.get("ready_for_formal_batch_build", False):
        warnings.append("p4c2_formal_data_readiness_blocked")

    cuda_required_for_smoke = runtime.device.startswith("cuda")
    smoke_errors = list(errors)
    if cuda_required_for_smoke and not cuda["ready_for_cuda_batch"]:
        smoke_errors.append("configured_cuda_device_unavailable")
    ready_for_server_smoke = not smoke_errors
    ready_for_server_migration = bool(
        environment["all_frozen_hashes_match"]
        and dependencies["all_python_modules_importable"]
        and runtime.project_root.is_dir()
    )
    ready_for_formal_registration = bool(
        ready_for_server_smoke
        and runtime.runtime_profile != "local_wsl"
        and environment["all_frozen_hashes_match"]
    )
    ready_for_formal_batch = bool(
        ready_for_server_smoke
        and p4c2_validation.get("ready_for_formal_batch_build", False)
        and storage["batch_plus_safety_available"]
        and (not runtime.require_cuda_for_formal_batch or cuda["ready_for_cuda_batch"])
    )
    formal_blockers = sorted(
        set(
            smoke_errors
            + list(p4c2_validation.get("blockers", ()))
            + ([] if storage["batch_plus_safety_available"] else ["insufficient_batch_storage"])
            + (
                []
                if not runtime.require_cuda_for_formal_batch or cuda["ready_for_cuda_batch"]
                else ["cuda_required_for_formal_batch_unavailable"]
            )
        )
    )
    return {
        "runtime_profile": runtime.runtime_profile,
        "runtime_location_metadata": runtime.location_metadata(),
        "environment": environment,
        "storage": storage,
        "cuda": cuda,
        "dependencies": dependencies,
        "ready_for_server_migration": ready_for_server_migration,
        "ready_for_server_smoke": ready_for_server_smoke,
        "ready_for_formal_dataset_registration": ready_for_formal_registration,
        "ready_for_formal_batch_execution": ready_for_formal_batch,
        "blocking_errors": sorted(set(errors)),
        "formal_batch_blocking_errors": formal_blockers,
        "warnings": sorted(set(warnings)),
        "formal_download_performed": False,
        "formal_model_inference_performed": False,
        "formal_batch_started": False,
        "model_training_performed": False,
        "test_performance_read": False,
    }


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# P4-C3A Server Environment Preflight",
        "",
        f"- runtime profile: `{report['runtime_profile']}`",
        f"- ready for server migration: `{str(report['ready_for_server_migration']).lower()}`",
        f"- ready for server smoke: `{str(report['ready_for_server_smoke']).lower()}`",
        f"- ready for formal dataset registration: `{str(report['ready_for_formal_dataset_registration']).lower()}`",
        f"- ready for formal batch execution: `{str(report['ready_for_formal_batch_execution']).lower()}`",
        "",
        "## CUDA",
        "",
        f"- torch: `{report['cuda']['torch_version']}`",
        f"- compiled CUDA: `{report['cuda']['torch_compiled_cuda_version']}`",
        f"- CUDA available: `{str(report['cuda']['cuda_available']).lower()}`",
        f"- device count: {report['cuda']['device_count']}",
        f"- minimal CUDA operation passed: `{str(report['cuda']['minimal_matrix_operation_passed']).lower()}`",
        "",
        "## Blocking Errors",
        "",
    ]
    lines.extend(f"- `{value}`" for value in report["blocking_errors"] or ["none"])
    lines.extend(["", "## Formal Batch Blockers", ""])
    lines.extend(f"- `{value}`" for value in report["formal_batch_blocking_errors"] or ["none"])
    lines.extend(["", "## Warnings", ""])
    lines.extend(f"- `{value}`" for value in report["warnings"] or ["none"])
    lines.extend(
        [
            "",
            "CPU protocol tests do not prove CUDA batch readiness. Provider failure and missing GPU evidence are quality-control states, not authenticity evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def write_preflight_artifacts(output_root: str | Path, report: Mapping[str, Any]) -> dict[str, Any]:
    """Write separated machine reports and a deterministic validation summary."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "environment_report.json", report["environment"])
    atomic_write_json(root / "storage_report.json", report["storage"])
    atomic_write_json(root / "cuda_report.json", report["cuda"])
    atomic_write_json(root / "dependency_report.json", report["dependencies"])
    atomic_write_bytes(root / "preflight_report.md", _markdown(report).encode("utf-8"))
    primary = (
        "environment_report.json",
        "storage_report.json",
        "cuda_report.json",
        "dependency_report.json",
        "preflight_report.md",
    )
    hashes = {name: sha256_file(root / name) for name in primary}
    validation = {
        "valid": not report["blocking_errors"],
        "ready_for_server_migration": report["ready_for_server_migration"],
        "ready_for_server_smoke": report["ready_for_server_smoke"],
        "ready_for_formal_dataset_registration": report[
            "ready_for_formal_dataset_registration"
        ],
        "ready_for_formal_batch_execution": report["ready_for_formal_batch_execution"],
        "blocking_errors": report["blocking_errors"],
        "formal_batch_blocking_errors": report["formal_batch_blocking_errors"],
        "warnings": report["warnings"],
        "artifact_sha256": hashes,
        "formal_batch_started": False,
        "model_training_performed": False,
        "test_performance_read": False,
    }
    atomic_write_json(root / "validation_report.json", validation)
    return validation

