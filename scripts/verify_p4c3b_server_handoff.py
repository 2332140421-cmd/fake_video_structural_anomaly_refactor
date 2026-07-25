#!/usr/bin/env python3
"""Verify P4-C3B-M2 source, frozen inputs, and optional local-only dependencies."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from semantic3d.runtime.external_sources import (
    ExternalSourceError,
    activate_unidepth_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "configs/handoff/p4c3b_m2_server_handoff_v1.yaml"


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _resolve(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _check_files(
    project_root: Path, expected: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    checks: dict[str, dict[str, Any]] = {}
    for relative, digest in sorted(expected.items()):
        path = project_root / relative
        actual = sha256_file(path) if path.is_file() else ""
        checks[relative] = {
            "exists": path.is_file(),
            "expected_sha256": digest,
            "actual_sha256": actual,
            "matches": actual == digest,
        }
    return checks


def _model_records(project_root: Path, registry_paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in registry_paths:
        path = project_root / relative
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("models"), list):
            for model in payload["models"]:
                records.append(
                    {
                        "model_id": model["model_id"],
                        "relative_path": model["filename"],
                        "size": int(model["size_bytes"]),
                        "sha256": model["sha256"],
                        "license_status": model["license_status"],
                        "required": bool(model.get("required_for_server_smoke", False)),
                        "registry": relative,
                    }
                )
        else:
            records.append(
                {
                    "model_id": f"{payload['model_name']}_{payload['variant']}",
                    "relative_path": str(payload["weights_path"]).removeprefix(
                        "checkpoints/"
                    ),
                    "size": int(payload["file_size"]),
                    "sha256": payload["sha256"],
                    "license_status": payload["license_status"],
                    "required": True,
                    "registry": relative,
                }
            )
            config_path = payload.get("config_path")
            if config_path:
                records.append(
                    {
                        "model_id": f"{payload['model_name']}_{payload['variant']}_config",
                        "relative_path": str(config_path).removeprefix("checkpoints/"),
                        "size": None,
                        "sha256": payload["config_sha256"],
                        "license_status": payload["license_status"],
                        "required": True,
                        "registry": relative,
                    }
                )
    return records


def _check_models(
    project_root: Path, model_root: Path, registry_paths: list[str]
) -> dict[str, Any]:
    checks = []
    for record in _model_records(project_root, registry_paths):
        candidates = [
            model_root / record["relative_path"],
            project_root / "checkpoints" / record["relative_path"],
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        actual_size = path.stat().st_size if path else 0
        actual_sha = sha256_file(path) if path else ""
        size_matches = record["size"] is None or actual_size == record["size"]
        checks.append(
            {
                **record,
                "path": "" if path is None else str(path),
                "exists": path is not None,
                "actual_size": actual_size,
                "actual_sha256": actual_sha,
                "size_matches": size_matches,
                "sha256_matches": actual_sha == record["sha256"],
                "valid": path is not None
                and size_matches
                and actual_sha == record["sha256"],
            }
        )
    required = [row for row in checks if row["required"]]
    return {
        "records": checks,
        "required_count": len(required),
        "valid_required_count": sum(row["valid"] for row in required),
        "all_required_valid": bool(required) and all(row["valid"] for row in required),
    }


def _check_videos(
    project_root: Path, data_root: Path, registry_path: str
) -> dict[str, Any]:
    payload = yaml.safe_load(
        (project_root / registry_path).read_text(encoding="utf-8")
    )
    checks = []
    for relative, expected in sorted(payload["expected_checksums"].items()):
        candidates = [
            data_root / "tests_videos" / relative,
            data_root / relative,
            project_root / "data" / "tests_videos" / relative,
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        actual = sha256_file(path) if path else ""
        checks.append(
            {
                "relative_path": relative,
                "path": "" if path is None else str(path),
                "exists": path is not None,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "valid": actual == expected,
            }
        )
    return {
        "dataset_role": payload["dataset_role"],
        "eligible_for_training": payload["eligible_for_training"],
        "eligible_for_final_evaluation": payload["eligible_for_final_evaluation"],
        "records": checks,
        "all_valid": bool(checks) and all(row["valid"] for row in checks),
    }


def _check_artifacts(
    project_root: Path, artifacts: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    checks = {}
    for name, metadata in sorted(artifacts.items()):
        sentinel = project_root / metadata["sentinel"]
        expected = metadata.get("sentinel_sha256", "")
        actual = sha256_file(sentinel) if sentinel.is_file() else ""
        checks[name] = {
            "root": metadata["root"],
            "sentinel": metadata["sentinel"],
            "exists": sentinel.is_file(),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "sha256_matches": bool(expected) and actual == expected,
            "valid": sentinel.is_file() and (not expected or actual == expected),
            "git_tracked": bool(metadata.get("git_tracked", False)),
            "rebuildable": bool(metadata.get("rebuildable", False)),
        }
    return checks


def build_handoff_report(
    *,
    project_root: str | Path,
    config_path: str | Path,
    model_root: str | Path,
    data_root: str | Path,
    source_only: bool = False,
) -> dict[str, Any]:
    """Build a read-only handoff report without running model inference."""

    root = Path(project_root).resolve()
    config_file = _resolve(root, config_path)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    source_checks = {
        relative: (root / relative).is_file()
        for relative in config["required_source_paths"]
    }
    frozen = _check_files(root, config["frozen_files"])
    model_checks = (
        {"records": [], "all_required_valid": False, "not_checked": True}
        if source_only
        else _check_models(
            root,
            Path(model_root).resolve(),
            list(config["model_registries"]),
        )
    )
    video_checks = (
        {"records": [], "all_valid": False, "not_checked": True}
        if source_only
        else _check_videos(
            root, Path(data_root).resolve(), config["smoke_data_registry"]
        )
    )
    artifacts = _check_artifacts(root, config["local_artifacts"])
    package = config["provider_source"]["package"]
    source_revision = ""
    source_worktree_clean = False
    import_mode = "python_distribution"
    if package == "unidepth":
        try:
            source = activate_unidepth_source()
        except ExternalSourceError:
            package_available = importlib.util.find_spec(package) is not None
        else:
            package_available = True
            source_revision = source.revision
            source_worktree_clean = source.worktree_clean
            import_mode = "verified_external_source_root"
    else:
        package_available = importlib.util.find_spec(package) is not None
    try:
        package_version = importlib.metadata.version(package) if package_available else ""
    except importlib.metadata.PackageNotFoundError:
        package_version = f"source@{source_revision}" if source_revision else ""
    source_ready = all(source_checks.values()) and all(
        row["matches"] for row in frozen.values()
    )
    m1_artifacts_ready = artifacts["m1_metric_provider"]["valid"]
    p4b5_artifacts_ready = artifacts["p4b5_formal_observations"]["valid"]
    models_ready = bool(model_checks.get("all_required_valid", False))
    videos_ready = bool(video_checks.get("all_valid", False))
    blockers = []
    if not all(source_checks.values()):
        blockers.append("required_source_paths_missing")
    if not all(row["matches"] for row in frozen.values()):
        blockers.append("frozen_hash_mismatch")
    if not source_only and not models_ready:
        blockers.append("registered_model_files_missing_or_invalid")
    if not source_only and not videos_ready:
        blockers.append("six_video_smoke_data_missing_or_invalid")
    if not package_available:
        blockers.append("unidepth_provider_source_not_installed")
    if not m1_artifacts_ready:
        blockers.append("m1_metric_provider_artifacts_missing_or_mismatched")
    if not p4b5_artifacts_ready:
        blockers.append("p4b5_formal_observation_artifacts_missing_or_mismatched")
    git_status = _git(root, "status", "--short")
    return {
        "schema_version": "semantic3d_server_handoff_validation_v1",
        "handoff_id": config["handoff_id"],
        "current_stage": config["current_stage"],
        "git": {
            "branch": _git(root, "branch", "--show-current"),
            "commit": _git(root, "rev-parse", "HEAD"),
            "worktree_clean": not bool(git_status),
            "status": git_status.splitlines(),
        },
        "handoff_config": {
            "path": str(config_file.relative_to(root)),
            "sha256": sha256_file(config_file),
        },
        "required_source_paths": source_checks,
        "frozen_files": frozen,
        "provider_source": {
            **dict(config["provider_source"]),
            "installed": package_available,
            "installed_version": package_version,
            "import_mode": import_mode,
            "verified_source_revision": source_revision,
            "source_worktree_clean": source_worktree_clean,
        },
        "models": model_checks,
        "videos": video_checks,
        "local_artifacts": artifacts,
        "source_checkout_ready": source_ready,
        "model_files_ready": models_ready,
        "smoke_videos_ready": videos_ready,
        "m1_artifacts_ready": m1_artifacts_ready,
        "p4b5_artifacts_ready": p4b5_artifacts_ready,
        "m2_historical_artifact_available": artifacts["m2_metric_scene3d"]["valid"],
        "ready_to_reproduce_m1": source_ready
        and package_available
        and models_ready
        and videos_ready,
        "ready_to_reproduce_m2": source_ready
        and m1_artifacts_ready
        and p4b5_artifacts_ready,
        "ready_for_next_stage": False,
        "next_stage_requires_user_prompt": True,
        "formal_operation_performed": False,
        "blocking_reasons": blockers,
    }


def main() -> int:
    """Run handoff validation and optionally require local dependencies."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--model-root",
        default=os.environ.get("MODEL_ROOT", str(PROJECT_ROOT / "checkpoints")),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DATA_ROOT", str(PROJECT_ROOT / "data")),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--require-models", action="store_true")
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--require-m2-inputs", action="store_true")
    args = parser.parse_args()
    report = build_handoff_report(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        model_root=args.model_root,
        data_root=args.data_root,
        source_only=args.source_only,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    valid = bool(report["source_checkout_ready"])
    if args.require_models:
        valid = valid and bool(report["model_files_ready"])
    if args.require_videos:
        valid = valid and bool(report["smoke_videos_ready"])
    if args.require_m2_inputs:
        valid = (
            valid
            and bool(report["m1_artifacts_ready"])
            and bool(report["p4b5_artifacts_ready"])
        )
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
