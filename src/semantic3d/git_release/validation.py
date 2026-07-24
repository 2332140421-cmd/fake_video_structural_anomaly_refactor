"""Release readiness validation for Git-based P4 server deployment."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import yaml

from .model_registry import validate_model_registry

FROZEN_HASHES = {
    "p4c0_config_file": (
        "configs/p4c0_experiment_protocol_v1.yaml",
        "8a4a8f5d6ac795646876042a84c9b0a4fdb1d06bec31045b734c3dfb64f8a304",
    ),
    "p4c1_manifest": (
        "outputs/p4c1_experiment_manifest/experiment_manifest.jsonl",
        "3bfa3cc030adc29d1bfd6374d36ee17ad9f4a6fedb48f7b882549bcba4ad57a3",
    ),
    "strict_v1": (
        "configs/scale_priors_strict_v1.yaml",
        "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
    ),
    "strict_v2": (
        "configs/scale_priors_strict_v2.yaml",
        "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    ),
    "p4c3a_server_smoke_config": (
        "configs/p4c3a_server_smoke_v1.yaml",
        "4742f9dc910e154bab03196568c4d33e8a6596a81874b87adc1d4553451b688d",
    ),
}
P4C0_SEMANTIC_HASH = "004fb982597091e038bbb833370169626326f0ad5ecd0998fd61132ed1dffc12"
P4C2_SEMANTIC_HASH = "2094a721bc58c7cae1d567a41d9fd31542990a95f87f0927b87c31119280e981"

_REQUIRED_GITIGNORE = (
    ".venv/", "venv/", "__pycache__/", "*.pyc", ".pytest_cache/",
    ".mypy_cache/", ".ruff_cache/", ".coverage", "htmlcov/", ".codex/",
    "*.sock", ".env", ".env.*", "*.pem", "*.key", "*credentials*",
    "*secret*", "*token*", "data/raw/", "data/downloads/", "data/cache/",
    "data/tmp/", "data/frames/", "data/depth/", "data/flow/", "data/tracks/",
    "data/semantic3d/", "data/large_artifacts/", "outputs/**", "logs/**",
    "tmp/**", "cache/**",
)
_REQUIRED_PATHS = (
    "AGENTS.md",
    ".gitignore",
    "pyproject.toml",
    "requirements-lock.txt",
    "requirements-inference-lock.txt",
    "configs/runtime/server_template.yaml",
    "configs/model_registry/yolo_weights_v1.yaml",
    "configs/model_registry/unidepth_v2_vits14_v1.yaml",
    "configs/handoff/p4c3b_m2_server_handoff_v1.yaml",
    "configs/p4c3b_metric_provider_smoke_v1.yaml",
    "configs/p4c3b_metric_scene3d_v1.yaml",
    "configs/data_registry/dataset_registry_schema_v1.yaml",
    "configs/data_registry/local_six_video_smoke_v1.yaml",
    "scripts/bootstrap_p4_server.sh",
    "scripts/verify_p4_git_checkout.py",
    "scripts/fetch_registered_models.py",
    "scripts/validate_git_release.py",
    "scripts/verify_p4c3b_server_handoff.py",
    "docs/SERVER_ENVIRONMENT.md",
    "docs/SERVER_DATA_SETUP.md",
    "docs/SERVER_HANDOFF_P4C3B_M2.md",
    "docs/GIT_UPLOAD_CHECKLIST.md",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tracked(project: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=project, capture_output=True, check=True
    )
    return {item.decode("utf-8") for item in result.stdout.split(b"\0") if item}


def _shell_syntax(project: Path) -> dict[str, Any]:
    script = project / "scripts/bootstrap_p4_server.sh"
    if not script.is_file():
        return {"valid": False, "error": "bootstrap_script_missing"}
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
    return {"valid": result.returncode == 0, "error": result.stderr.strip()}


def _pytest_status(report_path: Path) -> dict[str, Any]:
    if not report_path.is_file():
        return {"passed": False, "reason": "pytest_report_missing", "summary": "", "key_p4_skipped": []}
    text = report_path.read_text(encoding="utf-8", errors="replace")
    summaries = re.findall(r"(?:^|\n)([^\n]*(?:passed|failed)[^\n]*)", text)
    summary = summaries[-1].strip() if summaries else ""
    passed = bool(re.search(r"\b\d+ passed\b", summary)) and " failed" not in summary
    skip_lines = [line for line in text.splitlines() if line.startswith("SKIPPED")]
    skip_count = sum(
        int(match.group(1)) if (match := re.match(r"SKIPPED \[(\d+)\]", line)) else 1
        for line in skip_lines
    )
    key_p4_skipped = [line for line in skip_lines if "test_p4c" in line.lower()]
    return {
        "passed": passed,
        "reason": "" if passed else "pytest_summary_not_successful",
        "summary": summary,
        "skip_count": skip_count,
        "skip_reason_lines": skip_lines,
        "key_p4_skipped": key_p4_skipped,
        "report_path": str(report_path),
    }


def _data_registry(project: Path) -> dict[str, Any]:
    schema = yaml.safe_load(
        (project / "configs/data_registry/dataset_registry_schema_v1.yaml").read_text(encoding="utf-8")
    )
    smoke = yaml.safe_load(
        (project / "configs/data_registry/local_six_video_smoke_v1.yaml").read_text(encoding="utf-8")
    )
    missing = [field for field in schema["required_fields"] if field not in smoke]
    flags_false = all(
        smoke.get(field) is False
        for field in (
            "is_formal_dataset", "eligible_for_training", "eligible_for_model_selection",
            "eligible_for_threshold_selection", "eligible_for_final_evaluation",
        )
    )
    return {
        "valid": not missing and flags_false and smoke.get("license") == "unverified"
        and smoke.get("redistribution_allowed") is False,
        "missing_fields": missing,
        "smoke_formal_eligibility_all_false": flags_false,
        "redistribution_allowed": smoke.get("redistribution_allowed"),
        "checksum_count": len(smoke.get("expected_checksums", {})),
    }


def validate_git_release(
    project_root: str | Path,
    audit: Mapping[str, Any],
    model_registry: Mapping[str, Any],
    *,
    pytest_report_path: str | Path,
) -> dict[str, Any]:
    """Validate release content without modifying Git or running a formal operation."""

    project = Path(project_root).resolve()
    ignore_text = (project / ".gitignore").read_text(encoding="utf-8")
    ignore_checks = {pattern: pattern in ignore_text for pattern in _REQUIRED_GITIGNORE}
    tracked = _tracked(project)
    handoff_path = project / "configs/handoff/p4c3b_m2_server_handoff_v1.yaml"
    handoff_required = (
        tuple(
            yaml.safe_load(handoff_path.read_text(encoding="utf-8")).get(
                "required_source_paths", ()
            )
        )
        if handoff_path.is_file()
        else ()
    )
    required_paths = tuple(dict.fromkeys((*_REQUIRED_PATHS, *handoff_required)))
    required_exists = {path: (project / path).is_file() for path in required_paths}
    required_tracked = {path: path in tracked for path in required_paths}
    protocol_artifacts_not_ignored = {}
    for relative in (
        "outputs/p4c1_experiment_manifest/experiment_manifest.jsonl",
        "outputs/p4c2_formal_data_readiness/build_metadata.json",
        "outputs/p4c3a_batch_plan/batch_plan_metadata.json",
    ):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=project, check=False
        )
        protocol_artifacts_not_ignored[relative] = result.returncode != 0

    frozen: dict[str, Any] = {}
    for name, (relative, expected) in FROZEN_HASHES.items():
        path = project / relative
        actual = _sha(path) if path.is_file() else ""
        frozen[name] = {"path": relative, "expected": expected, "actual": actual, "matches": actual == expected}
    p4c2_metadata = json.loads(
        (project / "outputs/p4c2_formal_data_readiness/build_metadata.json").read_text(encoding="utf-8")
    )
    frozen["p4c0_semantic_protocol"] = {
        "path": "outputs/p4c2_formal_data_readiness/build_metadata.json",
        "expected": P4C0_SEMANTIC_HASH,
        "actual": p4c2_metadata.get("protocol_sha256", ""),
        "matches": p4c2_metadata.get("protocol_sha256") == P4C0_SEMANTIC_HASH,
    }
    frozen["p4c2_semantic_manifest"] = {
        "path": "outputs/p4c2_formal_data_readiness/build_metadata.json",
        "expected": P4C2_SEMANTIC_HASH,
        "actual": p4c2_metadata.get("p4c2_readiness_manifest_sha256", ""),
        "matches": p4c2_metadata.get("p4c2_readiness_manifest_sha256") == P4C2_SEMANTIC_HASH,
    }

    model_validation = validate_model_registry(model_registry, local_model_root=project / "checkpoints")
    data_validation = _data_registry(project)
    shell_validation = _shell_syntax(project)
    pytest_validation = _pytest_status(Path(pytest_report_path))
    absolute_blockers = [row for row in audit["absolute_path_findings"] if row["blocking"]]
    sensitive_commit_candidates = [
        row for row in audit["sensitive"] if row["recommended_action"] == "commit"
    ]
    ordinary_git_large_files = [
        row for row in (*audit["tracked"], *audit["untracked"])
        if row["size"] > 50 * 1024 * 1024 and row["recommended_action"] == "commit"
    ]
    required_available = all(required_exists.values())
    required_are_tracked = all(required_tracked.values())
    content_safe = (
        all(ignore_checks.values())
        and all(protocol_artifacts_not_ignored.values())
        and not sensitive_commit_candidates
        and not ordinary_git_large_files
        and not absolute_blockers
        and all(row["matches"] for row in frozen.values())
        and model_validation["valid"]
        and model_validation["server_smoke_hashes_complete"]
        and data_validation["valid"]
        and shell_validation["valid"]
        and pytest_validation["passed"]
        and not pytest_validation["key_p4_skipped"]
        and required_available
    )
    remote_configured = bool(audit.get("remotes"))
    ready_for_git_commit = content_safe
    ready_for_git_push = content_safe and bool(audit["worktree_clean"]) and required_are_tracked and remote_configured
    ready_for_server_clone = ready_for_git_push
    blockers: list[str] = []
    if not content_safe:
        blockers.append("release_content_validation_failed")
    if not required_are_tracked:
        blockers.append("required_release_files_not_yet_tracked")
    if not audit["worktree_clean"]:
        blockers.append("git_worktree_not_clean")
    return {
        "schema_version": "p4c3a_git_release_validation_v1",
        "valid": content_safe,
        "gitignore_checks": ignore_checks,
        "small_protocol_artifacts_not_ignored": protocol_artifacts_not_ignored,
        "required_paths_exist": required_exists,
        "required_paths_tracked": required_tracked,
        "all_required_paths_available": required_available,
        "all_required_paths_tracked": required_are_tracked,
        "absolute_path_blockers": absolute_blockers,
        "sensitive_commit_candidates": sensitive_commit_candidates,
        "ordinary_git_large_files": ordinary_git_large_files,
        "frozen_hashes": frozen,
        "model_registry": model_validation,
        "data_registry": data_validation,
        "bootstrap_shell_syntax": shell_validation,
        "pytest": pytest_validation,
        "remote_configured": remote_configured,
        "ready_for_git_commit": ready_for_git_commit,
        "ready_for_git_push": ready_for_git_push,
        "ready_for_server_clone": ready_for_server_clone,
        "ready_for_server_smoke": False,
        "ready_for_formal_batch_execution": False,
        "blockers": blockers,
        "automatic_git_action_performed": False,
        "formal_operation_performed": False,
    }
