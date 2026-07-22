"""Python, platform, Git, and frozen-hash environment probes."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from .runtime_config import RuntimeConfig


def _git(project: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout.strip() or result.stderr.strip()


def probe_environment(
    runtime: RuntimeConfig,
    required_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Probe host metadata and frozen files without including it in semantic hashes."""

    commit_code, commit = _git(runtime.project_root, "rev-parse", "HEAD")
    status_code, status = _git(runtime.project_root, "status", "--porcelain=v1")
    hash_paths = {
        "p4c0_protocol_sha256": runtime.project_root
        / "data/experiment_protocol_v1/experiment_protocol.json",
        "p4c1_manifest_sha256": runtime.project_root
        / "outputs/p4c1_experiment_manifest/experiment_manifest.jsonl",
        "p4c2_manifest_sha256": runtime.project_root
        / "outputs/p4c2_formal_data_readiness/source_lineage.jsonl",
        "strict_v1_sha256": runtime.project_root / "configs/scale_priors_strict_v1.yaml",
        "strict_v2_sha256": runtime.project_root / "configs/scale_priors_strict_v2.yaml",
    }
    # P4-C0/P4-C2 semantic hashes are stored metadata values, not raw artifact hashes.
    semantic_values = {
        "p4c0_protocol_sha256": _json_value(
            runtime.project_root / "outputs/p4c2_formal_data_readiness/build_metadata.json",
            "protocol_sha256",
        ),
        "p4c1_manifest_sha256": _json_value(
            runtime.project_root / "outputs/p4c1_experiment_manifest/manifest_metadata.json",
            "manifest_sha256",
        ),
        "p4c2_manifest_sha256": _json_value(
            runtime.project_root / "outputs/p4c2_formal_data_readiness/build_metadata.json",
            "p4c2_readiness_manifest_sha256",
        ),
        "strict_v1_sha256": _sha256(hash_paths["strict_v1_sha256"]),
        "strict_v2_sha256": _sha256(hash_paths["strict_v2_sha256"]),
    }
    hash_checks = {
        name: {
            "expected": str(expected),
            "actual": semantic_values.get(name, ""),
            "matches": semantic_values.get(name, "") == str(expected),
        }
        for name, expected in sorted(required_hashes.items())
    }
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "username": os.environ.get("USER", ""),
        "git_commit": commit if commit_code == 0 else "",
        "git_available": commit_code == 0 and status_code == 0,
        "git_worktree_clean": status_code == 0 and not status,
        "git_status_lines": status.splitlines(),
        "frozen_hash_checks": hash_checks,
        "all_frozen_hashes_match": all(row["matches"] for row in hash_checks.values()),
        "location_specific_metadata": True,
    }


def _sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(path: Path, key: str) -> str:
    import json

    if not path.is_file():
        return ""
    return str(json.loads(path.read_text(encoding="utf-8")).get(key, ""))

