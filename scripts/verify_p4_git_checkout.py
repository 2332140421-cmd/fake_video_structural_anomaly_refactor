#!/usr/bin/env python3
"""Verify a cloned P4 checkout and report whether server smoke can start."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.writer import atomic_write_json  # noqa: E402
from semantic3d.git_release.model_registry import (  # noqa: E402
    load_model_registry,
    validate_model_registry,
)
from semantic3d.git_release.validation import (  # noqa: E402
    FROZEN_HASHES,
    P4C0_SEMANTIC_HASH,
    P4C2_SEMANTIC_HASH,
)
from semantic3d.runtime.runtime_config import load_runtime_config  # noqa: E402


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", default="configs/runtime/server_template.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--tests-passed", action="store_true")
    parser.add_argument("--preflight-validation", default="")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--require-server-smoke", action="store_true")
    args = parser.parse_args()
    runtime = load_runtime_config(PROJECT_ROOT / args.runtime_config, environment=os.environ)
    status = _git("status", "--short")
    frozen = {}
    for name, (relative, expected) in FROZEN_HASHES.items():
        path = PROJECT_ROOT / relative
        actual = _sha(path) if path.is_file() else ""
        frozen[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
    p4c2 = json.loads(
        (PROJECT_ROOT / "outputs/p4c2_formal_data_readiness/build_metadata.json").read_text(encoding="utf-8")
    )
    frozen["p4c0_semantic_protocol"] = {
        "expected": P4C0_SEMANTIC_HASH,
        "actual": p4c2.get("protocol_sha256", ""),
        "matches": p4c2.get("protocol_sha256") == P4C0_SEMANTIC_HASH,
    }
    frozen["p4c2_semantic_manifest"] = {
        "expected": P4C2_SEMANTIC_HASH,
        "actual": p4c2.get("p4c2_readiness_manifest_sha256", ""),
        "matches": p4c2.get("p4c2_readiness_manifest_sha256") == P4C2_SEMANTIC_HASH,
    }
    model_registry = load_model_registry(PROJECT_ROOT / "configs/model_registry/yolo_weights_v1.yaml")
    model_validation = validate_model_registry(model_registry, local_model_root=runtime.model_root)
    smoke_registry = yaml.safe_load(
        (PROJECT_ROOT / "configs/data_registry/local_six_video_smoke_v1.yaml").read_text(encoding="utf-8")
    )
    video_checks = {}
    for relative, expected in sorted(smoke_registry["expected_checksums"].items()):
        path = runtime.data_root / "tests_videos" / relative
        video_checks[relative] = path.is_file() and _sha(path) == expected
    preflight_ready = False
    if args.preflight_validation:
        preflight = json.loads(Path(args.preflight_validation).read_text(encoding="utf-8"))
        preflight_ready = bool(preflight.get("ready_for_server_smoke", False))
    checkout_valid = (
        bool(_git("rev-parse", "--is-inside-work-tree") == "true")
        and (args.allow_dirty or not status)
        and all(row["matches"] for row in frozen.values())
    )
    server_smoke_ready = (
        checkout_valid
        and args.tests_passed
        and preflight_ready
        and model_validation["all_local_files_verified"]
        and bool(video_checks)
        and all(video_checks.values())
    )
    report = {
        "schema_version": "p4c3a_git_checkout_validation_v1",
        "branch": _git("branch", "--show-current"),
        "commit": _git("rev-parse", "HEAD"),
        "worktree_clean": not bool(status),
        "git_status": status.splitlines(),
        "runtime_profile": runtime.runtime_profile,
        "frozen_hashes": frozen,
        "model_registry": model_validation,
        "smoke_video_checks": video_checks,
        "tests_passed": args.tests_passed,
        "preflight_ready_for_server_smoke": preflight_ready,
        "ready_for_server_clone_checkout": checkout_valid,
        "ready_for_server_smoke": server_smoke_ready,
        "ready_for_formal_batch_execution": False,
        "formal_operation_performed": False,
    }
    output = Path(args.output) if args.output else runtime.output_root / "p4c3a_git_checkout/checkout_report.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if (server_smoke_ready if args.require_server_smoke else checkout_valid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
