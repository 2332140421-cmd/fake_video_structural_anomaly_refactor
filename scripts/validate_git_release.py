#!/usr/bin/env python3
"""Generate and validate the P4-C3A-G Git release audit package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.writer import atomic_write_bytes, atomic_write_json  # noqa: E402
from semantic3d.git_release.audit import (  # noqa: E402
    build_git_release_audit,
    write_git_release_artifacts,
)
from semantic3d.git_release.model_registry import load_model_registry  # noqa: E402
from semantic3d.git_release.validation import validate_git_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/p4c3a_git_release")
    parser.add_argument("--model-registry", default="configs/model_registry/yolo_weights_v1.yaml")
    parser.add_argument("--pytest-report", default="")
    args = parser.parse_args()
    output_root = PROJECT_ROOT / args.output_root
    pytest_report = (
        Path(args.pytest_report)
        if args.pytest_report
        else output_root / "pytest_full_q_rs.txt"
    )
    first = build_git_release_audit(PROJECT_ROOT)
    second = build_git_release_audit(PROJECT_ROOT)
    deterministic = first["release_content_sha256"] == second["release_content_sha256"]
    write_git_release_artifacts(output_root, first)
    registry = load_model_registry(PROJECT_ROOT / args.model_registry)
    validation = validate_git_release(
        PROJECT_ROOT,
        first,
        registry,
        pytest_report_path=pytest_report,
    )
    validation["deterministic_rebuild_matches"] = deterministic
    validation["first_release_content_sha256"] = first["release_content_sha256"]
    validation["second_release_content_sha256"] = second["release_content_sha256"]
    validation["valid"] = bool(validation["valid"] and deterministic)
    atomic_write_json(output_root / "git_release_validation.json", validation)
    report_path = output_root / "P4C3A_GIT_RELEASE_REPORT.md"
    report = report_path.read_text(encoding="utf-8")
    report += "\n## Readiness\n\n"
    for name in (
        "ready_for_git_commit", "ready_for_git_push", "ready_for_server_clone",
        "ready_for_server_smoke", "ready_for_formal_batch_execution",
    ):
        report += f"- {name}: `{str(validation[name]).lower()}`\n"
    report += f"- deterministic rebuild: `{str(deterministic).lower()}`\n"
    report += f"- pytest: `{validation['pytest']['summary']}`\n"
    atomic_write_bytes(report_path, report.encode("utf-8"))
    print(json.dumps(validation, ensure_ascii=False, indent=2))
    return 0 if validation["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
