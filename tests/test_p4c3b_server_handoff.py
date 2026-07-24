"""Tests for the repository-resident P4-C3B server handoff."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import yaml

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "verify_p4c3b_server_handoff.py"
)
SPEC = importlib.util.spec_from_file_location("verify_p4c3b_server_handoff", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs/handoff/p4c3b_m2_server_handoff_v1.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_handoff_required_source_and_frozen_hashes_are_valid() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert all((PROJECT_ROOT / path).is_file() for path in config["required_source_paths"])
    assert all(
        _sha(PROJECT_ROOT / path) == expected
        for path, expected in config["frozen_files"].items()
    )


def test_source_only_report_never_authorizes_next_stage() -> None:
    report = MODULE.build_handoff_report(
        project_root=PROJECT_ROOT,
        config_path=CONFIG_PATH,
        model_root=PROJECT_ROOT / "checkpoints",
        data_root=PROJECT_ROOT / "data",
        source_only=True,
    )
    assert report["source_checkout_ready"]
    assert not report["ready_for_next_stage"]
    assert report["next_stage_requires_user_prompt"]
    assert not report["formal_operation_performed"]


def test_local_only_artifacts_are_explicitly_not_git_tracked() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["local_artifacts"]
    assert all(
        item["git_tracked"] is False
        for item in config["local_artifacts"].values()
    )


def test_six_video_registry_remains_smoke_only() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    registry = yaml.safe_load(
        (PROJECT_ROOT / config["smoke_data_registry"]).read_text(encoding="utf-8")
    )
    assert registry["dataset_role"] == "geometry_validation_smoke"
    assert registry["eligible_for_training"] is False
    assert registry["eligible_for_model_selection"] is False
    assert registry["eligible_for_threshold_selection"] is False
    assert registry["eligible_for_final_evaluation"] is False
