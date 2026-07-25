"""P4-C3C-A3-A.2 public-asset and controlled-import acceptance tests."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from semantic3d.git_release.model_registry import (
    load_model_registry,
    sha256_file,
    validate_model_registry,
)
from semantic3d.runtime.external_sources import (
    UNIDEPTH_SOURCE_REVISION,
    UNIDEPTH_SOURCE_ROOT_ENV,
    activate_unidepth_source,
    verify_unidepth_source,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = PROJECT_ROOT / "configs/model_registry"
LIVE_ASSET_FLAG = "P4C3C_RUN_PUBLIC_ASSET_TESTS"
OLD_YOLO_REGISTRY_SHA256 = (
    "c2bd34b0829c6101a8be74542018a1ffaa0f881a0c16d3719a7130f93962cd2f"
)
UNIDEPTH_WEIGHT_SHA256 = (
    "93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648"
)
UNIDEPTH_CONFIG_SHA256 = (
    "ecc1f898690debe387d10329cca5d9d66a0a447ce83c3a2a7ae3673c7ce43cfe"
)


def _yaml(name: str) -> dict[str, object]:
    payload = yaml.safe_load((REGISTRY_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _require_live_assets() -> None:
    if os.environ.get(LIVE_ASSET_FLAG) != "1":
        pytest.skip(f"set {LIVE_ASSET_FLAG}=1 for server asset acceptance")


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_yolo_v2_registry_is_complete_and_tasks_are_distinct() -> None:
    registry = load_model_registry(REGISTRY_ROOT / "yolo_weights_v2.yaml")
    result = validate_model_registry(registry)
    assert result["valid"]
    assert result["model_count"] == 3
    assert registry["registry"]["status"] == "active"
    assert registry["registry"]["official_release_tag"] == "v8.4.0"
    assert registry["registry"]["expected_library_version"] == "8.4.90"
    by_name = {row["filename"]: row for row in registry["models"]}
    assert {name: row["task"] for name, row in by_name.items()} == {
        "yolov8n.pt": "detect",
        "yolov8n-seg.pt": "segment",
        "yolov8n-pose.pt": "pose",
    }
    assert all(row["verification_status"] == "VERIFIED" for row in by_name.values())
    assert all(row["source"].split("/")[-2] == "v8.4.0" for row in by_name.values())
    assert all(
        row["official_asset_digest"] == f"sha256:{row['sha256']}"
        for row in by_name.values()
    )


def test_yolo_v2_live_files_match_size_and_sha256() -> None:
    _require_live_assets()
    model_root = os.environ.get("MODEL_ROOT")
    assert model_root, "MODEL_ROOT must select the external model directory"
    registry = load_model_registry(REGISTRY_ROOT / "yolo_weights_v2.yaml")
    result = validate_model_registry(registry, local_model_root=model_root)
    assert result["valid"], result["errors"]
    assert result["all_local_files_verified"]
    assert all(row["local_size_matches"] for row in result["models"])
    assert all(row["local_sha256_matches"] for row in result["models"])


def test_old_yolo_registry_is_unchanged_and_historical() -> None:
    old_path = REGISTRY_ROOT / "yolo_weights_v1.yaml"
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == OLD_YOLO_REGISTRY_SHA256
    profile = _yaml("public_assets_active_v1.yaml")
    yolo = profile["registries"]["yolo"]
    assert yolo["active"] == "configs/model_registry/yolo_weights_v2.yaml"
    assert "configs/model_registry/yolo_weights_v1.yaml" in yolo["historical"]


def test_depth_anything_small_registry_semantics_and_provenance() -> None:
    registry = _yaml("depth_anything_v2_small_v1.yaml")
    assert registry["model_family"] == "Depth Anything V2"
    assert registry["model_variant"] == "Small"
    assert registry["task"] == "monocular_relative_depth_estimation"
    assert registry["source_revision"] == "a561b849ebae10a6f5ef49e26c83cbbcd36c71bf"
    assert registry["model_revision"] == "5426e4f0f36572d16453bbda7a8389317b1bef99"
    assert registry["filename"] == "model.safetensors"
    assert registry["file_size"] == 99173660
    assert registry["sha256"] == (
        "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"
    )
    assert registry["parameter_count"] == 24800000
    assert registry["license"] == "Apache-2.0"
    semantics = registry["output_semantics"]
    assert semantics["depth_type"] == "relative_depth"
    assert semantics["metric_depth"] is False
    assert semantics["physical_unit"] == "none"
    assert "metric_depth_in_meters" in semantics["must_not_be_interpreted_as"]
    usage = registry["active_pipeline_usage"]
    assert usage["stage"] == "P4-B.5"
    assert usage["component"] == "stage_06_depth_precomputation"


def test_depth_anything_small_live_snapshot_matches_registry() -> None:
    _require_live_assets()
    registry = _yaml("depth_anything_v2_small_v1.yaml")
    hf_home = os.environ.get("HF_HOME")
    assert hf_home, "HF_HOME must select the external Hugging Face cache"
    snapshot = (
        Path(hf_home)
        / "hub"
        / "models--depth-anything--Depth-Anything-V2-Small-hf"
        / "snapshots"
        / registry["model_revision"]
    )
    expected = {
        registry["filename"]: (registry["file_size"], registry["sha256"]),
        **{
            row["filename"]: (row["file_size"], row["sha256"])
            for row in registry["companion_files"]
        },
    }
    for filename, (size, digest) in expected.items():
        path = snapshot / filename
        assert path.stat().st_size == size
        assert sha256_file(path) == digest


def test_unidepth_live_source_weight_and_config_are_pinned() -> None:
    _require_live_assets()
    descriptor = verify_unidepth_source()
    assert descriptor.revision == UNIDEPTH_SOURCE_REVISION
    assert descriptor.worktree_clean
    model_root = os.environ.get("MODEL_ROOT")
    assert model_root
    root = Path(model_root) / "unidepth-v2-vits14"
    assert sha256_file(root / "model.safetensors") == UNIDEPTH_WEIGHT_SHA256
    assert sha256_file(root / "config.json") == UNIDEPTH_CONFIG_SHA256


def test_unidepth_adapter_and_handoff_accept_controlled_source_import() -> None:
    _require_live_assets()
    from semantic3d.method_completion.metric_depth_adapters import UniDepthV2Adapter

    model_root = Path(os.environ["MODEL_ROOT"])
    adapter = UniDepthV2Adapter(
        model_root / "unidepth-v2-vits14",
        expected_weight_sha256=UNIDEPTH_WEIGHT_SHA256,
    )
    descriptor = adapter.describe()
    assert descriptor.dependency_available
    assert descriptor.weight_hash_verified

    script = PROJECT_ROOT / "scripts/verify_p4c3b_server_handoff.py"
    spec = importlib.util.spec_from_file_location("a3a2_handoff_verifier", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module.build_handoff_report(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "configs/handoff/p4c3b_m2_server_handoff_v1.yaml",
        model_root=model_root,
        data_root=Path(os.environ["DATA_ROOT"]),
        source_only=True,
    )
    source = report["provider_source"]
    assert source["installed"]
    assert source["import_mode"] == "verified_external_source_root"
    assert source["verified_source_revision"] == UNIDEPTH_SOURCE_REVISION
    assert source["source_worktree_clean"] is True
    assert "unidepth_provider_source_not_installed" not in report["blocking_reasons"]


def test_controlled_unidepth_import_is_revision_checked_and_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "UniDepth"
    package = source / "unidepth"
    (package / "models").mkdir(parents=True)
    (package / "__init__.py").write_text("PINNED_TEST_SOURCE = True\n", encoding="utf-8")
    (package / "models" / "__init__.py").write_text("", encoding="utf-8")
    (source / ".gitignore").write_text("__pycache__/\n*.pyc\n", encoding="utf-8")
    _git(source, "init", "-b", "fixture")
    _git(source, "config", "user.name", "Asset Test")
    _git(source, "config", "user.email", "asset-test@example.invalid")
    _git(
        source,
        "add",
        ".gitignore",
        "unidepth/__init__.py",
        "unidepth/models/__init__.py",
    )
    _git(source, "commit", "-m", "fixture")
    revision = _git(source, "rev-parse", "HEAD")
    environment = {UNIDEPTH_SOURCE_ROOT_ENV: str(source)}
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in tuple(sys.modules):
        if name == "unidepth" or name.startswith("unidepth."):
            monkeypatch.delitem(sys.modules, name)
    original_sys_path = tuple(sys.path)

    def reject_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("controlled source activation must not use the network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    descriptor = activate_unidepth_source(
        environment=environment,
        expected_revision=revision,
    )
    repeated = activate_unidepth_source(
        environment=environment,
        expected_revision=revision,
    )
    assert descriptor.revision == revision
    assert repeated == descriptor
    assert descriptor.import_available
    assert tuple(sys.path) == original_sys_path
    imported = Path(sys.modules["unidepth.models"].__file__).resolve()
    assert source.resolve() in imported.parents


def test_core_gpu_stack_is_unchanged() -> None:
    _require_live_assets()
    import torch
    import torchvision
    import transformers
    import triton

    assert torch.__version__ == "2.12.1+cu130"
    assert torchvision.__version__ == "0.27.1+cu130"
    assert transformers.__version__ == "5.13.0"
    assert triton.__version__ == "3.7.1"
    assert torch.version.cuda == "13.0"
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 5090"
    assert torch.cuda.get_device_capability(0) == (12, 0)


def test_new_asset_configs_use_variables_not_server_absolute_paths() -> None:
    server_data_root = "/".join(("", "root", "autodl-tmp"))
    for name in (
        "yolo_weights_v2.yaml",
        "depth_anything_v2_small_v1.yaml",
        "public_assets_active_v1.yaml",
    ):
        text = (REGISTRY_ROOT / name).read_text(encoding="utf-8")
        assert server_data_root not in text
        assert "/mnt/" not in text
        assert re.search(r"(?m)(?:^|[\s\"'])[A-Za-z]:[\\/]", text) is None
    profile = _yaml("public_assets_active_v1.yaml")
    locations = profile["runtime_locations"]
    assert locations["yolo_model_root"] == "${MODEL_ROOT}"
    assert locations["unidepth_source_root"] == "${SEMANTIC3D_UNIDEPTH_SOURCE_ROOT}"
    assert locations["depth_anything_cache_root"] == "${HF_HOME}/hub"
