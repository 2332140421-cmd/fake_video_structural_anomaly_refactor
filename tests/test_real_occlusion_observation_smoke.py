from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_real_occlusion_observation_smoke import (
    _find_associated_observation,
    run_real_occlusion_observation_smoke,
)
from semantic3d.dynamic_3d import load_shared_geometry_cache


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = ROOT / "outputs/sequence_geometry_stabilization"
READINESS_ROOT = ROOT / "outputs/real_dynamic_3d_smoke"
OBSERVATION_ROOT = ROOT / "outputs/evaluation/pilot_6video"
REQUIRED_OUTPUTS = {
    "instance_masks.json",
    "tracked_masks.json",
    "visibility_states.csv",
    "predicted_support_masks.json",
    "occlusion_relations.csv",
    "depth_order_residuals.csv",
    "visibility_residuals.csv",
    "boundary_residuals.csv",
    "reappearance_observations.csv",
    "occlusion_diagnostics.png",
    "smoke_report.json",
}


def _assets(clip_id: str) -> tuple[Path, Path, Path]:
    manifest = GEOMETRY_ROOT / clip_id / "shared_geometry_cache/shared_3d_clip_manifest.json"
    readiness = READINESS_ROOT / clip_id / "dynamic_readiness.json"
    if not manifest.exists() or not readiness.exists():
        pytest.skip("Existing P3 shared-geometry smoke artifacts are unavailable.")
    cache = load_shared_geometry_cache(manifest)
    observation = _find_associated_observation(cache.clip.video_id, OBSERVATION_ROOT)
    return manifest, readiness, observation


def test_smoke_source_does_not_invoke_geometry_estimators() -> None:
    source = (ROOT / "scripts/run_real_occlusion_observation_smoke.py").read_text(encoding="utf-8")
    for forbidden in (
        "RealDepthProvider(",
        "estimate_adaptive_pose_candidates(",
        "stabilize_sequence_geometry(",
        "RealObjectProvider(",
    ):
        assert forbidden not in source
    assert "load_shared_geometry_cache(" in source


def test_static_real_smoke_keeps_bbox_fallback_diagnostic_only(tmp_path: Path) -> None:
    manifest, readiness, observation = _assets("static_camera")
    report = run_real_occlusion_observation_smoke(
        geometry_cache_manifest=manifest,
        readiness_path=readiness,
        associated_observation_path=observation,
        output_dir=tmp_path,
    )
    assert REQUIRED_OUTPUTS <= {item.name for item in tmp_path.iterdir()}
    assert report["geometry_mode"] == "static_camera_3d"
    assert report["shared_clip_reused"] is True
    assert report["depth_reestimated"] is False
    assert report["intrinsics_reestimated"] is False
    assert report["pose_reestimated"] is False
    assert report["current_frame_used_for_prediction"] is False
    assert report["formal_valid_mask_count"] == 0
    assert report["legacy_bbox_mask_count"] == report["diagnostic_valid_mask_count"]
    assert report["bbox_fallback_formal_evidence_count"] == 0
    assert report["formal_depth_order_evidence_count"] == 0
    assert report["formal_visibility_residual_count"] == 0
    assert report["formal_boundary_residual_count"] == 0


def test_unavailable_real_smoke_emits_no_formal_evidence(tmp_path: Path) -> None:
    manifest, readiness, observation = _assets("slowly_moving_camera")
    report = run_real_occlusion_observation_smoke(
        geometry_cache_manifest=manifest,
        readiness_path=readiness,
        associated_observation_path=observation,
        output_dir=tmp_path,
    )
    assert report["geometry_mode"] == "unavailable"
    assert report["dynamic_3d_ready"] is False
    assert report["formal_occlusion_relation_count"] == 0
    assert report["formal_depth_order_evidence_count"] == 0
    assert report["formal_visibility_residual_count"] == 0
    assert report["formal_boundary_residual_count"] == 0


@pytest.mark.parametrize(
    ("filename", "expected_hash"),
    (
        ("scale_priors_strict_v1.yaml", "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"),
        ("scale_priors_strict_v2.yaml", "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"),
    ),
)
def test_strict_scale_prior_hashes_remain_frozen(filename: str, expected_hash: str) -> None:
    content = (ROOT / "configs" / filename).read_bytes()
    assert hashlib.sha256(content).hexdigest() == expected_hash
