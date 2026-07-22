from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.run_real_3d_evidence_coverage import run_real_3d_evidence_coverage


ROOT = Path(__file__).resolve().parents[1]


def test_static_clip_coverage_outputs_and_missing_segmentation_are_explicit(tmp_path: Path) -> None:
    manifest = ROOT / "outputs/sequence_geometry_stabilization/static_camera/shared_geometry_cache/shared_3d_clip_manifest.json"
    if not manifest.exists():
        pytest.skip("P3 real shared cache unavailable.")
    summary = run_real_3d_evidence_coverage(
        geometry_root=ROOT / "outputs/sequence_geometry_stabilization",
        readiness_root=ROOT / "outputs/real_dynamic_3d_smoke",
        observation_root=ROOT / "outputs/evaluation/pilot_6video",
        output_root=tmp_path,
        mask_model_path=tmp_path / "missing-seg.pt",
        pose_model_path=ROOT / "checkpoints/yolov8n-pose.pt",
        clip_ids=("static_camera",),
    )
    for filename in ("mask_coverage.csv", "mask_object_association.csv", "keypoint_coverage.csv", "structure_graph_coverage.csv", "visibility_event_coverage.csv", "occlusion_evidence_coverage.csv", "per_video_summary.csv", "global_summary.json", "coverage_diagnostics.png"):
        assert (tmp_path / filename).exists()
    assert summary["real_instance_mask_provider_available"] is False
    assert summary["total_formal_masks"] == 0
    assert summary["truth_labels_used_for_selection"] is False


@pytest.mark.parametrize(("filename", "expected"), (
    ("scale_priors_strict_v1.yaml", "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"),
    ("scale_priors_strict_v2.yaml", "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"),
))
def test_strict_hashes_unchanged(filename: str, expected: str) -> None:
    assert hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest() == expected
