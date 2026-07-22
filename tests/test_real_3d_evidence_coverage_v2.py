from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from scripts.run_real_3d_evidence_coverage_v2 import run_real_3d_evidence_coverage_v2


ROOT = Path(__file__).resolve().parents[1]


def _video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (48, 32))
    for value in (0, 30, 60):
        writer.write(np.full((32, 48, 3), value, dtype=np.uint8))
    writer.release()


def test_v2_outputs_explicit_observation_missing_without_weights(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    _video(videos / "v1.mp4")
    output = tmp_path / "coverage"
    summary = run_real_3d_evidence_coverage_v2(
        video_root=videos,
        output_root=output,
        mask_model_path=tmp_path / "missing-seg.pt",
        pose_model_path=tmp_path / "missing-pose.pt",
        run_shared_3d_smoke=False,
    )
    for filename in (
        "model_metadata.json", "mask_coverage.csv", "mask_object_association.csv",
        "mask_tracking_quality.csv", "keypoint_coverage.csv",
        "structure_graph_coverage.csv", "structure_residual_coverage.csv",
        "visibility_event_coverage.csv", "occlusion_evidence_coverage.csv",
        "reappearance_coverage.csv", "per_video_summary.csv", "global_summary.json",
        "coverage_diagnostics.png",
    ):
        assert (output / filename).exists()
    assert summary["total_formal_masks"] == 0
    assert summary["ready_for_full_p4_videos"] == []
    metadata = json.loads((output / "model_metadata.json").read_text())
    assert metadata["missing_reason"] == "instance_segmentation_weights_missing"
    assert metadata["automatic_download_attempted"] is False


def test_strict_scale_prior_hashes_remain_frozen() -> None:
    expected = {
        "scale_priors_strict_v1.yaml": "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b",
        "scale_priors_strict_v2.yaml": "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b",
    }
    for filename, digest in expected.items():
        assert hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest() == digest
