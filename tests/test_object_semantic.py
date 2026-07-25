from pathlib import Path

import numpy as np

from data.schemas import ClipObservation, FrameObservation, ObjectObservation
from models.object_semantic import compute_object_semantic_residuals


def _prior(path: Path, orientation: str = "unknown") -> Path:
    path.write_text(
        "metric_scale_priors:\n"
        "- category: box\n"
        "  dimension: width\n"
        "  min_meters: 0.8\n"
        "  max_meters: 1.0\n"
        f"  orientation_requirement: {orientation}\n"
        "  minimum_observability: 0.5\n"
        "  source_note: synthetic geometry test\n",
        encoding="utf-8",
    )
    return path


def _clip(
    depths,
    *,
    category="box",
    valid_depth=True,
    truncated=False,
    occlusion=0.0,
    viewpoint="unknown",
):
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    intrinsics = np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]])
    frames = []
    for index, depth in enumerate(depths):
        obj = ObjectObservation(
            "object",
            "stable_track",
            category,
            (5, 5, 15, 15),
            1.0,
            instance_mask=mask,
            truncated=truncated,
            occlusion_ratio=occlusion,
            viewpoint=viewpoint,
        )
        frames.append(
            FrameObservation(
                "video",
                "clip",
                index,
                float(index),
                np.zeros((20, 20, 3), dtype=np.uint8),
                [obj],
                np.full((20, 20), depth, dtype=float),
                np.full((20, 20), valid_depth, dtype=bool),
                intrinsics=intrinsics,
                confidence={"metric_depth": 1.0},
            )
        )
    return ClipObservation("video", "clip", frames)


def _semantic(clip, prior):
    return compute_object_semantic_residuals(clip, prior_path=prior)


def test_metric_size_inside_above_and_below_prior(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    inside = _semantic(_clip([1.0]), prior)[0]
    above = _semantic(_clip([4.0]), prior)[0]
    below = _semantic(_clip([0.3]), prior)[0]
    assert inside.raw_value < 0.2
    assert above.raw_value > inside.raw_value
    assert below.raw_value > inside.raw_value


def test_semantic_gates_missing_inputs_and_observability(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    assert not _semantic(_clip([1.0], valid_depth=False), prior)[0].valid_mask
    assert _semantic(_clip([1.0], category="unknown"), prior)[0].reason == "missing_category_metric_prior"
    assert _semantic(_clip([1.0], truncated=True), prior)[0].reason == "severe_object_truncation"
    assert _semantic(_clip([1.0], occlusion=0.9), prior)[0].reason == "severe_object_occlusion"
    oriented = _prior(tmp_path / "oriented.yaml", orientation="front")
    assert _semantic(_clip([1.0], viewpoint="side"), oriented)[0].reason == "dimension_not_observable_from_current_view"


def test_same_track_temporal_stability_and_jump(tmp_path):
    prior = _prior(tmp_path / "prior.yaml")
    stable = [row for row in _semantic(_clip([1.0, 1.0]), prior) if row.name == "semantic_metric_temporal"][-1]
    jump = [row for row in _semantic(_clip([1.0, 4.0]), prior) if row.name == "semantic_metric_temporal"][-1]
    assert stable.valid_mask and stable.raw_value < 1e-8
    assert jump.valid_mask and jump.raw_value > 1.0
