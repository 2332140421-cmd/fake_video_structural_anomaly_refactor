from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from semantic3d.dimension_aligned_scale_depth import (
    compute_dimension_aligned_rsd,
    load_dimension_aligned_prior_resolver,
)
from semantic3d.keypoint_provider import (
    BaseKeypointProvider,
    Keypoint2D,
    MockKeypointProvider,
    RealHumanKeypointProvider,
)
from semantic3d.io import load_clip_observation
from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.pose_applicability import (
    CupHeightApplicabilityGate,
    PersonHeightApplicabilityGate,
)
from semantic3d.projected_measurement import load_projected_measurement_rules


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V1_HASH = "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"
V2_HASH = "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _point(name: str, x: float, y: float, confidence: float = 0.9) -> Keypoint2D:
    return Keypoint2D(name, x, y, confidence, confidence > 0, "mock_keypoints")


def _upright_points() -> list[Keypoint2D]:
    return [
        _point("left_shoulder", 40, 40),
        _point("right_shoulder", 60, 40),
        _point("left_hip", 44, 90),
        _point("right_hip", 56, 90),
        _point("left_knee", 44, 135),
        _point("right_knee", 56, 135),
        _point("left_ankle", 44, 180),
        _point("right_ankle", 56, 180),
    ]


def _prediction(points: list[Keypoint2D]):
    return MockKeypointProvider(points).predict(
        np.zeros((200, 100, 3), dtype=np.uint8), [20, 10, 80, 190], "person"
    )


def _person_result(points: list[Keypoint2D], bbox: list[float] | None = None):
    return PersonHeightApplicabilityGate().evaluate(
        bbox or [20, 10, 80, 190],
        (100, 200),
        _prediction(points),
        0.9,
    )


def test_complete_upright_person_passes() -> None:
    result = _person_result(_upright_points())
    assert result.applicable
    assert result.applicability_status == "applicable_upright_full_body"
    assert result.diagnostic_details["score_semantics"] == "heuristic_quality_not_probability"


def test_sitting_person_is_rejected() -> None:
    points = _upright_points()
    replacements = {
        "left_knee": (72, 108),
        "right_knee": (78, 108),
        "left_ankle": (78, 118),
        "right_ankle": (84, 118),
    }
    points = [
        _point(point.keypoint_name, *replacements.get(point.keypoint_name, (point.x, point.y)))
        for point in points
    ]
    result = _person_result(points)
    assert not result.applicable
    assert result.applicability_status == "sitting_or_bending"


def test_bending_person_is_rejected_or_unresolved() -> None:
    points = _upright_points()
    replacements = {
        "left_shoulder": (25, 60),
        "right_shoulder": (35, 60),
        "left_hip": (65, 90),
        "right_hip": (75, 90),
    }
    points = [
        _point(point.keypoint_name, *replacements.get(point.keypoint_name, (point.x, point.y)))
        for point in points
    ]
    result = _person_result(points)
    assert not result.applicable
    assert result.applicability_status in {"sitting_or_bending", "unresolved_pose"}


def test_missing_ankles_rejects_complete_height_prior() -> None:
    points = [
        point
        for point in _upright_points()
        if point.keypoint_name not in {"left_ankle", "right_ankle"}
    ]
    result = _person_result(points)
    assert not result.applicable
    assert result.applicability_status == "insufficient_keypoints"


def test_boundary_contact_is_rejected() -> None:
    result = _person_result(_upright_points(), [0, 10, 80, 190])
    assert not result.applicable
    assert result.applicability_status == "boundary_truncated"


def test_low_keypoint_confidence_is_rejected() -> None:
    points = [
        Keypoint2D(point.keypoint_name, point.x, point.y, 0.1, True, "mock_keypoints")
        for point in _upright_points()
    ]
    result = _person_result(points)
    assert not result.applicable
    assert result.applicability_status == "low_keypoint_confidence"


def test_non_person_keypoint_request_is_unsupported_not_exception() -> None:
    prediction = MockKeypointProvider(_upright_points()).predict(
        np.zeros((20, 20, 3), dtype=np.uint8), [1, 1, 10, 10], "cup"
    )
    assert prediction.status == "unsupported_label"
    assert not prediction.supported


def test_cup_too_small_is_rejected() -> None:
    result = CupHeightApplicabilityGate().evaluate(
        [10, 10, 16, 18],
        (100, 100),
        0.9,
        projection_history=[0.08, 0.081],
        depth_iqr=0.1,
        representative_depth=2.0,
        valid_depth_ratio=1.0,
    )
    assert not result.applicable
    assert result.applicability_status == "too_small"


def test_cup_unstable_depth_is_rejected() -> None:
    result = CupHeightApplicabilityGate().evaluate(
        [20, 20, 50, 60],
        (100, 100),
        0.9,
        projection_history=[0.4, 0.41],
        depth_iqr=1.0,
        representative_depth=2.0,
        valid_depth_ratio=1.0,
    )
    assert not result.applicable
    assert result.applicability_status == "unstable_depth"


def _strict_inputs():
    resolver = load_dimension_aligned_prior_resolver(
        PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml"
    )
    rules = load_projected_measurement_rules(
        PROJECT_ROOT / "configs/projected_measurement_rules.yaml"
    )
    person = ObjectObservationJSON(
        object_id="p",
        label="person",
        mask_area=10_000,
        frame_area=40_000,
        depth=4.0,
        confidence=0.9,
        bbox=[20, 10, 80, 190],
        track_id="person_track",
    )
    cup = ObjectObservationJSON(
        object_id="c",
        label="cup",
        mask_area=1_200,
        frame_area=40_000,
        depth=2.0,
        confidence=0.9,
        bbox=[120, 100, 150, 140],
        track_id="cup_track",
    )
    frame = FrameObservationJSON(
        frame_index=0,
        frame_id="f0",
        width=200,
        height=200,
        objects=[person, cup],
    )
    cup_gate = CupHeightApplicabilityGate().evaluate(
        cup.bbox or (),
        (200, 200),
        cup.confidence,
        projection_history=[0.2, 0.201],
        depth_iqr=0.1,
        representative_depth=2.0,
        valid_depth_ratio=1.0,
    )
    return resolver, rules, frame, person, cup, cup_gate


def test_failed_person_gate_produces_nan_not_zero() -> None:
    resolver, rules, frame, person, cup, cup_gate = _strict_inputs()
    failed_person = _person_result(
        [point for point in _upright_points() if "ankle" not in point.keypoint_name]
    )
    result = compute_dimension_aligned_rsd(
        frame,
        person,
        cup,
        resolver,
        rules,
        applicability_a=failed_person,
        applicability_b=cup_gate,
        require_applicability=True,
    )
    assert not result["valid"]
    assert result["skip_reason"] == "person_insufficient_keypoints"
    assert math.isnan(float(result["rsd_ratio"]))
    assert math.isnan(float(result["rsd_log"]))
    assert float(result["rsd_log"]) != 0.0


def test_passed_person_and_cup_gates_allow_rsd() -> None:
    resolver, rules, frame, person, cup, cup_gate = _strict_inputs()
    person_gate = _person_result(_upright_points())
    result = compute_dimension_aligned_rsd(
        frame,
        person,
        cup,
        resolver,
        rules,
        applicability_a=person_gate,
        applicability_b=cup_gate,
        require_applicability=True,
    )
    assert result["valid"]
    assert math.isfinite(float(result["rsd_log"]))


def test_old_observation_json_remains_readable() -> None:
    old = {
        "object_id": "old",
        "label": "person",
        "mask_area": 10,
        "frame_area": 100,
        "depth": 2,
    }
    obj = ObjectObservationJSON.from_dict(old)
    assert obj.keypoints_2d is None
    assert obj.pose_provider is None
    assert obj.person_track_id is None


def test_keypoint_fields_round_trip_for_future_tracks() -> None:
    payload = {
        "object_id": "p",
        "label": "person",
        "mask_area": 10,
        "frame_area": 100,
        "depth": 2,
        "keypoints_2d": [_upright_points()[0].to_dict()],
        "keypoint_confidences": [0.9],
        "pose_provider": "mock_keypoints",
        "keypoint_frame_index": 4,
        "person_track_id": "track_1",
    }
    obj = ObjectObservationJSON.from_dict(json.loads(json.dumps(payload)))
    assert obj.keypoints_2d == payload["keypoints_2d"]
    assert obj.keypoint_frame_index == 4
    assert obj.person_track_id == "track_1"
    assert issubclass(MockKeypointProvider, BaseKeypointProvider)


def test_strict_v1_v2_hashes_remain_frozen() -> None:
    assert _hash(PROJECT_ROOT / "configs/scale_priors_strict_v1.yaml") == V1_HASH
    assert _hash(PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml") == V2_HASH


def test_real3_does_not_appear_in_frozen_prior_config() -> None:
    text = (PROJECT_ROOT / "configs/scale_priors_strict_v2.yaml").read_text(
        encoding="utf-8"
    )
    assert "real_3" not in text
    assert "outputs/evaluation" not in text


def test_real_pose_provider_rejects_local_real3_seated_person_when_available() -> None:
    weight = PROJECT_ROOT / "checkpoints/yolov8n-pose.pt"
    directory = (
        PROJECT_ROOT
        / "outputs/evaluation/pilot_6video/videos/real_3/associated_observations"
    )
    paths = sorted(directory.rglob("*.json"))
    if not weight.exists() or not paths:
        pytest.skip("Local pose weight or real_3 observation is unavailable.")
    clip = load_clip_observation(paths[0])
    frame = next(frame for frame in clip.frames if frame.frame_index == 10)
    person = next(obj for obj in frame.objects if obj.label == "person")
    provider = RealHumanKeypointProvider(weight, device="cpu")
    prediction = provider.predict(frame.image_path or "", person.bbox or (), person.label)
    result = PersonHeightApplicabilityGate().evaluate(
        person.bbox or (),
        (frame.width, frame.height),
        prediction,
        person.confidence,
    )
    assert prediction.status == "ok"
    assert len(prediction.keypoints) == 17
    assert not result.applicable
    assert result.applicability_status in {
        "insufficient_keypoints", "sitting_or_bending", "incomplete_body"
    }
