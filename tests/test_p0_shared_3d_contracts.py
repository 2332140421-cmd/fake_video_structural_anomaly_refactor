from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic3d.build_observations import _replace_object_depth
from semantic3d.depth_provider import (
    DepthRepresentation,
    DepthScaleStatus,
    LegacyDepthProviderAdapter,
    LargerValueMeans,
    RealDepthProvider,
)
from semantic3d.depth_temporal_consistency import (
    compute_depth_temporal_residual,
    depth_transition_evidence,
    r_depth_cons_2p5d,
)
from semantic3d.geometry.camera import CameraObservation, CoordinateConvention
from semantic3d.io import load_frame_observation
from semantic3d.multilevel_residuals import (
    ObjectMaskObservation,
    build_object_level_residual_evidence,
    build_object_level_residuals,
)
from semantic3d.observations import ObjectObservationJSON
from semantic3d.scale_depth import (
    ObjectObservation,
    ScalePrior,
    rsd_2d_coarse,
    scale_depth_residual,
)
from semantic3d.shared_3d_observation import (
    GeometryScaleStatus,
    Object3DObservation,
    Point3DObservation,
    Shared3DFrameObservation,
    VisibilityStatus,
)
from semantic3d.validity import MissingReason, ResidualEvidence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STRICT_HASHES = {
    "configs/scale_priors_strict_v1.yaml": (
        "e86466e19fe3e1663fa855fdf73843cf7ab8b5b6c8fa8771aa46172f6b726a6b"
    ),
    "configs/scale_priors_strict_v2.yaml": (
        "3c55c8eb42e19d2447b00794085d8e6fa233c37a2071ef51f23b447d23ef268b"
    ),
}


def _write_image(path: Path, width: int = 8, height: int = 6) -> None:
    image = np.full((height, width, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _object(object_id: str, depth: float = 5.0) -> ObjectObservationJSON:
    return ObjectObservationJSON(
        object_id=object_id,
        label="person",
        mask_area=100.0,
        frame_area=1_000.0,
        depth=depth,
        confidence=0.9,
        bbox=[1.0, 2.0, 11.0, 22.0],
        mask_path="mask.npy",
        track_id="trk_1",
        canonical_label="person",
        keypoints_2d=[{"keypoint_name": "nose", "x": 4.0, "y": 5.0}],
        keypoint_confidences=[0.8],
        pose_provider="mock_pose",
        keypoint_frame_index=3,
        person_track_id="person_trk_1",
        provenance={"detector": "unit_test"},
        quality=0.75,
        metadata={"source": "fixture"},
    )


def _relative_point(point_id: str, xyz: tuple[float, float, float]) -> Point3DObservation:
    return Point3DObservation(
        point_id=point_id,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        coordinate_frame="camera",
        scale_status=GeometryScaleStatus.RELATIVE_3D,
        confidence=0.9,
        valid=True,
    )


def _normalized_point(point_id: str, xyz: tuple[float, float, float]) -> Point3DObservation:
    return Point3DObservation(
        point_id=point_id,
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        coordinate_frame="object_normalized",
        scale_status=GeometryScaleStatus.NORMALIZED_SHAPE,
        confidence=0.9,
        valid=True,
    )


def test_geometry_depth_and_visualization_depth_are_separate(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path)
    raw_inverse = np.asarray([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    provider = RealDepthProvider(
        invert_depth=True,
        pipeline_instance=lambda _image: {"predicted_depth": raw_inverse},
    )

    observation = provider.predict_observation(frame_path, frame_index=7)

    geometry = observation.require_geometry_depth()
    assert observation.depth_representation == DepthRepresentation.RELATIVE_DEPTH
    assert observation.scale_status == DepthScaleStatus.RELATIVE_PER_FRAME
    assert observation.larger_value_means == LargerValueMeans.FARTHER
    assert observation.frame_index == 7
    assert observation.visualization_depth is not None
    assert geometry.shape == observation.visualization_depth.shape == (6, 8)
    assert not np.allclose(geometry, observation.visualization_depth)
    assert observation.raw_model_output is not None
    assert observation.raw_model_output.shape == raw_inverse.shape
    assert observation.metadata["metric_depth"] is False
    assert observation.metadata["conversion"] == "reciprocal_inverse_to_relative"


def test_legacy_normalized_depth_cannot_enter_geometry(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path)

    class LegacyProvider:
        def predict_depth(self, _frame_path: Path) -> np.ndarray:
            return np.linspace(1.0, 10.0, 48, dtype=np.float32).reshape(6, 8)

    observation = LegacyDepthProviderAdapter(LegacyProvider()).predict_observation(
        frame_path, frame_index=0
    )

    assert not observation.valid
    assert observation.metadata["legacy_normalized_depth"] is True
    with pytest.raises(ValueError, match="legacy_normalized_depth|invalid"):
        observation.require_geometry_depth()


def test_real_provider_legacy_array_remains_affine_1_to_10(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path)
    provider = RealDepthProvider(
        invert_depth=True,
        pipeline_instance=lambda _image: {
            "predicted_depth": np.asarray([[1.0, 2.0], [3.0, 5.0]])
        },
    )

    legacy = provider.predict_depth(frame_path)
    metadata = provider.legacy_depth_metadata()

    assert float(np.nanmin(legacy)) == pytest.approx(1.0)
    assert float(np.nanmax(legacy)) == pytest.approx(10.0)
    assert metadata["legacy_normalized_depth"] is True
    assert metadata["depth_representation"] == "legacy_normalized_depth"


def test_missing_camera_is_invalid_without_fake_matrices() -> None:
    camera = CameraObservation.missing(1280, 720, MissingReason.MISSING_INTRINSICS)

    assert not camera.valid
    assert camera.K is None
    assert camera.T_world_camera is None
    assert camera.T_camera_world is None
    assert camera.missing_reason == "missing_intrinsics"


def test_identity_matrix_cannot_impersonate_intrinsics() -> None:
    with pytest.raises(ValueError, match="identity matrix"):
        CameraObservation(
            K=np.eye(3),
            distortion=np.zeros(5),
            T_world_camera=np.eye(4),
            T_camera_world=np.eye(4),
            image_width=640,
            image_height=480,
            coordinate_convention=CoordinateConvention.OPENCV,
            intrinsics_source="fake_identity",
            pose_source="known_static_camera",
            valid=True,
            quality=1.0,
        )


def test_missing_object3d_has_no_zero_filled_points() -> None:
    obj = Object3DObservation.missing(
        "video", 3, "person", "det_3", track_id="trk_1"
    )

    assert not obj.valid
    assert obj.center_3d is None
    assert obj.boundary_points_3d == ()
    assert obj.keypoints_3d == ()
    assert obj.observed_scale_3d is None


def test_observed_scale_and_normalized_structure_are_separate() -> None:
    obj = Object3DObservation(
        video_id="video",
        frame_index=1,
        track_id="trk_1",
        semantic_label="person",
        canonical_label="person",
        center_3d=_relative_point("center", (1.0, 2.0, 5.0)),
        boundary_points_3d=(),
        keypoints_3d=(),
        structure_points_3d=(),
        observed_scale_3d=1.8,
        normalized_structure_points=(
            _normalized_point("shape_1", (0.0, 0.0, 0.0)),
            _normalized_point("shape_2", (1.0, 0.0, 0.0)),
        ),
        scale_status=GeometryScaleStatus.RELATIVE_3D,
        visibility=VisibilityStatus.VISIBLE,
        reconstruction_quality=0.8,
        valid=True,
        missing_reason="",
        source_object_2d_id="det_1",
    )

    assert obj.require_observed_scale() == pytest.approx(1.8)
    assert obj.normalized_structure_points[1].as_array()[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Metric scale"):
        obj.require_observed_scale(metric=True)


def test_normalized_points_cannot_be_declared_as_observed_object_scale() -> None:
    with pytest.raises(ValueError, match="Observed 3D scale"):
        Object3DObservation(
            video_id="video",
            frame_index=1,
            track_id=None,
            semantic_label="cup",
            canonical_label="cup",
            center_3d=_relative_point("center", (0.0, 0.0, 2.0)),
            boundary_points_3d=(),
            keypoints_3d=(),
            structure_points_3d=(),
            observed_scale_3d=1.0,
            normalized_structure_points=(),
            scale_status=GeometryScaleStatus.NORMALIZED_SHAPE,
            visibility=VisibilityStatus.VISIBLE,
            reconstruction_quality=0.8,
            valid=True,
            missing_reason="",
            source_object_2d_id="cup_1",
        )


def test_shared_frame_can_represent_missing_3d_without_fabrication(tmp_path: Path) -> None:
    frame_path = tmp_path / "frame.png"
    _write_image(frame_path)
    depth = RealDepthProvider(
        invert_depth=True,
        pipeline_instance=lambda _image: {
            "predicted_depth": np.asarray([[1.0, 2.0], [3.0, 4.0]])
        },
    ).predict_observation(frame_path, 0)
    camera = CameraObservation.missing(8, 6, MissingReason.MISSING_CAMERA)
    frame = Shared3DFrameObservation.missing(
        "video", 0, 8, 6, camera, depth, reason=MissingReason.MISSING_CAMERA
    )

    assert not frame.valid
    assert frame.objects == ()
    assert frame.camera.K is None


def test_missing_residual_evidence_is_nan() -> None:
    evidence = ResidualEvidence.missing("r_track_3d", MissingReason.NOT_OBSERVED)

    assert not evidence.valid
    assert math.isnan(evidence.value)


def test_valid_normal_residual_can_be_zero() -> None:
    evidence = ResidualEvidence.observed("r_semantic_size_3d", 0.0, quality=0.7)

    assert evidence.valid
    assert evidence.value == pytest.approx(0.0)


def test_legacy_residual_aliases_preserve_values() -> None:
    a = ObjectObservation("a", "ball", 100.0, 10_000.0, 2.0)
    b = ObjectObservation("b", "person", 2_500.0, 10_000.0, 10.0)
    priors = {
        "ball": ScalePrior(0.2, 0.3),
        "person": ScalePrior(1.5, 2.0),
    }

    legacy = scale_depth_residual(a, b, priors)
    explicit = rsd_2d_coarse(a, b, priors)

    assert explicit == legacy


def test_legacy_depth_cons_zero_missing_has_nan_evidence_adapter() -> None:
    previous = _object("obj_f0", depth=5.0)
    current = _object("obj_f1", depth=0.0)

    legacy = compute_depth_temporal_residual(previous, current, 5.0, 5.0)
    renamed = r_depth_cons_2p5d(previous, current, 5.0, 5.0)
    evidence = depth_transition_evidence(legacy)

    assert legacy == renamed
    assert legacy.residual == pytest.approx(0.0)
    assert not evidence.valid
    assert math.isnan(evidence.value)


def test_evidence_aware_multilevel_missing_is_not_zero() -> None:
    obj = ObjectMaskObservation("obj", "person", np.ones((4, 4), dtype=bool))

    legacy = build_object_level_residuals([obj])
    evidence = build_object_level_residual_evidence([obj])

    assert legacy[0].flow == pytest.approx(0.0)
    assert not evidence[0].flow.valid
    assert math.isnan(evidence[0].flow.value)
    assert not evidence[0].track.valid
    assert math.isnan(evidence[0].track.value)


def test_depth_replacement_preserves_all_object_metadata() -> None:
    original = _object("obj_f3")

    replaced = _replace_object_depth(original, 8.5)

    assert replaced.depth == pytest.approx(8.5)
    for field_name in (
        "track_id",
        "canonical_label",
        "keypoints_2d",
        "keypoint_confidences",
        "pose_provider",
        "keypoint_frame_index",
        "person_track_id",
        "mask_path",
        "provenance",
        "quality",
        "metadata",
    ):
        assert getattr(replaced, field_name) == getattr(original, field_name)


def test_old_observation_json_without_p0_fields_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "old_frame.json"
    path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "frame_id": "old",
                "width": 16,
                "height": 12,
                "objects": [
                    {
                        "object_id": "old_obj",
                        "label": "person",
                        "mask_area": 20,
                        "frame_area": 192,
                        "depth": 5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    frame = load_frame_observation(path)

    assert frame.objects[0].provenance == {}
    assert frame.objects[0].quality is None
    assert frame.objects[0].metadata == {}


def test_strict_v1_v2_hashes_are_unchanged() -> None:
    for relative_path, expected_hash in STRICT_HASHES.items():
        content = (PROJECT_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(content).hexdigest() == expected_hash
