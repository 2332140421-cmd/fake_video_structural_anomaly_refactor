"""P4-C3B-M1 metric provider and smoke tests without large-model inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

from semantic3d.depth_provider import (
    DepthRepresentation,
    DepthScaleStatus,
    LargerValueMeans,
)
from semantic3d.method_completion.metric_depth_adapters import (
    MetricProviderRuntimeError,
    UniDepthV2Adapter,
    sha256_file,
)
from semantic3d.method_completion.metric_provider_smoke import (
    build_smoke_frame_specs,
    run_metric_provider_smoke,
)
from semantic3d.method_completion.metric_scale import (
    MetricDepthDefinition,
    MetricScaleStatus,
)


class _FakeUniDepthModel:
    def __init__(self) -> None:
        self.resolution_level = -1

    def to(self, device: str) -> "_FakeUniDepthModel":
        self.device = device
        return self

    def eval(self) -> "_FakeUniDepthModel":
        return self

    def half(self) -> "_FakeUniDepthModel":
        return self

    def infer(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        _, height, width = image.shape
        y = torch.linspace(1.0, 2.0, height).view(1, 1, height, 1)
        depth = y.expand(1, 1, height, width).clone()
        radius = depth * 1.1
        uncertainty = torch.linspace(0.2, 2.0, height * width).reshape(
            1, 1, height, width
        )
        K = torch.tensor(
            [[[700.0, 0.0, width / 2.0], [0.0, 710.0, height / 2.0], [0.0, 0.0, 1.0]]]
        )
        return {
            "depth": depth,
            "radius": radius,
            "confidence": uncertainty,
            "intrinsics": K,
        }


def _image(path: Path, height: int = 24, width: int = 32) -> Path:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, 1] = 128
    assert cv2.imwrite(str(path), image)
    return path


def _adapter(tmp_path: Path, factory=None) -> UniDepthV2Adapter:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"verified-local-test-weight")
    adapter = UniDepthV2Adapter(
        weights_path=weight,
        expected_weight_sha256=sha256_file(weight),
        device="cpu",
        precision="fp32",
        model_factory=factory or (lambda _: _FakeUniDepthModel()),
    )
    adapter.expected_module = "numpy"
    return adapter


def _video(path: Path, frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (32, 24)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((24, 32, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_unidepth_adapter_returns_canonical_metric_depth(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    result = adapter.predict_frame(_image(tmp_path / "frame.jpg"), frame_index=4)

    observation = result.depth_observation
    assert observation.depth_representation == DepthRepresentation.METRIC_DEPTH
    assert observation.scale_status == DepthScaleStatus.METRIC_CALIBRATED
    assert observation.larger_value_means == LargerValueMeans.FARTHER
    assert observation.metadata["metric_scale_status"] == MetricScaleStatus.MODEL_PREDICTED.value
    assert observation.metadata["sensor_ground_truth"] is False
    assert result.raw_depth_definition == MetricDepthDefinition.RAY_DISTANCE.value
    assert result.standardized_depth_definition == MetricDepthDefinition.Z_DEPTH.value
    assert np.all(observation.raw_model_output >= observation.depth_map)
    assert observation.require_geometry_depth(require_metric=True).shape == (24, 32)


def test_model_predicted_intrinsics_preserve_original_coordinates(tmp_path: Path) -> None:
    result = _adapter(tmp_path).predict_frame(_image(tmp_path / "wide.jpg", 20, 40))
    camera = result.camera_observation
    assert camera.valid
    assert camera.intrinsics_source == "model_predicted"
    assert camera.pose_valid is False
    assert camera.quality == 0.0
    assert np.isnan(result.intrinsics_confidence)
    assert camera.image_width == 40 and camera.image_height == 20
    assert camera.K[0, 0] == pytest.approx(700.0)
    assert camera.K[1, 1] == pytest.approx(710.0)
    assert camera.K[0, 2] == pytest.approx(20.0)
    assert camera.K[1, 2] == pytest.approx(10.0)


def test_unidepth_loading_is_forced_offline(tmp_path: Path) -> None:
    observed: dict[str, str | None] = {}

    def factory(_: Path) -> _FakeUniDepthModel:
        observed.update(
            {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "WANDB_MODE": os.environ.get("WANDB_MODE"),
            }
        )
        return _FakeUniDepthModel()

    adapter = _adapter(tmp_path, factory=factory)
    adapter.predict_frame(_image(tmp_path / "offline.jpg"))
    assert observed == {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "WANDB_MODE": "offline",
    }


def test_weight_sha_is_required_and_verified(tmp_path: Path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weight")
    adapter = UniDepthV2Adapter(
        weight,
        expected_weight_sha256="0" * 64,
        model_factory=lambda _: _FakeUniDepthModel(),
    )
    adapter.expected_module = "numpy"
    with pytest.raises(MetricProviderRuntimeError, match="does not match") as error:
        adapter.predict_frame(_image(tmp_path / "sha.jpg"))
    assert error.value.reason == "metric_provider_weight_sha256_mismatch"


def test_missing_dependency_has_stable_failure(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    adapter.expected_module = "semantic3d_dependency_that_does_not_exist"
    with pytest.raises(MetricProviderRuntimeError) as error:
        adapter.predict_frame(_image(tmp_path / "missing.jpg"))
    assert error.value.reason == "metric_provider_dependency_missing"


def test_cpu_fp16_is_rejected() -> None:
    with pytest.raises(ValueError, match="not supported on CPU"):
        UniDepthV2Adapter(device="cpu", precision="fp16")


def test_repeated_fake_inference_is_identical(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    frame = _image(tmp_path / "repeat.jpg")
    first = adapter.predict_frame(frame).depth_observation.depth_map
    second = adapter.predict_frame(frame).depth_observation.depth_map
    np.testing.assert_array_equal(first, second)


def test_smoke_frame_order_is_deterministic(tmp_path: Path) -> None:
    config = {
        "inputs": [
            {"source_name": "z", "video_path": str(tmp_path / "z.mp4"), "frame_indices": [2, 0]},
            {"source_name": "a", "video_path": str(tmp_path / "a.mp4"), "frame_indices": [1, 0]},
        ]
    }
    specs = build_smoke_frame_specs(config)
    assert [(item.source_name, item.frame_index) for item in specs] == [
        ("a", 0), ("a", 1), ("z", 0), ("z", 2)
    ]


def test_small_smoke_writes_required_artifacts(tmp_path: Path) -> None:
    first_video = tmp_path / "source_a.mp4"
    second_video = tmp_path / "source_b.mp4"
    _video(first_video)
    _video(second_video)
    config = {
        "schema_version": "semantic3d_p4c3b_metric_provider_smoke_v1",
        "stage": "P4-C3B-M1",
        "provider": {"name": "unidepth_v2"},
        "inputs": [
            {"source_name": "source_a", "video_path": str(first_video), "frame_indices": [0, 1]},
            {"source_name": "source_b", "video_path": str(second_video), "frame_indices": [0, 1]},
        ],
        "reference_observations": {"dataset_root": str(tmp_path / "missing_reference")},
        "quality_audit": {
            "boundary_fraction": 0.05,
            "rank_sample_stride": 2,
            "repeat_first_frame": True,
        },
    }
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    output_dir = tmp_path / "smoke_output"
    report = run_metric_provider_smoke(
        config_path=config_path,
        output_dir=output_dir,
        adapter=_adapter(tmp_path),
    )

    expected = {
        "provider_environment.json",
        "provider_weight_manifest.json",
        "metric_depth_frame_manifest.csv",
        "metric_depth_numeric_audit.json",
        "intrinsics_audit.json",
        "temporal_scale_drift.json",
        "object_region_depth_audit.csv",
        "provider_failure_audit.json",
        "validation_report.json",
        "METRIC_PROVIDER_SMOKE_REPORT.md",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})
    assert report["metric_provider_real_inference_executed"] is True
    assert report["metric_depth_output_verified"] is True
    assert report["ready_for_full_984_frame_build"] is False
    assert report["method_effectiveness_established"] is False
    numeric = json.loads((output_dir / "metric_depth_numeric_audit.json").read_text())
    assert numeric["summary"]["total_nan"] == 0
    assert numeric["repeatability"]["deterministic_within_1e_5"] is True
