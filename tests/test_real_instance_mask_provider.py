from __future__ import annotations

from pathlib import Path
import hashlib

import cv2
import numpy as np

from semantic3d.observations import FrameObservationJSON, ObjectObservationJSON
from semantic3d.occlusion import RealInstanceMaskProvider


def _frame(tmp_path: Path) -> FrameObservationJSON:
    image_path = tmp_path / "frame.png"
    cv2.imwrite(str(image_path), np.zeros((64, 64, 3), dtype=np.uint8))
    return FrameObservationJSON(
        0, "f0", 64, 64,
        [ObjectObservationJSON("det_1", "person", 400.0, 4096.0, 5.0, bbox=[10, 10, 40, 50], track_id="person_1")],
        image_path=str(image_path),
    )


def test_missing_real_segmentation_weights_are_explicit(tmp_path: Path) -> None:
    provider = RealInstanceMaskProvider(model_path=tmp_path / "missing-seg.pt")
    result = provider.predict(video_id="v", frame=_frame(tmp_path))[0]
    assert not provider.available
    assert not result.valid and result.visible_mask is None
    assert "weights_missing" in result.missing_reason


def test_real_provider_callback_mask_is_visible_not_filled_bbox(tmp_path: Path) -> None:
    def predictor(frame, obj):
        mask = np.zeros((frame.height, frame.width), dtype=bool)
        cv2.circle(mask.view(np.uint8), (25, 30), 10, 1, -1)
        return mask

    result = RealInstanceMaskProvider(predictor=predictor, model_name="frozen_test_segmenter").predict(video_id="v", frame=_frame(tmp_path))[0]
    assert result.valid and result.is_visible_mask and not result.is_amodal_mask
    assert result.amodal_mask is None
    assert result.mask_area < (40 - 10) * (50 - 10)
    assert result.metadata["formal_mask_evidence"] is True
    assert result.metadata["truth_label_used"] is False


def test_frozen_segmentation_weight_hash_and_mask_output_are_recorded(
    tmp_path: Path, monkeypatch,
) -> None:
    import torch
    import ultralytics

    weight = tmp_path / "seg.pt"
    weight.write_bytes(b"frozen-segmentation-test-weight")

    class Boxes:
        cls = torch.tensor([0.0])
        conf = torch.tensor([0.9])
        id = None

        def __len__(self):
            return 1

    class Masks:
        data = torch.zeros((1, 64, 64))

    Masks.data[0, 12:48, 18:38] = 1.0

    class Result:
        boxes = Boxes()
        masks = Masks()
        names = {0: "person"}

    class SegmentModel:
        task = "segment"

        def predict(self, **kwargs):
            return [Result()]

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: SegmentModel())
    provider = RealInstanceMaskProvider(model_path=weight)
    candidates = provider.predict_candidates(_frame(tmp_path))
    assert provider.available and len(candidates) == 1
    assert provider.model_metadata.output_contains_instance_masks
    assert provider.model_metadata.weight_sha256 == hashlib.sha256(weight.read_bytes()).hexdigest()
    assert candidates[0].weight_sha256 == provider.model_metadata.weight_sha256
    assert np.count_nonzero(candidates[0].visible_mask) < 30 * 40


def test_detection_task_weights_cannot_enter_segmentation_path(tmp_path: Path, monkeypatch) -> None:
    import ultralytics

    weight = tmp_path / "detect.pt"
    weight.write_bytes(b"detection-only")

    class DetectionModel:
        task = "detect"

    monkeypatch.setattr(ultralytics, "YOLO", lambda _: DetectionModel())
    provider = RealInstanceMaskProvider(model_path=weight)
    assert not provider.available
    assert provider.unavailable_reason == "instance_segmentation_model_task_mismatch"
