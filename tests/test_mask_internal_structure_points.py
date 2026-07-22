from __future__ import annotations

import cv2
import numpy as np

from semantic3d.occlusion import (
    InstanceMaskObservation,
    eroded_mask_interior,
    select_formal_mask_internal_points,
    track_formal_mask_internal_points,
)


def _observation(frame: int, mask: np.ndarray, *, legacy: bool = False) -> InstanceMaskObservation:
    return InstanceMaskObservation.from_visible_mask(
        video_id="v", frame_index=frame, object_track_id="cup_1",
        semantic_label="cup", mask=mask, confidence=0.9,
        source_provider="real_instance_mask_provider",
        metadata={"formal_mask_evidence": not legacy, "legacy_bbox_fallback": legacy},
    )


def test_ordinary_structure_points_are_inside_eroded_formal_mask() -> None:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    mask = np.zeros((80, 100), dtype=bool)
    mask[15:65, 20:80] = True
    for y in range(25, 61, 10):
        for x in range(30, 76, 10):
            cv2.rectangle(image, (x, y), (x + 3, y + 3), (255, 255, 255), -1)
    points = select_formal_mask_internal_points(image, _observation(0, mask))
    interior = eroded_mask_interior(mask)
    assert len(points) > 0
    assert all(interior[int(round(y)), int(round(x))] for x, y in points)


def test_bbox_diagnostic_cannot_seed_formal_structure_points() -> None:
    image = np.full((64, 64, 3), 127, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    mask[8:56, 8:56] = True
    assert len(select_formal_mask_internal_points(image, _observation(0, mask, legacy=True))) == 0


def test_current_mask_only_validates_klt_prediction() -> None:
    images, masks = {}, []
    for frame in range(3):
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:54, 10 + frame:54 + frame] = True
        cv2.rectangle(image, (22 + frame, 22), (28 + frame, 28), (255, 255, 255), -1)
        cv2.rectangle(image, (38 + frame, 38), (44 + frame, 44), (255, 255, 255), -1)
        images[frame] = image
        masks.append(_observation(frame, mask))
    points = track_formal_mask_internal_points(images, masks, max_points=8)
    assert any(point.valid and point.frame_index == 2 for point in points)
    assert all(not point.metadata.get("current_mask_used_for_prediction", True) for point in points if point.valid)
