from __future__ import annotations

from _bootstrap import ensure_project_test_environment

ensure_project_test_environment(__file__)

import json
from pathlib import Path

from semantic3d.io import load_clip_observation
from semantic3d.object_association import (
    ObjectAssociator,
    bbox_area_ratio,
    bbox_iou,
    deduplicate_frames_by_index,
    normalize_object_label,
    normalized_center_distance,
)
from semantic3d.observations import (
    ClipObservationJSON,
    FrameObservationJSON,
    ObjectObservationJSON,
)


FRAME_AREA = 640.0 * 480.0


def _object(
    object_id: str,
    label: str,
    bbox: list[float],
    depth: float = 5.0,
) -> ObjectObservationJSON:
    x1, y1, x2, y2 = bbox
    return ObjectObservationJSON(
        object_id=object_id,
        label=label,
        bbox=bbox,
        mask_area=(x2 - x1) * (y2 - y1),
        frame_area=FRAME_AREA,
        depth=depth,
        confidence=1.0,
    )


def _frame(index: int, objects: list[ObjectObservationJSON]) -> FrameObservationJSON:
    return FrameObservationJSON(
        frame_index=index,
        frame_id=f"frame_{index:06d}",
        width=640,
        height=480,
        objects=objects,
    )


def test_bbox_helpers() -> None:
    box_a = [10, 10, 50, 50]
    box_b = [20, 20, 60, 60]

    assert bbox_iou(box_a, box_b) > 0.0
    assert normalized_center_distance(box_a, box_b, 640, 480) < 0.1
    assert bbox_area_ratio(box_a, box_b) == 1.0
    assert normalize_object_label("sports ball") == "ball"


def test_same_label_high_iou_gets_same_track_id() -> None:
    frames = [
        _frame(0, [_object("person_f0", "person", [100, 100, 180, 260])]),
        _frame(1, [_object("person_f1", "person", [104, 103, 184, 263])]),
    ]

    associated = ObjectAssociator().associate(frames)

    assert associated[0].objects[0].track_id == associated[1].objects[0].track_id
    assert associated[0].objects[0].canonical_label == "person"


def test_different_label_not_associated() -> None:
    frames = [
        _frame(0, [_object("person_f0", "person", [100, 100, 180, 260])]),
        _frame(1, [_object("car_f1", "car", [104, 103, 184, 263])]),
    ]

    associated = ObjectAssociator().associate(frames)

    assert associated[0].objects[0].track_id != associated[1].objects[0].track_id


def test_center_distance_too_large_not_associated() -> None:
    frames = [
        _frame(0, [_object("person_f0", "person", [100, 100, 180, 260])]),
        _frame(1, [_object("person_f1", "person", [420, 320, 500, 470])]),
    ]

    associated = ObjectAssociator(center_distance_threshold=0.1).associate(frames)

    assert associated[0].objects[0].track_id != associated[1].objects[0].track_id


def test_new_object_creates_new_track() -> None:
    frames = [
        _frame(0, [_object("person_f0", "person", [100, 100, 180, 260])]),
        _frame(
            1,
            [
                _object("person_f1", "person", [104, 103, 184, 263]),
                _object("cup_f1", "cup", [300, 200, 330, 245]),
            ],
        ),
    ]

    associated = ObjectAssociator().associate(frames)

    assert associated[0].objects[0].track_id == associated[1].objects[0].track_id
    assert associated[1].objects[1].track_id != associated[1].objects[0].track_id


def test_same_track_same_frame_is_unique() -> None:
    frames = [
        _frame(
            0,
            [
                _object("person_a_f0", "person", [100, 100, 180, 260]),
                _object("person_b_f0", "person", [300, 100, 380, 260]),
            ],
        ),
        _frame(
            1,
            [
                _object("person_a_f1", "person", [104, 103, 184, 263]),
                _object("person_b_f1", "person", [304, 103, 384, 263]),
            ],
        ),
    ]

    associated = ObjectAssociator().associate(frames)

    for frame in associated:
        track_ids = [obj.track_id for obj in frame.objects]
        assert len(track_ids) == len(set(track_ids))


def test_empty_frame_does_not_crash() -> None:
    frames = [
        _frame(0, [_object("person_f0", "person", [100, 100, 180, 260])]),
        _frame(1, []),
        _frame(2, [_object("person_f2", "person", [106, 104, 186, 264])]),
    ]

    associated = ObjectAssociator(max_frame_gap=1).associate(frames)

    assert len(associated) == 3
    assert associated[1].objects == []


def test_duplicate_frame_index_is_deduplicated() -> None:
    frames = [
        _frame(0, [_object("person_f0_a", "person", [100, 100, 180, 260])]),
        _frame(0, [_object("person_f0_b", "person", [300, 100, 380, 260])]),
        _frame(1, [_object("person_f1", "person", [104, 103, 184, 263])]),
    ]

    deduped = deduplicate_frames_by_index(frames)
    associated = ObjectAssociator().associate(frames)

    assert [frame.frame_index for frame in deduped] == [0, 1]
    assert len(associated) == 2
    assert associated[0].objects[0].object_id == "person_f0_a"


def test_old_observation_json_without_track_id_can_be_read(tmp_path: Path) -> None:
    path = tmp_path / "old_clip.json"
    data = {
        "clip_id": "clip_old",
        "video_id": "video_old",
        "frame_indices": [0],
        "frames": [
            {
                "frame_index": 0,
                "frame_id": "frame_000000",
                "width": 640,
                "height": 480,
                "objects": [
                    {
                        "object_id": "person_f0",
                        "label": "person",
                        "mask_area": 12000,
                        "frame_area": FRAME_AREA,
                        "depth": 5.0,
                    }
                ],
            }
        ],
    }
    path.write_text(json.dumps(data), encoding="utf-8")

    clip = load_clip_observation(path)

    assert isinstance(clip, ClipObservationJSON)
    assert clip.frames[0].objects[0].track_id is None
    assert clip.frames[0].objects[0].canonical_label is None
