#!/usr/bin/env python3
"""Smoke-test Ultralytics YOLO detection on a local video."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

import cv2  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run a small YOLO detection smoke test on video frames."
    )
    parser.add_argument("--video_path", required=True, help="Path to the input video.")
    parser.add_argument(
        "--model_path",
        default=str(PROJECT_ROOT / "checkpoints" / "yolov8n.pt"),
        help="Path to YOLO weights. If missing, tries Ultralytics 'yolov8n.pt'.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(PROJECT_ROOT / "outputs" / "yolo_smoke_test"),
        help="Directory for annotated frame images.",
    )
    parser.add_argument("--max_frames", type=int, default=5)
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def bbox_area(bbox: Sequence[float]) -> float:
    """Return area of an xyxy bounding box."""

    x1, y1, x2, y2 = [float(value) for value in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def load_yolo_model(model_path: Path) -> Any:
    """Load YOLO from a local path, or try the small yolov8n.pt fallback."""

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Run: python -m pip install ultralytics"
        ) from exc

    if model_path.exists():
        print(f"Loading local YOLO weights: {model_path}")
        return YOLO(str(model_path))

    print(
        f"Local model_path does not exist: {model_path}\n"
        "Trying Ultralytics fallback model name: yolov8n.pt"
    )
    try:
        return YOLO("yolov8n.pt")
    except Exception as exc:
        raise RuntimeError(
            "Could not load yolov8n.pt automatically. If the environment cannot "
            f"download weights, manually place yolov8n.pt at: {model_path}"
        ) from exc


def draw_detection(
    frame: Any,
    label: str,
    confidence: float,
    bbox: Sequence[float],
) -> None:
    """Draw one detection on a BGR frame."""

    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 80, 240), 2)
    text = f"{label} {confidence:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(16, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (40, 80, 240),
        2,
        cv2.LINE_AA,
    )


def run_smoke_test(
    video_path: Path,
    model_path: Path,
    output_dir: Path,
    max_frames: int = 5,
    confidence_threshold: float = 0.3,
    device: str = "cpu",
) -> int:
    """Run YOLO over the first max_frames frames and save annotated images."""

    if max_frames <= 0:
        raise ValueError(f"max_frames must be > 0, got {max_frames}.")
    if confidence_threshold < 0 or confidence_threshold > 1:
        raise ValueError(
            f"confidence_threshold must be in [0, 1], got {confidence_threshold}."
        )
    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    model = load_yolo_model(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video with OpenCV: {video_path}")

    frame_index = 0
    total_detections = 0
    try:
        while frame_index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            results = model.predict(
                source=frame,
                conf=confidence_threshold,
                device=device,
                verbose=False,
            )
            detections_this_frame = 0
            for result in results:
                names = getattr(result, "names", {})
                boxes = getattr(result, "boxes", None)
                if boxes is None:
                    continue
                xyxy = boxes.xyxy.cpu().numpy().tolist()
                conf = boxes.conf.cpu().numpy().tolist()
                cls = boxes.cls.cpu().numpy().tolist()
                for bbox, confidence, class_id in zip(xyxy, conf, cls):
                    label = str(names.get(int(class_id), int(class_id)))
                    area = bbox_area(bbox)
                    detections_this_frame += 1
                    total_detections += 1
                    print(
                        f"frame_index={frame_index}, label={label}, "
                        f"confidence={confidence:.4f}, bbox={bbox}, "
                        f"bbox_area={area:.2f}"
                    )
                    draw_detection(frame, label, float(confidence), bbox)

            if detections_this_frame == 0:
                print(f"frame_index={frame_index}, no detections")

            output_path = output_dir / f"frame_{frame_index:06d}.jpg"
            if not cv2.imwrite(str(output_path), frame):
                raise IOError(f"Failed to write annotated image: {output_path}")
            frame_index += 1
    finally:
        capture.release()

    print(f"saved annotated frames to: {output_dir}")
    print(f"processed_frames={frame_index}, total_detections={total_detections}")
    return total_detections


def main() -> None:
    """Command-line entrypoint."""

    args = parse_args()
    try:
        run_smoke_test(
            video_path=Path(args.video_path),
            model_path=Path(args.model_path),
            output_dir=Path(args.output_dir),
            max_frames=args.max_frames,
            confidence_threshold=args.confidence_threshold,
            device=args.device,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
