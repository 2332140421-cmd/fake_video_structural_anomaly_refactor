#!/usr/bin/env python3
"""Build validated pilot manifests from the project's existing video folders."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
MANIFEST_FIELDS = ["video_id", "video_path", "label", "label_name", "split"]


def _ensure_project_environment() -> None:
    """Re-execute with the project-local interpreter when needed."""

    if not PROJECT_PYTHON.exists():
        raise RuntimeError(f"Project environment is missing: {PROJECT_PYTHON}")
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), *sys.argv])


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build real/fake manifests without copying or moving videos."
    )
    parser.add_argument(
        "--real_dir",
        default=str(PROJECT_ROOT / "data/tests_videos/tests_real_videos"),
    )
    parser.add_argument(
        "--fake_dir",
        default=str(PROJECT_ROOT / "data/tests_videos/tests_fake_videos"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(PROJECT_ROOT / "data/manifests/pilot_real_fake.csv"),
    )
    parser.add_argument(
        "--smoke_output_csv",
        default=str(PROJECT_ROOT / "data/manifests/pilot_smoke_2video.csv"),
    )
    parser.add_argument("--split", default="val")
    return parser.parse_args()


def find_videos(video_dir: Path) -> list[Path]:
    """Return sorted supported video files from an existing directory."""

    if not video_dir.is_dir():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")
    videos = [
        path.resolve()
        for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    if not videos:
        raise FileNotFoundError(f"No supported videos found in: {video_dir}")
    return videos


def _display_path(path: Path) -> str:
    """Prefer a repository-relative path in the manifest."""

    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def build_manifest_rows(
    real_videos: Iterable[Path],
    fake_videos: Iterable[Path],
    split: str = "val",
) -> list[dict[str, object]]:
    """Build and validate manifest rows with real=0 and fake=1."""

    if not split.strip():
        raise ValueError("split must be a non-empty string.")
    rows: list[dict[str, object]] = []
    for videos, label, label_name in (
        (real_videos, 0, "real"),
        (fake_videos, 1, "fake"),
    ):
        for raw_path in videos:
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Manifest video does not exist: {path}")
            rows.append(
                {
                    "video_id": path.stem,
                    "video_path": _display_path(path),
                    "label": label,
                    "label_name": label_name,
                    "split": split,
                }
            )

    video_ids = [str(row["video_id"]) for row in rows]
    duplicates = sorted({item for item in video_ids if video_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"Duplicate video_id values are not allowed: {duplicates}")
    return rows


def validate_manifest_rows(rows: list[dict[str, object]]) -> None:
    """Validate labels, split, paths, and unique video identifiers."""

    if not rows:
        raise ValueError("Manifest must contain at least one video.")
    seen: set[str] = set()
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            raise ValueError("Manifest row has an empty video_id.")
        if video_id in seen:
            raise ValueError(f"Duplicate video_id is not allowed: {video_id}")
        seen.add(video_id)
        label = int(row["label"])
        if label not in {0, 1}:
            raise ValueError(f"label must be 0 or 1 for {video_id}, got {label}.")
        expected_name = "real" if label == 0 else "fake"
        if str(row.get("label_name")) != expected_name:
            raise ValueError(f"Invalid label_name for {video_id}: expected {expected_name}.")
        if str(row.get("split")) != "val":
            raise ValueError(f"Pilot split must be val for {video_id}.")
        raw_path = Path(str(row["video_path"]))
        path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        if not path.is_file():
            raise FileNotFoundError(f"Manifest path does not exist: {path}")


def select_smoke_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Select one real and one fake video for the smoke manifest."""

    real = next((row for row in rows if int(row["label"]) == 0), None)
    fake = next((row for row in rows if int(row["label"]) == 1), None)
    if real is None or fake is None:
        raise ValueError("Smoke manifest requires at least one real and one fake video.")
    return [real, fake]


def save_manifest(rows: list[dict[str, object]], output_csv: Path) -> None:
    """Write a validated manifest CSV and create its parent directory."""

    validate_manifest_rows(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build the six-video and two-video pilot manifests."""

    args = parse_args()
    rows = build_manifest_rows(
        find_videos(Path(args.real_dir)),
        find_videos(Path(args.fake_dir)),
        split=args.split,
    )
    smoke_rows = select_smoke_rows(rows)
    save_manifest(rows, Path(args.output_csv))
    save_manifest(smoke_rows, Path(args.smoke_output_csv))
    print(f"Saved {len(rows)} videos to {args.output_csv}")
    print(f"Saved {len(smoke_rows)} videos to {args.smoke_output_csv}")
    print("Labels: real=0, fake=1; split=val")


if __name__ == "__main__":
    _ensure_project_environment()
    main()
