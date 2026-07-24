#!/usr/bin/env python3
"""Build validated pilot manifests from the project's existing video folders."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.formal_schema import FORMAL_SAMPLE_FIELDS  # noqa: E402
from semantic3d.dataset_builder.manifest import (  # noqa: E402
    VIDEO_EXTENSIONS,
    build_formal_manifest_from_directory,
    scan_video_files,
)

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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan legacy real/fake directories.",
    )
    parser.add_argument(
        "--formal-dir",
        default="",
        help="Build the unified formal manifest from this directory instead of pilot manifests.",
    )
    parser.add_argument(
        "--data-root",
        default="",
        help="Explicit external data root used by --formal-dir.",
    )
    parser.add_argument("--source-dataset", default="")
    parser.add_argument(
        "--formal-split",
        choices=("train", "validation", "test"),
        default=None,
    )
    parser.add_argument(
        "--formal-output-csv",
        default="",
        help="Required output path for --formal-dir.",
    )
    parser.add_argument(
        "--formal-path-mode",
        choices=("absolute", "data_root_relative"),
        default="data_root_relative",
    )
    return parser.parse_args()


def find_videos(video_dir: Path, *, recursive: bool = False) -> list[Path]:
    """Return sorted supported video files from an existing directory."""

    videos = scan_video_files(video_dir, recursive=recursive)
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


def save_formal_manifest(
    *,
    video_dir: Path,
    data_root: Path,
    source_dataset: str,
    split: str | None,
    output_csv: Path,
    path_mode: str,
) -> int:
    """Write the generic schema without inferring labels or source lineage."""

    if not source_dataset.strip():
        raise ValueError("--source-dataset is required with --formal-dir")
    samples = build_formal_manifest_from_directory(
        video_dir,
        data_root=data_root,
        source_dataset=source_dataset,
        split=split,
        recursive=True,
        path_mode=path_mode,
    )
    if not samples:
        raise FileNotFoundError(f"No supported videos found in: {video_dir}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FORMAL_SAMPLE_FIELDS)
        writer.writeheader()
        for sample in samples:
            row = sample.to_dict()
            for name in (
                "source_lineage",
                "temporal_annotation",
                "spatial_annotation",
                "metadata_status",
            ):
                row[name] = json.dumps(row[name], ensure_ascii=False, sort_keys=True)
            writer.writerow(row)
    return len(samples)


def main() -> None:
    """Build the six-video and two-video pilot manifests."""

    args = parse_args()
    if args.formal_dir:
        if not args.data_root or not args.formal_output_csv:
            raise ValueError(
                "--data-root and --formal-output-csv are required with --formal-dir"
            )
        count = save_formal_manifest(
            video_dir=Path(args.formal_dir),
            data_root=Path(args.data_root),
            source_dataset=args.source_dataset,
            split=args.formal_split,
            output_csv=Path(args.formal_output_csv),
            path_mode=args.formal_path_mode,
        )
        print(f"Saved {count} formal sample(s) to {args.formal_output_csv}")
        return
    rows = build_manifest_rows(
        find_videos(Path(args.real_dir), recursive=args.recursive),
        find_videos(Path(args.fake_dir), recursive=args.recursive),
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
