#!/usr/bin/env python3
"""Build a separate label table after structural construction has finished.

This script is intentionally outside the P4-B builder and cache graph. Structural
stages never call it and never read its output.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.reader import DatasetReader  # noqa: E402
from semantic3d.dataset_builder.writer import write_parquet  # noqa: E402


LABEL_COLUMNS = (
    "video_id",
    "label",
    "label_name",
    "split",
    "temporal_annotation",
    "spatial_annotation",
    "metadata_status",
    "source_manifest",
)


def build_label_rows(
    manifest_rows: Iterable[Mapping[str, Any]],
    structural_videos: Iterable[Mapping[str, Any]],
    *,
    source_manifest: str,
) -> list[dict[str, Any]]:
    """Match explicit IDs safely and preserve missing annotations as null."""

    videos = list(structural_videos)
    by_video_id = {str(row["video_id"]): str(row["video_id"]) for row in videos}
    by_source_name: dict[str, list[str]] = defaultdict(list)
    for row in videos:
        by_source_name[str(row["source_name"])].append(str(row["video_id"]))

    output = []
    for row in manifest_rows:
        identifier = str(row.get("sample_id") or row.get("video_id") or "").strip()
        if not identifier:
            raise ValueError("Label row requires sample_id or video_id")
        if identifier in by_video_id:
            video_id = by_video_id[identifier]
        else:
            candidates = by_source_name.get(identifier, [])
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous source_name label match; use a stable video_id: {identifier}"
                )
            if not candidates:
                raise ValueError(f"Label row has no structural video match: {identifier}")
            video_id = candidates[0]

        raw_label = row.get("label")
        if raw_label in {None, ""}:
            label = None
            label_name = None
            label_status = "missing"
        else:
            label = int(raw_label)
            if label not in {0, 1}:
                raise ValueError(f"label must be 0, 1, or missing for {identifier}")
            label_name = str(row.get("label_name") or ("real" if label == 0 else "fake"))
            label_status = "provided"
        split = str(row.get("split") or "").strip() or None
        if split not in {None, "val", "train", "validation", "test"}:
            raise ValueError(f"Unsupported split for {identifier}: {split!r}")
        temporal = row.get("temporal_annotation") or None
        spatial = row.get("spatial_annotation") or None
        output.append(
            {
                "video_id": video_id,
                "label": label,
                "label_name": label_name,
                "split": split,
                "temporal_annotation": temporal,
                "spatial_annotation": spatial,
                "metadata_status": json.dumps(
                    {
                        "label": label_status,
                        "split": "provided" if split is not None else "missing",
                        "temporal_annotation": (
                            "provided" if temporal is not None else "missing"
                        ),
                        "spatial_annotation": (
                            "provided" if spatial is not None else "missing"
                        ),
                    },
                    sort_keys=True,
                ),
                "source_manifest": source_manifest,
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    videos = DatasetReader(dataset_root).rows("manifests/videos.parquet")
    input_manifest = Path(args.input_manifest).expanduser().resolve()
    with input_manifest.open("r", encoding="utf-8", newline="") as handle:
        output = build_label_rows(
            csv.DictReader(handle),
            videos,
            source_manifest=input_manifest.as_posix(),
        )
    path = write_parquet(
        dataset_root / "labels_manifest.parquet",
        output,
        columns=LABEL_COLUMNS,
    )
    print(f"Saved {len(output)} isolated label row(s) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
