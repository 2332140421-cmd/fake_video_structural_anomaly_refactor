#!/usr/bin/env python3
"""Build a separate label table after structural construction has finished.

This script is intentionally outside the P4-B builder and cache graph. Structural
stages never call it and never read its output.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.dataset_builder.reader import DatasetReader  # noqa: E402
from semantic3d.dataset_builder.writer import write_parquet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    videos = DatasetReader(dataset_root).rows("manifests/videos.parquet")
    by_source_name = {str(row["source_name"]): str(row["video_id"]) for row in videos}
    output = []
    with Path(args.input_manifest).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_name = str(row["video_id"])
            if source_name not in by_source_name:
                raise ValueError(f"Label row has no structural video match: {source_name}")
            output.append(
                {
                    "video_id": by_source_name[source_name],
                    "label": int(row["label"]),
                    "label_name": str(row.get("label_name", "")),
                    "split": str(row.get("split", "")),
                    "temporal_annotation": "",
                    "spatial_annotation": "",
                    "source_manifest": str(Path(args.input_manifest)),
                }
            )
    path = write_parquet(
        dataset_root / "labels_manifest.parquet",
        output,
        columns=("video_id", "label", "label_name", "split", "temporal_annotation", "spatial_annotation", "source_manifest"),
    )
    print(f"Saved {len(output)} isolated label row(s) to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
