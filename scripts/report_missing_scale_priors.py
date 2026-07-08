#!/usr/bin/env python3
"""Report scale-prior coverage for labels found in observation JSON files."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from semantic3d.io import load_clip_observation  # noqa: E402
from semantic3d.scale_prior import (  # noqa: E402
    ScalePriorResolver,
    load_scale_prior_resolver,
    normalize_label,
)


FIELDS = [
    "label",
    "count",
    "has_exact_prior",
    "has_alias_prior",
    "resolved_label",
    "reliable",
    "status",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Report missing or unreliable scale priors in observations."
    )
    parser.add_argument(
        "--observation_dir",
        default=str(PROJECT_ROOT / "outputs" / "real_observations_depth"),
    )
    parser.add_argument(
        "--scale_prior_path",
        default=str(PROJECT_ROOT / "configs" / "scale_priors.yaml"),
    )
    parser.add_argument(
        "--output_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "missing_scale_prior_report.csv"),
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Only include labels that appear at least this many times.",
    )
    return parser.parse_args()


def collect_label_counts(observation_dir: Path) -> dict[str, int]:
    """Collect object label counts from clip observation JSON files."""

    counts: dict[str, int] = {}
    json_paths = sorted(observation_dir.rglob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No observation JSON files found in {observation_dir}.")

    for json_path in json_paths:
        clip = load_clip_observation(json_path)
        for frame in clip.frames:
            for obj in frame.objects:
                label = normalize_label(obj.label)
                counts[label] = counts.get(label, 0) + 1
    return counts


def build_report_rows(
    label_counts: dict[str, int],
    resolver: ScalePriorResolver,
    min_count: int = 1,
) -> list[dict[str, object]]:
    """Build report rows for label counts and scale-prior resolver status."""

    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}.")

    rows: list[dict[str, object]] = []
    for label, count in sorted(label_counts.items(), key=lambda item: (-item[1], item[0])):
        if count < min_count:
            continue
        normalized = normalize_label(label)
        resolved = resolver.resolve(normalized, require_reliable=True)
        has_exact_prior = normalized in resolver.scale_priors
        has_alias_prior = normalized in resolver.aliases
        rows.append(
            {
                "label": label,
                "count": count,
                "has_exact_prior": has_exact_prior,
                "has_alias_prior": has_alias_prior,
                "resolved_label": "" if resolved.source == "missing" else resolved.resolved_label,
                "reliable": resolved.reliable,
                "status": resolved.source,
            }
        )
    return rows


def save_report(rows: list[dict[str, object]], output_csv: Path) -> None:
    """Save scale-prior coverage report rows to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in FIELDS})


def main() -> None:
    """Run the reporting script."""

    args = parse_args()
    resolver = load_scale_prior_resolver(args.scale_prior_path)
    rows = build_report_rows(
        collect_label_counts(Path(args.observation_dir)),
        resolver,
        min_count=args.min_count,
    )
    save_report(rows, Path(args.output_csv))
    print(f"Saved scale-prior coverage report to {args.output_csv}")
    for row in rows:
        print(
            f"{row['label']}: count={row['count']}, status={row['status']}, "
            f"resolved={row['resolved_label']}, reliable={row['reliable']}"
        )


if __name__ == "__main__":
    main()
