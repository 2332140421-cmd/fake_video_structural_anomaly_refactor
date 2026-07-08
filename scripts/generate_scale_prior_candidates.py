#!/usr/bin/env python3
"""Generate review-only scale-prior candidates from a coverage report."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _ensure_project_environment() -> None:
    """Re-execute with the project .venv Python when available."""

    project_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    current_python = Path(sys.executable).resolve()
    if project_python.exists() and current_python != project_python.resolve():
        os.execv(str(project_python), [str(project_python), *sys.argv])


_ensure_project_environment()

import yaml  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate review-only scale prior candidate YAML."
    )
    parser.add_argument(
        "--missing_report_csv",
        default=str(PROJECT_ROOT / "outputs" / "results" / "missing_scale_prior_report.csv"),
    )
    parser.add_argument(
        "--output_yaml",
        default=str(PROJECT_ROOT / "outputs" / "results" / "scale_prior_candidates.yaml"),
    )
    parser.add_argument("--min_count", type=int, default=2)
    return parser.parse_args()


def load_report_rows(report_csv: Path) -> list[dict[str, str]]:
    """Load report rows from CSV."""

    if not report_csv.exists():
        raise FileNotFoundError(f"Missing scale-prior report CSV: {report_csv}")
    with report_csv.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def build_candidate_data(
    report_rows: list[dict[str, str]],
    min_count: int = 2,
) -> dict[str, dict[str, Any]]:
    """Build candidate scale-prior records from missing/unreliable report rows."""

    if min_count < 1:
        raise ValueError(f"min_count must be >= 1, got {min_count}.")

    candidates: dict[str, dict[str, Any]] = {}
    for row in report_rows:
        label = str(row["label"])
        count = int(row["count"])
        status = str(row["status"])
        if count < min_count or status not in {"missing", "unreliable"}:
            continue

        if status == "unreliable":
            candidates[label] = {
                "min": None,
                "max": None,
                "reliable": False,
                "action": "review_or_skip",
                "count": count,
                "note": (
                    "Highly variable physical size; recommended to skip unless "
                    "manually reviewed."
                ),
            }
        else:
            candidates[label] = {
                "min": None,
                "max": None,
                "reliable": None,
                "action": "review",
                "count": count,
                "note": "Please manually review or map to a coarse category.",
            }
    return candidates


def save_candidates(candidates: dict[str, dict[str, Any]], output_yaml: Path) -> None:
    """Save candidate records without modifying the main scale-prior config."""

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    data = {"candidate_scale_priors": candidates}
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=True)


def main() -> None:
    """Run the candidate YAML generator."""

    args = parse_args()
    candidates = build_candidate_data(
        load_report_rows(Path(args.missing_report_csv)),
        min_count=args.min_count,
    )
    save_candidates(candidates, Path(args.output_yaml))
    print(f"Saved {len(candidates)} candidate scale-prior record(s) to {args.output_yaml}")
    if not candidates:
        print("No missing/unreliable labels met the min_count threshold.")


if __name__ == "__main__":
    main()
