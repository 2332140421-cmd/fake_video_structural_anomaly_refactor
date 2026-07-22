#!/usr/bin/env python3
"""Plan or explicitly fetch registered model weights with SHA-256 validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic3d.git_release.model_registry import (  # noqa: E402
    load_model_registry,
    sha256_file,
)


def fetch_registered_models(
    registry_path: Path,
    model_root: Path,
    *,
    execute: bool = False,
    selected_ids: set[str] | None = None,
) -> dict[str, object]:
    """Return an acquisition plan; download only after explicit execute=True."""

    registry = load_model_registry(registry_path)
    model_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for model in sorted(registry["models"], key=lambda row: str(row["model_id"])):
        model_id = str(model["model_id"])
        if selected_ids and model_id not in selected_ids:
            continue
        target = model_root / str(model["filename"])
        expected = str(model["sha256"])
        exists = target.is_file()
        valid_before = exists and target.stat().st_size == int(model["size_bytes"]) and sha256_file(target) == expected
        action = "already_verified" if valid_before else "would_download_and_verify"
        if execute and not valid_before:
            fd, temporary_name = tempfile.mkstemp(
                prefix=f".{target.name}.", suffix=".partial", dir=model_root
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                urllib.request.urlretrieve(str(model["source"]), temporary)
                if temporary.stat().st_size != int(model["size_bytes"]):
                    raise RuntimeError(f"Size mismatch for {model_id}")
                if sha256_file(temporary) != expected:
                    raise RuntimeError(f"SHA-256 mismatch for {model_id}")
                os.replace(temporary, target)
                action = "downloaded_and_verified"
            finally:
                temporary.unlink(missing_ok=True)
        valid_after = target.is_file() and target.stat().st_size == int(model["size_bytes"]) and sha256_file(target) == expected
        rows.append(
            {
                "model_id": model_id,
                "source": model["source"],
                "target": str(target),
                "sha256": expected,
                "action": action,
                "valid": valid_after,
            }
        )
    return {
        "execute": execute,
        "downloads_performed": execute and any(row["action"] == "downloaded_and_verified" for row in rows),
        "models": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/model_registry/yolo_weights_v1.yaml")
    parser.add_argument("--model-root", default=os.environ.get("MODEL_ROOT", "checkpoints"))
    parser.add_argument("--model-id", action="append", default=[])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Explicitly perform registered downloads. Default is dry-run.",
    )
    args = parser.parse_args()
    result = fetch_registered_models(
        PROJECT_ROOT / args.registry,
        Path(args.model_root),
        execute=args.execute,
        selected_ids=set(args.model_id) or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
