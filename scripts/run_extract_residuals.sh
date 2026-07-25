#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PROJECT_PYTHON:-python}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/extract_residual_dataset.py" "$@"
