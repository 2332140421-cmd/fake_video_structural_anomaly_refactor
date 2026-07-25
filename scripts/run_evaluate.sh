#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "Usage: run_evaluate.sh --manifest PATH --checkpoint PATH --split SPLIT --output PATH [options]"
  echo "Options: --batch-size N --device NAME --num-workers N --classification-threshold F"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MANIFEST=""
CHECKPOINT=""
SPLIT=""
OUTPUT=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --batch-size|--device|--num-workers|--classification-threshold)
      FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST" || -z "$CHECKPOINT" || -z "$SPLIT" || -z "$OUTPUT" ]]; then
  echo "--manifest, --checkpoint, --split, and --output are required." >&2
  usage >&2
  exit 2
fi
mkdir -p "$OUTPUT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_EXECUTABLE="python3"
fi

exec > >(tee "$OUTPUT/console.log") 2>&1
COMMAND=(
  "$PYTHON_EXECUTABLE" -m experiments.evaluate
  --manifest "$MANIFEST"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --output "$OUTPUT"
  "${FORWARD_ARGS[@]}"
)
echo "[COMMAND]"
printf '%q ' "${COMMAND[@]}"
echo
cd "$PROJECT_ROOT"
"${COMMAND[@]}"
