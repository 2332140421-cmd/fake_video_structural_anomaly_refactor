#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  echo "Usage: run_train.sh --manifest PATH --output PATH [options]"
  echo "Options: --config PATH --run-name NAME --epochs N --batch-size N"
  echo "         --learning-rate F --weight-decay F --seed N --device NAME"
  echo "         --num-workers N --log-every N --classification-threshold F"
  echo "         --channel-schema PATH --resume PATH --amp --no-amp"
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

MANIFEST=""
CONFIG="configs/training_default.yaml"
OUTPUT=""
RUN_NAME=""
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --epochs|--batch-size|--learning-rate|--weight-decay|--seed|--device|--num-workers|--log-every|--classification-threshold|--channel-schema|--resume)
      FORWARD_ARGS+=("$1" "$2"); shift 2 ;;
    --amp|--no-amp) FORWARD_ARGS+=("$1"); shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$MANIFEST" || -z "$OUTPUT" ]]; then
  echo "--manifest and --output are required." >&2
  usage >&2
  exit 2
fi
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$PROJECT_ROOT/$CONFIG"
fi
RUN_OUTPUT="$OUTPUT"
if [[ -n "$RUN_NAME" ]]; then
  RUN_OUTPUT="$OUTPUT/$RUN_NAME"
fi
mkdir -p "$RUN_OUTPUT"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_EXECUTABLE="python3"
fi

exec > >(tee "$RUN_OUTPUT/console.log") 2>&1
COMMAND=(
  "$PYTHON_EXECUTABLE" -m inference.cli train
  --manifest "$MANIFEST"
  --config "$CONFIG"
  --output "$RUN_OUTPUT"
  "${FORWARD_ARGS[@]}"
)
echo "[COMMAND]"
printf '%q ' "${COMMAND[@]}"
echo
echo "[CONFIG] $CONFIG"
sed -n '1,240p' "$CONFIG"
cd "$PROJECT_ROOT"
"${COMMAND[@]}"
