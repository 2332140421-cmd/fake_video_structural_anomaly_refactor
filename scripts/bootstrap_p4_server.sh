#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${P4_VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${P4_PYTHON_BIN:-python3}"
RUNTIME_CONFIG="${P4_RUNTIME_CONFIG:-configs/runtime/server_template.yaml}"

cd "${PROJECT_ROOT}"
git rev-parse --is-inside-work-tree >/dev/null
git rev-parse HEAD

: "${DATA_ROOT:?Set DATA_ROOT before server bootstrap}"
: "${DOWNLOAD_ROOT:?Set DOWNLOAD_ROOT before server bootstrap}"
: "${CACHE_ROOT:?Set CACHE_ROOT before server bootstrap}"
: "${MODEL_ROOT:?Set MODEL_ROOT before server bootstrap}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT before server bootstrap}"
: "${LOG_ROOT:?Set LOG_ROOT before server bootstrap}"
export PROJECT_ROOT DATA_ROOT DOWNLOAD_ROOT CACHE_ROOT MODEL_ROOT OUTPUT_ROOT LOG_ROOT

command -v ffmpeg >/dev/null || { echo "ffmpeg is required; install it with the server package manager." >&2; exit 2; }
command -v ffprobe >/dev/null || { echo "ffprobe is required; install it with the server package manager." >&2; exit 2; }

mkdir -p "${DATA_ROOT}" "${DOWNLOAD_ROOT}" "${CACHE_ROOT}" "${CACHE_ROOT}/tmp" \
  "${MODEL_ROOT}" "${OUTPUT_ROOT}" "${LOG_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r requirements-lock.txt

if ! "${VENV_DIR}/bin/python" -c 'import torch' >/dev/null 2>&1; then
  if [[ -z "${P4_TORCH_INSTALL_COMMAND:-}" ]]; then
    echo "PyTorch is missing. Set P4_TORCH_INSTALL_COMMAND to a command compatible with this server's driver/CUDA." >&2
    exit 2
  fi
  /bin/bash -lc "${P4_TORCH_INSTALL_COMMAND}"
fi

"${VENV_DIR}/bin/python" -m pip install -r requirements-inference-lock.txt
"${VENV_DIR}/bin/python" -m pip install -e . --no-deps

PREFLIGHT_ROOT="${OUTPUT_ROOT}/p4c3a_server_preflight"
"${VENV_DIR}/bin/python" scripts/preflight_p4_server_environment.py \
  --runtime-config "${RUNTIME_CONFIG}" \
  --output-root "${PREFLIGHT_ROOT}"

"${VENV_DIR}/bin/python" -m pytest -q -rs | tee "${PREFLIGHT_ROOT}/pytest_full_q_rs.txt"
"${VENV_DIR}/bin/python" -m pytest -q \
  tests/test_p4c_experiment_protocol.py \
  tests/test_p4c1_experiment_manifest.py \
  tests/test_p4c1_leakage_audit.py \
  tests/test_p4c2_formal_data_readiness.py \
  tests/test_p4c2_source_lineage.py \
  tests/test_p4c3a_runtime_config.py \
  tests/test_p4c3a_server_preflight.py \
  tests/test_p4c3a_batch_state.py \
  tests/test_p4c3a_migration_manifest.py

"${VENV_DIR}/bin/python" -m pytest -q \
  tests/test_p4c3b_metric_provider.py \
  tests/test_p4c3b_metric_scene3d.py \
  tests/test_p4c3b_server_handoff.py

"${VENV_DIR}/bin/python" scripts/verify_p4c3b_server_handoff.py \
  --source-only \
  --output "${PREFLIGHT_ROOT}/p4c3b_handoff_source_report.json"

"${VENV_DIR}/bin/python" scripts/verify_p4_git_checkout.py \
  --runtime-config "${RUNTIME_CONFIG}" \
  --tests-passed \
  --preflight-validation "${PREFLIGHT_ROOT}/validation_report.json" \
  --require-server-smoke

echo "P4 server checkout and smoke prerequisites verified. Formal batch execution remains disabled."
