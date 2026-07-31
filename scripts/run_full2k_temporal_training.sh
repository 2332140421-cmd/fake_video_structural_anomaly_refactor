#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT="${BASE_ROOT:-/media/njupt/8FCD3869E2FAB06F/chenyh/fake_video_structural_anomaly}"
PROJECT_ROOT="${PROJECT_ROOT:-${BASE_ROOT}/projects/fake_video_structural_anomaly_refactor}"
PROJECT_PYTHON="${PROJECT_PYTHON:-${BASE_ROOT}/envs/fake_video_structural_anomaly/bin/python}"
TRAINING_MANIFEST="${TRAINING_MANIFEST:-${PROJECT_ROOT}/provenance/aigvdbench_pilot2k_v1/manifests/aigvdbench_open_sora_paired_2k_residual_v1.csv}"
RUNTIME_PATH_MANIFEST="${RUNTIME_PATH_MANIFEST:-${BASE_ROOT}/migration/aigvd_full2k_v1/reports/server_restore_audit/full2k_media_manifest_server.csv}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${PROJECT_ROOT}/configs/training_default.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_ROOT}/outputs/training_pilot/full2k_temporal_head_1epoch_v1}"
EPOCHS="${EPOCHS:-1}"

if [[ ! -x "${PROJECT_PYTHON}" ]]; then
  echo "Project Python is not executable: ${PROJECT_PYTHON}" >&2
  exit 2
fi
for required_path in \
  "${PROJECT_ROOT}" \
  "${TRAINING_MANIFEST}" \
  "${RUNTIME_PATH_MANIFEST}" \
  "${TRAINING_CONFIG}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Required path is missing: ${required_path}" >&2
    exit 2
  fi
done

timestamp="$(date '+%Y%m%dT%H%M%S%z')"
run_dir="${OUTPUT_ROOT}/${timestamp}"
if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite existing run directory: ${run_dir}" >&2
  exit 2
fi
mkdir -p "${run_dir}/mpl_cache"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export MPLCONFIGDIR="${run_dir}/mpl_cache"

command=(
  "${PROJECT_PYTHON}" -m inference.cli train
  --config "${TRAINING_CONFIG}"
  --manifest "${TRAINING_MANIFEST}"
  --runtime-path-manifest "${RUNTIME_PATH_MANIFEST}"
  --epochs "${EPOCHS}"
  --output "${run_dir}"
)

finish() {
  status=$?
  if [[ ${status} -ne 0 ]]; then
    printf 'FAIL\n' >"${run_dir}/FAILED"
  fi
}
trap finish EXIT

cd "${PROJECT_ROOT}"
printf 'RUN_DIR=%s\n' "${run_dir}"
printf 'PYTHON=%s\n' "${PROJECT_PYTHON}"
printf 'COMMAND='
printf '%q ' "${command[@]}"
printf '\n'
"${command[@]}" 2>&1 | tee "${run_dir}/terminal.log"
printf 'PASS\n' >"${run_dir}/COMPLETED"
trap - EXIT
