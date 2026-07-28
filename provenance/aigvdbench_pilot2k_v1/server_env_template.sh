#!/usr/bin/env bash
set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/projects/fake_video_structural_anomaly_refactor}"
export DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/datasets}"
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/autodl-tmp/checkpoints}"
export MODEL_ROOT="${MODEL_ROOT:-${CHECKPOINT_ROOT}}"
export CACHE_ROOT="${CACHE_ROOT:-/root/autodl-tmp/caches}"
export DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${CACHE_ROOT}/downloads}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
export LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"

export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${CACHE_ROOT}/transformers}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"

export SEMANTIC3D_UNIDEPTH_SOURCE_ROOT="${SEMANTIC3D_UNIDEPTH_SOURCE_ROOT:-/root/autodl-tmp/projects/third_party/UniDepth}"
export SEMANTIC3D_UNIDEPTH_SOURCE_COMMIT="8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
export PYTHONPATH="${SEMANTIC3D_UNIDEPTH_SOURCE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
