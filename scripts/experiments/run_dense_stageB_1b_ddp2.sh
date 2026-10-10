#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/storage/yjl/moe-matmulfree
CONFIG_PATH="${PROJECT_ROOT}/configs/pretrain/dense_matmulfree_scratch_relaxed30b_stageB_1b_ddp2.yaml"
OUTPUT_DIR="${STAGEB_OUTPUT_DIR:?Set STAGEB_OUTPUT_DIR to a new output path}"
TRITON_CACHE_HOST="${TRITON_CACHE_HOST:-/tmp/stageB_triton_cache_20260723}"
IMAGE="${STAGEB_IMAGE:-matmulfreellm:latest}"
LOG_DIR="${PROJECT_ROOT}/outputs/stageB_launch_logs"
RUN_NAME="$(basename "${OUTPUT_DIR}")"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${TRITON_CACHE_HOST}" "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "output_dir=${OUTPUT_DIR}"
echo "config_path=${CONFIG_PATH}"
echo "triton_cache_host=${TRITON_CACHE_HOST}"
echo "triton_cache_container=/tmp/triton-cache"

run_stage() {
  docker run --rm \
    --gpus '"device=0,1"' \
    --ipc=host \
    --user 1026:1027 \
    -v /home/storage/yjl:/home/storage/yjl \
    -v "${TRITON_CACHE_HOST}:/tmp/triton-cache" \
    -w "${PROJECT_ROOT}" \
    -e HOME=/tmp \
    -e HF_HOME=/tmp/hf-cache \
    -e TRITON_CACHE_DIR=/tmp/triton-cache \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e PYTHONHASHSEED=20260721 \
    "${IMAGE}" \
    /opt/venv/bin/torchrun \
    --standalone \
    --nproc_per_node=2 \
    scripts/run_dense_stageB_ddp.py \
    "$@"
}

echo "Stage B phase 1: scratch training to controlled 250M-token boundary"
run_stage \
  --config-path "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --stop-after-tokens 250000000 \
  --checkpoint-name checkpoint_250m

CHECKPOINT_250M="${OUTPUT_DIR}/checkpoint_250m"
test -s "${CHECKPOINT_250M}/trainer_state.pt"
test -s "${CHECKPOINT_250M}/checkpoint_manifest.json"
test -s "${CHECKPOINT_250M}/dataset_state_rank0.pt"
test -s "${CHECKPOINT_250M}/dataset_state_rank1.pt"

echo "Stage B phase 2: new torchrun, exact resume from 250M to 1B"
run_stage \
  --config-path "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --resume-checkpoint "${CHECKPOINT_250M}" \
  --checkpoint-name checkpoint_final_1b

test -s "${OUTPUT_DIR}/checkpoint_final_1b/trainer_state.pt"
test -s "${OUTPUT_DIR}/checkpoint_final_1b/checkpoint_manifest.json"
echo "Stage B completed: ${OUTPUT_DIR}"
