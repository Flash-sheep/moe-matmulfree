#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/storage/yjl/moe-matmulfree
CONFIG_PATH="${PROJECT_ROOT}/configs/pretrain/dense_matmulfree_scratch_relaxed30b_single15b.yaml"
OUTPUT_DIR="${DENSE_SINGLE_OUTPUT_DIR:?Set DENSE_SINGLE_OUTPUT_DIR to a new output path}"
IMAGE="${DENSE_SINGLE_IMAGE:-matmulfreellm:latest}"
GPU_DEVICE="${DENSE_SINGLE_GPU:-0}"
LOG_DIR="${PROJECT_ROOT}/outputs/dense_single15b_launch_logs"
RUN_NAME="$(basename "${OUTPUT_DIR}")"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "output_dir=${OUTPUT_DIR}"
echo "config_path=${CONFIG_PATH}"
echo "gpu_device=${GPU_DEVICE}"
echo "triton_cache_container=/tmp/triton-cache"

run_stage() {
  docker run --rm \
    --gpus "device=${GPU_DEVICE}" \
    --ipc=host \
    --user 1026:1027 \
    -v /home/storage/yjl:/home/storage/yjl \
    -w "${PROJECT_ROOT}" \
    -e HOME=/tmp \
    -e HF_HOME=/tmp/hf-cache \
    -e TRITON_CACHE_DIR=/tmp/triton-cache \
    -e CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    -e PYTHONHASHSEED=20260721 \
    "${IMAGE}" \
    /opt/venv/bin/python \
    scripts/run_dense_stageB_ddp.py \
    "$@"
}

previous_checkpoint=
for target in 1000000000 3000000000 6000000000 10000000000 15000000000; do
  target_billions=$((target / 1000000000))
  checkpoint_name="checkpoint_${target_billions}b"
  args=(
    --config-path "${CONFIG_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --stop-after-tokens "${target}"
    --checkpoint-name "${checkpoint_name}"
  )
  if [[ -n "${previous_checkpoint}" ]]; then
    args+=(--resume-checkpoint "${previous_checkpoint}")
  fi

  echo "Training single-GPU dense scratch model to ${target} tokens"
  run_stage "${args[@]}"

  checkpoint_dir="${OUTPUT_DIR}/${checkpoint_name}"
  test -s "${checkpoint_dir}/trainer_state.pt"
  test -s "${checkpoint_dir}/checkpoint_manifest.json"
  test -s "${checkpoint_dir}/checkpoint_write_complete.json"
  test -s "${checkpoint_dir}/dataset_state_rank0.pt"
  previous_checkpoint="${checkpoint_dir}"
done

echo "Dense single-GPU 15B-token training completed: ${OUTPUT_DIR}"
echo "Final checkpoint: ${previous_checkpoint}"
echo "Launch log: ${LOG_PATH}"
