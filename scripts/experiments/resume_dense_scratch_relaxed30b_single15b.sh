#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/storage/yjl/moe-matmulfree
CONFIG_PATH="${PROJECT_ROOT}/configs/pretrain/dense_matmulfree_scratch_relaxed30b_single15b.yaml"
OUTPUT_DIR="${DENSE_SINGLE_OUTPUT_DIR:-${PROJECT_ROOT}/outputs/dense_matmulfree_scratch_relaxed30b_single15b_20260724_112405}"
IMAGE="${DENSE_SINGLE_IMAGE:-matmulfreellm:latest}"
GPU_DEVICE="${DENSE_SINGLE_GPU:-0}"
RUN_RESUME="${RUN_RESUME:-0}"
LOG_DIR="${PROJECT_ROOT}/outputs/dense_single15b_launch_logs"
RUN_NAME="$(basename "${OUTPUT_DIR}")"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.resume.log"

checkpoint_complete() {
  local checkpoint_dir=$1
  [[ -s "${checkpoint_dir}/trainer_state.pt" ]] \
    && [[ -s "${checkpoint_dir}/checkpoint_manifest.json" ]] \
    && [[ -s "${checkpoint_dir}/checkpoint_write_complete.json" ]] \
    && [[ -s "${checkpoint_dir}/dataset_state_rank0.pt" ]] \
    && [[ -s "${checkpoint_dir}/rng_state_rank0.pt" ]]
}

if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "Missing output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi

previous_checkpoint=
previous_target=0
previous_actual_tokens=0
for target in 1000000000 3000000000 6000000000 10000000000 15000000000; do
  target_billions=$((target / 1000000000))
  candidate="${OUTPUT_DIR}/checkpoint_${target_billions}b"
  if checkpoint_complete "${candidate}"; then
    previous_checkpoint="${candidate}"
    previous_target="${target}"
    previous_actual_tokens="$(
      /usr/bin/python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["global_tokens"])' \
        "${candidate}/checkpoint_manifest.json"
    )"
  fi
done

if [[ -z "${previous_checkpoint}" ]]; then
  echo "No complete checkpoint found in ${OUTPUT_DIR}" >&2
  exit 1
fi

echo "output_dir=${OUTPUT_DIR}"
echo "resume_checkpoint=${previous_checkpoint}"
echo "resume_global_tokens=${previous_actual_tokens}"
echo "gpu_device=${GPU_DEVICE}"
echo "triton_cache_container=/tmp/triton-cache"

if [[ "${RUN_RESUME}" != "1" ]]; then
  echo "RUN_RESUME is not 1; no training started."
  echo "Set RUN_RESUME=1 to continue from the checkpoint above."
  exit 0
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

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

for target in 6000000000 10000000000 15000000000; do
  if (( target <= previous_target )); then
    continue
  fi
  target_billions=$((target / 1000000000))
  checkpoint_name="checkpoint_${target_billions}b"
  checkpoint_dir="${OUTPUT_DIR}/${checkpoint_name}"
  if [[ -e "${checkpoint_dir}" ]]; then
    echo "Refusing to overwrite incomplete checkpoint: ${checkpoint_dir}" >&2
    exit 1
  fi

  echo "Resuming single-GPU dense scratch model to ${target} tokens"
  run_stage \
    --config-path "${CONFIG_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --resume-checkpoint "${previous_checkpoint}" \
    --stop-after-tokens "${target}" \
    --checkpoint-name "${checkpoint_name}"

  checkpoint_complete "${checkpoint_dir}"
  previous_checkpoint="${checkpoint_dir}"
  previous_target="${target}"
done

echo "Dense single-GPU 15B-token training completed: ${OUTPUT_DIR}"
echo "Final checkpoint: ${previous_checkpoint}"
echo "Resume log: ${LOG_PATH}"
