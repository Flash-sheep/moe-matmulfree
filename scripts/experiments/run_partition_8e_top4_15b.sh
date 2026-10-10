#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/storage/yjl/moe-matmulfree
CONFIG_PATH="${PROJECT_ROOT}/configs/pretrain/partition_8e_top4_all_layers_random_scratch_relaxed30b_15b.yaml"
IMAGE="${PARTITION_8E_IMAGE:-matmulfreellm:latest}"
GPU_DEVICE="${PARTITION_8E_GPU:-1}"
USE_HOST_CONDA="${PARTITION_8E_USE_HOST_CONDA:-0}"
HOST_PYTHON="${PARTITION_8E_PYTHON:-/home/fyf/anaconda3/bin/python}"
OUTPUT_DIR="${PARTITION_8E_OUTPUT_DIR:?Set PARTITION_8E_OUTPUT_DIR to a new output path}"
RESUME_CHECKPOINT="${PARTITION_8E_RESUME_CHECKPOINT:-}"
RUN_TRAIN="${RUN_TRAIN:-1}"
DETACH="${DETACH:-1}"
IN_TMUX="${PARTITION_8E_IN_TMUX:-0}"
LOG_DIR="${PROJECT_ROOT}/outputs/partition_8e_top4_launch_logs"
RUN_NAME="$(basename "${OUTPUT_DIR}")"
SESSION_NAME="${PARTITION_8E_SESSION:-partition_8e_top4_${RUN_NAME}}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"

if [[ -z "${RESUME_CHECKPOINT}" && -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  [[ -d "${OUTPUT_DIR}" ]] || { echo "Missing resume output: ${OUTPUT_DIR}" >&2; exit 1; }
  [[ -f "${RESUME_CHECKPOINT}/trainer_state.pt" ]] || { echo "Missing trainer state: ${RESUME_CHECKPOINT}" >&2; exit 1; }
fi
if [[ "${IN_TMUX}" != "1" ]] && tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: ${SESSION_NAME}" >&2
  exit 1
fi
if [[ "${USE_HOST_CONDA}" == "1" && ! -x "${HOST_PYTHON}" ]]; then
  echo "Missing host conda Python: ${HOST_PYTHON}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

run_stage() {
  local target_tokens="$1"
  local checkpoint_name="$2"
  local resume_arg=()
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    resume_arg=(--resume-checkpoint "${RESUME_CHECKPOINT}")
  fi
  if [[ "${USE_HOST_CONDA}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
      TRITON_CACHE_DIR=/tmp/triton-cache \
      HF_HOME=/tmp/hf-cache \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=20260721 \
      "${HOST_PYTHON}" scripts/run_dense_stageB_ddp.py \
      --config-path "${CONFIG_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --stop-after-tokens "${target_tokens}" \
      --checkpoint-name "${checkpoint_name}" \
      "${resume_arg[@]}"
  else
    docker run --rm --ipc=host --gpus "device=${GPU_DEVICE}" --user 1026:1027 \
      -v /home/storage/yjl:/home/storage/yjl \
      -w "${PROJECT_ROOT}" \
      -e HOME=/tmp -e HF_HOME=/tmp/hf-cache \
      -e TRITON_CACHE_DIR=/tmp/triton-cache \
      -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e PYTHONHASHSEED=20260721 \
      -e CUDA_VISIBLE_DEVICES=0 \
      "${IMAGE}" /opt/venv/bin/python scripts/run_dense_stageB_ddp.py \
      --config-path "${CONFIG_PATH}" \
      --output-dir "${OUTPUT_DIR}" \
      --stop-after-tokens "${target_tokens}" \
      --checkpoint-name "${checkpoint_name}" \
      "${resume_arg[@]}"
  fi
}

if [[ "${USE_HOST_CONDA}" == "1" ]]; then
  echo "partition_8e_top4 output=${OUTPUT_DIR} gpu=${GPU_DEVICE} runtime=host_conda python=${HOST_PYTHON}"
else
  echo "partition_8e_top4 output=${OUTPUT_DIR} gpu=${GPU_DEVICE} image=${IMAGE} runtime=container"
fi
echo "resume=${RESUME_CHECKPOINT:-none} triton_cache=/tmp/triton-cache"
echo "stages=1B,3B,6B,10B,15B; checkpoints include model/optimizer/scheduler/RNG/dataset packing state"

if [[ "${RUN_TRAIN}" != "1" ]]; then
  echo "RUN_TRAIN is not 1; command template only. No training started."
  exit 0
fi

if [[ "${DETACH}" == "1" && "${IN_TMUX}" != "1" ]]; then
  launch_command="cd $(printf '%q' "${PROJECT_ROOT}") && env"
  for env_arg in \
    "PARTITION_8E_IN_TMUX=1" \
    "PARTITION_8E_IMAGE=${IMAGE}" \
    "PARTITION_8E_GPU=${GPU_DEVICE}" \
    "PARTITION_8E_USE_HOST_CONDA=${USE_HOST_CONDA}" \
    "PARTITION_8E_PYTHON=${HOST_PYTHON}" \
    "PARTITION_8E_OUTPUT_DIR=${OUTPUT_DIR}" \
    "PARTITION_8E_RESUME_CHECKPOINT=${RESUME_CHECKPOINT}" \
    "PARTITION_8E_SESSION=${SESSION_NAME}" \
    "RUN_TRAIN=1" \
    "DETACH=0"; do
    launch_command+=" $(printf '%q' "${env_arg}")"
  done
  launch_command+=" bash $(printf '%q' "${BASH_SOURCE[0]}")"
  tmux new-session -d -s "${SESSION_NAME}" "${launch_command}"
  echo "Training detached in tmux session: ${SESSION_NAME}"
  echo "Attach with: tmux attach -t ${SESSION_NAME}"
  echo "Host log: ${LOG_PATH}"
  exit 0
fi

exec > >(tee -a "${LOG_PATH}") 2>&1
targets=(1000000000 3000000000 6000000000 10000000000 15000000000)
names=(checkpoint_1b checkpoint_3b checkpoint_6b checkpoint_10b checkpoint_15b)

start_index=0
if [[ -n "${PARTITION_8E_RESUME_CHECKPOINT:-}" ]]; then
  resume_manifest="${PARTITION_8E_RESUME_CHECKPOINT}/checkpoint_manifest.json"
  [[ -s "${resume_manifest}" ]] || { echo "Missing checkpoint manifest: ${resume_manifest}" >&2; exit 1; }
  resume_global_tokens="$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["global_tokens"]))' "${resume_manifest}")"
  while (( start_index < ${#targets[@]} && ${targets[$start_index]} <= resume_global_tokens )); do
    start_index=$((start_index + 1))
  done
  if (( start_index >= ${#targets[@]} )); then
    echo "partition_8e_top4 is already at or beyond checkpoint_15b; nothing to resume."
    exit 0
  fi
fi

for index in $(seq "${start_index}" 4); do
  run_stage "${targets[$index]}" "${names[$index]}"
  if [[ -s "${OUTPUT_DIR}/stageB_process_summary.json" ]] && python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("stop_requested", False) else 1)' "${OUTPUT_DIR}/stageB_process_summary.json"; then
    echo "Graceful interruption checkpoint saved: ${OUTPUT_DIR}/checkpoint_signal_interrupt"
    echo "Resume with PARTITION_8E_RESUME_CHECKPOINT=${OUTPUT_DIR}/checkpoint_signal_interrupt"
    exit 0
  fi
  checkpoint_dir="${OUTPUT_DIR}/${names[$index]}"
  test -s "${checkpoint_dir}/trainer_state.pt"
  test -s "${checkpoint_dir}/checkpoint_manifest.json"
  test -s "${checkpoint_dir}/checkpoint_write_complete.json"
  test -s "${checkpoint_dir}/dataset_state_rank0.pt"
  RESUME_CHECKPOINT="${checkpoint_dir}"
done

echo "partition_8e_top4 15B training completed: ${OUTPUT_DIR}"
echo "final checkpoint: ${OUTPUT_DIR}/checkpoint_15b"
echo "launch log: ${LOG_PATH}"
