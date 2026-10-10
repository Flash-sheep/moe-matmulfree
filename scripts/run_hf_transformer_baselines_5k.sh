#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/storage/yjl/moe-matmulfree"
PY="${PY:-/opt/venv/bin/python}"
DATA_SOURCE="${DATA_SOURCE:-datasets/SlimPajama-6B/data}"
VAL_DATA_SOURCE="${VAL_DATA_SOURCE:-datasets/SlimPajama-6B/data}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
GPU_MAP="${GPU_MAP:-0,1}"

IFS=',' read -r GPU0 GPU1 <<< "${GPU_MAP}"
GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"

cd "${WORKDIR}"

echo "pwd: $(pwd)"
echo "python: $(${PY} --version)"
nvidia-smi || true
tmux ls || true

launch_one() {
  local model_path="$1"
  local output_dir="$2"
  local session_name="$3"
  local gpu_id="$4"

  if [ -d "${output_dir}" ] && [ "$(find "${output_dir}" -mindepth 1 -maxdepth 1 | wc -l)" -gt 0 ]; then
    echo "refusing to overwrite non-empty output dir: ${output_dir}"
    return 1
  fi
  if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session already exists: ${session_name}"
    return 1
  fi

  mkdir -p "${output_dir}"

  tmux new-session -d -s "${session_name}" \
    "cd '${WORKDIR}' && CUDA_VISIBLE_DEVICES=${gpu_id} ${PY} scripts/train_hf_transformer_baseline.py \
      --model-path '${model_path}' \
      --tokenizer-path '${model_path}' \
      --data-source '${DATA_SOURCE}' \
      --val-data-source '${VAL_DATA_SOURCE}' \
      --output-dir '${output_dir}' \
      --max-steps 5000 \
      --batch-size 2 \
      --gradient-accumulation-steps 16 \
      --max-length 2048 \
      --learning-rate 2e-5 \
      --weight-decay 0.01 \
      --warmup-steps 500 \
      --grad-clip 1.0 \
      --log-interval 25 \
      --eval-interval 1000 \
      --save-interval 1000 \
      --max-eval-batches 64 \
      --max-val-samples 1024 \
      --eval-batch-size 4 \
      --precision bf16 \
      --device cuda > '${output_dir}/host.log' 2>&1"
}

launch_one "checkpoints/SmolLM2-360M" "outputs/smollm2_360m_ft_5k" "smollm2_360m_5k" "${GPU0}"
launch_one "checkpoints/Qwen2.5-0.5B" "outputs/qwen25_05b_ft_5k" "qwen25_05b_5k" "${GPU1}"

echo "launched:"
echo "  tmux attach -t smollm2_360m_5k"
echo "  tmux attach -t qwen25_05b_5k"
echo "logs:"
echo "  tail -f outputs/smollm2_360m_ft_5k/host.log"
echo "  tail -f outputs/qwen25_05b_ft_5k/host.log"
