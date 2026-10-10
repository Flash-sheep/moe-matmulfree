#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/storage/yjl/moe-matmulfree
CONFIG="$ROOT/configs/pretrain/partition_capacity_4e_all_layers_random_scratch_relaxed30b_15b.yaml"
GPU_ID="${GPU_ID:-1}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/partition_capacity_4e_all_layers_random_scratch_relaxed30b_15b_${STAMP}}"
SESSION_NAME="${SESSION_NAME:-partition_capacity_4e_15b}"
LOG_PATH="${OUTPUT_DIR}.host.log"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  exit 1
fi
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Refusing to reuse active tmux session: $SESSION_NAME" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DIR")"

TRAIN_CMD="docker run --rm --gpus device=${GPU_ID} --user 1026:1027 \
  -e HOME=/tmp \
  -e TRITON_CACHE_DIR=/tmp/triton-cache \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e PYTHONHASHSEED=20260721 \
  -v /home/storage/yjl:/home/storage/yjl \
  -w /home/storage/yjl/moe-matmulfree \
  matmulfreellm:latest \
  bash -lc '/opt/venv/bin/python scripts/run_dense_stageB_ddp.py \
    --config-path configs/pretrain/partition_capacity_4e_all_layers_random_scratch_relaxed30b_15b.yaml \
    --output-dir ${OUTPUT_DIR}'"

echo "Starting background training session: $SESSION_NAME"
echo "Output directory: $OUTPUT_DIR"
echo "Host log: $LOG_PATH"
echo "$TRAIN_CMD"

tmux new-session -d -s "$SESSION_NAME" "cd '$ROOT' && $TRAIN_CMD 2>&1 | tee '$LOG_PATH'"
echo "Training started in tmux session $SESSION_NAME"
