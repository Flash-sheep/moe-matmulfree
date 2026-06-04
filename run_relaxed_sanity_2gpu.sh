#!/usr/bin/env bash
set -euo pipefail

cd /home/storage/yjl/moe-matmulfree

echo "========== Environment check =========="
pwd
nvidia-smi || true
tmux ls || true
ps aux | grep run_sparse_upcycling | grep -v grep || true
echo

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

EXP_A="sanity_relaxed_lambda000_${TIMESTAMP}"
EXP_B="sanity_relaxed_lambda020_${TIMESTAMP}"

CONFIG_A="experiments/sparse_upcycling/configs/complement6e_relaxed_top2_lambda000_alpha005_aux0001_5000step.json"
CONFIG_B="experiments/sparse_upcycling/configs/complement6e_relaxed_top2_lambda020_alpha005_aux0001_5000step.json"

OUT_A="outputs/${EXP_A}"
OUT_B="outputs/${EXP_B}"

LOG_A="${OUT_A}.host.log"
LOG_B="${OUT_B}.host.log"

echo "========== Config check =========="
if [ ! -f "${CONFIG_A}" ]; then
  echo "[ERROR] Missing config: ${CONFIG_A}"
  exit 1
fi

if [ ! -f "${CONFIG_B}" ]; then
  echo "[ERROR] Missing config: ${CONFIG_B}"
  exit 1
fi

echo "[OK] Found config A: ${CONFIG_A}"
echo "[OK] Found config B: ${CONFIG_B}"
echo

echo "========== Create output dirs =========="
mkdir -p "${OUT_A}" "${OUT_B}"
echo "[OK] OUT_A=${OUT_A}"
echo "[OK] OUT_B=${OUT_B}"
echo

echo "========== Preflight A: lambda=0.0 on GPU0 =========="
CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path "${CONFIG_A}" \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir "${OUT_A}_preflight" \
  --preflight-only \
  2>&1 | tee "${OUT_A}_preflight.log"

echo "========== Preflight B: lambda=0.2 on GPU1 =========="
CUDA_VISIBLE_DEVICES=1 /opt/venv/bin/python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path "${CONFIG_B}" \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir "${OUT_B}_preflight" \
  --preflight-only \
  2>&1 | tee "${OUT_B}_preflight.log"

echo
echo "========== Launch 2 short sanity runs in parallel =========="

CUDA_VISIBLE_DEVICES=0 /opt/venv/bin/python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path "${CONFIG_A}" \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir "${OUT_A}" \
  --max-steps 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  2>&1 | tee "${LOG_A}" &
PID_A=$!

CUDA_VISIBLE_DEVICES=1 /opt/venv/bin/python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path "${CONFIG_B}" \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir "${OUT_B}" \
  --max-steps 3 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  2>&1 | tee "${LOG_B}" &
PID_B=$!

echo "[INFO] Started A PID=${PID_A}, GPU0, output=${OUT_A}"
echo "[INFO] Started B PID=${PID_B}, GPU1, output=${OUT_B}"
echo

wait "${PID_A}"
STATUS_A=$?

wait "${PID_B}"
STATUS_B=$?

echo
echo "========== Run status =========="
echo "A status: ${STATUS_A}"
echo "B status: ${STATUS_B}"

if [ "${STATUS_A}" -ne 0 ]; then
  echo "[ERROR] Experiment A failed. Check log: ${LOG_A}"
fi

if [ "${STATUS_B}" -ne 0 ]; then
  echo "[ERROR] Experiment B failed. Check log: ${LOG_B}"
fi

echo
echo "========== Output file check =========="

check_file() {
  local path="$1"
  if [ -f "$path" ]; then
    echo "[OK] $path"
  else
    echo "[MISSING] $path"
  fi
}

echo "--- Experiment A files ---"
check_file "${OUT_A}/init_verification.json"
check_file "${OUT_A}/trainable_param_summary.json"
check_file "${OUT_A}/optimizer_param_groups.json"
check_file "${OUT_A}/train_log.jsonl"
check_file "${OUT_A}/training_report.json"
check_file "${LOG_A}"

echo
echo "--- Experiment B files ---"
check_file "${OUT_B}/init_verification.json"
check_file "${OUT_B}/trainable_param_summary.json"
check_file "${OUT_B}/optimizer_param_groups.json"
check_file "${OUT_B}/train_log.jsonl"
check_file "${OUT_B}/training_report.json"
check_file "${LOG_B}"

echo
echo "========== Last train logs =========="
echo "--- A train_log tail ---"
if [ -f "${OUT_A}/train_log.jsonl" ]; then
  tail -n 5 "${OUT_A}/train_log.jsonl"
fi

echo
echo "--- B train_log tail ---"
if [ -f "${OUT_B}/train_log.jsonl" ]; then
  tail -n 5 "${OUT_B}/train_log.jsonl"
fi

echo
echo "========== Done =========="
echo "A output: ${OUT_A}"
echo "B output: ${OUT_B}"
echo "A log: ${LOG_A}"
echo "B log: ${LOG_B}"