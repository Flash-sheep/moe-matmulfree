#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?Usage: finalize_dense_stageB_when_complete.sh OUTPUT_DIR}"
FINAL_CHECKPOINT="${OUTPUT_DIR}/checkpoint_final_1b/trainer_state.pt"
FINAL_MANIFEST="${OUTPUT_DIR}/checkpoint_final_1b/checkpoint_manifest.json"
PROCESS_SUMMARY="${OUTPUT_DIR}/stageB_process_summary.json"

while true; do
  if [[ -f "${OUTPUT_DIR}/failure_report.json" ]]; then
    echo "Stage B failure report detected: ${OUTPUT_DIR}/failure_report.json" >&2
    exit 1
  fi
  if [[ -s "${FINAL_CHECKPOINT}" ]] \
    && [[ -s "${FINAL_MANIFEST}" ]] \
    && [[ -s "${PROCESS_SUMMARY}" ]] \
    && grep -q '"final_optimizer_step": 7630' "${PROCESS_SUMMARY}"; then
    break
  fi
  sleep 60
done

docker run --rm \
  --user 1026:1027 \
  -v /home/storage/yjl:/home/storage/yjl \
  -w /home/storage/yjl/moe-matmulfree \
  matmulfreellm:latest \
  /opt/venv/bin/python \
  scripts/diagnostics/finalize_dense_stageB.py \
  --output-dir "${OUTPUT_DIR}"
