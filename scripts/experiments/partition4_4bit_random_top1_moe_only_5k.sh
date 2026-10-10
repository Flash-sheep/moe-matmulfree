#!/usr/bin/env bash
set -euo pipefail

# DO NOT RUN TRAINING BY DEFAULT
# This script is a template only.
# Remove DRY_RUN=1 manually before launching training.
DRY_RUN="${DRY_RUN:-1}"

REPO_ROOT="/home/storage/yjl/moe-matmulfree"
cd "${REPO_ROOT}"

CMD=(
  /opt/venv/bin/python
  scripts/run_sparse_upcycling.py
  --pretrained-path checkpoints/MMfreeLM-370M
  --config-path experiments/sparse_upcycling/configs/partition4_4bit_random_top1_moe_only_5k.json
  --data-source datasets/SlimPajama-6B/data
  --val-data-source datasets/SlimPajama-6B/data
  --tokenizer-path checkpoints/MMfreeLM-370M
  --output-dir outputs/partition4_4bit_random_top1_moe_only_5k
)

echo "Command template:"
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1, not launching training."
  exit 0
fi

exec "${CMD[@]}"
