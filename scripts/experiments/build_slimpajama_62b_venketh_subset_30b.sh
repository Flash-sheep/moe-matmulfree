#!/usr/bin/env bash
set -euo pipefail

# DO NOT RUN 30B BUILD BY DEFAULT.
# Set RUN_BUILD=1 manually after confirming disk space and previous sanity reports.
RUN_BUILD=${RUN_BUILD:-0}

CMD="./.venv/bin/python scripts/build_mixed_pretrain_dataset.py \
  --config configs/datasets/slimpajama_62b_venketh_local.yaml \
  --output-dir /home/storage/yjl/matmulfreellm_assets/datasets/slimpajama_62b_venketh_subset_30b \
  --target-tokens 30000000000 \
  --dry-run false"

echo "$CMD"

if [ "$RUN_BUILD" = "1" ]; then
  eval "$CMD"
else
  echo "RUN_BUILD is not 1; command printed only. No 30B build started."
fi
