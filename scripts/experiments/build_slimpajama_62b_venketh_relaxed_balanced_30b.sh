#!/usr/bin/env bash
set -euo pipefail

RUN_BUILD=${RUN_BUILD:-0}
ROOT="/home/storage/yjl/moe-matmulfree"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="$ROOT/configs/datasets/slimpajama_62b_venketh_relaxed_balanced_30b.yaml"
OUTPUT_DIR="/home/storage/yjl/matmulfreellm_assets/datasets/slimpajama_62b_venketh_relaxed_balanced_30b"
LOG_DIR="$ROOT/outputs/build_logs/slimpajama_62b_venketh_relaxed_balanced_30b"
BUILD_LOG="$LOG_DIR/build.log"
INSPECT_LOG="$LOG_DIR/inspect.log"

BUILD_CMD="$PYTHON_BIN $ROOT/scripts/build_mixed_pretrain_dataset.py --config $CONFIG_PATH --output-dir $OUTPUT_DIR --target-tokens 30000000000 --dry-run false"
INSPECT_CMD="$PYTHON_BIN $ROOT/scripts/inspect_mixed_dataset.py --dataset-dir $OUTPUT_DIR --max-samples 100"

echo "Relaxed-balanced 30B build command:"
echo "$BUILD_CMD"
echo
echo "Inspection command:"
echo "$INSPECT_CMD"

if [ "$RUN_BUILD" != "1" ]; then
  echo "RUN_BUILD is not 1; commands printed only. No dataset build started."
  exit 0
fi

if [ -d "$OUTPUT_DIR" ] && [ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 2>/dev/null | head -n 1)" ]; then
  echo "Output directory already exists and is non-empty: $OUTPUT_DIR"
  exit 1
fi

mkdir -p "$LOG_DIR"
echo "Writing logs to $LOG_DIR"

eval "$BUILD_CMD" 2>&1 | tee "$BUILD_LOG"

test -f "$OUTPUT_DIR/dataset_manifest.json"
test -f "$OUTPUT_DIR/dataset_build_config_resolved.yaml"

eval "$INSPECT_CMD" 2>&1 | tee "$INSPECT_LOG"

test -f "$OUTPUT_DIR/dataset_inspection_report.json"
grep -q '"success": true' "$OUTPUT_DIR/dataset_inspection_report.json"

echo "Build complete."
echo "Manifest: $OUTPUT_DIR/dataset_manifest.json"
echo "Inspection: $OUTPUT_DIR/dataset_inspection_report.json"
