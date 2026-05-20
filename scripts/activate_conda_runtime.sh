#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/yjl/a100-r760/matmulfreellm"
CONDA_ROOT="/home/yjl/anaconda3"
CONDA_ENV_PREFIX="/home/yjl/conda_envs/matmulfreellm_runtime"

if [ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  echo "Missing conda initialization script: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

. "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_PREFIX}"

# Keep the project environment isolated from user-site packages.
export PYTHONNOUSERSITE=1
export HF_HOME="/home/data/yjl/hf_home"
export TRANSFORMERS_CACHE="/home/data/yjl/hf_home/transformers"
export HF_DATASETS_CACHE="/home/data/yjl/hf_home/datasets"

cd "${REPO_ROOT}"

echo "Activated conda runtime: ${CONDA_ENV_PREFIX}"
echo "Repo root: ${REPO_ROOT}"
echo "Assets:"
echo "  checkpoints -> $(readlink -f "${REPO_ROOT}/checkpoints")"
echo "  data        -> $(readlink -f "${REPO_ROOT}/data")"
echo "  outputs     -> $(readlink -f "${REPO_ROOT}/outputs")"
