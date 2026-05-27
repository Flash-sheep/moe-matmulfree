# Runtime Environment And Storage Layout

## Large-file storage

This repository is expected to live under a movable storage root, for example:

- `/home/storage/yjl/moe-matmulfree`

Large-file assets should live in the sibling directory:

- `../matmulfreellm_assets/checkpoints`
- `../matmulfreellm_assets/data`
- `../matmulfreellm_assets/datasets`
- `../matmulfreellm_assets/outputs`

The worktree paths below should be symlinks into that sibling assets directory:

- `checkpoints -> ../matmulfreellm_assets/checkpoints`
- `data -> ../matmulfreellm_assets/data`
- `datasets -> ../matmulfreellm_assets/datasets`
- `outputs -> ../matmulfreellm_assets/outputs`

As long as training and export scripts keep writing into those repo-relative paths, checkpoints and model weights will land under the mounted assets directory rather than a machine-specific absolute path.

## Conda runtime

The Conda environment prefix is:

- `${CONDA_ENV_PREFIX:-/home/yjl/conda_envs/matmulfreellm_runtime}`

Project-local reference:

- `./.conda_runtime_env`

Activation helper:

- `./scripts/activate_conda_runtime.sh`

Usage:

```bash
source ./scripts/activate_conda_runtime.sh
```

That helper:

- activates the Conda environment
- sets `PYTHONNOUSERSITE=1`
- pins Hugging Face caches to the sibling `../hf_home`
- allows overriding `CONDA_ROOT` / `CONDA_ENV_PREFIX` through environment variables
- prints the resolved asset directories

## Current state

The Conda environment itself has been created, but GPU runtime packages may still need to finish installing if a network-interrupted install was left incomplete.

Recommended verification command:

```bash
source ./scripts/activate_conda_runtime.sh
python -c "import torch, transformers, mmfreelm; print(torch.__version__, transformers.__version__)"
```
