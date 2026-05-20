# Runtime Environment And Storage Layout

## Large-file storage

The project worktree remains in:

- `/home/yjl/a100-r760/matmulfreellm`

Large files are stored under:

- `/home/data/yjl/matmulfreellm_assets/checkpoints`
- `/home/data/yjl/matmulfreellm_assets/data`
- `/home/data/yjl/matmulfreellm_assets/outputs`

The worktree paths below are symlinks into `/home/data/yjl`:

- `checkpoints`
- `data`
- `outputs`

As long as training and export scripts keep writing into those repo-relative paths, checkpoints and model weights will land under `/home/data/yjl`.

## Conda runtime

The Conda environment prefix is:

- `/home/yjl/conda_envs/matmulfreellm_runtime`

Project-local reference:

- `/home/yjl/a100-r760/matmulfreellm/.conda_runtime_env`

Activation helper:

- `/home/yjl/a100-r760/matmulfreellm/scripts/activate_conda_runtime.sh`

Usage:

```bash
source /home/yjl/a100-r760/matmulfreellm/scripts/activate_conda_runtime.sh
```

That helper:

- activates the Conda environment
- sets `PYTHONNOUSERSITE=1`
- pins Hugging Face caches to `/home/data/yjl/hf_home`
- prints the resolved asset directories

## Current state

The Conda environment itself has been created, but GPU runtime packages may still need to finish installing if a network-interrupted install was left incomplete.

Recommended verification command:

```bash
source /home/yjl/a100-r760/matmulfreellm/scripts/activate_conda_runtime.sh
python -c "import torch, transformers, mmfreelm; print(torch.__version__, transformers.__version__)"
```
