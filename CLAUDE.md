# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project for **MatMul-Free Language Modeling** with **Mixture-of-Experts (MoE) sparse upcycling**. Eliminates matrix multiplications from LLM inference by using ternary-quantized weights ({-1, 0, 1}) and 8-bit quantized activations, combined with recurrent HGRN attention (not Transformer self-attention). Implemented in Python with PyTorch + Triton CUDA kernels, HuggingFace `transformers` integration.

Paper: [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)

## Environment & Package Management

- **Package manager**: `uv` (with `.venv` at repo root)
- **Python**: >= 3.9
- **Package index**: Tsinghua PyPI mirror configured in `pyproject.toml`
- **Runtime options**: Docker image `matmulfreellm:cu126-py310` or Conda via `source scripts/activate_conda_runtime.sh`
- **Import caveat**: `import mmfreelm` works from repo root only; the package is not installed globally

## Path Conventions

All paths must be repo-relative (from `/home/storage/yjl/moe-matmulfree`):

- `checkpoints/MMfreeLM-370M` — pretrained model
- `datasets/SlimPajama-6B/data` — training/eval data (parquet shards with `train-*`, `validation-*`, `test-*` splits)
- `outputs/<experiment_name>` — experiment outputs
- `data/` — raw evaluation datasets (wikitext2, wikitext103)

These are symlinks into `/home/storage/yjl/matmulfreellm_assets/`. Never use legacy absolute paths like `/home/yjl/...` or `/home/data/yjl/...`.

HuggingFace cache is at `/home/storage/yjl/hf_home/`.

## Key Commands

### Preflight check (verify config, model, and data without training)

```bash
python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path experiments/sparse_upcycling/configs/<config>.json \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir outputs/<new_experiment_name> \
  --preflight-only
```

### Standard training

```bash
python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path experiments/sparse_upcycling/configs/<config>.json \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir outputs/<new_experiment_name>
```

### Short sanity run (max-steps 2–5, batch-size 1, new output dir only)

```bash
python scripts/run_sparse_upcycling.py \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --config-path experiments/sparse_upcycling/configs/<config>.json \
  --data-source datasets/SlimPajama-6B/data \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M \
  --output-dir outputs/logging_fix_sanity_<timestamp> \
  --max-steps 2 --batch-size 1 --gradient-accumulation-steps 1
```

### Offline postprocess / formal evaluation (after training completes)

```bash
python scripts/postprocess_sparse_upcycling.py \
  --config-path experiments/sparse_upcycling/configs/<config>.json \
  --pretrained-path checkpoints/MMfreeLM-370M \
  --output-dir outputs/<experiment_name> \
  --val-data-source datasets/SlimPajama-6B/data \
  --tokenizer-path checkpoints/MMfreeLM-370M
```

### Replot training curves from existing logs

```bash
python scripts/plot_training_curves.py \
  --log outputs/<experiment_name>/train_log.jsonl \
  --eval1024 outputs/<experiment_name>/eval_results_1024.json \
  --label <experiment_name> \
  --out-dir outputs/<experiment_name>/logging_fix_preview \
  --window 100
```

### Static compile checks

```bash
python -m py_compile scripts/run_sparse_upcycling.py
python -m py_compile scripts/postprocess_sparse_upcycling.py
python -m py_compile scripts/plot_training_curves.py
python -m py_compile scripts/train_moe_lm.py
```

### Docker launch pattern

```bash
docker run --rm --gpus 'device=0' \
  -v /home/storage/yjl:/home/storage/yjl \
  -w /home/storage/yjl/moe-matmulfree \
  matmulfreellm:cu126-py310 \
  bash -lc '<command>'
```

## Architecture

The codebase has four layers, all under `mmfreelm/`:

### 1. `ops/` — Low-level GPU kernels (Triton)
- `fusedbitnet.py` — Fused BitLinear with RMSNorm pre-normalization (ternary weight quant + int8 activation quant + STE backward)
- `bitnet.py` — Non-fused BitLinear reference
- `hgrn/` — Recurrent HGRN kernels (fused, chunked, naive)

### 2. `modules/` — Reusable neural network building blocks
- `moe.py` — **Core MoE**: `TopKRouter` (6 routing modes), `ExpertMLP` (quantized or FP), `SparseMoEBlock`, `build_expert_from_mlp_state`
- `layernorm.py` — `RMSNorm`, `LayerNorm`, `RMSNormLinear`
- `fused_norm_gate.py` — `FusedRMSNormSwishGate`
- `fused_cross_entropy.py` — Triton-fused cross entropy loss
- `convolution.py` — Implicit/long/short convolutions
- `activations.py` — Activation functions including JIT-compiled CUDA `swiglu`

### 3. `layers/` — Semantic network sub-structures
- `hgrn_bit.py` — `HGRNBitAttention` (recurrent attention, not Transformer self-attention)

### 4. `models/` — HuggingFace-compatible model wrappers
- `hgrn_bit/configuration_hgrn_bit.py` — `HGRNBitConfig` with 60+ MoE parameters
- `hgrn_bit/modeling_hgrn_bit.py` — `HGRNBitBlock`, `HGRNBitMLP`, `HGRNBitForCausalLM`
- `hgrn_bit/__init__.py` — Registers with HF `AutoConfig`/`AutoModel`/`AutoModelForCausalLM`

### Sparse Upcycling Pipeline: `mmfreelm/upcycling/`
- `sparse_upcycling.py` — `upcycle_dense_to_moe()`: converts dense → MoE with 7 init methods (copy_noise, svd_orthogonal, partition, virtual_group_partition, complement_pair_6e, complement_copy_12e, etc.)
- `freeze.py` — Selective parameter freezing for upcycling
- `param_groups.py` — Builds 4-group optimizer param groups (moe, backbone, norm_or_bias, embed_lm_head) with separate LRs
- `data_utils.py` — `StreamingTextDataset`: streaming iterable dataset for parquet/jsonl/txt/md with split filtering
- `expert_monitor.py` — Tracks expert similarity norms and router statistics
- `svd_init.py` — SVD and partition-based expert initialization

### Training Scripts (top-level entrypoints)
- `scripts/run_sparse_upcycling.py` — Primary training main loop (load checkpoint → convert to MoE → train with logging/checkpointing)
- `scripts/postprocess_sparse_upcycling.py` — Offline formal eval producing `eval_results_64.json`, `eval_results_1024.json`, `pair_usage_*.json`, `training_report.json`
- `scripts/plot_training_curves.py` — Render `loss_curve.png`, `lm_loss_curve.png`, `lr_curve.png`
- `scripts/train_moe_lm.py` — General MoE LM training (dense/FP-MoE/ternary-MoE)
- `scripts/preflight_sparse_upcycling.py` — Config verification

## Experiment Config Structure

JSON configs in `experiments/sparse_upcycling/configs/` have three sections:
- **`moe`**: layer_indices, num_experts, top_k, routing_mode, init_method, noise parameters, quantized_experts, aux loss coefficient
- **`freeze`**: freeze_mode, freeze_embeddings, freeze_lm_head
- **`training`**: batch_size, gradient_accumulation_steps, max_steps, learning_rate, precision, warmup, eval/save intervals

## Validation & Checkpoint Semantics

Two distinct evaluation scopes — do not conflate them:

- **Proxy validation** (training-time periodic): `proxy_val_loss`, `proxy_val_lm_loss`, `proxy_val_ppl`. Training best checkpoint is `checkpoint_best_proxy` (metric: `proxy_val_loss`).
- **Formal evaluation** (offline postprocess): Primary metric is `eval_results_1024.json` (`PPL@1024`). Do not use `PPL@64` as the main decision criterion when `PPL@1024` exists.

## Logging Semantics (post-May 2026 fix)

- `train_loss` = normalized optimizer-step loss (not raw accumulated micro-batch loss)
- `normalized_lm_loss`, `normalized_router_aux_loss` — per-step normalized components
- `lm_loss`, `router_aux_loss` — log-window averages
- LR fields: `lr`, `lr_moe`, `lr_backbone`, `lr_norm_or_bias`, `lr_embed_lm_head`
- Historical `train_loss ~= 45` in older logs was a normalization artifact, not real LM loss

## Safety Rules

- **Never** overwrite existing experiment output directories
- **Never** delete historical `outputs/`
- **Never** modify `current_experiment_matrix*` unless explicitly asked
- Before any GPU launch, check: `nvidia-smi`, `tmux ls`, `ps aux | grep run_sparse_upcycling`
- If another training job is active, avoid starting GPU work unless explicitly intended
- New runs must always target a new `--output-dir`
- Sanity runs: `max_steps <= 5`, new output dir, no reuse of historical dirs
