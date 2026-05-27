# Manual Training Guide

This guide describes how to run the MoE quantization validation workflow manually with the training infrastructure currently implemented in this repository.

It complements:

- `moe_quant_validation_guide.html`
- `moe_quant_phase0_baseline.md`
- `experiments/moe_quant/phase_matrix.md`

## 1. What is already implemented

The repository now contains:

- dense HGRNBit training path
- floating-point MoE HGRNBit training path
- ternary-expert MoE HGRNBit training path
- router auxiliary loss logging
- router entropy and expert usage logging
- validation loss / perplexity reporting
- checkpoint save / resume support

The main training entrypoint is:

- `scripts/train_moe_lm.py`

## 2. What you still need to provide

Before running real training, you still need to provide:

- a tokenizer name or local tokenizer path
- a training dataset path
- a validation dataset path
- the final hardware budget and target model scale

## 3. Supported dataset formats

The training script currently supports:

- `.txt`
- `.md`
- `.jsonl`
- `.json`
- `.pt`
- a directory that contains any of the text-based formats above

### Text-based formats

For `.txt` and `.md`, the file is read as plain text and tokenized directly.

For `.jsonl` and `.json`, the script reads a text field. By default that field is `text`, but you can override it with:

`--text-field your_field_name`

### Pretokenized format

For `.pt`, the script expects one of:

- a 1D token tensor
- a list of token ids
- a dict with `input_ids`

## 4. Recommended execution order

Run the experiments in this order:

1. `dense_fp_baseline`
2. `fp_moe_baseline`
3. `ternary_expert_moe_baseline`

Do not start the ternary-expert run until the floating-point MoE run is stable.

## 5. How to launch

### Dense baseline

```bash
cd /path/to/moe-matmulfree
python3 scripts/train_moe_lm.py \
  --config-file experiments/moe_quant/configs/dense_fp_baseline.json \
  --tokenizer YOUR_TOKENIZER \
  --train-data YOUR_TRAIN_DATA \
  --val-data YOUR_VAL_DATA \
  --output-dir experiments/moe_quant/runs/dense_fp_baseline
```

### Floating-point MoE baseline

```bash
cd /path/to/moe-matmulfree
python3 scripts/train_moe_lm.py \
  --config-file experiments/moe_quant/configs/fp_moe_baseline.json \
  --tokenizer YOUR_TOKENIZER \
  --train-data YOUR_TRAIN_DATA \
  --val-data YOUR_VAL_DATA \
  --output-dir experiments/moe_quant/runs/fp_moe_baseline
```

### Ternary-expert MoE baseline

```bash
cd /path/to/moe-matmulfree
python3 scripts/train_moe_lm.py \
  --config-file experiments/moe_quant/configs/ternary_expert_moe_baseline.json \
  --tokenizer YOUR_TOKENIZER \
  --train-data YOUR_TRAIN_DATA \
  --val-data YOUR_VAL_DATA \
  --output-dir experiments/moe_quant/runs/ternary_expert_moe_baseline
```

## 6. Resume training

To resume from a checkpoint:

```bash
python3 scripts/train_moe_lm.py \
  --config-file ... \
  --tokenizer ... \
  --train-data ... \
  --val-data ... \
  --output-dir ... \
  --resume-from PATH_TO_CHECKPOINT_DIR
```

`PATH_TO_CHECKPOINT_DIR` should point to a directory like:

`experiments/moe_quant/runs/fp_moe_baseline/checkpoints/step_0000200`

## 7. Key outputs

Each run writes:

- `run_args.json`
- `train_log.jsonl`
- `checkpoints/`

Important fields in `train_log.jsonl` include:

- `train_loss`
- `lm_loss`
- `router_aux_loss`
- `grad_norm`
- `lr`
- `router_entropy`
- `tokens_per_expert`
- `val_loss`
- `val_ppl`

## 8. Phase acceptance checklist

### Phase 1

The floating-point MoE run passes Phase 1 only if:

- loss remains finite
- validation perplexity is produced
- `router_entropy` remains finite
- `tokens_per_expert` is not degenerate

### Phase 2

The ternary-expert MoE run passes Phase 2 only if:

- loss remains finite
- router metrics stay reasonable
- no immediate expert collapse appears
- relative validation degradation stays within the target threshold

## 9. Recommended next ablations after the first three runs

After the first three runs are complete, expand in this order:

1. multi-seed repeat
2. `top-1` versus `top-2`
3. different `moe_router_aux_loss_coef`
4. warm-start ternary experts from FP MoE checkpoint
5. optional quantized `lm_head`

## 10. Known limitations of the current training script

The current implementation is intentionally minimal. It does not yet provide:

- distributed multi-GPU training
- packed streaming datasets
- external experiment trackers
- automatic phase pass/fail decisions
- downstream task evaluation

It is designed to be the first real training shell for the MoE quantization validation project, not the final production trainer.
