# MoE Quant Phase 0 Baseline

This file is the Phase 0 execution baseline for validating whether the MatmulFreeLLM quantization scheme can work in an MoE architecture without unacceptable accuracy loss.

It is meant to be read together with `moe_quant_validation_guide.html`, which remains the higher-level project guide.

## Current repo state

- The repository currently contains one Hugging Face model family: `HGRNBit`.
- The repository does not contain an MoE implementation yet.
- The repository does not contain a full training pipeline yet.
- The repository already contains the quantized linear building blocks needed for first-pass MoE validation:
  - `mmfreelm.ops.bitnet.BitLinear`
  - `mmfreelm.ops.fusedbitnet.FusedBitLinear`
  - `RMSNorm`, `SwiGLU`, recurrent HGRN mixer, and fused loss components

## Phase 0 objective

Lock the first executable validation scope so Phase 1 and Phase 2 can proceed without baseline drift.

## First validation scope

The first validation target is intentionally narrow:

- Keep the current `HGRNBitAttention` path unchanged.
- Replace only the dense `HGRNBitMLP` with an MoE MLP.
- Quantize only the expert MLP internal linear layers using the existing MatmulFreeLLM scheme.
- Keep the router in floating point.
- Keep embeddings, recurrent mixer, norms, and `lm_head` in floating point for the first pass.

## Default implementation assumptions

Unless later overridden, the default implementation assumptions are:

- Backbone: current `HGRNBitModel`
- MoE insertion point: replace `HGRNBitMLP` inside `HGRNBitBlock`
- Expert architecture: `SwiGLU` MLP with `FusedBitLinear`
- Router architecture: standard floating-point linear layer
- Routing mode: `top-2` preferred, `top-1` as fallback ablation
- First comparison target: `FP/BF16 dense` vs `FP/BF16 MoE` vs `ternary-expert MoE`
- Success metric: relative validation perplexity degradation no worse than `3% to 5%`

## Initial code work breakdown

The first implementation slice should be:

1. Add MoE-capable config fields to a new or extended config path.
2. Implement a router module.
3. Implement a quantized expert MLP module using existing `FusedBitLinear`.
4. Implement token dispatch/combine in plain PyTorch first.
5. Integrate the MoE MLP into `HGRNBitBlock`.
6. Keep generation/cache semantics unchanged by avoiding MoE changes in the recurrent attention path.

## Phase 1 acceptance gate

Before ternary expert validation starts, the following must be true:

- A floating-point MoE version of the current HGRNBit block trains stably.
- Router statistics can be logged.
- Expert token distribution can be measured.
- There is a reproducible dense baseline and a reproducible floating-point MoE baseline.

If any of the above is not true, do not move on to ternary expert validation.

## Phase 2 acceptance gate

The first quantized MoE experiment passes only if:

- Training does not diverge.
- No obvious expert collapse appears.
- Validation perplexity degradation stays within the target threshold.
- The result is repeatable across more than one seed.

## Known gaps to close next

- No MoE module exists yet.
- No experiment runner exists yet.
- No logging schema for router/expert metrics exists yet.
- No explicit baseline config file exists yet.

## Immediate next task

Implement the minimum MoE model path needed to support:

- `FP/BF16 MoE baseline`
- `ternary expert MoE`
- later ablations on `top-k`, load-balance coefficient, and warm-start
