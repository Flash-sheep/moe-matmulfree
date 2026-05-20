# MoE Quant Phase Matrix

This file translates the project guide into concrete experiment groups.

## Completed local self-check groups

- `dense_smoke`
- `fp_moe_smoke`
- `ternary_moe_smoke`

Artifacts are stored in `experiments/moe_quant/logs/`.

## Next real validation groups

1. Dense floating-point baseline
2. MoE floating-point baseline
3. MoE ternary experts
4. MoE ternary experts + warm-start
5. MoE ternary experts with `top-1`
6. MoE ternary experts with `top-2`
7. MoE ternary experts with different router auxiliary coefficients
8. Optional: MoE ternary experts + quantized `lm_head`

## Required per-run logs

- validation loss / perplexity
- train loss
- router auxiliary loss
- router entropy
- tokens per expert
- route load per expert
- training divergence markers
- seed

## Phase gates

- Phase 1 passes only if the floating-point MoE baseline is stable and logs expert usage.
- Phase 2 passes only if ternary experts remain stable and do not immediately collapse.
- Phase 3 requires repeated runs across multiple seeds.
- Phase 4 starts only after Phase 2 is accepted.
