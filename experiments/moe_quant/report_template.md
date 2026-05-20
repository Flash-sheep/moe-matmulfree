# MoE Quant Report Template

## 1. Experiment setup

- model:
- dataset:
- tokenizer:
- sequence length:
- optimizer:
- learning rate schedule:
- batch size:
- training steps:
- seed:
- `use_moe`:
- `moe_num_experts`:
- `moe_num_experts_per_tok`:
- `moe_use_quantized_experts`:
- `moe_router_aux_loss_coef`:

## 2. Main results

| Run | Quant scope | Val loss | Val ppl | Main task metric | Stable | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Dense FP | none |  |  |  |  |  |
| MoE FP | none |  |  |  |  |  |
| MoE ternary experts | expert MLP |  |  |  |  |  |

## 3. Router and expert behavior

- router entropy:
- average tokens per expert:
- route load:
- expert collapse observed:
- auxiliary loss behavior:

## 4. Acceptance decision

- Phase 1 pass/fail:
- Phase 2 pass/fail:
- Phase 3 pass/fail:

## 5. Final conclusion

Use this sentence structure:

`Under [quant scope], relative to the FP MoE baseline, [metric] changed by [x%]. Training was [stable/unstable]. The project goal was [met/not met]. The main risk concentrated in [router/load-balance/convergence].`
