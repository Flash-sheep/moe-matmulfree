# Current Experiment Matrix

本文档用于维护当前 sparse upcycling 实验的横向对照矩阵，优先记录已经完成统一评测的实验。

数据来源：

- `outputs/*/train_log.jsonl`
- `outputs/*/eval_results_64.json`
- `outputs/*/eval_results_1024.json`
- `current_project_status.md`

口径说明：

- `2026-05-25` 已对本文件涉及的主线实验做统一离线复评。
- 这里的 `eval_results_64` 统一指当前口径：`batch_size=4`, `max_samples=256`。
- 更早记录里的 `15.x` 级 `PPL@64` 属于旧口径结果，不再用于这里的横向比较。

## 1. `router_aux_loss_coef` 5k sweep

实验前缀：

- `complement6e_half_top2_alpha005_*_5000step`

固定条件：

- `6 experts`
- `top-2 strict complement pair routing`
- `noise_alpha = 0.05`
- `5000 training steps`

| Experiment | `router_aux_loss_coef` | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `complement6e_half_top2_alpha005_aux001_5000step` | `0.001` | `15.8242` | `17.0793` | `17.1550` | `1.6612` | `0.272489` | completed |
| `complement6e_half_top2_alpha005_aux0005_5000step` | `0.0005` | `15.8199` | `17.0664` | `17.1369` | `1.6879` | `0.272484` | completed |
| `complement6e_half_top2_alpha005_aux0001_5000step` | `0.0001` | `15.7974` | `17.0382` | `17.1159` | `1.7346` | `0.272485` | completed |
| `complement6e_half_top2_alpha005_aux00005_5000step` | `0.00005` | `15.8008` | `17.0391` | `17.1125` | `1.7414` | `0.272483` | completed |

当前读取结论：

- 按 `eval_results_64` 看，`aux=0.0001` 最优。
- 按 `eval_results_1024` 看，`aux=0.00005` 最优。
- 这四组的 `avg expert similarity` 几乎相同，差异主要体现在验证集指标和 router entropy。

## 2. Completed Follow-Ups

| Experiment | Change | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complement6e_half_top2_alpha005_aux0001_20000step` | keep layer `12-23`, extend best 5k `aux=0.0001` setting to `20000` steps | `15.4997` | `16.7260` | `16.8139` | `1.7675` | `0.268614` | completed |
| `complement6e_half_top2_alpha005_aux0001_layers6_23_5000step` | expand MoE layers from `12-23` to `6-23`, keep `aux=0.0001` and `5000` steps | `15.9158` | `17.1535` | `17.2318` | `1.7359` | `0.272888` | completed |

当前读取结论：

- `aux=0.0001` 的 scratch `20k` 已经明显优于之前的 `aux=0.005` `20k` 主线。
- 将 MoE layer 从 `12-23` 扩到 `6-23` 后，`5k` 指标反而劣于标准 `12-layer` 版本，说明“多替几层”当前不是更高优先级方向。

## 3. Local Adaptation Follow-Ups

| Experiment | Change | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `complement6e_half_top2_alpha005_aux0001_unfreeze_moe_norm_5000step` | unfreeze only `layer12-23` `mlp_norm` around the MoE blocks, keep fixed `moe_output_scale=2.0` | `15.8002` | `17.0522` | `17.1132` | `1.7351` | `0.272481` | completed with offline re-eval |
| `complement6e_half_top2_alpha005_aux0001_learnable_pair_scale_5000step` | keep RMSNorm frozen, add learnable positive per-layer/per-pair output scales initialized at `2.0` | `15.8222` | `17.0573` | `17.1193` | `1.7356` | `0.272543` | completed with offline re-eval |

当前读取结论：

- 这两组本地适配增强都没有实质性超过标准 `aux=0.0001` `5k` 基线的 `17.0382 / 17.1159`。
- `unfreeze moe norm` 在 `1024-sample` 上略好于基线，但 `256-sample` 上也略差于基线，整体更像是“基本打平”而不是明确增益。
- `learnable pair-scale` 学到的 scale 明显低于初始 `2.0`，但在统一复评口径下仍略差于 fixed-scale 基线，因此当前优先级也不高。
