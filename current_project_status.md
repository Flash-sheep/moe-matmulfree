# Current Project Status

本文档总结当前 `matmulfreellm` 仓库在以下几个方面的状态：

1. 代码结构
2. 模型训练能力的支持程度
3. 对 MoE 模型训练的支持程度
4. Sparse upcycling 实验进度
5. 当前环境状态

## 1. 当前代码结构

仓库目前的核心代码都集中在 `mmfreelm/` 下，结构可以分成四层。

### 1.1 `models/`

这一层是 Hugging Face 兼容的模型封装层。

- 当前主模型家族是 `HGRNBit`
- 配置定义在 `mmfreelm/models/hgrn_bit/configuration_hgrn_bit.py`
- 模型定义在 `mmfreelm/models/hgrn_bit/modeling_hgrn_bit.py`
- Hugging Face 注册入口在 `mmfreelm/models/hgrn_bit/__init__.py`

当前已经从纯 `HGRNBit` 扩展到了可选 `MoE` 路径：

- `use_moe=false` 时走原始 dense MLP
- `use_moe=true` 时在 block 内改走 `MoE MLP`

### 1.2 `layers/`

这一层承载语义级别的网络子结构。

- 当前最关键的是 `mmfreelm/layers/hgrn_bit.py`
- 它实现了 `HGRNBitAttention`
- 这一部分仍然是 recurrent HGRN 路径，不是 Transformer 自注意力

### 1.3 `modules/`

这一层是可复用功能模块。

已有模块包括：

- `RMSNorm / LayerNorm`
- `FusedRMSNormSwishGate`
- `ShortConvolution`
- `FusedCrossEntropyLoss`
- activation helpers，如 `swiglu`

当前新增的 `MoE` 相关模块在：

- `mmfreelm/modules/moe.py`

其中包含：

- `TopKRouter`
- `ExpertMLP`
- `SparseMoEBlock`

### 1.4 `ops/`

这一层是底层算子实现，包含 Triton kernel 与量化线性层。

关键文件包括：

- `mmfreelm/ops/fusedbitnet.py`
- `mmfreelm/ops/bitnet.py`
- `mmfreelm/ops/hgrn/recurrent_fuse.py`

当前量化方案的核心仍然是：

- activation `per-token int8 fake quant`
- weight ternary projection `{-1, 0, 1}`
- `STE` 反传

注意：当前仓库仍然是“量化训练逻辑 + fused kernel”模式，不是完整的专用 ternary runtime。

## 2. 当前模型训练能力的支持程度

### 2.1 已具备的训练能力

当前仓库已经具备一个最小可执行训练闭环，入口为：

- `scripts/train_moe_lm.py`

该脚本当前支持：

- dense HGRNBit 训练
- floating-point MoE HGRNBit 训练
- ternary-expert MoE HGRNBit 训练
- `train/val` 双数据集输入
- tokenizer 加载
- 固定长度 token packing
- 训练日志输出
- 验证 `loss / ppl`
- checkpoint 保存与恢复

支持的数据格式包括：

- `.txt`
- `.md`
- `.jsonl`
- `.json`
- `.pt`
- 目录形式的文本数据集合

### 2.2 当前训练日志能力

训练脚本可以记录以下关键字段：

- `train_loss`
- `lm_loss`
- `router_aux_loss`
- `grad_norm`
- `lr`
- `router_entropy`
- `tokens_per_expert`
- `val_loss`
- `val_ppl`

这意味着当前已经具备执行 `Phase 1 / Phase 2` 所需最基本日志条件。

### 2.3 当前训练能力的限制

当前训练基础设施仍然是最小版本，还没有这些能力：

- 多机多卡分布式训练
- `DDP / torchrun` 正式支持
- 数据流式读取与超大语料高效 packing
- 训练自动恢复策略之外的实验调度
- WandB / TensorBoard 等外部追踪器
- 下游任务评估脚本
- 自动 phase pass/fail 判定

另外，当前仓库本身依赖 Triton CUDA kernel，因此真实训练默认要求：

- CUDA 可用
- GPU 环境可运行 Triton kernel

## 3. 对 MoE 模型训练的支持程度

### 3.1 已实现的 MoE 训练支持

当前仓库已经支持一个“可训练的最小 MoE 版本”，目标与项目 guide 保持一致：

- 保持 `HGRNBitAttention` 不变
- 将 `HGRNBitBlock` 中的 dense `MLP` 替换为 `MoE MLP`
- router 保持浮点
- expert 可切换为浮点或 ternary quantized

对应配置字段已加入 `HGRNBitConfig`，包括：

- `use_moe`
- `moe_num_experts`
- `moe_num_experts_per_tok`
- `moe_router_aux_loss_coef`
- `moe_router_jitter_noise`
- `moe_router_bias`
- `moe_normalize_topk_prob`
- `moe_output_router_logits`
- `moe_use_quantized_experts`

### 3.2 当前 MoE 训练可以覆盖的实验

当前已经能支持的实验类型包括：

1. dense baseline
2. FP MoE baseline
3. ternary expert MoE baseline
4. 后续 `top-1 / top-2` ablation
5. 后续 router aux loss 系数 ablation
6. 后续 warm-start ablation

配套配置样例已放在：

- `experiments/moe_quant/configs/dense_fp_baseline.json`
- `experiments/moe_quant/configs/fp_moe_baseline.json`
- `experiments/moe_quant/configs/ternary_expert_moe_baseline.json`

### 3.3 已完成的 MoE 自检

当前已经完成本地 smoke 检查：

- dense smoke
- FP MoE smoke
- ternary-expert MoE smoke

日志文件保存在：

- `experiments/moe_quant/logs/dense_smoke.json`
- `experiments/moe_quant/logs/fp_moe_smoke.json`
- `experiments/moe_quant/logs/ternary_moe_smoke.json`

这些 smoke 结果说明：

- 前向可运行
- 反向可运行
- `router_aux_loss` 能接入总 loss
- router entropy 和 expert 使用统计能正常输出

这意味着当前仓库已经通过了 “最小训练闭环” 层面的自检。

### 3.4 当前 MoE 支持的边界

尽管最小 MoE 路径已经打通，但目前仍有明显边界：

- 还没有多 seed 稳定性验证
- 还没有正式 `val ppl` 对照表
- 还没有 expert 分化的长期观察结果
- 还没有 router quantization 支持
- 还没有 quantized `lm_head` 的正式训练路径
- 还没有专门的 expert dispatch 高性能 kernel

换句话说，当前阶段的结论是：

- “MoE 模型结构与训练入口已具备”
- “MoE 最小训练闭环已可运行”
- “真实实验结论仍依赖后续补充数据集、tokenizer 和正式训练运行”

## 4. 当前项目结论

截至目前，仓库状态可以概括为：

- 原始 `HGRNBit` 结构仍是主干
- 已成功扩展出一个最小可训练的 `MoE` 路径
- 已具备 dense / FP MoE / ternary-expert MoE 三类训练入口
- 已具备基本日志、验证与 checkpoint 能力
- 已通过最小 smoke 检查
- 已完成一轮真实 sparse upcycling pilot，但还没有足够长的训练来判断最终 expert 分化与精度上限

### 4.1 Shared-Family 近期结论

`shared_residual / full_shared` 家族在 `2026-06-03` 前后的实验需要严格区分：

- 旧的 `full_shared` 与 `shared_residual` 结果中，有一批曾经受到 shared-copy bug 污染：
  - `initialize_shared_expert_from_dense()` 起初没有完整复制内部 `norm` 参数；
  - 新建 shared / sparse expert 模块的 `dtype` 也一度没有对齐到原 dense MLP。
- 该问题修复后，`full_shared_no_residual identity eval` 已经与 dense baseline 严格等价，说明 shared copy 路径现在是可信的。

基于修复后的 clean rerun，目前可以成立的结论是：

- `full_shared` 控制组在 `shared_width = 2816` 时已经回到 dense baseline 附近：
  - 最好的是 `full_shared_8x32_top2_alpha005_local_backbone_ft_5k`
  - `PPL@1024 ≈ 17.288`
  - 略优于 dense baseline `≈ 17.294`
- 因此，之前 `full_shared` 明显差于 baseline 的现象主要来自实现 bug，而不是结构本身。

- `shared_residual topchannel + discarded residual` 路线在修复后反而更差，而且现在这个更差的结果更可信：
  - `4x128 top1` / `8x64 top2` clean rerun 都落在 `PPL@1024 ≈ 20.06 ~ 20.12`
  - 明显差于 dense baseline，也远差于 `full_shared`
  - 其中 `8x64 top2` 虽然略优于 `4x128 top1`，但 router entropy 更低、load imbalance 更高，路由健康度更差

- 额外的 `shared-only-from-dense 2288` eval-only 诊断进一步说明：
  - 在同样 `resolved_shared_width = 2288` 下，
  - 如果完全拿掉 sparse residual，只保留从 dense checkpoint 直接构建的 shared-only 路径，
  - `PPL@1024 ≈ 20.548`
  - 这比训练后的 `shared_residual` 结果还要更差约 `0.43 ~ 0.49`

所以当前更准确的结构判断是：

- `shared_width = 2288` 这一级别的 top-channel 压缩本身就已经造成很大损伤；
- sparse residual 不是当前 20.x PPL 的唯一根因；
- 在这条路线里，residual branch 实际上是在“部分补回 shared 压缩损失”，但补回幅度远远不够；
- 也就是说，当前失败的主因是 `shared path` 压缩过重，而不是“加了 sparse residual 以后才变差”。

## 5. 后续最小必需输入

为了继续进入更严格的真实实验阶段，仍需要进一步明确或补齐以下内容：

- 更长训练预算
- 最终可接受精度损失阈值
- 正式实验对照表设计
- 多 seed 或更长步数的资源预算

在这些输入补齐后，当前仓库可以继续沿现有 sparse upcycling 路径做正式验证。

## 6. Sparse Upcycling 增量状态

在原始 `MoE` 最小训练闭环之外，当前仓库还新增了 sparse upcycling 相关能力，主要包括：

- `mmfreelm/upcycling/sparse_upcycling.py`
- `mmfreelm/upcycling/freeze.py`
- `mmfreelm/upcycling/expert_monitor.py`
- `mmfreelm/upcycling/data_utils.py`

同时新增了两个入口脚本：

- `scripts/run_sparse_upcycling.py`
- `scripts/evaluate_lm.py`

这些改动使仓库现在支持以下 sparse upcycling 工作流：

1. 从已有 dense checkpoint 加载 `HGRNBitForCausalLM`
2. 仅将指定层的 dense `MLP` 替换为 `SparseMoEBlock`
3. 将原始 dense MLP 权重复制到每个 expert，并加入小扰动
4. 冻结非 MoE 参数，仅训练 router + experts
5. 在训练期间记录 expert 相似度与 router 统计
6. 使用流式数据接口驱动较大规模文本训练

当前 sparse upcycling 路径已经不只是代码级集成，而是完成了基于真实 checkpoint 与真实数据的一轮 pilot 验证。

### 6.1 已完成的真实 sparse upcycling pilot

本地已完成一轮真实 sparse upcycling 试运行，产物目录为：

- `outputs/first_real_upcycling/`

主要产物包括：

- `outputs/first_real_upcycling/training_report.json`
- `outputs/first_real_upcycling/execution_timing.json`
- `outputs/first_real_upcycling/baseline_eval.json`
- `outputs/first_real_upcycling/expert_metrics.json`
- `outputs/first_real_upcycling/train_log.jsonl`
- `outputs/first_real_upcycling/checkpoint_best/`

这轮实验的关键事实是：

- 使用了真实 `370M dense checkpoint`
- 使用了真实 `WikiText-2` 数据
- 实际执行了 sparse upcycling 训练流程
- 出于时间预算限制，实际执行的是 `100-step pilot`，不是原计划的 `5000 step`

### 6.2 当前已知实验结果

这轮 pilot 的关键结果如下：

- dense baseline：`val_ppl = 24.43`
- best upcycled checkpoint：`val_ppl = 18.12`
- 训练过程稳定，`train_loss` 从约 `26.27` 降到 `23.37`
- `grad_norm` 保持有限，没有明显发散
- `router_aux_loss` 始终为正
- router 没有 usage collapse
- 各层 `tokens_per_expert` 不为 0
- `router_entropy` 大多落在 `1.24 ~ 1.35`
- `avg_expert_similarity` 仅从 `0.99760` 变化到 `0.99758`

### 6.3 当前对实验结果的解释

目前可以确认：

- sparse upcycling pipeline 已经在真实环境中跑通
- ternary expert 的训练是稳定的
- router 行为目前正常
- 但 `100 step / 819,200 tokens` 还不足以观察到明显的 expert 分化

因此，当前更准确的结论是：

- 最小 `MoE` 训练闭环：已完成代码与 smoke 自检
- sparse upcycling 训练闭环：已完成真实 checkpoint + 真实数据 + 真实训练 pilot 验证
- 尚未完成“长训练下 expert 是否真正分化”的验证

### 6.4 当前最合理的下一步

下一步建议不是继续补框架代码，而是直接扩展训练长度：

1. 优先做 `1000 step` 长训练，而不是一次直接冲 `5000 step`
2. 保持当前结构不动，先观察 `avg_expert_similarity` 是否能明显下降，例如降到 `< 0.95`
3. 如果 expert 仍不分化，再调：
   - `noise_scale`
   - learning rate
   - `moe_router_aux_loss_coef`

### 6.5 `router_aux_loss_coef` 5k 对照总表

截至 `2026-05-25`，`complement6e_half_top2_alpha005_*_5000step` 这一批 `router aux loss` 对照已经按统一离线口径重新评测。

| Experiment | `router_aux_loss_coef` | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complement6e_half_top2_alpha005_aux001_5000step` | `0.001` | `15.8242` | `17.0793` | `17.1550` | `1.6612` | `0.272489` |
| `complement6e_half_top2_alpha005_aux0005_5000step` | `0.0005` | `15.8199` | `17.0664` | `17.1369` | `1.6879` | `0.272484` |
| `complement6e_half_top2_alpha005_aux0001_5000step` | `0.0001` | `15.7974` | `17.0382` | `17.1159` | `1.7346` | `0.272485` |
| `complement6e_half_top2_alpha005_aux00005_5000step` | `0.00005` | `15.8008` | `17.0391` | `17.1125` | `1.7414` | `0.272483` |

从统一复评后的结果看，排序没有变化：按 `eval_results_64` 看，`aux0001` 仍略优；按 `eval_results_1024` 看，`aux00005` 仍最好。这说明之前混入旧口径 `PPL@64` 的问题主要影响绝对数值，不改变这组 `aux sweep` 的相对结论。

### 6.6 新增完成实验结果

截至 `2026-05-24`，又有两组 complement-pair 后续实验完成并拿到统一评测结果。

| Experiment | Change | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `complement6e_half_top2_alpha005_aux0001_20000step` | 保持 `layer 12-23`，将 `aux=0.0001` 最优 5k 设定延长到 scratch `20000 step` | `15.4997` | `16.7260` | `16.8139` | `1.7675` | `0.268614` |
| `complement6e_half_top2_alpha005_aux0001_layers6_23_5000step` | 将 MoE 化层数从 `layer 12-23` 扩大到 `layer 6-23`，其余保持 `aux=0.0001` / `5000 step` | `15.9158` | `17.1535` | `17.2318` | `1.7359` | `0.272888` |

这两组结果给出的结论比较明确：

- `aux=0.0001` 的 scratch `20k` 已经优于之前 `aux=0.005` 的 scratch `20k` 主线，说明更低 router aux loss 在 complement-pair 结构上不只是 `5k` 有利，拉长训练后收益仍然保留。
- `layer 6-23` 的 `18-layer` MoE 化并没有超过标准 `layer 12-23` 的 `12-layer` 版本；在参数规模明显增大的前提下，`5k` 指标反而更差，因此当前不值得优先扩展到这条更深的路线。

### 6.7 本地适配增强实验结果

截至 `2026-05-25`，又完成了两组围绕 `complement6e_half_top2_alpha005_aux0001_5000step` 主结构的本地适配增强实验，并进行了离线复评。

| Experiment | Change | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `complement6e_half_top2_alpha005_aux0001_unfreeze_moe_norm_5000step` | 仅解冻 `layer12-23` 的 `mlp_norm`，其余仍保持固定 `moe_output_scale=2.0` | `15.8002` | `17.0522` | `17.1132` | `1.7351` | `0.272481` |
| `complement6e_half_top2_alpha005_aux0001_learnable_pair_scale_5000step` | 保持 RMSNorm 冻结，引入按 `layer/pair` 学习的正值 output scale，初始值 `2.0` | `15.8222` | `17.0573` | `17.1193` | `1.7356` | `0.272543` |

离线复评补充结论：

- 这两组都没有实质性超过标准 `aux=0.0001` `5k` 基线的 `17.0382 / 17.1159`。
- `unfreeze moe norm` 在 `1024-sample` 上仅有极小幅改善，同时在 `256-sample` 上也仅有极小幅退化；统一口径下，它更接近“基本打平”而不是“明显变差”。
- `learnable pair-scale` 的 learned scale 明显从 `2.0` 下调，最终 `mean/min/max ≈ 1.819 / 1.721 / 1.890`，说明模型确实在利用这组额外自由度；但 PPL 没有提升，因此“让 scale 可学”本身当前还不构成有效收益。
- 两组实验的 `avg expert similarity`、`router entropy` 和 `zero ratio` 都与基线几乎一致，说明它们没有改变 complement-pair 路由的整体工作点，只是在输出幅度或局部归一化上做了有限微调。
- 现阶段更合理的判断是：这两个方向都不应该优先排到 scratch `20k` 前面，主线仍应优先沿标准 `layer12-23` + fixed norm + fixed scale 的 `aux=0.0001` complement-pair 继续推进。

### 6.8 统一复评后的口径纠偏

`2026-05-25` 这次排查确认，历史文档中部分 `PPL@64` 混入了旧评测口径，不能直接和当前离线复评结果横向比较。

当前统一口径为：

- `eval_results_64`: `batch_size=4`, `max_samples=256`
- `eval_results_1024`: `batch_size=4`, `max_samples=1024`

纠偏后可以明确看到：

- `PPL@1024` 基本一直是稳定的，主线实验相对排序几乎不变。
- `PPL@64` 的绝对值整体上移了约 `1.2` 到 `1.3`，包括 non-MoE baseline 自身也从旧记录的 `15.9501` 变为统一复评后的约 `17.214`。在 `2026-05-28` 按当前 metadata-aware 评测路径重跑后，baseline 仍稳定在 `17.2137 / 17.2942`。
- 因此，之前“某些新实验在 `PPL@64` 上比 baseline 差很多”的现象，主要不是训练代码坏掉，而是旧 baseline 和新复评结果不在同一评测口径。
- 在统一口径下，标准 `complement6e aux=0.0001 5k` 基线是 `17.0382 / 17.1159`，`unfreeze moe norm` 是 `17.0522 / 17.1132`，`learnable pair-scale` 是 `17.0573 / 17.1193`；三者实际上非常接近。
- 当前唯一还没能按同口径重刷的主线 active-path-fair 行是 `virtual-group 8e half 20k`，因为原始 output 目录目前找不到，所以总表里它暂时仍保留历史数值。

### 6.9 Dense-Init Local Backbone Joint-Train 20k

截至 `2026-05-28`，`complement6e_aux0001_init_local_backbone_ft_20000step` 已完成训练，并在 `matmulfreellm:cu126-py310` 容器环境中补做了统一离线复评。

| Experiment | Change | best `val_ppl` in train log | `eval_results_64` `val_ppl` | `eval_results_1024` `val_ppl` | `eval_results_1024` router entropy | avg expert similarity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `complement6e_aux0001_init_local_backbone_ft_20000step` | 从 dense `MMfreeLM-370M` 直接 upcycle 成 `6E complement-pair`，同时训练 MoE 与 `layer12-23` local backbone，训练 `20000 step` | `15.5688` | `16.8309` | `16.9046` | `1.5947` | `0.273919` |

补充诊断：

- `pair_fractions@1024 = [0.33435, 0.33325, 0.33239]`
- `pair_entropy@1024 = 1.098609`
- `normalized_pair_entropy@1024 = 0.999997`
- `tokens_per_expert@1024 = [0.33435, 0.33325, 0.33239, 0.33239, 0.33325, 0.33435]`
- `zero_ratio_avg ≈ 18.851%`
- `trainable_params = 362,093,568`

这组结果给出的结论也很明确：

- 训练过程本身是稳定的，没有 collapse，`pair usage` 也几乎是完美均匀分配。
- 但最终 `PPL@1024 = 16.9046` 明显弱于当前 best `complement6e_half_top2_alpha005_aux0001_20000step` 的 `16.8139`。
- 相比直接从最优 `20k` MoE checkpoint 继续做 `partial-full-ft 10k` 或 `local-tm-norm-ft 10k`，这条“dense-init + local backbone joint-train”路线也没有体现出优势。
- 因此当前更合理的判断是：这条路线不应替代现有 scratch complement-pair 主线，也不应优先于基于 best `20k` checkpoint 的小幅 continuation follow-up。

## 7. 当前环境状态

当前项目使用 `uv` 管理本地虚拟环境，已确认：

- 虚拟环境路径：`.venv`
- `uv pip check` 通过
- 当前已安装并通过导入检查的关键依赖包括：
  - `torch`
  - `triton`
  - `einops`
  - `transformers`
  - `pyarrow`
  - `packaging`

另外已确认：

- `HGRNBitConfig`
- `HGRNBitForCausalLM`
- `AutoModel.from_config(HGRNBitConfig())`

都可以在当前仓库目录下正常导入和构造。

当前仍有一个环境边界：

- 项目还没有作为已安装包写入 `.venv`
- `import mmfreelm` 在仓库根目录下可用，但在仓库外路径下仍不可用

### 6.10 Recent 20k/40k Follow-up Results

| Experiment | Formal `PPL@64` | Formal `PPL@1024` | Router entropy | Avg expert similarity | Zero ratio | Note |
| --- | --- | --- | --- | --- | --- | --- |
| `moe8e_half_top2_random_extreme_local_backbone_ft_40000step` | `22.5616` | `22.6484` | `1.4258` | `0.002544` | `15.561%` | Extreme specialization with near-zero similarity, but catastrophic formal perplexity. |
| `moe4e_full_top1_random_extreme_local_backbone_ft_40000step` | `25.2194` | `25.4243` | `1.3856` | `0.002140` | `15.596%` | Top-1 full-width random-extreme local-backbone fine-tuning fails even more severely. |
| `complement6e_pair_plus_free_top3_scale050_alpha005_aux0001_20000step` | `16.7786` | `16.8707` | `1.7872` | `0.268994` | `18.807%` | Stable and useful, but still behind the best strict 6E complement scratch-20K mainline. |

结论：`random_extreme_local_backbone_ft` 这条线虽然把 expert similarity 压到了几乎 0，但正式 `PPL@1024` 明显崩坏，说明这类“强行极端分化”不是可行主线。相比之下，`pair+free scale=0.50` 延长到 `20k` 后仍然稳定，但还没有超过当前最强 `6E complement aux=1e-4 20k` 主线。

## 6. Shared-Residual Iso-Area 5K Screening Update

最新完成两组 strict-parameter-fair shared-residual 5K screening：

- `shared_residual_2304_4x128_top1_alpha010_isoarea_local_backbone_ft_5000step`
- `shared_residual_2304_4x128_top1_alpha025_isoarea_local_backbone_ft_5000step`

两组都满足：

- total params `373,616,460 <= 374,108,160` baseline
- resolved shared width = `2288`
- active width ratio vs dense = `0.858x`
- sparse residual experts 使用 `random_ternary_matched`，zero ratio 约 `36.6%`

正式结果显示：

- `alpha=0.10`: `PPL@1024 = 19.118`
- `alpha=0.25`: `PPL@1024 = 19.085`

结论：

- 该 shared-residual iso-area 设计当前可以稳定训练、保存、复评并绘图；
- `alpha=0.25` 略好于 `alpha=0.10`；
- 但两者都显著差于 non-MoE baseline 和当前 6E complement 主线，因此这版结构暂时不构成可行主线。
