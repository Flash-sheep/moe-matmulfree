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
