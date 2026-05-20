# MoE Sparse Upcycling 实现指南：基于 MatMul-free LM 370M

## 文档目的

本文档指导 agent 将现有 `matmulfreellm` 仓库从当前状态（最小 MoE 训练闭环已通过 smoke 检查）推进到**可执行 sparse upcycling 实验**的完整状态。

核心目标：加载预训练的 370M MatMul-free LM dense checkpoint，将其 Channel Mixer 层 MoE 化，冻结非 MoE 部分，只训练 router + expert 权重，使 expert 从预训练权重分化出专门化能力。

---

## 0. 当前仓库状态假设

根据 `current_project_status.md`，以下能力已具备：

```
已有：
  ✓ HGRNBitConfig 支持 use_moe 等字段
  ✓ modules/moe.py 包含 TopKRouter / ExpertMLP / SparseMoEBlock
  ✓ models/hgrn_bit/modeling_hgrn_bit.py 支持 MoE 路径
  ✓ scripts/train_moe_lm.py 训练入口
  ✓ dense / FP MoE / ternary-expert MoE smoke 通过
  ✓ 日志记录 router_entropy / tokens_per_expert / router_aux_loss
  ✓ checkpoint 保存与恢复

未有：
  ✗ 从 pretrained dense checkpoint 初始化 MoE expert 的逻辑
  ✗ 参数冻结策略
  ✗ expert 分化监控
  ✗ 大规模数据集支持（流式读取）
  ✗ 下游评估脚本
  ✗ 多 GPU 训练支持
```

---

## 1. 需要新增 / 修改的文件清单

```
新增文件：
  mmfreelm/utils/sparse_upcycling.py       # 核心：从 dense checkpoint 构建 MoE 模型
  mmfreelm/utils/freeze.py                 # 参数冻结策略
  mmfreelm/utils/expert_monitor.py         # expert 分化与 routing 监控
  mmfreelm/utils/data_utils.py             # 大规模数据集流式读取
  scripts/run_sparse_upcycling.py          # sparse upcycling 训练入口
  scripts/evaluate_lm.py                   # 评估脚本（PPL / zero-shot）
  experiments/sparse_upcycling/configs/     # 实验配置目录

修改文件：
  mmfreelm/models/hgrn_bit/configuration_hgrn_bit.py   # 新增 upcycling 相关配置字段
  mmfreelm/models/hgrn_bit/modeling_hgrn_bit.py         # 支持 partial MoE（部分层 MoE 化）
  mmfreelm/modules/moe.py                               # 补充 expert 初始化接口
  scripts/train_moe_lm.py                               # 集成冻结与监控
```

---

## 2. 核心实现：Sparse Upcycling

### 2.1 原理

Sparse upcycling 的核心步骤：

```
1. 加载 pretrained 370M dense HGRNBit checkpoint
2. 对指定层（如 layer 12-23），将 Channel Mixer（GLU MLP）替换为 SparseMoEBlock
3. 每个 expert 用原始 Channel Mixer 权重初始化 + 随机扰动
4. 新建 router（随机初始化）
5. 冻结所有非 MoE 参数
6. 只训练 router + expert 权重
```

### 2.2 实现文件：`mmfreelm/utils/sparse_upcycling.py`

```python
"""
sparse_upcycling.py

从 pretrained dense HGRNBit checkpoint 构建 MoE 模型。
将指定层的 Channel Mixer 替换为 SparseMoEBlock，
每个 expert 从原始 MLP 权重初始化。
"""

import copy
import torch
import torch.nn as nn
from typing import List, Optional, Set


def upcycle_dense_to_moe(
    model,
    moe_layer_indices: List[int],
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    noise_scale: float = 0.05,
    use_quantized_experts: bool = True,
    router_aux_loss_coef: float = 0.01,
    router_jitter_noise: float = 0.0,
):
    """
    将 dense model 的指定层 Channel Mixer 替换为 MoE。

    参数说明：
        model: 已加载 pretrained weights 的 HGRNBitForCausalLM
        moe_layer_indices: 要 MoE 化的层索引列表，如 [12, 13, ..., 23]
        num_experts: expert 数量
        num_experts_per_tok: top-k routing
        noise_scale: expert 初始化时加在 latent weight 上的扰动幅度
        use_quantized_experts: expert 是否使用 BitLinear（ternary）
        router_aux_loss_coef: load balance auxiliary loss 系数
        router_jitter_noise: router 输入 jitter 噪声

    返回：
        修改后的 model（in-place 修改）
    """
    from mmfreelm.modules.moe import SparseMoEBlock

    num_layers = len(model.model.layers)
    assert all(0 <= idx < num_layers for idx in moe_layer_indices), \
        f"moe_layer_indices 超出范围，模型共 {num_layers} 层"

    for layer_idx in moe_layer_indices:
        block = model.model.layers[layer_idx]
        original_mlp = block.mlp  # 原始 Channel Mixer（GLU）

        # 提取原始 MLP 的结构参数
        # 需要根据实际 MLP 类的属性名调整
        hidden_size = _get_hidden_size(original_mlp)
        intermediate_size = _get_intermediate_size(original_mlp)

        # 构建 SparseMoEBlock
        moe_block = SparseMoEBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            router_aux_loss_coef=router_aux_loss_coef,
            router_jitter_noise=router_jitter_noise,
            use_quantized=use_quantized_experts,
        )

        # 将原始 MLP 权重复制到每个 expert
        _init_experts_from_dense(
            moe_block=moe_block,
            source_mlp=original_mlp,
            num_experts=num_experts,
            noise_scale=noise_scale,
        )

        # 替换 block 中的 MLP
        block.mlp = moe_block
        block.use_moe = True  # 标记该层使用 MoE

        print(f"[Upcycling] Layer {layer_idx}: Channel Mixer → MoE "
              f"({num_experts} experts, top-{num_experts_per_tok})")

    # 更新 model config
    model.config.use_moe = True
    model.config.moe_num_experts = num_experts
    model.config.moe_num_experts_per_tok = num_experts_per_tok
    model.config.moe_layer_indices = moe_layer_indices
    model.config.moe_router_aux_loss_coef = router_aux_loss_coef

    return model


def _get_hidden_size(mlp):
    """
    从原始 MLP 模块推断 hidden_size。
    需要根据实际 MLP 类的属性名适配。

    常见可能的属性名：
      - mlp.gate_proj.weight.shape[1]
      - mlp.W_g.weight.shape[1]
      - mlp.linear_gate.weight.shape[1]

    请 agent 检查 modeling_hgrn_bit.py 中 MLP 类的实际属性名并适配。
    """
    # 以下为示例，agent 需要根据实际代码调整
    for name, param in mlp.named_parameters():
        if 'weight' in name:
            # 第一个遇到的 weight 的输入维度即为 hidden_size
            return param.shape[1]
    raise RuntimeError("无法从 MLP 推断 hidden_size")


def _get_intermediate_size(mlp):
    """
    从原始 MLP 模块推断 intermediate_size。
    GLU 结构中 gate_proj 的输出维度即为 intermediate_size。
    请 agent 检查实际属性名。
    """
    for name, param in mlp.named_parameters():
        if 'weight' in name:
            return param.shape[0]
    raise RuntimeError("无法从 MLP 推断 intermediate_size")


def _init_experts_from_dense(
    moe_block,
    source_mlp,
    num_experts: int,
    noise_scale: float,
):
    """
    将原始 dense MLP 的权重复制到 MoE 中每个 expert，并加随机扰动。

    关键点：
    1. 深拷贝权重，不是引用
    2. 扰动加在 latent float weight 上（STE 训练中的实际可学习参数）
    3. 扰动幅度相对于权重标准差按比例缩放
    """
    source_state = source_mlp.state_dict()

    for expert_idx in range(num_experts):
        expert = moe_block.experts[expert_idx]
        expert_state = expert.state_dict()

        # 逐个参数复制 + 扰动
        new_state = {}
        for key in expert_state:
            if key in source_state:
                src_param = source_state[key].clone()
                if src_param.dtype in (torch.float32, torch.float16, torch.bfloat16):
                    # 对 float 权重加扰动
                    noise = torch.randn_like(src_param) * noise_scale * src_param.std()
                    new_state[key] = src_param + noise
                else:
                    new_state[key] = src_param
            else:
                # expert 有但 source 没有的参数，保持默认初始化
                # 可能是 quantization 相关的 scale/offset
                new_state[key] = expert_state[key]

        expert.load_state_dict(new_state, strict=False)

    print(f"[Upcycling] Initialized {num_experts} experts from dense MLP "
          f"(noise_scale={noise_scale})")
```

**Agent 注意事项：**

```
1. _get_hidden_size 和 _get_intermediate_size 是占位实现。
   Agent 必须检查 modeling_hgrn_bit.py 中 MLP / Channel Mixer 类的实际结构，
   找到 gate_proj / up_proj / down_proj（或 W_g / W_u / W_d）的属性名。

2. _init_experts_from_dense 中的 source_state key 映射需要确认：
   - 原始 MLP 的 state_dict key 格式
   - ExpertMLP 的 state_dict key 格式
   - 两者的 key 是否能直接对应
   - 如果不能，需要建立一个 key mapping dict

3. SparseMoEBlock 的构造函数签名需要和 modules/moe.py 中的实现一致。
   如果 SparseMoEBlock 当前不支持 use_quantized 参数，需要补充。
```

---

## 3. 参数冻结策略

### 3.1 实现文件：`mmfreelm/utils/freeze.py`

```python
"""
freeze.py

实现 sparse upcycling 的参数冻结策略。
冻结所有非 MoE 参数，只训练 router + expert。
"""

from typing import List, Set, Optional


def apply_freeze_for_upcycling(
    model,
    moe_layer_indices: List[int],
    freeze_embeddings: bool = True,
    freeze_lm_head: bool = True,
    freeze_token_mixer: bool = True,
    freeze_non_moe_mlp: bool = True,
    freeze_rmsnorm: bool = True,
    trainable_extra_patterns: Optional[List[str]] = None,
):
    """
    冻结 sparse upcycling 中不需要训练的参数。

    默认策略：
      冻结：embedding, lm_head, 所有层的 token_mixer (MLGRU),
            非 MoE 层的 MLP, 所有 RMSNorm
      训练：MoE 层的 router + expert 权重

    参数说明：
        model: HGRNBitForCausalLM
        moe_layer_indices: MoE 化的层索引
        freeze_embeddings: 是否冻结 embedding 层
        freeze_lm_head: 是否冻结 LM head
        freeze_token_mixer: 是否冻结所有层的 token mixer (MLGRU/HGRN attention)
        freeze_non_moe_mlp: 是否冻结非 MoE 层的 MLP
        freeze_rmsnorm: 是否冻结 RMSNorm scale 参数
        trainable_extra_patterns: 额外允许训练的参数名 pattern 列表

    返回：
        trainable_params: int, 可训练参数数量
        frozen_params: int, 冻结参数数量
    """

    moe_layer_set = set(moe_layer_indices)
    trainable_patterns = trainable_extra_patterns or []

    # 第一步：全部冻结
    for param in model.parameters():
        param.requires_grad = False

    # 第二步：解冻 MoE 层的 router + expert
    for layer_idx in moe_layer_indices:
        block = model.model.layers[layer_idx]
        # 解冻整个 MoE MLP（包含 router + 所有 expert）
        if hasattr(block, 'mlp'):
            for param in block.mlp.parameters():
                param.requires_grad = True

    # 第三步：按策略决定是否解冻其他模块
    if not freeze_rmsnorm:
        # 解冻所有 RMSNorm（可能帮助适配 activation 分布变化）
        for name, param in model.named_parameters():
            if 'norm' in name.lower() and 'weight' in name:
                param.requires_grad = True

    # 第四步：处理额外的可训练 pattern
    for name, param in model.named_parameters():
        for pattern in trainable_patterns:
            if pattern in name:
                param.requires_grad = True

    # 统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"[Freeze] Trainable: {trainable_params:,} | "
          f"Frozen: {frozen_params:,} | "
          f"Ratio: {trainable_params/(trainable_params+frozen_params)*100:.1f}%")

    # 打印可训练参数的分组明细
    _print_trainable_summary(model)

    return trainable_params, frozen_params


def _print_trainable_summary(model):
    """按模块打印可训练参数统计"""
    module_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 提取顶层模块名
            parts = name.split('.')
            if 'layers' in parts:
                layer_idx = parts[parts.index('layers') + 1]
                module_key = f"layer.{layer_idx}.{parts[parts.index(layer_idx)+1]}"
            else:
                module_key = parts[0]

            if module_key not in module_params:
                module_params[module_key] = 0
            module_params[module_key] += param.numel()

    print("[Freeze] Trainable parameter breakdown:")
    for key, count in sorted(module_params.items()):
        print(f"  {key}: {count:,}")
```

**Agent 注意事项：**

```
1. 属性名需要适配：
   - block.mlp 是否是 MoE 化后 Channel Mixer 的属性名
   - token mixer 可能叫 block.attn 或 block.token_mixer
   - RMSNorm 的命名可能是 block.norm1 / block.norm2 / block.attn_norm 等
   Agent 需要检查 modeling_hgrn_bit.py 中 HGRNBitBlock 的属性定义

2. 冻结后必须确认 optimizer 只接收 requires_grad=True 的参数：
   optimizer = torch.optim.AdamW(
       filter(lambda p: p.requires_grad, model.parameters()),
       lr=learning_rate
   )
```

---

## 4. Expert 分化与 Routing 监控

### 4.1 实现文件：`mmfreelm/utils/expert_monitor.py`

```python
"""
expert_monitor.py

监控 expert 分化程度和 routing 行为。
提供训练过程中的关键诊断指标。
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import defaultdict
import json


class ExpertMonitor:
    """
    在训练过程中收集和分析 expert 行为指标。

    使用方法：
        monitor = ExpertMonitor(model, moe_layer_indices)

        # 每 N 步调用
        metrics = monitor.compute_metrics()
        monitor.log(step, metrics)

        # 训练结束后
        monitor.save("expert_metrics.json")
    """

    def __init__(self, model, moe_layer_indices: List[int]):
        self.model = model
        self.moe_layer_indices = moe_layer_indices
        self.history = []

    def compute_metrics(self) -> Dict:
        """计算当前所有 MoE 层的 expert 指标"""
        metrics = {}

        for layer_idx in self.moe_layer_indices:
            block = self.model.model.layers[layer_idx]
            moe = block.mlp  # SparseMoEBlock

            layer_metrics = {}

            # 1. Expert 间权重余弦相似度
            layer_metrics['expert_weight_similarity'] = \
                self._compute_expert_similarity(moe)

            # 2. Expert 权重范数
            layer_metrics['expert_weight_norms'] = \
                self._compute_expert_norms(moe)

            # 3. Router 权重分布统计
            layer_metrics['router_weight_stats'] = \
                self._compute_router_stats(moe)

            metrics[f'layer_{layer_idx}'] = layer_metrics

        # 汇总指标
        metrics['summary'] = self._compute_summary(metrics)

        return metrics

    def _compute_expert_similarity(self, moe) -> Dict:
        """
        计算 expert 间的平均余弦相似度。
        值越低表示 expert 分化越好。
        初始时（刚 upcycle）应接近 1.0，训练后应下降。
        """
        expert_vectors = []
        for expert in moe.experts:
            # 将 expert 所有参数展平成一个向量
            params = []
            for p in expert.parameters():
                params.append(p.data.detach().float().reshape(-1))
            expert_vectors.append(torch.cat(params))

        n = len(expert_vectors)
        if n < 2:
            return {'mean': 1.0, 'min': 1.0, 'max': 1.0}

        similarities = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = F.cosine_similarity(
                    expert_vectors[i].unsqueeze(0),
                    expert_vectors[j].unsqueeze(0)
                ).item()
                similarities.append(sim)

        return {
            'mean': sum(similarities) / len(similarities),
            'min': min(similarities),
            'max': max(similarities),
        }

    def _compute_expert_norms(self, moe) -> List[float]:
        """计算每个 expert 的 L2 权重范数"""
        norms = []
        for expert in moe.experts:
            total_norm = 0.0
            for p in expert.parameters():
                total_norm += p.data.detach().float().norm().item() ** 2
            norms.append(total_norm ** 0.5)
        return norms

    def _compute_router_stats(self, moe) -> Dict:
        """Router 权重的统计信息"""
        # Agent 需要根据 TopKRouter 的实际属性名适配
        router = moe.router  # 或 moe.gate
        if hasattr(router, 'weight'):
            w = router.weight.data.detach().float()
        elif hasattr(router, 'linear'):
            w = router.linear.weight.data.detach().float()
        else:
            return {}

        return {
            'weight_mean': w.mean().item(),
            'weight_std': w.std().item(),
            'weight_norm': w.norm().item(),
        }

    def _compute_summary(self, metrics: Dict) -> Dict:
        """跨层汇总"""
        all_sims = []
        for key, layer_m in metrics.items():
            if key.startswith('layer_') and 'expert_weight_similarity' in layer_m:
                all_sims.append(layer_m['expert_weight_similarity']['mean'])

        return {
            'avg_expert_similarity': sum(all_sims) / len(all_sims) if all_sims else 0.0,
            'num_moe_layers': len(self.moe_layer_indices),
        }

    def log(self, step: int, metrics: Dict):
        """记录一个时间步的指标"""
        entry = {'step': step}
        entry['avg_expert_similarity'] = metrics['summary']['avg_expert_similarity']

        # 提取每层的平均相似度
        for key, layer_m in metrics.items():
            if key.startswith('layer_'):
                entry[f'{key}_sim'] = layer_m['expert_weight_similarity']['mean']

        self.history.append(entry)

    def save(self, path: str):
        """保存历史指标到 JSON"""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def check_expert_collapse(self, threshold: float = 0.99) -> bool:
        """
        检查是否发生 expert collapse。
        如果平均 expert 相似度 > threshold，发出警告。
        """
        if not self.history:
            return False

        latest = self.history[-1]
        sim = latest.get('avg_expert_similarity', 0)
        if sim > threshold:
            print(f"[WARNING] Expert collapse detected! "
                  f"avg_similarity={sim:.4f} > {threshold}")
            return True
        return False
```

---

## 5. 数据准备

### 5.1 数据集选择

原论文使用 SlimPajama 数据集训练。对于 sparse upcycling fine-tuning，建议：

```
首选方案：
  使用 SlimPajama 的一个 subset（保持与预训练数据分布一致）
  HuggingFace: cerebras/SlimPajama-627B
  取其中一个 chunk 作为训练集（约 5-10B tokens）

备选方案：
  RedPajama-Data-1T-Sample（更小，适合快速验证）
  HuggingFace: togethercomputer/RedPajama-Data-1T-Sample

Tokenizer：
  原论文使用的 tokenizer 需要从 pretrained checkpoint 中获取
  通常是 HuggingFace 格式，可直接 AutoTokenizer.from_pretrained() 加载
```

### 5.2 数据处理：`mmfreelm/utils/data_utils.py`

```python
"""
data_utils.py

提供流式数据加载，支持大规模语料的 token packing。
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from typing import Optional
import os
import json


class StreamingTextDataset(IterableDataset):
    """
    流式读取大规模文本数据集，自动做 token packing。

    支持：
      - HuggingFace datasets（streaming mode）
      - 本地 .jsonl 文件
      - 本地文本文件目录

    所有文本被 tokenize 后拼接并按 max_length 切分，
    不需要 padding，最大化训练效率。
    """

    def __init__(
        self,
        data_source: str,
        tokenizer_path: str,
        max_length: int = 2048,
        split: str = 'train',
        text_field: str = 'text',
        max_samples: Optional[int] = None,
    ):
        self.data_source = data_source
        self.max_length = max_length
        self.split = split
        self.text_field = text_field
        self.max_samples = max_samples

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        buffer = []
        sample_count = 0

        for text in self._text_iterator():
            if self.max_samples and sample_count >= self.max_samples:
                break

            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)

            while len(buffer) >= self.max_length + 1:
                chunk = buffer[:self.max_length + 1]
                buffer = buffer[self.max_length:]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {'input_ids': input_ids, 'labels': labels}
                sample_count += 1

                if self.max_samples and sample_count >= self.max_samples:
                    return

    def _text_iterator(self):
        """根据 data_source 类型返回文本迭代器"""

        if os.path.isdir(self.data_source):
            # 本地文本目录
            for fname in sorted(os.listdir(self.data_source)):
                fpath = os.path.join(self.data_source, fname)
                if fname.endswith('.jsonl'):
                    with open(fpath) as f:
                        for line in f:
                            obj = json.loads(line)
                            yield obj.get(self.text_field, '')
                elif fname.endswith('.txt') or fname.endswith('.md'):
                    with open(fpath) as f:
                        yield f.read()

        elif self.data_source.endswith('.jsonl'):
            # 单个 JSONL 文件
            with open(self.data_source) as f:
                for line in f:
                    obj = json.loads(line)
                    yield obj.get(self.text_field, '')

        else:
            # 假设是 HuggingFace dataset name
            try:
                from datasets import load_dataset
                ds = load_dataset(
                    self.data_source,
                    split=self.split,
                    streaming=True
                )
                for item in ds:
                    yield item.get(self.text_field, '')
            except Exception as e:
                raise RuntimeError(
                    f"无法加载数据源 '{self.data_source}': {e}"
                )
```

---

## 6. 训练入口脚本

### 6.1 实现文件：`scripts/run_sparse_upcycling.py`

```python
"""
run_sparse_upcycling.py

Sparse upcycling 训练的完整入口脚本。

用法：
  python scripts/run_sparse_upcycling.py \
      --pretrained_path <370M checkpoint 路径> \
      --config_path <实验配置 JSON 路径> \
      --data_source <数据集路径或 HF dataset name> \
      --output_dir <输出目录>
"""

import argparse
import json
import os
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 项目内导入
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.utils.sparse_upcycling import upcycle_dense_to_moe
from mmfreelm.utils.freeze import apply_freeze_for_upcycling
from mmfreelm.utils.expert_monitor import ExpertMonitor
from mmfreelm.utils.data_utils import StreamingTextDataset


def parse_args():
    parser = argparse.ArgumentParser()

    # 模型
    parser.add_argument('--pretrained_path', type=str, required=True,
                        help='预训练 370M dense checkpoint 路径')
    parser.add_argument('--config_path', type=str, required=True,
                        help='实验配置 JSON 路径')

    # 数据
    parser.add_argument('--data_source', type=str, required=True,
                        help='训练数据路径或 HuggingFace dataset name')
    parser.add_argument('--val_data_source', type=str, default=None,
                        help='验证数据路径')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='tokenizer 路径，默认使用 pretrained_path')

    # 输出
    parser.add_argument('--output_dir', type=str, required=True)

    # 训练超参（可被 config JSON 覆盖）
    parser.add_argument('--max_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=None)

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    config = load_config(args.config_path)

    # 合并 CLI 参数和 config（CLI 优先）
    for key in ['max_steps', 'batch_size', 'gradient_accumulation_steps']:
        cli_val = getattr(args, key)
        if cli_val is not None:
            config[key] = cli_val

    os.makedirs(args.output_dir, exist_ok=True)

    # 保存实验配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ========================================
    # Step 1: 加载预训练 dense model
    # ========================================
    print("=" * 60)
    print("Step 1: Loading pretrained dense model")
    print("=" * 60)

    model = HGRNBitForCausalLM.from_pretrained(
        args.pretrained_path,
        torch_dtype=torch.bfloat16,
    )
    print(f"Loaded model: {sum(p.numel() for p in model.parameters()):,} params")

    # ========================================
    # Step 2: Sparse upcycling（MoE 化）
    # ========================================
    print("=" * 60)
    print("Step 2: Sparse upcycling")
    print("=" * 60)

    moe_config = config.get('moe', {})
    moe_layer_indices = moe_config.get('layer_indices', list(range(12, 24)))

    model = upcycle_dense_to_moe(
        model=model,
        moe_layer_indices=moe_layer_indices,
        num_experts=moe_config.get('num_experts', 8),
        num_experts_per_tok=moe_config.get('num_experts_per_tok', 2),
        noise_scale=moe_config.get('noise_scale', 0.05),
        use_quantized_experts=moe_config.get('use_quantized_experts', True),
        router_aux_loss_coef=moe_config.get('router_aux_loss_coef', 0.01),
    )

    print(f"After upcycling: {sum(p.numel() for p in model.parameters()):,} params")

    # ========================================
    # Step 3: 冻结参数
    # ========================================
    print("=" * 60)
    print("Step 3: Freezing parameters")
    print("=" * 60)

    freeze_config = config.get('freeze', {})
    trainable, frozen = apply_freeze_for_upcycling(
        model=model,
        moe_layer_indices=moe_layer_indices,
        freeze_rmsnorm=freeze_config.get('freeze_rmsnorm', True),
    )

    # ========================================
    # Step 4: 准备数据
    # ========================================
    print("=" * 60)
    print("Step 4: Preparing data")
    print("=" * 60)

    tokenizer_path = args.tokenizer_path or args.pretrained_path
    train_config = config.get('training', {})
    max_length = train_config.get('max_length', 2048)
    batch_size = config.get('batch_size', 4)

    train_dataset = StreamingTextDataset(
        data_source=args.data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = None
    if args.val_data_source:
        val_dataset = StreamingTextDataset(
            data_source=args.val_data_source,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            max_samples=1000,  # 验证集取有限样本
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ========================================
    # Step 5: 配置优化器
    # ========================================
    print("=" * 60)
    print("Step 5: Configuring optimizer")
    print("=" * 60)

    lr = train_config.get('learning_rate', 5e-4)
    weight_decay = train_config.get('weight_decay', 0.01)
    max_steps = config.get('max_steps', 50000)
    warmup_steps = train_config.get('warmup_steps', 1000)
    grad_accum_steps = config.get('gradient_accumulation_steps', 8)
    grad_clip = train_config.get('grad_clip', 1.0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # Cosine LR scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ========================================
    # Step 6: 训练循环
    # ========================================
    print("=" * 60)
    print("Step 6: Starting training")
    print("=" * 60)

    model.to(device)
    model.train()

    monitor = ExpertMonitor(model, moe_layer_indices)

    log_interval = config.get('log_interval', 50)
    eval_interval = config.get('eval_interval', 500)
    save_interval = config.get('save_interval', 2000)
    monitor_interval = config.get('monitor_interval', 500)

    global_step = 0
    accum_loss = 0.0
    accum_lm_loss = 0.0
    accum_aux_loss = 0.0
    best_val_loss = float('inf')
    train_log = []

    data_iter = iter(train_loader)

    while global_step < max_steps:
        optimizer.zero_grad()

        for accum_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)

            # loss 分解
            # Agent 需要确认模型输出的 loss 结构
            # 可能是 outputs.loss 包含 lm_loss + aux_loss
            # 也可能需要手动从 outputs 中提取 router_aux_loss
            loss = outputs.loss / grad_accum_steps
            loss.backward()

            accum_loss += loss.item()

            # 如果模型输出中有分开的 loss 信息
            if hasattr(outputs, 'lm_loss'):
                accum_lm_loss += outputs.lm_loss.item() / grad_accum_steps
            if hasattr(outputs, 'router_aux_loss'):
                accum_aux_loss += outputs.router_aux_loss.item() / grad_accum_steps

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            grad_clip
        )

        optimizer.step()
        scheduler.step()

        global_step += 1

        # ---- Logging ----
        if global_step % log_interval == 0:
            log_entry = {
                'step': global_step,
                'loss': accum_loss,
                'lm_loss': accum_lm_loss,
                'aux_loss': accum_aux_loss,
                'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                'lr': scheduler.get_last_lr()[0],
            }
            train_log.append(log_entry)
            print(f"[Step {global_step}] loss={accum_loss:.4f} "
                  f"lm={accum_lm_loss:.4f} aux={accum_aux_loss:.4f} "
                  f"grad={log_entry['grad_norm']:.4f} lr={log_entry['lr']:.6f}")

            accum_loss = 0.0
            accum_lm_loss = 0.0
            accum_aux_loss = 0.0

        # ---- Expert monitoring ----
        if global_step % monitor_interval == 0:
            metrics = monitor.compute_metrics()
            monitor.log(global_step, metrics)
            print(f"[Monitor] avg_expert_similarity="
                  f"{metrics['summary']['avg_expert_similarity']:.4f}")
            monitor.check_expert_collapse()

        # ---- Validation ----
        if val_loader and global_step % eval_interval == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            print(f"[Eval] step={global_step} val_loss={val_loss:.4f} "
                  f"val_ppl={val_ppl:.2f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, global_step, args.output_dir,
                                tag='best')

        # ---- Save checkpoint ----
        if global_step % save_interval == 0:
            save_checkpoint(model, optimizer, global_step, args.output_dir,
                            tag=f'step_{global_step}')

    # 训练结束
    save_checkpoint(model, optimizer, global_step, args.output_dir, tag='final')
    monitor.save(os.path.join(args.output_dir, 'expert_metrics.json'))

    with open(os.path.join(args.output_dir, 'train_log.json'), 'w') as f:
        json.dump(train_log, f, indent=2)

    print("Training complete!")


def evaluate(model, val_loader, device) -> tuple:
    """计算验证集 loss 和 perplexity"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            # 只用 LM loss 计算 PPL，不含 aux loss
            lm_loss = outputs.lm_loss if hasattr(outputs, 'lm_loss') else outputs.loss
            total_loss += lm_loss.item() * input_ids.shape[0]
            total_tokens += input_ids.shape[0]

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100))  # 防溢出
    model.train()
    return avg_loss, ppl


def save_checkpoint(model, optimizer, step, output_dir, tag='latest'):
    """保存 checkpoint"""
    path = os.path.join(output_dir, f'checkpoint_{tag}')
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    torch.save({
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, os.path.join(path, 'training_state.pt'))
    print(f"[Save] Checkpoint saved to {path}")


if __name__ == '__main__':
    main()
```

---

## 7. 实验配置

### 7.1 主实验配置：`experiments/sparse_upcycling/configs/upcycling_370m_8e_top2.json`

```json
{
    "experiment_name": "sparse_upcycling_370m_8expert_top2",
    "description": "Sparse upcycling: 370M dense -> MoE (8 experts, top-2) on layers 12-23",

    "moe": {
        "layer_indices": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "noise_scale": 0.05,
        "use_quantized_experts": true,
        "router_aux_loss_coef": 0.01
    },

    "freeze": {
        "freeze_rmsnorm": true,
        "freeze_embeddings": true,
        "freeze_lm_head": true,
        "freeze_token_mixer": true,
        "freeze_non_moe_mlp": true
    },

    "training": {
        "learning_rate": 5e-4,
        "weight_decay": 0.01,
        "warmup_steps": 1000,
        "max_length": 2048,
        "grad_clip": 1.0
    },

    "max_steps": 50000,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,

    "log_interval": 50,
    "eval_interval": 500,
    "save_interval": 5000,
    "monitor_interval": 500
}
```

### 7.2 对照实验配置

Agent 需要创建以下额外配置文件用于对照：

```
configs/upcycling_370m_4e_top1.json     # 4 experts, top-1
configs/upcycling_370m_4e_top2.json     # 4 experts, top-2
configs/upcycling_370m_8e_top1.json     # 8 experts, top-1
configs/upcycling_370m_fp_expert.json   # 8 experts, FP expert（非 ternary）
configs/upcycling_370m_all_layers.json  # 所有 24 层 MoE 化
```

每个配置只修改相关字段，其余继承主配置。

---

## 8. 模型代码修改详细说明

### 8.1 `modeling_hgrn_bit.py` 需要修改的点

```
修改 1：HGRNBitBlock 支持 per-layer MoE 控制

当前状态：use_moe 是全局开关（所有层统一）
目标状态：支持 moe_layer_indices，只有指定层使用 MoE

实现方式：
  在 HGRNBitBlock.__init__ 中增加 use_moe 参数（per-block），
  而不仅是从 config 全局读取。

  class HGRNBitBlock(nn.Module):
      def __init__(self, config, layer_idx, use_moe=False):
          ...
          if use_moe:
              self.mlp = SparseMoEBlock(...)
          else:
              self.mlp = OriginalMLP(...)
          self.use_moe = use_moe

修改 2：HGRNBitModel 根据 moe_layer_indices 构建

  class HGRNBitModel(nn.Module):
      def __init__(self, config):
          ...
          moe_indices = set(getattr(config, 'moe_layer_indices', []))
          self.layers = nn.ModuleList([
              HGRNBitBlock(
                  config,
                  layer_idx=i,
                  use_moe=(i in moe_indices and config.use_moe)
              )
              for i in range(config.num_hidden_layers)
          ])

修改 3：Forward 中收集 router auxiliary loss

  模型的 forward 方法需要从每个 MoE block 收集 router_aux_loss，
  汇总后加到总 loss 上。

  def forward(self, ...):
      total_aux_loss = 0.0
      for layer in self.layers:
          hidden_states = layer(hidden_states, ...)
          if layer.use_moe and hasattr(layer.mlp, 'aux_loss'):
              total_aux_loss += layer.mlp.aux_loss

      # 计算 LM loss
      lm_loss = cross_entropy(...)

      # 总 loss
      loss = lm_loss + self.config.moe_router_aux_loss_coef * total_aux_loss

      # 返回时分开报告
      return CausalLMOutput(
          loss=loss,
          lm_loss=lm_loss,            # 新增字段
          router_aux_loss=total_aux_loss,  # 新增字段
          logits=logits,
      )
```

### 8.2 `configuration_hgrn_bit.py` 需要新增的字段

```python
# 在 HGRNBitConfig.__init__ 中新增：

self.moe_layer_indices = kwargs.get('moe_layer_indices', [])
# 指定哪些层使用 MoE，空列表 = 全部层（当 use_moe=True 时）

self.moe_noise_scale = kwargs.get('moe_noise_scale', 0.05)
# sparse upcycling 时 expert 初始化的扰动幅度
```

### 8.3 `modules/moe.py` 需要补充的接口

```
补充 1：ExpertMLP 需要支持从外部 state_dict 初始化

  确保 ExpertMLP 的参数名与原始 MLP 类的参数名兼容，
  或者提供一个 key mapping 方法。

补充 2：SparseMoEBlock 需要暴露 aux_loss 属性

  class SparseMoEBlock(nn.Module):
      def forward(self, hidden_states):
          router_logits = self.router(hidden_states)
          ...
          # 计算 auxiliary load balance loss
          self.aux_loss = self._compute_aux_loss(router_logits, ...)
          return output

补充 3：SparseMoEBlock 需要暴露 routing 统计信息

  用于日志记录：
    self.router_entropy: float
    self.tokens_per_expert: List[float]
    self.expert_probs: Tensor

  这些在 forward 中计算并存为属性，
  训练循环通过 block.mlp.router_entropy 等读取。
```

---

## 9. 多 GPU 支持

### 9.1 最小 DDP 集成

对于 2×A100，使用 PyTorch DDP 即可：

```python
# 在 run_sparse_upcycling.py 中添加 DDP 支持

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group('nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return local_rank

# 训练循环中：
local_rank = setup_distributed()
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
# find_unused_parameters=True 因为 MoE 的 expert 并非每次都被激活
```

**启动命令：**

```bash
torchrun --nproc_per_node=2 scripts/run_sparse_upcycling.py \
    --pretrained_path ./checkpoints/matmulfree-370m \
    --config_path experiments/sparse_upcycling/configs/upcycling_370m_8e_top2.json \
    --data_source cerebras/SlimPajama-627B \
    --output_dir ./outputs/upcycling_370m_8e_top2
```

**Agent 注意事项：**

```
1. DDP + MoE 有一个已知问题：未被路由到的 expert 不参与 forward，
   但 DDP 默认要求所有参数都参与 gradient all-reduce。
   设置 find_unused_parameters=True 可以解决，但会增加少量开销。

2. 更高效的方案是给未使用的 expert 添加一个 dummy gradient：
   for expert in unused_experts:
       for param in expert.parameters():
           if param.requires_grad:
               param.grad = torch.zeros_like(param)

3. 数据并行时 IterableDataset 需要 worker 分片：
   在 StreamingTextDataset 中加入 worker_info 处理逻辑，
   或者改用 map-style dataset。
```

---

## 10. 评估脚本

### 10.1 实现文件：`scripts/evaluate_lm.py`

```python
"""
evaluate_lm.py

评估语言模型的 perplexity 和 MoE 行为统计。

用法：
  python scripts/evaluate_lm.py \
      --model_path <checkpoint 路径> \
      --eval_data <验证数据> \
      --output_path <结果输出>
"""

import argparse
import json
import math
import torch
from torch.utils.data import DataLoader
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.utils.data_utils import StreamingTextDataset
from mmfreelm.utils.expert_monitor import ExpertMonitor


def evaluate_ppl(model, dataloader, device):
    """计算 perplexity"""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            lm_loss = outputs.lm_loss if hasattr(outputs, 'lm_loss') else outputs.loss
            total_loss += lm_loss.item() * input_ids.shape[0]
            total_samples += input_ids.shape[0]

    avg_loss = total_loss / max(total_samples, 1)
    ppl = math.exp(min(avg_loss, 100))
    return avg_loss, ppl


def evaluate_routing(model, dataloader, device, moe_layer_indices):
    """
    在验证数据上跑 forward，收集 routing 统计：
    - 每层 router entropy
    - 每层 expert activation frequency
    - 每层 load balance
    """
    model.eval()
    layer_stats = {idx: {
        'entropy': [],
        'tokens_per_expert': [],
    } for idx in moe_layer_indices}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            _ = model(input_ids=input_ids)

            for idx in moe_layer_indices:
                block = model.model.layers[idx]
                moe = block.mlp
                if hasattr(moe, 'router_entropy'):
                    layer_stats[idx]['entropy'].append(moe.router_entropy)
                if hasattr(moe, 'tokens_per_expert'):
                    layer_stats[idx]['tokens_per_expert'].append(
                        moe.tokens_per_expert.tolist()
                        if isinstance(moe.tokens_per_expert, torch.Tensor)
                        else moe.tokens_per_expert
                    )

    # 汇总
    results = {}
    for idx, stats in layer_stats.items():
        results[f'layer_{idx}'] = {
            'avg_entropy': sum(stats['entropy']) / max(len(stats['entropy']), 1),
            'avg_tokens_per_expert': [
                sum(col) / len(col)
                for col in zip(*stats['tokens_per_expert'])
            ] if stats['tokens_per_expert'] else [],
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--eval_data', type=str, required=True)
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default='eval_results.json')
    parser.add_argument('--max_samples', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=2048)
    args = parser.parse_args()

    device = torch.device('cuda')

    model = HGRNBitForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    model.to(device)

    tokenizer_path = args.tokenizer_path or args.model_path
    dataset = StreamingTextDataset(
        data_source=args.eval_data,
        tokenizer_path=tokenizer_path,
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # PPL
    avg_loss, ppl = evaluate_ppl(model, dataloader, device)
    print(f"Eval Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")

    # Routing stats
    moe_indices = getattr(model.config, 'moe_layer_indices', [])
    if moe_indices:
        routing_results = evaluate_routing(model, dataloader, device, moe_indices)
    else:
        routing_results = {}

    # Expert similarity
    if moe_indices:
        monitor = ExpertMonitor(model, moe_indices)
        expert_metrics = monitor.compute_metrics()
    else:
        expert_metrics = {}

    results = {
        'loss': avg_loss,
        'ppl': ppl,
        'routing': routing_results,
        'expert_metrics': expert_metrics,
    }

    with open(args.output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {args.output_path}")


if __name__ == '__main__':
    main()
```

---

## 11. Agent 执行清单

以下是 agent 应按顺序执行的步骤，每步完成后进行验证再继续。

### Phase A：代码修改（预计 1-2 天）

```
A1. 阅读现有代码
    - 通读 mmfreelm/models/hgrn_bit/modeling_hgrn_bit.py
    - 通读 mmfreelm/modules/moe.py
    - 通读 mmfreelm/ops/fusedbitnet.py 或 bitnet.py
    - 确认 MLP / Channel Mixer 类的属性名、参数名、state_dict key 格式
    - 确认 SparseMoEBlock 的构造函数签名和 forward 输出格式
    - 确认 ExpertMLP 与原始 MLP 的参数结构差异

A2. 修改 configuration_hgrn_bit.py
    - 新增 moe_layer_indices 字段
    - 新增 moe_noise_scale 字段

A3. 修改 modeling_hgrn_bit.py
    - HGRNBitBlock 支持 per-layer MoE 控制
    - HGRNBitModel 根据 moe_layer_indices 构建
    - Forward 中收集并汇总 router_aux_loss
    - 输出中分开报告 lm_loss 和 router_aux_loss

A4. 补充 modules/moe.py
    - ExpertMLP 参数名与原始 MLP 兼容（或提供 key mapping）
    - SparseMoEBlock 暴露 aux_loss 属性
    - SparseMoEBlock 暴露 router_entropy / tokens_per_expert

A5. 新建 utils/ 模块
    - sparse_upcycling.py（本文档第 2 节）
    - freeze.py（本文档第 3 节）
    - expert_monitor.py（本文档第 4 节）
    - data_utils.py（本文档第 5 节）

A6. 新建训练脚本
    - scripts/run_sparse_upcycling.py（本文档第 6 节）
    - scripts/evaluate_lm.py（本文档第 10 节）
```

### Phase B：本地验证（预计 0.5 天）

```
B1. Smoke 测试：upcycling 流程
    - 加载 pretrained 370M（或用随机初始化 370M 模拟）
    - 调用 upcycle_dense_to_moe，确认无报错
    - 确认模型参数量增长符合预期
    - 确认冻结后可训练参数数量合理

B2. Smoke 测试：训练一步
    - 用少量 dummy 数据跑 1 个 training step
    - 确认 loss 能反传
    - 确认 optimizer 只更新 MoE 参数
    - 确认 router_aux_loss 有值
    - 确认 frozen 参数的 grad 确实为 None

B3. Smoke 测试：expert monitor
    - 调用 ExpertMonitor.compute_metrics()
    - 确认 expert_weight_similarity 接近 1.0（刚 upcycle 完应该很高）
    - 确认各指标能正常输出

B4. Smoke 测试：evaluation
    - 跑 evaluate_lm.py 用 dummy 数据
    - 确认 PPL 和 routing 统计能正常输出
```

### Phase C：实验准备（预计 1 天）

```
C1. 获取预训练 checkpoint
    - 从 HuggingFace 下载 370M MatMul-free LM
    - 确认能 from_pretrained 加载成功
    - 记录 baseline PPL

C2. 准备数据集
    - 下载 SlimPajama subset 或 RedPajama-Data-1T-Sample
    - 确认 tokenizer 兼容
    - 确认流式数据加载能正常工作

C3. 创建实验配置
    - 主实验：8E top-2, layers 12-23, ternary experts
    - 对照实验配置（见 7.2 节）
```

### Phase D：正式训练（预计 3-7 天）

```
D1. 主实验
    - 启动 upcycling_370m_8e_top2
    - 每 500 步检查 loss 曲线和 expert similarity
    - 监控是否有 expert collapse

D2. 对照实验（串行或并行）
    - 4E top-1
    - 4E top-2
    - 8E top-1
    - FP expert（非 ternary）
    - 每个 2-3 天
```

---

## 12. 预期输出（给硬件设计的参数）

训练完成后，以下数据应被提取并整理为硬件设计输入：

```
算法侧输出参数                  对应硬件设计决策
────────────────────────      ─────────────────────────
ternary weight ratio           ReRAM 存储密度与稀疏性利用
  {-1: x%, 0: y%, +1: z%}

activation bitwidth             activation 数据通路精度
  per-layer INT8/INT16

expert activation frequency     array allocation / 冷热分层
  per-layer distribution

router entropy                  routing 计算复杂度
  per-layer values

load balance                    expert array utilization
  per-layer std

loss gap (MoE vs dense)         精度代价 vs 硬件收益 tradeoff
  PPL comparison

expert weight similarity        expert deduplication 可行性
  per-layer after training
```

---

## 13. 常见问题排查

```
问题：Expert 不分化（similarity 一直接近 1.0）
排查：
  1. noise_scale 是否太小 → 增大到 0.1-0.2
  2. learning rate 是否太小 → 增大到 1e-3
  3. load balance loss 是否太强 → 减小 aux_loss_coef 到 0.001
  4. 训练步数是否足够 → 至少 5000-10000 步才能看到分化趋势

问题：Router collapse（少数 expert 获得 >90% tokens）
排查：
  1. aux_loss_coef 是否太小 → 增大到 0.05
  2. router jitter noise → 设为 0.1-0.2
  3. expert capacity factor → 添加 capacity 限制

问题：Loss 在 upcycling 后大幅上升
排查：
  1. noise_scale 是否太大 → 减小到 0.01
  2. 是否有参数没正确从 dense 复制到 expert → 检查 key mapping
  3. RMSNorm 是否需要解冻 → 尝试 freeze_rmsnorm=False
  4. top-k routing 是否正确 → 验证 top-k 后概率和是否合理

问题：DDP 报错 unused parameters
排查：
  1. 确保 find_unused_parameters=True
  2. 或者在 forward 中对未使用 expert 添加 dummy loss：
     dummy = sum(0 * p.sum() for p in unused_expert.parameters())
     total_loss = total_loss + dummy

问题：OOM on 2×A100
排查：
  1. 减小 batch_size 到 2
  2. 增大 gradient_accumulation_steps 到 16
  3. 使用 bf16 mixed precision
  4. 减少 MoE 层数量（先做 layers 18-23）
  5. 减少 expert 数量（先做 4 experts）
```
