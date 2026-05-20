# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmfreelm.modules.activations import swiglu
from mmfreelm.ops.fusedbitnet import FusedBitLinear


def _resolve_intermediate_size(
    hidden_size: int,
    hidden_ratio: Optional[int] = None,
    intermediate_size: Optional[int] = None,
) -> Tuple[int, int]:
    if hidden_ratio is None:
        hidden_ratio = 4
    if intermediate_size is None:
        intermediate_size = int(hidden_size * hidden_ratio * 2 / 3)
        intermediate_size = 256 * ((intermediate_size + 256 - 1) // 256)
    return hidden_ratio, intermediate_size


class TopKRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        bias: bool = False,
        jitter_noise: float = 0.0,
        normalize_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("`top_k` must be at least 1.")
        if top_k > num_experts:
            raise ValueError("`top_k` cannot exceed `num_experts`.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.normalize_topk_prob = normalize_topk_prob
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.jitter_noise > 0:
            low = 1.0 - self.jitter_noise
            high = 1.0 + self.jitter_noise
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(low, high)

        router_logits = self.gate(hidden_states)
        router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.normalize_topk_prob:
            denom = topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            topk_weights = topk_weights / denom
        topk_weights = topk_weights.to(hidden_states.dtype)
        return router_logits, router_probs, topk_weights, topk_indices


class ExpertMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        quantized: bool = False,
    ) -> None:
        super().__init__()
        _, intermediate_size = _resolve_intermediate_size(hidden_size, hidden_ratio, intermediate_size)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        linear_cls = FusedBitLinear if quantized else nn.Linear
        self.gate_proj = linear_cls(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = linear_cls(intermediate_size, hidden_size, bias=False)
        self.down_proj._is_residual_projection = True

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_value = self.gate_proj(hidden_states)
        gate, value = gate_value.chunk(2, dim=-1)
        return self.down_proj(swiglu(gate, value))


def build_expert_from_mlp_state(
    moe_block,
    source_mlp,
    noise_scale: float,
) -> None:
    source_state = source_mlp.state_dict()
    source_gate_weight = source_mlp.gate_proj.weight.detach()
    source_std = float(source_gate_weight.float().std().item())
    scale = noise_scale * source_std

    for expert in moe_block.experts:
        target_state = expert.state_dict()
        new_state = {}
        for key, target_value in target_state.items():
            if key not in source_state:
                new_state[key] = target_value
                continue
            source_value = source_state[key].detach().clone().float()
            if source_value.shape != target_value.shape:
                if key == "gate_proj.weight":
                    source_intermediate = source_value.shape[0] // 2
                    target_intermediate = target_value.shape[0] // 2
                    source_gate = source_value[:source_intermediate]
                    source_up = source_value[source_intermediate:]
                    source_value = torch.cat(
                        [source_gate[:target_intermediate], source_up[:target_intermediate]],
                        dim=0,
                    )
                elif key == "down_proj.weight":
                    target_intermediate = target_value.shape[1]
                    source_value = source_value[:, :target_intermediate]
                else:
                    source_value = source_value.narrow(0, 0, min(source_value.shape[0], target_value.shape[0]))
            if source_value.is_floating_point() and scale > 0:
                source_value = source_value + torch.randn_like(source_value) * scale
            new_state[key] = source_value.to(target_value.dtype)
        expert.load_state_dict(new_state, strict=False)


class SparseMoEBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hidden_ratio: Optional[int],
        intermediate_size: Optional[int],
        num_experts: int,
        top_k: int,
        quantized_experts: bool = False,
        expert_intermediate_factor: float = 1.0,
        expert_intermediate_size: Optional[int] = None,
        router_bias: bool = False,
        router_jitter_noise: float = 0.0,
        normalize_topk_prob: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.quantized_experts = quantized_experts
        _, base_intermediate_size = _resolve_intermediate_size(hidden_size, hidden_ratio, intermediate_size)
        if expert_intermediate_size is None:
            expert_intermediate_size = max(1, int(round(base_intermediate_size * float(expert_intermediate_factor))))
        self.intermediate_size = base_intermediate_size
        self.expert_intermediate_size = int(expert_intermediate_size)
        self.expert_intermediate_factor = float(expert_intermediate_factor)
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            bias=router_bias,
            jitter_noise=router_jitter_noise,
            normalize_topk_prob=normalize_topk_prob,
        )
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size=hidden_size,
                    hidden_ratio=hidden_ratio,
                    intermediate_size=self.expert_intermediate_size,
                    quantized=quantized_experts,
                )
                for _ in range(num_experts)
            ]
        )

    def _build_router_metrics(
        self,
        router_probs: torch.Tensor,
        topk_indices: torch.Tensor,
        num_active_tokens: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        assignment_mask = F.one_hot(topk_indices, num_classes=self.num_experts).sum(dim=1).clamp(max=1)
        assignment_mask = assignment_mask.to(router_probs.dtype)
        tokens_per_expert = assignment_mask.mean(dim=0)
        router_prob_per_expert = router_probs.mean(dim=0)
        router_entropy = (-router_probs.clamp_min(1e-9) * router_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
        route_load = F.one_hot(topk_indices.reshape(-1), num_classes=self.num_experts).to(router_probs.dtype).sum(dim=0)
        aux_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
        metrics = {
            "tokens_per_expert": tokens_per_expert.detach(),
            "router_prob_per_expert": router_prob_per_expert.detach(),
            "router_entropy": router_entropy.detach(),
            "route_load": route_load.detach(),
            "active_tokens": torch.tensor(float(num_active_tokens), device=router_probs.device),
        }
        return aux_loss, metrics

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        output = flat_hidden.new_zeros(flat_hidden.shape)

        if attention_mask is None:
            active_mask = torch.ones(flat_hidden.shape[0], dtype=torch.bool, device=flat_hidden.device)
        else:
            active_mask = attention_mask.reshape(-1).to(torch.bool)
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)

        if active_indices.numel() == 0:
            zero = flat_hidden.new_zeros(())
            metrics = {
                "tokens_per_expert": flat_hidden.new_zeros(self.num_experts),
                "router_prob_per_expert": flat_hidden.new_zeros(self.num_experts),
                "router_entropy": flat_hidden.new_zeros(()),
                "route_load": flat_hidden.new_zeros(self.num_experts),
                "active_tokens": flat_hidden.new_zeros(()),
            }
            return output.reshape(original_shape), zero, None, metrics

        active_hidden = flat_hidden.index_select(0, active_indices)
        router_logits, router_probs, topk_weights, topk_indices = self.router(active_hidden)
        aux_loss, metrics = self._build_router_metrics(router_probs, topk_indices, active_hidden.shape[0])

        mixed_output = active_hidden.new_zeros(active_hidden.shape)
        for expert_idx, expert in enumerate(self.experts):
            expert_assignments = (topk_indices == expert_idx).nonzero(as_tuple=False)
            if expert_assignments.numel() == 0:
                continue
            token_positions = expert_assignments[:, 0]
            route_positions = expert_assignments[:, 1]
            expert_input = active_hidden.index_select(0, token_positions)
            expert_output = expert(expert_input)
            expert_weight = topk_weights[token_positions, route_positions].unsqueeze(-1).to(expert_output.dtype)
            mixed_output.index_add_(0, token_positions, expert_output * expert_weight)

        output.index_copy_(0, active_indices, mixed_output)
        router_logits_out = router_logits if output_router_logits else None
        return output.reshape(original_shape), aux_loss, router_logits_out, metrics
