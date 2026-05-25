# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

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
        grouped_topk: bool = False,
        num_virtual_groups: int = 1,
        topk_per_group: int = 1,
        routing_mode: str = "standard",
        pair_weights: str = "router",
        complement_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        super().__init__()
        if top_k < 1:
            raise ValueError("`top_k` must be at least 1.")
        if top_k > num_experts:
            raise ValueError("`top_k` cannot exceed `num_experts`.")
        if routing_mode not in {"standard", "strict_complement_pair", "strict_complement_copy_pair"}:
            raise ValueError(f"Unsupported routing_mode `{routing_mode}`.")
        if pair_weights not in {"router", "uniform"}:
            raise ValueError(f"Unsupported pair_weights `{pair_weights}`.")
        if grouped_topk and routing_mode != "standard":
            raise ValueError("grouped_topk cannot be combined with non-standard routing modes.")
        if grouped_topk:
            if num_virtual_groups < 1:
                raise ValueError("`num_virtual_groups` must be at least 1 when grouped_topk is enabled.")
            if num_experts % num_virtual_groups != 0:
                raise ValueError("`num_experts` must be divisible by `num_virtual_groups`.")
            if topk_per_group < 1:
                raise ValueError("`topk_per_group` must be at least 1 when grouped_topk is enabled.")
            experts_per_group = num_experts // num_virtual_groups
            if topk_per_group > experts_per_group:
                raise ValueError("`topk_per_group` cannot exceed experts per virtual group.")
            if top_k != num_virtual_groups * topk_per_group:
                raise ValueError("`top_k` must equal `num_virtual_groups * topk_per_group` when grouped_topk is enabled.")
        if routing_mode == "strict_complement_pair" and not complement_pairs and num_experts == 6:
            complement_pairs = [(0, 5), (1, 4), (2, 3)]
        if routing_mode in {"strict_complement_pair", "strict_complement_copy_pair"}:
            if top_k != 2:
                raise ValueError(f"{routing_mode} routing requires top_k=2.")
            if not complement_pairs:
                raise ValueError(f"{routing_mode} routing requires explicit complement_pairs.")
            for pair in complement_pairs:
                if len(pair) != 2:
                    raise ValueError(f"Each complement pair must contain exactly 2 experts, got {pair}.")
                lhs, rhs = pair
                if lhs == rhs:
                    raise ValueError(f"Complement pair cannot repeat the same expert: {pair}.")
                if not (0 <= lhs < num_experts and 0 <= rhs < num_experts):
                    raise ValueError(f"Complement pair indices out of range for {num_experts} experts: {pair}.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.normalize_topk_prob = normalize_topk_prob
        self.grouped_topk = grouped_topk
        self.num_virtual_groups = num_virtual_groups
        self.topk_per_group = topk_per_group
        self.experts_per_group = num_experts // num_virtual_groups if num_virtual_groups > 0 else num_experts
        self.routing_mode = routing_mode
        self.pair_weights = pair_weights
        self.complement_pairs = [tuple(pair) for pair in (complement_pairs or [])]
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias)
        self.eval_pair_weights_override: Optional[str] = None

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training and self.jitter_noise > 0:
            low = 1.0 - self.jitter_noise
            high = 1.0 + self.jitter_noise
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(low, high)

        router_logits = self.gate(hidden_states.to(self.gate.weight.dtype))
        router_probs = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        if self.routing_mode in {"strict_complement_pair", "strict_complement_copy_pair"}:
            pair_weights_mode = self.pair_weights
            if not self.training and self.eval_pair_weights_override is not None:
                pair_weights_mode = self.eval_pair_weights_override
            pair_scores = []
            for lhs_idx, rhs_idx in self.complement_pairs:
                pair_scores.append(router_logits[..., lhs_idx] + router_logits[..., rhs_idx])
            pair_scores = torch.stack(pair_scores, dim=-1)
            best_pair = torch.argmax(pair_scores, dim=-1)
            pair_index_tensor = torch.tensor(self.complement_pairs, device=router_logits.device, dtype=torch.long)
            topk_indices = pair_index_tensor.index_select(0, best_pair)
            if pair_weights_mode == "uniform":
                topk_weights = torch.full(
                    topk_indices.shape,
                    0.5,
                    dtype=router_probs.dtype,
                    device=router_probs.device,
                )
            else:
                gathered = router_probs.gather(-1, topk_indices)
                denom = gathered.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                topk_weights = gathered / denom
        elif self.grouped_topk:
            grouped_weights: List[torch.Tensor] = []
            grouped_indices: List[torch.Tensor] = []
            for group_idx in range(self.num_virtual_groups):
                start = group_idx * self.experts_per_group
                end = start + self.experts_per_group
                group_probs = router_probs[..., start:end]
                group_weights, group_indices = torch.topk(group_probs, self.topk_per_group, dim=-1)
                grouped_weights.append(group_weights)
                grouped_indices.append(group_indices + start)
            topk_weights = torch.cat(grouped_weights, dim=-1)
            topk_indices = torch.cat(grouped_indices, dim=-1)
        else:
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
    noise_mode: str = "legacy_global_std",
    noise_alpha: Optional[float] = None,
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
            if source_value.is_floating_point():
                if noise_mode == "legacy_global_std":
                    if scale > 0:
                        source_value = source_value + torch.randn_like(source_value) * scale
                elif noise_mode == "relative_std":
                    alpha = float(noise_alpha or 0.0)
                    tensor_std = float(source_value.std().item())
                    if alpha > 0 and tensor_std > 0:
                        source_value = source_value + torch.randn_like(source_value) * (alpha * tensor_std)
                else:
                    raise ValueError(f"Unsupported copy-noise mode `{noise_mode}`.")
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
        grouped_topk: bool = False,
        num_virtual_groups: int = 1,
        topk_per_group: int = 1,
        routing_mode: str = "standard",
        pair_weights: str = "router",
        complement_pairs: Optional[List[Tuple[int, int]]] = None,
        output_scale: float = 1.0,
        enable_learnable_output_scale: bool = False,
        output_scale_granularity: str = "global",
        initial_output_scale: float = 1.0,
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
        self.grouped_topk = bool(grouped_topk)
        self.num_virtual_groups = int(num_virtual_groups)
        self.topk_per_group = int(topk_per_group)
        self.routing_mode = routing_mode
        self.pair_weights = pair_weights
        if self.routing_mode == "strict_complement_pair" and not complement_pairs and num_experts == 6:
            complement_pairs = [(0, 5), (1, 4), (2, 3)]
        self.complement_pairs = [tuple(pair) for pair in (complement_pairs or [])]
        self.output_scale = float(output_scale)
        self.eval_output_scale_override: Optional[float] = None
        self.enable_learnable_output_scale = bool(enable_learnable_output_scale)
        self.output_scale_granularity = output_scale_granularity
        self.initial_output_scale = float(initial_output_scale)
        if self.enable_learnable_output_scale:
            if self.initial_output_scale <= 0.0:
                raise ValueError("Learnable output scale requires a strictly positive initial_output_scale.")
            if self.output_scale_granularity != "per_layer_per_pair":
                raise ValueError(
                    "Learnable output scale currently supports only `per_layer_per_pair` granularity."
                )
            if self.routing_mode not in {"strict_complement_pair", "strict_complement_copy_pair"}:
                raise ValueError(
                    "Learnable per-pair output scale requires strict complement-pair style routing."
                )
            if not self.complement_pairs:
                raise ValueError("Learnable per-pair output scale requires configured complement_pairs.")
            init_log_scale = math.log(self.initial_output_scale)
            self.pair_log_scales = nn.Parameter(torch.full((len(self.complement_pairs),), init_log_scale))
        else:
            self.register_parameter("pair_log_scales", None)
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            bias=router_bias,
            jitter_noise=router_jitter_noise,
            normalize_topk_prob=normalize_topk_prob,
            grouped_topk=self.grouped_topk,
            num_virtual_groups=self.num_virtual_groups,
            topk_per_group=self.topk_per_group,
            routing_mode=self.routing_mode,
            pair_weights=self.pair_weights,
            complement_pairs=self.complement_pairs,
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

    def get_output_scales(self) -> torch.Tensor:
        if self.enable_learnable_output_scale and self.pair_log_scales is not None:
            return torch.exp(self.pair_log_scales)
        if self.complement_pairs:
            return torch.full(
                (len(self.complement_pairs),),
                float(self.output_scale),
                dtype=torch.float32,
                device=self.router.gate.weight.device,
            )
        return torch.tensor([float(self.output_scale)], dtype=torch.float32, device=self.router.gate.weight.device)

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
        if self.routing_mode in {"strict_complement_pair", "strict_complement_copy_pair"}:
            pair_route_load = []
            for lhs_idx, rhs_idx in self.complement_pairs:
                pair_mask = (topk_indices[:, 0] == lhs_idx) & (topk_indices[:, 1] == rhs_idx)
                pair_route_load.append(pair_mask.sum())
            if pair_route_load:
                pair_route_load = torch.stack(pair_route_load).to(router_probs.dtype)
                pair_fraction = pair_route_load / max(float(num_active_tokens), 1.0)
                pair_fraction_safe = pair_fraction.clamp_min(1e-9)
                pair_entropy = -(pair_fraction_safe * pair_fraction_safe.log()).sum()
                pair_entropy_normalized = pair_entropy / math.log(len(self.complement_pairs))
                dominant_pair_fraction = pair_fraction.max()
                pair_load_imbalance = pair_fraction.max() - pair_fraction.min()
                metrics.update(
                    {
                        "pair_route_load": pair_route_load.detach(),
                        "pair_fraction": pair_fraction.detach(),
                        "pair_entropy": pair_entropy.detach(),
                        "pair_entropy_normalized": pair_entropy_normalized.detach(),
                        "dominant_pair_fraction": dominant_pair_fraction.detach(),
                        "pair_load_imbalance": pair_load_imbalance.detach(),
                    }
                )
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
        if self.grouped_topk:
            metrics["grouped_topk"] = torch.tensor(1.0, device=router_probs.device)
            metrics["num_virtual_groups"] = torch.tensor(float(self.num_virtual_groups), device=router_probs.device)
            metrics["topk_per_group"] = torch.tensor(float(self.topk_per_group), device=router_probs.device)
        if self.routing_mode == "strict_complement_pair":
            metrics["strict_complement_pair"] = torch.tensor(1.0, device=router_probs.device)
            metrics["num_complement_pairs"] = torch.tensor(float(len(self.complement_pairs)), device=router_probs.device)
        elif self.routing_mode == "strict_complement_copy_pair":
            metrics["strict_complement_copy_pair"] = torch.tensor(1.0, device=router_probs.device)
            metrics["num_complement_paths"] = torch.tensor(float(len(self.complement_pairs)), device=router_probs.device)

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

        output_scale = self.output_scale
        if not self.training and self.eval_output_scale_override is not None:
            output_scale = float(self.eval_output_scale_override)
            if output_scale != 1.0:
                mixed_output = mixed_output * output_scale
        elif self.enable_learnable_output_scale and self.pair_log_scales is not None:
            if self.routing_mode not in {"strict_complement_pair", "strict_complement_copy_pair"}:
                raise RuntimeError("Learnable pair scales are only supported for strict complement-pair routing.")
            scale_values = torch.exp(self.pair_log_scales).to(mixed_output.dtype)
            pair_scale_per_token = active_hidden.new_ones(active_hidden.shape[0], dtype=mixed_output.dtype)
            for pair_idx, pair in enumerate(self.complement_pairs):
                pair_tensor = torch.tensor(pair, device=topk_indices.device, dtype=topk_indices.dtype)
                pair_mask = (topk_indices == pair_tensor).all(dim=-1)
                if pair_mask.any():
                    pair_scale_per_token[pair_mask] = scale_values[pair_idx]
            mixed_output = mixed_output * pair_scale_per_token.unsqueeze(-1)
        elif output_scale != 1.0:
            mixed_output = mixed_output * output_scale

        output.index_copy_(0, active_indices, mixed_output)
        router_logits_out = router_logits if output_router_logits else None
        return output.reshape(original_shape), aux_loss, router_logits_out, metrics
