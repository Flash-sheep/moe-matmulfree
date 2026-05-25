# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List

import torch

from mmfreelm.modules.moe import SparseMoEBlock, build_expert_from_mlp_state
from mmfreelm.upcycling.svd_init import (
    complement_copy_12e_init,
    complement_pair_6e_init,
    partition_init,
    svd_orthogonal_init,
    virtual_group_partition_copy_noise_init,
)


def _infer_hidden_and_intermediate_from_mlp(mlp) -> tuple[int, int]:
    if not hasattr(mlp, "gate_proj") or not hasattr(mlp, "down_proj"):
        raise RuntimeError("Dense MLP structure does not expose `gate_proj` and `down_proj`.")
    gate_weight = mlp.gate_proj.weight
    down_weight = mlp.down_proj.weight
    hidden_size = gate_weight.shape[1]
    intermediate_size = down_weight.shape[1]
    return hidden_size, intermediate_size


def upcycle_dense_to_moe(
    model,
    moe_layer_indices: List[int],
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
    noise_scale: float = 0.05,
    use_quantized_experts: bool = True,
    router_aux_loss_coef: float = 0.01,
    router_jitter_noise: float = 0.0,
    router_bias: bool = False,
    normalize_topk_prob: bool = True,
    expert_intermediate_factor: float = 1.0,
    init_method: str = "copy_noise",
    noise_alpha: float | None = None,
    noise_mode: str = "legacy_global_std",
    grouped_topk: bool = False,
    num_virtual_groups: int = 1,
    topk_per_group: int = 1,
    routing_mode: str = "standard",
    pair_weights: str = "router",
    moe_output_scale: float = 1.0,
    enable_learnable_output_scale: bool = False,
    output_scale_granularity: str = "global",
    initial_moe_output_scale: float | None = None,
):
    num_layers = len(model.model.layers)
    if not all(0 <= idx < num_layers for idx in moe_layer_indices):
        raise ValueError(f"`moe_layer_indices` must be within [0, {num_layers - 1}]")

    for layer_idx in moe_layer_indices:
        block = model.model.layers[layer_idx]
        original_mlp = block.mlp
        hidden_size, intermediate_size = _infer_hidden_and_intermediate_from_mlp(original_mlp)
        requested_expert_intermediate_size = max(1, int(round(intermediate_size * float(expert_intermediate_factor))))
        if init_method == "partition":
            # Partition init requires non-overlapping row/column assignments, so each expert can
            # consume at most floor(intermediate_size / num_experts) channels from the dense MLP.
            expert_intermediate_size = min(requested_expert_intermediate_size, intermediate_size // num_experts)
        else:
            expert_intermediate_size = requested_expert_intermediate_size
        effective_expert_intermediate_factor = expert_intermediate_size / max(intermediate_size, 1)
        expert_group_assignments = None
        expert_copy_group_assignments = None
        complement_pairs = None
        if routing_mode == "strict_complement_copy_pair" and num_experts == 12:
            complement_pairs = [
                (0, 10), (0, 11), (1, 10), (1, 11),
                (2, 8), (2, 9), (3, 8), (3, 9),
                (4, 6), (4, 7), (5, 6), (5, 7),
            ]

        moe_block = SparseMoEBlock(
            hidden_size=hidden_size,
            hidden_ratio=None,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=num_experts_per_tok,
            quantized_experts=use_quantized_experts,
            expert_intermediate_factor=effective_expert_intermediate_factor,
            expert_intermediate_size=expert_intermediate_size,
            router_bias=router_bias,
            router_jitter_noise=router_jitter_noise,
            normalize_topk_prob=normalize_topk_prob,
            grouped_topk=grouped_topk,
            num_virtual_groups=num_virtual_groups,
            topk_per_group=topk_per_group,
            routing_mode=routing_mode,
            pair_weights=pair_weights,
            complement_pairs=complement_pairs,
            output_scale=moe_output_scale,
            enable_learnable_output_scale=enable_learnable_output_scale,
            output_scale_granularity=output_scale_granularity,
            initial_output_scale=float(initial_moe_output_scale if initial_moe_output_scale is not None else moe_output_scale),
        )
        if init_method == "copy_noise":
            build_expert_from_mlp_state(
                moe_block=moe_block,
                source_mlp=original_mlp,
                noise_scale=noise_scale,
                noise_mode=noise_mode,
                noise_alpha=noise_alpha,
            )
        elif init_method == "copy_noise_relative_std":
            build_expert_from_mlp_state(
                moe_block=moe_block,
                source_mlp=original_mlp,
                noise_scale=0.0,
                noise_mode="relative_std",
                noise_alpha=noise_alpha,
            )
        elif init_method == "svd_orthogonal":
            expert_states = svd_orthogonal_init(
                original_mlp=original_mlp,
                num_experts=num_experts,
                expert_intermediate=expert_intermediate_size,
                assignment="interleaved",
            )
            for expert, state in zip(moe_block.experts, expert_states):
                target_state = expert.state_dict()
                cast_state = {key: value.to(target_state[key].dtype) for key, value in state.items() if key in target_state}
                expert.load_state_dict(cast_state, strict=False)
        elif init_method == "partition":
            expert_states = partition_init(
                original_mlp=original_mlp,
                num_experts=num_experts,
                expert_intermediate=expert_intermediate_size,
                assignment="interleaved",
            )
            for expert, state in zip(moe_block.experts, expert_states):
                target_state = expert.state_dict()
                cast_state = {key: value.to(target_state[key].dtype) for key, value in state.items() if key in target_state}
                expert.load_state_dict(cast_state, strict=False)
        elif init_method == "virtual_group_partition_copy_noise":
            expert_states, expert_group_assignments = virtual_group_partition_copy_noise_init(
                original_mlp=original_mlp,
                num_experts=num_experts,
                expert_intermediate=expert_intermediate_size,
                num_virtual_groups=num_virtual_groups,
                noise_alpha=float(noise_alpha or 0.0),
            )
            for expert, state in zip(moe_block.experts, expert_states):
                target_state = expert.state_dict()
                cast_state = {key: value.to(target_state[key].dtype) for key, value in state.items() if key in target_state}
                expert.load_state_dict(cast_state, strict=False)
        elif init_method == "complement_pair_6e":
            expert_states, expert_group_assignments, complement_pairs = complement_pair_6e_init(
                original_mlp=original_mlp,
                expert_intermediate=expert_intermediate_size,
                noise_alpha=float(noise_alpha or 0.0),
            )
            for expert, state in zip(moe_block.experts, expert_states):
                target_state = expert.state_dict()
                cast_state = {key: value.to(target_state[key].dtype) for key, value in state.items() if key in target_state}
                expert.load_state_dict(cast_state, strict=False)
        elif init_method == "complement_copy_12e":
            expert_states, expert_group_assignments, expert_copy_group_assignments, complement_pairs = complement_copy_12e_init(
                original_mlp=original_mlp,
                expert_intermediate=expert_intermediate_size,
                noise_alpha=float(noise_alpha or 0.0),
            )
            for expert, state in zip(moe_block.experts, expert_states):
                target_state = expert.state_dict()
                cast_state = {key: value.to(target_state[key].dtype) for key, value in state.items() if key in target_state}
                expert.load_state_dict(cast_state, strict=False)
        else:
            raise ValueError(f"Unsupported moe init method `{init_method}`.")

        if complement_pairs is not None:
            moe_block.complement_pairs = [tuple(pair) for pair in complement_pairs]
            moe_block.router.complement_pairs = [tuple(pair) for pair in complement_pairs]

        block.mlp = moe_block
        block.use_moe = True

    model.config.use_moe = True
    model.config.moe_num_experts = num_experts
    model.config.moe_num_experts_per_tok = num_experts_per_tok
    model.config.moe_layer_indices = list(moe_layer_indices)
    model.config.moe_router_aux_loss_coef = router_aux_loss_coef
    model.config.moe_router_jitter_noise = router_jitter_noise
    model.config.moe_router_bias = router_bias
    model.config.moe_normalize_topk_prob = normalize_topk_prob
    model.config.moe_output_router_logits = True
    model.config.moe_use_quantized_experts = use_quantized_experts
    model.config.moe_upcycling_noise_scale = noise_scale
    model.config.moe_noise_alpha = noise_alpha
    model.config.moe_noise_mode = noise_mode
    model.config.moe_expert_intermediate_factor = effective_expert_intermediate_factor
    model.config.moe_expert_intermediate_size = expert_intermediate_size
    model.config.moe_init_method = init_method
    model.config.moe_grouped_topk = grouped_topk
    model.config.moe_num_virtual_groups = num_virtual_groups
    model.config.moe_topk_per_group = topk_per_group
    model.config.moe_routing_mode = routing_mode
    model.config.moe_pair_weights = pair_weights
    model.config.moe_output_scale = moe_output_scale
    model.config.moe_enable_learnable_output_scale = enable_learnable_output_scale
    model.config.moe_output_scale_granularity = output_scale_granularity
    model.config.moe_initial_output_scale = float(initial_moe_output_scale if initial_moe_output_scale is not None else moe_output_scale)
    if expert_group_assignments is not None:
        model.config.moe_expert_group_assignments = expert_group_assignments
    if expert_copy_group_assignments is not None:
        model.config.moe_expert_copy_group_assignments = expert_copy_group_assignments
    if complement_pairs is not None:
        model.config.moe_complement_pairs = [list(pair) for pair in complement_pairs]
    return model
