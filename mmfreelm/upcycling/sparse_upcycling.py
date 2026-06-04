# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from mmfreelm.modules.moe import (
    SharedResidualMoEBlock,
    SparseMoEBlock,
    build_expert_from_mlp_state,
    initialize_sparse_expert_from_dense,
    initialize_shared_expert_from_dense,
    select_dense_channel_indices,
    select_discarded_channel_indices,
    split_channel_indices_for_sparse_experts,
)
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


def _infer_module_device_and_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for param in module.parameters():
        return param.device, param.dtype
    for buf in module.buffers():
        return buf.device, buf.dtype
    raise RuntimeError("Unable to infer device/dtype from source module: it has no parameters or buffers.")


def count_module_parameters(module) -> int:
    return sum(param.numel() for param in module.parameters())


def compute_baseline_param_count(model) -> int:
    return sum(param.numel() for param in model.parameters())


def _collect_ternary_stats_from_experts(experts) -> Dict[str, object]:
    expert_stats: Dict[str, object] = {}
    zero_values = []
    for expert_idx, expert in enumerate(experts):
        layer_entries = []
        for name, param in expert.named_parameters():
            if "weight" not in name or param.ndim < 2:
                continue
            quantized = param.detach().float().round().clamp(-1, 1)
            total = max(quantized.numel(), 1)
            zero_ratio = float((quantized == 0).sum().item() / total)
            positive_ratio = float((quantized == 1).sum().item() / total)
            negative_ratio = float((quantized == -1).sum().item() / total)
            layer_entries.append(
                {
                    "name": name,
                    "zero_ratio": zero_ratio,
                    "positive_ratio": positive_ratio,
                    "negative_ratio": negative_ratio,
                    "mean": float(param.detach().float().mean().item()),
                    "std": float(param.detach().float().std().item()),
                }
            )
            zero_values.append(zero_ratio)
        expert_stats[f"expert_{expert_idx}"] = layer_entries
    return {
        "experts": expert_stats,
        "zero_ratio_avg": float(sum(zero_values) / max(len(zero_values), 1)),
    }


def _build_shared_residual_block_from_mlp(
    original_mlp,
    *,
    shared_width: int,
    enable_sparse_residual: bool,
    num_sparse_experts: int,
    sparse_top_k: int,
    sparse_expert_width: int,
    use_quantized_experts: bool,
    router_bias: bool,
    router_jitter_noise: float,
    normalize_topk_prob: bool,
    shared_init: str,
    sparse_init: str,
    residual_scale_init: float,
    residual_scale_learnable: bool,
    residual_scale_max: float,
    parameter_budget_delta: float,
    noise_alpha: Optional[float],
) -> Tuple[SharedResidualMoEBlock, Dict[str, object]]:
    hidden_size, intermediate_size = _infer_hidden_and_intermediate_from_mlp(original_mlp)
    source_device, source_dtype = _infer_module_device_and_dtype(original_mlp.gate_proj)
    block = SharedResidualMoEBlock(
        hidden_size=hidden_size,
        hidden_ratio=None,
        intermediate_size=intermediate_size,
        shared_intermediate_size=shared_width,
        enable_sparse_residual=enable_sparse_residual,
        num_sparse_experts=num_sparse_experts,
        sparse_top_k=sparse_top_k,
        sparse_expert_width=sparse_expert_width,
        quantized_experts=use_quantized_experts,
        router_bias=router_bias,
        router_jitter_noise=router_jitter_noise,
        normalize_topk_prob=normalize_topk_prob,
        residual_scale_init=residual_scale_init,
        residual_scale_learnable=residual_scale_learnable,
        residual_scale_max=residual_scale_max,
        dense_intermediate_size=intermediate_size,
        parameter_budget_delta=parameter_budget_delta,
    )
    block.shared_expert.to(device=source_device, dtype=source_dtype)
    if enable_sparse_residual and block.sparse_experts:
        block.sparse_experts.to(device=source_device, dtype=source_dtype)
    channel_indices = select_dense_channel_indices(
        original_mlp,
        target_width=shared_width,
        selection_mode=shared_init,
    )
    initialize_shared_expert_from_dense(block.shared_expert, original_mlp, channel_indices)
    block.shared_channel_indices = [int(index) for index in channel_indices.tolist()]
    block.shared_init_method = shared_init
    block.sparse_init_method = sparse_init
    sparse_init_stats: Dict[str, object] = {"experts": {}, "zero_ratio_avg": None}
    sparse_channel_indices: List[int] = []
    sparse_expert_channel_assignments: Dict[str, List[int]] = {}
    if enable_sparse_residual and num_sparse_experts > 0:
        if sparse_init == "random_ternary_matched":
            _init_experts_random_ternary_matched(block, noise_alpha)
        elif sparse_init == "dense_discarded_channel_split":
            total_sparse_width = int(num_sparse_experts) * int(sparse_expert_width)
            selected_sparse_channels = select_discarded_channel_indices(
                original_mlp,
                shared_channel_indices=channel_indices,
                target_width=total_sparse_width,
                selection_mode=sparse_init,
            )
            sparse_channel_indices = [int(index) for index in selected_sparse_channels.tolist()]
            expert_channel_chunks = split_channel_indices_for_sparse_experts(
                selected_sparse_channels,
                num_experts=int(num_sparse_experts),
                expert_width=int(sparse_expert_width),
            )
            for expert_idx, (expert, expert_channel_indices) in enumerate(zip(block.sparse_experts, expert_channel_chunks)):
                initialize_sparse_expert_from_dense(expert, original_mlp, expert_channel_indices)
                sparse_expert_channel_assignments[f"expert_{expert_idx}"] = [
                    int(index) for index in expert_channel_indices.tolist()
                ]
        else:
            raise ValueError(f"Unsupported shared-residual sparse_init `{sparse_init}`.")
        sparse_init_stats = _collect_ternary_stats_from_experts(block.sparse_experts)
    block.sparse_channel_indices = sparse_channel_indices
    block.sparse_expert_channel_assignments = sparse_expert_channel_assignments
    return block, {
        "shared_channel_indices": block.shared_channel_indices,
        "shared_init": shared_init,
        "sparse_init": sparse_init,
        "sparse_channel_indices": sparse_channel_indices,
        "sparse_expert_channel_assignments": sparse_expert_channel_assignments,
        "sparse_init_stats": sparse_init_stats,
        "shared_expert_params": count_module_parameters(block.shared_expert),
        "sparse_expert_params": sum(count_module_parameters(expert) for expert in block.sparse_experts),
        "router_params": 0 if block.router is None else count_module_parameters(block.router),
        "residual_scale_params": 0 if block.raw_residual_scale is None else int(block.raw_residual_scale.numel()),
        "active_width": block.active_width,
        "active_width_ratio_vs_dense": block.active_width_ratio_vs_dense,
    }


def resolve_shared_width_for_param_budget(
    model,
    moe_layer_indices: List[int],
    *,
    nominal_shared_width: int,
    min_shared_width: int,
    shared_width_step: int,
    enable_sparse_residual: bool,
    num_sparse_experts: int,
    sparse_top_k: int,
    sparse_expert_width: int,
    use_quantized_experts: bool,
    router_bias: bool,
    router_jitter_noise: float,
    normalize_topk_prob: bool,
    shared_init: str,
    sparse_init: str,
    residual_scale_init: float,
    residual_scale_learnable: bool,
    residual_scale_max: float,
    noise_alpha: Optional[float],
) -> Dict[str, object]:
    baseline_total_params = compute_baseline_param_count(model)
    _, dense_width_budget = _infer_hidden_and_intermediate_from_mlp(model.model.layers[moe_layer_indices[0]].mlp)
    original_target_params = 0
    for layer_idx in moe_layer_indices:
        original_target_params += count_module_parameters(model.model.layers[layer_idx].mlp)

    shared_width_candidates = range(int(nominal_shared_width), int(min_shared_width) - 1, -int(shared_width_step))
    selected_payload: Optional[Dict[str, object]] = None
    last_candidate_error: Optional[Exception] = None
    for shared_width in shared_width_candidates:
        total_new_target_params = 0
        shared_expert_params = 0
        sparse_expert_params = 0
        router_params = 0
        residual_scale_params = 0
        zero_ratio_values = []
        active_width = None
        active_width_ratio = None
        shared_channel_indices = None
        sparse_channel_indices = None
        sparse_expert_channel_assignments = None
        for layer_idx in moe_layer_indices:
            original_mlp = model.model.layers[layer_idx].mlp
            try:
                block, block_stats = _build_shared_residual_block_from_mlp(
                    original_mlp,
                    shared_width=shared_width,
                    enable_sparse_residual=enable_sparse_residual,
                    num_sparse_experts=num_sparse_experts,
                    sparse_top_k=sparse_top_k,
                    sparse_expert_width=sparse_expert_width,
                    use_quantized_experts=use_quantized_experts,
                    router_bias=router_bias,
                    router_jitter_noise=router_jitter_noise,
                    normalize_topk_prob=normalize_topk_prob,
                    shared_init=shared_init,
                    sparse_init=sparse_init,
                    residual_scale_init=residual_scale_init,
                    residual_scale_learnable=residual_scale_learnable,
                    residual_scale_max=residual_scale_max,
                    parameter_budget_delta=0.0,
                    noise_alpha=noise_alpha,
                )
            except ValueError as exc:
                if "Insufficient discarded channels" in str(exc):
                    last_candidate_error = exc
                    block = None
                    block_stats = None
                    break
                raise
            total_new_target_params += count_module_parameters(block)
            shared_expert_params += int(block_stats["shared_expert_params"])
            sparse_expert_params += int(block_stats["sparse_expert_params"])
            router_params += int(block_stats["router_params"])
            residual_scale_params += int(block_stats["residual_scale_params"])
            if block_stats["sparse_init_stats"]["zero_ratio_avg"] is not None:
                zero_ratio_values.append(float(block_stats["sparse_init_stats"]["zero_ratio_avg"]))
            active_width = int(block_stats["active_width"])
            active_width_ratio = float(block_stats["active_width_ratio_vs_dense"])
            if shared_channel_indices is None:
                shared_channel_indices = block_stats["shared_channel_indices"]
            if sparse_channel_indices is None:
                sparse_channel_indices = block_stats.get("sparse_channel_indices")
            if sparse_expert_channel_assignments is None:
                sparse_expert_channel_assignments = block_stats.get("sparse_expert_channel_assignments")
        if active_width is None:
            continue
        new_total_params = baseline_total_params - original_target_params + total_new_target_params
        delta_params = int(new_total_params - baseline_total_params)
        payload = {
            "baseline_total_params": int(baseline_total_params),
            "target_total_params": int(baseline_total_params),
            "new_total_params": int(new_total_params),
            "delta_params": delta_params,
            "delta_percent": float(delta_params / max(baseline_total_params, 1)),
            "strict_total_param_fair_passed": bool(new_total_params <= baseline_total_params),
            "nominal_shared_width": int(nominal_shared_width),
            "resolved_shared_width": int(shared_width),
            "sparse_expert_width": int(sparse_expert_width if enable_sparse_residual else 0),
            "num_sparse_experts": int(num_sparse_experts if enable_sparse_residual else 0),
            "router_params": int(router_params),
            "shared_expert_params": int(shared_expert_params),
            "sparse_expert_params": int(sparse_expert_params),
            "residual_scale_params": int(residual_scale_params),
            "active_width": int(active_width or shared_width),
            "active_width_ratio_vs_dense": float(active_width_ratio or 0.0),
            "total_width_budget": int(dense_width_budget),
            "width_budget_passed": bool((active_width or shared_width) <= int(dense_width_budget)),
            "shared_init": shared_init,
            "sparse_init": sparse_init,
            "shared_channel_indices": shared_channel_indices,
            "sparse_channel_indices": sparse_channel_indices,
            "sparse_expert_channel_assignments": sparse_expert_channel_assignments,
            "sparse_zero_ratio_avg": None if not zero_ratio_values else float(sum(zero_ratio_values) / len(zero_ratio_values)),
        }
        if payload["strict_total_param_fair_passed"]:
            selected_payload = payload
            break
    if selected_payload is None:
        message = "Unable to satisfy strict total parameter fairness within the configured shared_width search range."
        if last_candidate_error is not None:
            raise ValueError(message) from last_candidate_error
        raise ValueError(message)
    return selected_payload


def build_shared_residual_parameter_budget_payload(
    model,
    moe_layer_indices: List[int],
    *,
    shared_width: int,
    enable_sparse_residual: bool,
    num_sparse_experts: int,
    sparse_top_k: int,
    sparse_expert_width: int,
    use_quantized_experts: bool,
    router_bias: bool,
    router_jitter_noise: float,
    normalize_topk_prob: bool,
    shared_init: str,
    sparse_init: str,
    residual_scale_init: float,
    residual_scale_learnable: bool,
    residual_scale_max: float,
    noise_alpha: Optional[float],
    enforce_baseline_fair: bool,
    enforce_active_width_below_dense: bool,
) -> Dict[str, object]:
    baseline_total_params = compute_baseline_param_count(model)
    _, dense_width_budget = _infer_hidden_and_intermediate_from_mlp(model.model.layers[moe_layer_indices[0]].mlp)
    original_target_params = 0
    total_new_target_params = 0
    shared_expert_params = 0
    sparse_expert_params = 0
    router_params = 0
    residual_scale_params = 0
    zero_ratio_values = []
    active_width = None
    active_width_ratio = None
    shared_channel_indices = None
    sparse_channel_indices = None
    sparse_expert_channel_assignments = None

    for layer_idx in moe_layer_indices:
        original_mlp = model.model.layers[layer_idx].mlp
        original_target_params += count_module_parameters(original_mlp)
        block, block_stats = _build_shared_residual_block_from_mlp(
            original_mlp,
            shared_width=int(shared_width),
            enable_sparse_residual=enable_sparse_residual,
            num_sparse_experts=num_sparse_experts,
            sparse_top_k=sparse_top_k,
            sparse_expert_width=sparse_expert_width,
            use_quantized_experts=use_quantized_experts,
            router_bias=router_bias,
            router_jitter_noise=router_jitter_noise,
            normalize_topk_prob=normalize_topk_prob,
            shared_init=shared_init,
            sparse_init=sparse_init,
            residual_scale_init=residual_scale_init,
            residual_scale_learnable=residual_scale_learnable,
            residual_scale_max=residual_scale_max,
            parameter_budget_delta=0.0,
            noise_alpha=noise_alpha,
        )
        total_new_target_params += count_module_parameters(block)
        shared_expert_params += int(block_stats["shared_expert_params"])
        sparse_expert_params += int(block_stats["sparse_expert_params"])
        router_params += int(block_stats["router_params"])
        residual_scale_params += int(block_stats["residual_scale_params"])
        if block_stats["sparse_init_stats"]["zero_ratio_avg"] is not None:
            zero_ratio_values.append(float(block_stats["sparse_init_stats"]["zero_ratio_avg"]))
        active_width = int(block_stats["active_width"])
        active_width_ratio = float(block_stats["active_width_ratio_vs_dense"])
        if shared_channel_indices is None:
            shared_channel_indices = block_stats["shared_channel_indices"]
        if sparse_channel_indices is None:
            sparse_channel_indices = block_stats.get("sparse_channel_indices")
        if sparse_expert_channel_assignments is None:
            sparse_expert_channel_assignments = block_stats.get("sparse_expert_channel_assignments")

    new_total_params = baseline_total_params - original_target_params + total_new_target_params
    delta_params = int(new_total_params - baseline_total_params)
    return {
        "baseline_total_params": int(baseline_total_params),
        "target_total_params": int(baseline_total_params),
        "new_total_params": int(new_total_params),
        "delta_params": delta_params,
        "delta_percent": float(delta_params / max(baseline_total_params, 1)),
        "strict_total_param_fair_passed": bool(new_total_params <= baseline_total_params),
        "nominal_shared_width": int(shared_width),
        "resolved_shared_width": int(shared_width),
        "sparse_expert_width": int(sparse_expert_width if enable_sparse_residual else 0),
        "num_sparse_experts": int(num_sparse_experts if enable_sparse_residual else 0),
        "router_params": int(router_params),
        "shared_expert_params": int(shared_expert_params),
        "sparse_expert_params": int(sparse_expert_params),
        "residual_scale_params": int(residual_scale_params),
        "active_width": int(active_width or shared_width),
        "active_width_ratio_vs_dense": float(active_width_ratio or 0.0),
        "total_width_budget": int(dense_width_budget),
        "width_budget_passed": bool((active_width or shared_width) <= int(dense_width_budget)),
        "shared_init": shared_init,
        "sparse_init": sparse_init,
        "shared_channel_indices": shared_channel_indices,
        "sparse_channel_indices": sparse_channel_indices,
        "sparse_expert_channel_assignments": sparse_expert_channel_assignments,
        "sparse_zero_ratio_avg": None if not zero_ratio_values else float(sum(zero_ratio_values) / len(zero_ratio_values)),
        "enforce_baseline_fair": bool(enforce_baseline_fair),
        "enforce_active_width_below_dense": bool(enforce_active_width_below_dense),
        "shared_width_resolution_mode": "exact_requested_width",
    }


def _init_experts_random_ternary_matched(moe_block, noise_alpha: float | None = None):
    """Initialize MoE experts with random weights matching ternary distribution.

    Target zero_ratio ≈ 0.37, pos/neg ≈ 0.315 each.
    This matches typical ternary patterns observed after training (36-37% zeros).

    Strategy:
    - Weights drawn from zero-centered Gaussian with std tuned per layer
    - FusedBitLinear's forward pass will quantize to {-1, 0, 1} via STE
    - Scale parameters set to stable initial values
    - RMSNorm weights initialized to 1.0 (identity)
    """
    import math
    target_zero_ratio = 0.37
    # For N(0, sigma), fraction within [-delta, delta] approximates zero ratio
    # delta = sqrt(2) * sigma * erf^{-1}(target_zero_ratio)
    # ~= 1.414 * sigma * 0.4827 ≈ 0.683 * sigma
    # So sigma ≈ delta / 0.683
    # With delta ≈ 0.33 (roughly 1/3 of weight range), sigma ≈ 0.48

    # Under the current preflight sanity proxy we approximate ternary symbols with
    # `round().clamp(-1, 1)`, so zero covers roughly the interval (-0.5, 0.5).
    # sigma≈1.05 yields P(|x|<0.5)≈0.36~0.37 for a zero-centered Gaussian.
    target_std = 1.05

    for expert in moe_block.experts:
        for name, param in expert.named_parameters():
            if "norm" in name.lower():
                # RMSNorm weights: initialize to 1.0
                param.data.fill_(1.0)
            elif "weight" in name:
                # BitLinear / FusedBitLinear weight: random ternary-matched Gaussian
                # For FusedBitLinear, the weight is the core ternary matrix
                if param.ndim >= 2:
                    # Match the std to produce desired zero ratio
                    param.data.normal_(mean=0.0, std=target_std)
                elif param.ndim == 1:
                    param.data.zero_()
            else:
                param.data.zero_()

    # Log init stats for verification (only first expert, first layer)
    first_expert = moe_block.experts[0]
    for name, param in first_expert.named_parameters():
        if "weight" in name and param.ndim >= 2:
            # Estimate zero ratio after rounding to nearest integer (approximate ternary)
            w = param.data.detach().float()
            rounded = w.round().clamp(-1, 1)
            total = rounded.numel()
            zero_r = float((rounded == 0).sum().item() / total)
            pos_r = float((rounded == 1).sum().item() / total)
            neg_r = float((rounded == -1).sum().item() / total)
            print(f"[random_ternary_matched] expert_0 {name}: zero={zero_r:.4f} pos={pos_r:.4f} neg={neg_r:.4f} "
                  f"std={w.std().item():.4f} mean={w.mean().item():.4f}")

    # If noise_alpha is specified, add small per-expert noise for differentiation
    if noise_alpha is not None and noise_alpha > 0:
        for expert in moe_block.experts:
            for name, param in expert.named_parameters():
                if "weight" in name and param.ndim >= 2:
                    noise = torch.randn_like(param) * noise_alpha * target_std
                    param.data.add_(noise)


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
    coverage_penalty_lambda: float = 0.0,
    free_expert_scale: float = 0.5,
    free_expert_exclude_pair_experts: bool = True,
    enable_learnable_output_scale: bool = False,
    output_scale_granularity: str = "global",
    initial_moe_output_scale: float | None = None,
    moe_arch: str = "standard",
    enable_sparse_residual: bool = True,
    nominal_shared_width: Optional[int] = None,
    auto_resolve_shared_width: bool = False,
    min_shared_width: int = 2048,
    shared_width_step: int = 16,
    strict_total_param_fair: bool = False,
    shared_init: str = "dense_prefix",
    sparse_init: str = "random_ternary_matched",
    sparse_expert_width: int = 128,
    sparse_top_k: int = 1,
    residual_scale_init: float = 0.1,
    residual_scale_learnable: bool = True,
    residual_scale_max: float = 0.5,
    skip_param_budget_resolver: bool = False,
):
    num_layers = len(model.model.layers)
    if not all(0 <= idx < num_layers for idx in moe_layer_indices):
        raise ValueError(f"`moe_layer_indices` must be within [0, {num_layers - 1}]")

    parameter_budget_payload: Optional[Dict[str, object]] = None
    expert_group_assignments = None
    expert_copy_group_assignments = None
    complement_pairs = None
    effective_expert_intermediate_factor = float(expert_intermediate_factor)
    expert_intermediate_size = None
    if moe_arch == "shared_residual":
        if nominal_shared_width is None:
            nominal_shared_width = max(int(model.config.intermediate_size or 2816) - int(num_experts * sparse_expert_width), min_shared_width)
        if skip_param_budget_resolver:
            parameter_budget_payload = build_shared_residual_parameter_budget_payload(
                model,
                moe_layer_indices=moe_layer_indices,
                shared_width=int(nominal_shared_width),
                enable_sparse_residual=bool(enable_sparse_residual),
                num_sparse_experts=int(num_experts),
                sparse_top_k=int(sparse_top_k),
                sparse_expert_width=int(sparse_expert_width),
                use_quantized_experts=use_quantized_experts,
                router_bias=router_bias,
                router_jitter_noise=router_jitter_noise,
                normalize_topk_prob=normalize_topk_prob,
                shared_init=shared_init,
                sparse_init=sparse_init,
                residual_scale_init=float(residual_scale_init),
                residual_scale_learnable=bool(residual_scale_learnable),
                residual_scale_max=float(residual_scale_max),
                noise_alpha=noise_alpha,
                enforce_baseline_fair=bool(strict_total_param_fair),
                enforce_active_width_below_dense=bool(strict_total_param_fair),
            )
        else:
            parameter_budget_payload = resolve_shared_width_for_param_budget(
                model,
                moe_layer_indices=moe_layer_indices,
                nominal_shared_width=int(nominal_shared_width),
                min_shared_width=int(min_shared_width),
                shared_width_step=int(shared_width_step),
                enable_sparse_residual=bool(enable_sparse_residual),
                num_sparse_experts=int(num_experts),
                sparse_top_k=int(sparse_top_k),
                sparse_expert_width=int(sparse_expert_width),
                use_quantized_experts=use_quantized_experts,
                router_bias=router_bias,
                router_jitter_noise=router_jitter_noise,
                normalize_topk_prob=normalize_topk_prob,
                shared_init=shared_init,
                sparse_init=sparse_init,
                residual_scale_init=float(residual_scale_init),
                residual_scale_learnable=bool(residual_scale_learnable),
                residual_scale_max=float(residual_scale_max),
                noise_alpha=noise_alpha,
            )
            parameter_budget_payload["enforce_baseline_fair"] = bool(strict_total_param_fair)
            parameter_budget_payload["enforce_active_width_below_dense"] = bool(strict_total_param_fair)
            parameter_budget_payload["shared_width_resolution_mode"] = "budget_resolved"
        if strict_total_param_fair and not parameter_budget_payload["strict_total_param_fair_passed"]:
            raise ValueError("Strict total parameter fairness failed for shared_residual architecture.")

    resolved_shared_width = None if parameter_budget_payload is None else int(parameter_budget_payload["resolved_shared_width"])
    shared_residual_init_stats: Dict[str, object] = {}
    for layer_idx in moe_layer_indices:
        block = model.model.layers[layer_idx]
        original_mlp = block.mlp
        hidden_size, intermediate_size = _infer_hidden_and_intermediate_from_mlp(original_mlp)
        if moe_arch == "shared_residual":
            moe_block, block_stats = _build_shared_residual_block_from_mlp(
                original_mlp,
                shared_width=int(resolved_shared_width or nominal_shared_width or intermediate_size),
                enable_sparse_residual=bool(enable_sparse_residual),
                num_sparse_experts=int(num_experts),
                sparse_top_k=int(sparse_top_k),
                sparse_expert_width=int(sparse_expert_width),
                use_quantized_experts=use_quantized_experts,
                router_bias=router_bias,
                router_jitter_noise=router_jitter_noise,
                normalize_topk_prob=normalize_topk_prob,
                shared_init=shared_init,
                sparse_init=sparse_init,
                residual_scale_init=float(residual_scale_init),
                residual_scale_learnable=bool(residual_scale_learnable),
                residual_scale_max=float(residual_scale_max),
                parameter_budget_delta=float(parameter_budget_payload["delta_params"]) if parameter_budget_payload else 0.0,
                noise_alpha=noise_alpha,
            )
            shared_residual_init_stats[f"layer_{layer_idx}"] = block_stats
            block.mlp = moe_block
            block.use_moe = True
            continue
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
            coverage_penalty_lambda=coverage_penalty_lambda,
            free_expert_scale=free_expert_scale,
            free_expert_exclude_pair_experts=free_expert_exclude_pair_experts,
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
        elif init_method == "random_ternary_matched":
            # Random initialization matching ternary distribution expectations
            # Target: ~37% zeros, ~31.5% +1, ~31.5% -1 per expert
            _init_experts_random_ternary_matched(moe_block, noise_alpha)
        else:
            raise ValueError(f"Unsupported moe init method `{init_method}`.")

        if complement_pairs is not None:
            moe_block.complement_pairs = [tuple(pair) for pair in complement_pairs]
            moe_block.router.complement_pairs = [tuple(pair) for pair in complement_pairs]
        if expert_group_assignments is not None:
            normalized_group_assignments = {
                int(expert_idx): [int(q) for q in quarters]
                for expert_idx, quarters in expert_group_assignments.items()
            }
            moe_block.expert_group_assignments = normalized_group_assignments
            moe_block.router.configure_expert_group_assignments(normalized_group_assignments)
            if routing_mode == "relaxed_complement_coverage":
                model.config.moe_relaxed_candidate_pairs = [list(pair) for pair in moe_block.router.candidate_pairs]
                model.config.moe_relaxed_pair_penalties = [
                    {
                        "pair": list(pair),
                        "coverage_penalty": int(penalty),
                        "repeated_quarters": int(repeated),
                        "missing_quarters": int(missing),
                        "is_strict_complement": bool(strict_flag),
                    }
                    for pair, penalty, repeated, missing, strict_flag in zip(
                        moe_block.router.candidate_pairs,
                        moe_block.router.candidate_pair_penalties,
                        moe_block.router.candidate_pair_repeated_quarters,
                        moe_block.router.candidate_pair_missing_quarters,
                        moe_block.router.strict_pair_mask,
                    )
                ]

        block.mlp = moe_block
        block.use_moe = True

    model.config.use_moe = True
    model.config.moe_arch = moe_arch
    model.config.moe_num_experts = num_experts
    model.config.moe_num_experts_per_tok = int(sparse_top_k if moe_arch == "shared_residual" and enable_sparse_residual else num_experts_per_tok)
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
    model.config.moe_expert_intermediate_size = int(sparse_expert_width if moe_arch == "shared_residual" else expert_intermediate_size)
    model.config.moe_init_method = sparse_init if moe_arch == "shared_residual" else init_method
    model.config.moe_grouped_topk = grouped_topk
    model.config.moe_num_virtual_groups = num_virtual_groups
    model.config.moe_topk_per_group = topk_per_group
    model.config.moe_routing_mode = (
        f"shared_residual_top{int(sparse_top_k)}"
        if moe_arch == "shared_residual" and enable_sparse_residual
        else ("shared_residual_shared_only" if moe_arch == "shared_residual" else routing_mode)
    )
    model.config.moe_pair_weights = pair_weights
    model.config.moe_output_scale = moe_output_scale
    model.config.moe_coverage_penalty_lambda = coverage_penalty_lambda
    model.config.moe_free_expert_scale = free_expert_scale
    model.config.moe_free_expert_exclude_pair_experts = free_expert_exclude_pair_experts
    model.config.moe_free_expert_selection_rule = (
        "argmax over non-pair experts" if free_expert_exclude_pair_experts else "argmax over all experts"
    )
    model.config.moe_enable_learnable_output_scale = enable_learnable_output_scale
    model.config.moe_output_scale_granularity = output_scale_granularity
    model.config.moe_initial_output_scale = float(initial_moe_output_scale if initial_moe_output_scale is not None else moe_output_scale)
    model.config.moe_enable_sparse_residual = bool(enable_sparse_residual)
    model.config.moe_nominal_shared_width = nominal_shared_width
    model.config.moe_shared_intermediate_size = resolved_shared_width
    model.config.moe_sparse_expert_width = int(sparse_expert_width)
    model.config.moe_sparse_top_k = int(sparse_top_k)
    model.config.moe_shared_init = shared_init
    model.config.moe_sparse_init = sparse_init
    model.config.moe_residual_scale_init = float(residual_scale_init)
    model.config.moe_residual_scale_learnable = bool(residual_scale_learnable)
    model.config.moe_residual_scale_max = float(residual_scale_max)
    model.config.moe_strict_total_param_fair = bool(strict_total_param_fair)
    model.config.moe_auto_resolve_shared_width = bool(auto_resolve_shared_width)
    model.config.moe_skip_param_budget_resolver = bool(skip_param_budget_resolver)
    model.config.moe_parameter_budget_verification = parameter_budget_payload
    model.config.moe_shared_residual_init_stats = shared_residual_init_stats
    if expert_group_assignments is not None:
        model.config.moe_expert_group_assignments = expert_group_assignments
    if expert_copy_group_assignments is not None:
        model.config.moe_expert_copy_group_assignments = expert_copy_group_assignments
    if complement_pairs is not None:
        model.config.moe_complement_pairs = [list(pair) for pair in complement_pairs]
    return model
