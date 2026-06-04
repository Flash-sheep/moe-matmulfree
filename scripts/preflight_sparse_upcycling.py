#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import ExpertMonitor, StreamingTextDataset, apply_freeze_for_upcycling, upcycle_dense_to_moe
from mmfreelm.upcycling.param_groups import (
    build_optimizer_param_groups,
    resolve_optimizer_hparams,
    run_strict_trainable_checks,
)
from mmfreelm.upcycling.trainable_scope import (
    infer_freeze_mode,
    infer_norm_scope,
    infer_strict_trainable_check,
    resolve_local_backbone_layer_indices,
    summarize_trainable_parameters as summarize_trainable_scope,
)
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, evaluate, flatten_router_metrics, json_default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight initialization check for sparse upcycling configs.")
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--run-output-dir", type=Path)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_loader(data_source: str, tokenizer_path: str, max_length: int, text_field: str, batch_size: int) -> DataLoader:
    dataset = StreamingTextDataset(
        data_source=data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        split="validation",
        text_field=text_field,
        max_samples=max(batch_size * 8, 64),
    )
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_streaming_batch)


def quantize_symbols(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.detach().float().cpu()
    scale = 1.0 / weight.abs().mean().clamp(min=1e-8)
    return (weight * scale).round().clamp(-1, 1)


def concat_weight_tensors(module) -> torch.Tensor:
    tensors = []
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            tensors.append(param.detach().float().cpu().reshape(-1))
    if not tensors:
        raise RuntimeError("No matrix weights found for similarity computation.")
    return torch.cat(tensors, dim=0)


def concat_quantized_weight_tensors(module) -> torch.Tensor:
    tensors = []
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            tensors.append(quantize_symbols(param).reshape(-1).float())
    if not tensors:
        raise RuntimeError("No matrix weights found for ternary similarity computation.")
    return torch.cat(tensors, dim=0)


def pairwise_cosine(vectors: List[torch.Tensor]) -> Dict[str, float]:
    if len(vectors) < 2:
        return {
            "mean": None,
            "min": None,
            "max": None,
        }
    sims = []
    for lhs_idx, rhs_idx in combinations(range(len(vectors)), 2):
        lhs = vectors[lhs_idx]
        rhs = vectors[rhs_idx]
        sims.append(float(F.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0)).item()))
    sims_tensor = torch.tensor(sims, dtype=torch.float32)
    return {
        "mean": float(sims_tensor.mean().item()),
        "min": float(sims_tensor.min().item()),
        "max": float(sims_tensor.max().item()),
    }


def compute_zero_ratio(expert) -> Dict[str, float]:
    zeros = []
    for name, param in expert.named_parameters():
        if "weight" in name and param.dim() >= 2:
            q = quantize_symbols(param)
            zeros.append(float((q == 0).float().mean().item()))
    zeros_tensor = torch.tensor(zeros, dtype=torch.float32)
    return {
        "mean": float(zeros_tensor.mean().item()),
        "min": float(zeros_tensor.min().item()),
        "max": float(zeros_tensor.max().item()),
    }


def compute_hamming_vs_dense(expert, dense_mlp) -> Dict[str, float]:
    dense_state = dense_mlp.state_dict()
    distances = []
    for name, param in expert.named_parameters():
        if name not in dense_state or "weight" not in name or param.dim() < 2:
            continue
        dense_q = quantize_symbols(dense_state[name])
        expert_q = quantize_symbols(param)
        if dense_q.shape != expert_q.shape:
            continue
        distances.append(float((dense_q != expert_q).float().mean().item()))
    if not distances:
        return {
            "mean": None,
            "min": None,
            "max": None,
        }
    dist_tensor = torch.tensor(distances, dtype=torch.float32)
    return {
        "mean": float(dist_tensor.mean().item()),
        "min": float(dist_tensor.min().item()),
        "max": float(dist_tensor.max().item()),
    }


def router_probe(model, moe_layer_indices: List[int], dataloader: DataLoader, device: torch.device) -> Dict[str, object]:
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=input_ids, output_router_logits=True, return_dict=True)
    flat = flatten_router_metrics(outputs.router_metrics)
    probe = {
        "router_entropy": flat.get("router_entropy"),
        "tokens_per_expert": flat.get("tokens_per_expert"),
    }
    if moe_layer_indices:
        first_router = model.model.layers[moe_layer_indices[0]].mlp.router
        if first_router is None:
            probe["router_present"] = False
            model.train()
            return probe
        probe["router_present"] = True
        probe_hidden = torch.randn(64, model.config.hidden_size, device=device, dtype=input_ids.dtype if input_ids.dtype.is_floating_point else torch.float32)
        probe_hidden = probe_hidden.float()
        _, _, _, topk_indices, route_info = first_router(probe_hidden)
        if getattr(first_router, "grouped_topk", False):
            experts_per_group = first_router.experts_per_group
            group_ids = topk_indices // experts_per_group
            group_counts = []
            group_pass = True
            for group_idx in range(first_router.num_virtual_groups):
                counts = (group_ids == group_idx).sum(dim=-1)
                group_counts.append({
                    "group_id": group_idx,
                    "unique_counts": sorted({int(v) for v in counts.detach().cpu().tolist()}),
                })
                if not torch.all(counts == first_router.topk_per_group):
                    group_pass = False
            probe["group_assignment_sanity"] = {
                "pass": group_pass,
                "num_virtual_groups": first_router.num_virtual_groups,
                "topk_per_group": first_router.topk_per_group,
                "counts": group_counts,
            }
        if getattr(first_router, "routing_mode", "standard") in {"strict_complement_pair", "strict_complement_copy_pair"}:
            complement_pairs = [tuple(pair) for pair in getattr(first_router, "complement_pairs", [])]
            pair_index = {pair: idx for idx, pair in enumerate(complement_pairs)}
            sorted_pair_index = {tuple(sorted(pair)): idx for idx, pair in enumerate(complement_pairs)}
            pair_counts = {str(pair): 0 for pair in complement_pairs}
            pair_pass = True
            for row in topk_indices.detach().cpu().tolist():
                row_pair = tuple(row)
                sorted_row_pair = tuple(sorted(row_pair))
                if sorted_row_pair not in sorted_pair_index:
                    pair_pass = False
                    continue
                canonical_pair = complement_pairs[sorted_pair_index[sorted_row_pair]]
                pair_counts[str(canonical_pair)] += 1
            probe["complement_pair_sanity"] = {
                "pass": pair_pass,
                "pairs": [list(pair) for pair in complement_pairs],
                "selected_counts": pair_counts,
            }
        elif getattr(first_router, "routing_mode", "standard") == "relaxed_complement_coverage":
            candidate_pairs = [tuple(pair) for pair in getattr(first_router, "candidate_pairs", [])]
            selected_pair_index = route_info.get("selected_pair_index")
            selected_penalties = route_info.get("coverage_penalty_per_token")
            strict_pair_mask = route_info.get("strict_pair_mask")
            probe["relaxed_pair_sanity"] = {
                "pass": selected_pair_index is not None and selected_penalties is not None and len(candidate_pairs) == 15,
                "num_candidate_pairs": len(candidate_pairs),
                "strict_pair_count": int(strict_pair_mask.sum().item()) if strict_pair_mask is not None else None,
                "selected_pair_index_min": int(selected_pair_index.min().item()) if selected_pair_index is not None else None,
                "selected_pair_index_max": int(selected_pair_index.max().item()) if selected_pair_index is not None else None,
                "average_selected_penalty": float(selected_penalties.float().mean().item()) if selected_penalties is not None else None,
            }
        elif getattr(first_router, "routing_mode", "standard") == "complement_pair_plus_free":
            free_overlap = route_info.get("free_expert_overlap")
            probe["pair_plus_free_sanity"] = {
                "pass": free_overlap is not None and float(free_overlap.float().max().item()) == 0.0 and topk_indices.shape[-1] == 3,
                "top_k": int(topk_indices.shape[-1]),
                "free_expert_overlap_rate": float(free_overlap.float().mean().item()) if free_overlap is not None else None,
            }
    model.train()
    return probe


def build_relaxed_pair_penalty_table(
    expert_mapping: Dict[str, List[int]],
    strict_complement_pairs: List[List[int]],
) -> Dict[str, object]:
    strict_set = {tuple(sorted(pair)) for pair in strict_complement_pairs}
    table = []
    for pair in combinations(sorted(int(k) for k in expert_mapping.keys()), 2):
        quarter_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for expert_idx in pair:
            for quarter in expert_mapping[str(expert_idx)]:
                quarter_counts[int(quarter)] += 1
        repeated_quarters = sum(max(count - 1, 0) for count in quarter_counts.values())
        missing_quarters = sum(1 for count in quarter_counts.values() if count == 0)
        table.append(
            {
                "pair": list(pair),
                "covered_quarters": [q for q, count in quarter_counts.items() for _ in range(count)],
                "coverage_penalty": int(repeated_quarters + missing_quarters),
                "repeated_quarters": int(repeated_quarters),
                "missing_quarters": int(missing_quarters),
                "is_strict_complement": tuple(sorted(pair)) in strict_set,
            }
        )
    return {
        "all_pairs": [entry["pair"] for entry in table],
        "coverage_penalty_table": table,
    }


def build_eval_args(precision: str, max_eval_batches: int | None) -> argparse.Namespace:
    return type(
        "EvalArgs",
        (),
        {
            "precision": precision,
            "max_eval_batches": max_eval_batches,
            "use_moe": True,
        },
    )()


def estimate_active_param_budget(model, moe_layer_indices: List[int], num_experts_per_tok: int) -> Tuple[int, int]:
    total_params = sum(param.numel() for param in model.parameters())
    total_expert_params = 0
    params_per_expert = 0
    if moe_layer_indices:
        first_layer_experts = model.model.layers[moe_layer_indices[0]].mlp.experts
        if len(first_layer_experts) > 0:
            first_expert = first_layer_experts[0]
            params_per_expert = sum(param.numel() for param in first_expert.parameters())
    for layer_idx in moe_layer_indices:
        total_expert_params += sum(
            param.numel()
            for expert in model.model.layers[layer_idx].mlp.experts
            for param in expert.parameters()
        )
    active_expert_params = len(moe_layer_indices) * num_experts_per_tok * params_per_expert
    active_total_params = total_params - total_expert_params + active_expert_params
    return active_expert_params, active_total_params


def validate_complement_pairs(
    expert_mapping: Dict[str, object],
    complement_pairs: List[List[int]],
    expected_quarters: List[int],
) -> Dict[str, object]:
    pair_results = []
    passed = True
    for pair in complement_pairs:
        covered = []
        for expert_idx in pair:
            quarters = expert_mapping[str(expert_idx)]
            covered.extend(int(q) for q in quarters)
        covered_sorted = sorted(covered)
        pair_ok = covered_sorted == expected_quarters
        if not pair_ok:
            passed = False
        pair_results.append(
            {
                "pair": pair,
                "covered_quarters": covered_sorted,
                "covers_all_quarters_exactly_once": pair_ok,
            }
        )
    return {"pass": passed, "pairs": pair_results}


def summarize_trainable_parameters(
    model,
    moe_layer_indices: List[int],
    freeze_mode: str,
    local_backbone_layer_indices: List[int],
    norm_scope: str,
    trainable_extra_patterns: List[str],
) -> Dict[str, object]:
    summary = summarize_trainable_scope(
        model=model,
        freeze_mode=freeze_mode,
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=local_backbone_layer_indices,
        norm_scope=norm_scope,
        trainable_extra_patterns=trainable_extra_patterns,
    )
    by_module_type = summary["trainable_parameter_names_by_module_type"]
    counts_by_type = summary["trainable_parameter_counts_by_module_type"]
    norm_names = by_module_type["norm"]
    allowed_mlp_norm_names = {f"model.layers.{layer_idx}.mlp_norm.weight" for layer_idx in moe_layer_indices}
    mlp_norm_names = [name for name in summary["selected_norm_parameter_names"] if name in allowed_mlp_norm_names]
    other_names = by_module_type["other"]
    return {
        **summary,
        "trainable_parameter_name_summary": {
            "router": by_module_type["moe_router"],
            "experts": by_module_type["moe_experts"],
            "moe_layer_mlp_norm": mlp_norm_names,
            "learnable_output_scale": by_module_type["moe_pair_scales"],
            "other": other_names,
        },
        "trainable_parameter_count_summary": {
            "router": counts_by_type["moe_router"],
            "experts": counts_by_type["moe_experts"],
            "moe_layer_mlp_norm": sum(
                next(param.numel() for param_name, param in model.named_parameters() if param_name == name)
                for name in mlp_norm_names
            ),
            "learnable_output_scale": counts_by_type["moe_pair_scales"],
            "other": counts_by_type["other"],
        },
        "trainable_rmsnorm_parameter_names": norm_names,
        "num_trainable_rmsnorm_parameters": len(norm_names),
        "trainable_rmsnorm_parameter_elements": counts_by_type["norm"],
        "trainable_moe_layer_rmsnorm_names": mlp_norm_names,
        "num_trainable_moe_layer_rmsnorm_parameters": len(mlp_norm_names),
        "other_trainable_parameter_names": other_names,
    }


def collect_output_scale_payload(model, moe_layer_indices: List[int]) -> Dict[str, object]:
    layer_scales: Dict[str, List[float]] = {}
    scale_values: List[float] = []
    parameter_names: List[str] = []
    positive_by_construction = True
    for layer_idx in moe_layer_indices:
        moe_block = model.model.layers[layer_idx].mlp
        if hasattr(moe_block, "pair_log_scales") and moe_block.pair_log_scales is not None:
            parameter_names.append(f"model.layers.{layer_idx}.mlp.pair_log_scales")
            current_scales = torch.exp(moe_block.pair_log_scales.detach().float().cpu())
            layer_scales[f"layer_{layer_idx}"] = [float(v) for v in current_scales.tolist()]
            scale_values.extend(float(v) for v in current_scales.tolist())
        else:
            positive_by_construction = False
    if not scale_values:
        return {
            "number_of_learnable_output_scale_parameters": 0,
            "learnable_output_scale_parameter_names": [],
            "initial_scale_values": {},
            "scale_mean": None,
            "scale_min": None,
            "scale_max": None,
            "scale_positive_by_construction": False,
        }
    scale_tensor = torch.tensor(scale_values, dtype=torch.float32)
    return {
        "number_of_learnable_output_scale_parameters": len(scale_values),
        "learnable_output_scale_parameter_names": parameter_names,
        "initial_scale_values": layer_scales,
        "scale_mean": float(scale_tensor.mean().item()),
        "scale_min": float(scale_tensor.min().item()),
        "scale_max": float(scale_tensor.max().item()),
        "scale_positive_by_construction": positive_by_construction,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)
    moe_cfg = config.get("moe", {})
    freeze_cfg = config.get("freeze", {})
    training_cfg = config.get("training", {})
    dense_baseline = bool(config.get("dense_baseline", False))
    tokenizer_path = args.tokenizer_path or args.pretrained_path
    device = ensure_cuda_device(args.device)
    issues: List[str] = []
    freeze_mode = infer_freeze_mode(freeze_cfg, dense_baseline=dense_baseline)
    norm_scope = infer_norm_scope(freeze_cfg, freeze_mode)
    optimizer_hparams = resolve_optimizer_hparams(config, training_cfg, freeze_cfg, freeze_mode)

    if args.run_output_dir is not None:
        checkpoint_best = args.run_output_dir / "checkpoint_best"
        checkpoints_dir = args.run_output_dir / "checkpoints"
        has_checkpoints = checkpoint_best.exists() or (
            checkpoints_dir.exists() and any(checkpoints_dir.iterdir())
        )
        if has_checkpoints:
            issues.append(f"Run output dir already contains checkpoints: {args.run_output_dir}")

    base_model = HGRNBitForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16)
    source_model = HGRNBitForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16)
    moe_layer_indices = list(moe_cfg.get("layer_indices", list(range(12, 24))))
    local_backbone_layer_indices = resolve_local_backbone_layer_indices(
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=freeze_cfg.get("local_backbone_layer_indices"),
    )
    strict_trainable_check = infer_strict_trainable_check(freeze_cfg, config)
    baseline_hidden = source_model.model.layers[moe_layer_indices[0]].mlp.hidden_size
    baseline_intermediate = source_model.model.layers[moe_layer_indices[0]].mlp.intermediate_size

    model = upcycle_dense_to_moe(
        model=base_model,
        moe_layer_indices=moe_layer_indices,
        num_experts=moe_cfg.get("num_experts", 4),
        num_experts_per_tok=moe_cfg.get("num_experts_per_tok", 2),
        noise_scale=moe_cfg.get("noise_scale", 0.05),
        use_quantized_experts=moe_cfg.get("use_quantized_experts", True),
        router_aux_loss_coef=moe_cfg.get("router_aux_loss_coef", 0.01),
        router_jitter_noise=moe_cfg.get("router_jitter_noise", 0.0),
        router_bias=moe_cfg.get("router_bias", False),
        normalize_topk_prob=moe_cfg.get("normalize_topk_prob", True),
        expert_intermediate_factor=moe_cfg.get("expert_intermediate_factor", 1.0),
        init_method=moe_cfg.get("init_method", "copy_noise"),
        noise_alpha=moe_cfg.get("noise_alpha"),
        noise_mode=moe_cfg.get("noise_mode", "legacy_global_std"),
        grouped_topk=moe_cfg.get("grouped_topk", False),
        num_virtual_groups=moe_cfg.get("num_virtual_groups", 1),
        topk_per_group=moe_cfg.get("topk_per_group", 1),
        routing_mode=moe_cfg.get("routing_mode", "standard"),
        pair_weights=moe_cfg.get("pair_weights", "router"),
        moe_output_scale=moe_cfg.get("moe_output_scale", 1.0),
        coverage_penalty_lambda=moe_cfg.get("coverage_penalty_lambda", 0.0),
        free_expert_scale=moe_cfg.get("free_expert_scale", 0.5),
        free_expert_exclude_pair_experts=moe_cfg.get("free_expert_exclude_pair_experts", True),
        enable_learnable_output_scale=moe_cfg.get("enable_learnable_moe_output_scale", False),
        output_scale_granularity=moe_cfg.get("scale_granularity", "global"),
        initial_moe_output_scale=moe_cfg.get("initial_moe_output_scale"),
        moe_arch=moe_cfg.get("moe_arch", "standard"),
        enable_sparse_residual=moe_cfg.get("enable_sparse_residual", True),
        nominal_shared_width=moe_cfg.get("nominal_shared_width"),
        auto_resolve_shared_width=moe_cfg.get("auto_resolve_shared_width", False),
        min_shared_width=moe_cfg.get("min_shared_width", 2048),
        shared_width_step=moe_cfg.get("shared_width_step", 16),
        strict_total_param_fair=moe_cfg.get("strict_total_param_fair", False),
        shared_init=moe_cfg.get("shared_init", "dense_prefix"),
        sparse_init=moe_cfg.get("sparse_init", "random_ternary_matched"),
        sparse_expert_width=moe_cfg.get("sparse_expert_width", 128),
        sparse_top_k=moe_cfg.get("sparse_top_k", 1),
        residual_scale_init=moe_cfg.get("residual_scale_init", 0.1),
        residual_scale_learnable=moe_cfg.get("residual_scale_learnable", True),
        residual_scale_max=moe_cfg.get("residual_scale_max", 0.5),
        skip_param_budget_resolver=moe_cfg.get("skip_param_budget_resolver", False),
    )
    trainable_params, frozen_params = apply_freeze_for_upcycling(
        model=model,
        moe_layer_indices=moe_layer_indices,
        freeze_embeddings=freeze_cfg.get("freeze_embeddings", True),
        freeze_lm_head=freeze_cfg.get("freeze_lm_head", True),
        freeze_token_mixer=freeze_cfg.get("freeze_token_mixer", True),
        freeze_non_moe_mlp=freeze_cfg.get("freeze_non_moe_mlp", True),
        freeze_rmsnorm=freeze_cfg.get("freeze_rmsnorm", True),
        trainable_extra_patterns=freeze_cfg.get("trainable_extra_patterns", []),
        freeze_mode=freeze_mode,
        local_backbone_layer_indices=local_backbone_layer_indices,
        norm_scope=norm_scope,
        dense_baseline=dense_baseline,
    )
    trainable_summary = summarize_trainable_parameters(
        model,
        moe_layer_indices,
        freeze_mode=freeze_mode,
        local_backbone_layer_indices=local_backbone_layer_indices,
        norm_scope=norm_scope,
        trainable_extra_patterns=freeze_cfg.get("trainable_extra_patterns", []),
    )
    _, optimizer_group_summary = build_optimizer_param_groups(
        model=model,
        freeze_mode=freeze_mode,
        moe_lr=optimizer_hparams["moe_lr"],
        shared_expert_lr=optimizer_hparams["shared_expert_lr"],
        backbone_lr=optimizer_hparams["backbone_lr"],
        norm_lr=optimizer_hparams["norm_lr"],
        embed_lr=optimizer_hparams["embed_lr"],
        weight_decay=optimizer_hparams["weight_decay"],
        local_backbone_layer_indices=local_backbone_layer_indices,
    )
    if strict_trainable_check:
        issues.extend(
            run_strict_trainable_checks(
                trainable_summary=trainable_summary,
                optimizer_group_summary=optimizer_group_summary,
                freeze_mode=freeze_mode,
                freeze_embeddings=freeze_cfg.get("freeze_embeddings", True),
                freeze_lm_head=freeze_cfg.get("freeze_lm_head", True),
                local_backbone_layer_indices=local_backbone_layer_indices,
                require_moe_router=not (
                    moe_cfg.get("moe_arch", "standard") == "shared_residual"
                    and not moe_cfg.get("enable_sparse_residual", True)
                ),
            )
        )
    warnings: List[str] = []
    if trainable_summary["extra_pattern_enabled_parameter_count"] > 32:
        warnings.append(
            "trainable_extra_patterns enabled more than 32 additional parameters; verify that scope expansion is intended."
        )
    model = model.to(device)

    dataloader = build_loader(
        data_source=args.data_source,
        tokenizer_path=tokenizer_path,
        max_length=training_cfg.get("max_length", 2048),
        text_field=training_cfg.get("text_field", "text"),
        batch_size=training_cfg.get("batch_size", 2),
    )
    probe = router_probe(model, moe_layer_indices, dataloader, device)
    eval_args = build_eval_args(training_cfg.get("precision", "bf16"), training_cfg.get("max_eval_batches", 64))
    init_eval_metrics = evaluate(model, dataloader, device, eval_args)

    layers: Dict[str, object] = {}
    zero_ratio_values = []
    for layer_idx in moe_layer_indices:
        source_mlp = source_model.model.layers[layer_idx].mlp
        moe_block = model.model.layers[layer_idx].mlp
        experts = list(moe_block.experts)
        if experts:
            latent_vectors = [concat_weight_tensors(expert) for expert in experts]
            ternary_vectors = [concat_quantized_weight_tensors(expert) for expert in experts]
            latent_stats = pairwise_cosine(latent_vectors)
            ternary_stats = pairwise_cosine(ternary_vectors)
            expert_zero = {f"expert_{idx}": compute_zero_ratio(expert) for idx, expert in enumerate(experts)}
            zero_ratio_values.extend([stats["mean"] for stats in expert_zero.values()])
        else:
            latent_stats = {"mean": None, "min": None, "max": None}
            ternary_stats = {"mean": None, "min": None, "max": None}
            expert_zero = {}
        layer_payload: Dict[str, object] = {
            "latent_cosine_similarity": latent_stats,
            "ternary_projected_similarity": ternary_stats,
            "ternary_zero_ratio": expert_zero,
        }
        if moe_cfg.get("expert_intermediate_factor", 1.0) == 1.0:
            layer_payload["ternary_hamming_vs_dense_quantized"] = {
                f"expert_{idx}": compute_hamming_vs_dense(expert, source_mlp) for idx, expert in enumerate(experts)
            }
        layers[f"layer_{layer_idx}"] = layer_payload

    zero_ratio_avg = None if not zero_ratio_values else float(torch.tensor(zero_ratio_values, dtype=torch.float32).mean().item())
    if zero_ratio_avg is not None and (zero_ratio_avg > 0.50 or zero_ratio_avg < 0.20):
        issues.append(f"Ternary zero ratio out of expected range: {zero_ratio_avg:.4f}")

    trainable_mlp_norm_names = set(trainable_summary["trainable_moe_layer_rmsnorm_names"])
    expected_mlp_norm_names = {f"model.layers.{layer_idx}.mlp_norm.weight" for layer_idx in moe_layer_indices}
    configured_extra_patterns = freeze_cfg.get("trainable_extra_patterns", [])
    if configured_extra_patterns:
        if trainable_mlp_norm_names != expected_mlp_norm_names:
            issues.append(
                "Configured trainable extra patterns did not yield exactly the expected MoE-layer mlp_norm weights."
            )

    group_assignment = None
    if moe_cfg.get("grouped_topk", False):
        experts_per_group = moe_cfg["num_experts"] // moe_cfg.get("num_virtual_groups", 1)
        group_assignment = {str(expert_idx): expert_idx // experts_per_group for expert_idx in range(moe_cfg["num_experts"])}
        sanity = probe.get("group_assignment_sanity", {})
        if not sanity.get("pass", False):
            issues.append("Grouped top-k sanity check failed: tokens were not constrained to one expert per group.")

    complement_payload = None
    if moe_cfg.get("routing_mode") in {"strict_complement_pair", "strict_complement_copy_pair"}:
        expert_mapping = {
            str(expert_idx): list(value)
            for expert_idx, value in getattr(model.config, "moe_expert_group_assignments", {}).items()
        }
        complement_pairs = [list(pair) for pair in getattr(model.config, "moe_complement_pairs", [])]
        complement_sanity = validate_complement_pairs(expert_mapping, complement_pairs, [0, 1, 2, 3])
        router_sanity = probe.get("complement_pair_sanity", {"pass": False})
        if not complement_sanity["pass"]:
            issues.append("Complement-pair coverage sanity failed: at least one legal pair does not cover Q0-Q3 exactly once.")
        if not router_sanity.get("pass", False):
            issues.append("Router complement-pair sanity failed: probe selected an illegal expert pair.")
        complement_payload = {
            "expert_composition_mapping": expert_mapping,
            "legal_complement_pairs": complement_pairs,
            "complement_pair_coverage_sanity": complement_sanity,
            "router_complement_pair_sanity": router_sanity,
            "moe_output_scale": moe_cfg.get("moe_output_scale", 1.0),
        }
        if hasattr(model.config, "moe_expert_copy_group_assignments"):
            complement_payload["copy_group_mapping"] = {
                str(expert_idx): int(group_idx)
                for expert_idx, group_idx in getattr(model.config, "moe_expert_copy_group_assignments", {}).items()
            }
        if moe_cfg.get("routing_mode") == "strict_complement_copy_pair":
            complement_payload["legal_path_count"] = len(complement_pairs)
        if float(moe_cfg.get("moe_output_scale", 1.0)) != 2.0:
            issues.append(f"Expected moe_output_scale=2.0 for complement-pair experiment, got {moe_cfg.get('moe_output_scale')}.")
    elif moe_cfg.get("routing_mode") == "relaxed_complement_coverage":
        expert_mapping = {
            str(expert_idx): list(value)
            for expert_idx, value in getattr(model.config, "moe_expert_group_assignments", {}).items()
        }
        complement_pairs = [list(pair) for pair in getattr(model.config, "moe_complement_pairs", [])]
        relaxed_pair_payload = build_relaxed_pair_penalty_table(expert_mapping, complement_pairs)
        strict_zero_penalty = all(
            entry["coverage_penalty"] == 0
            for entry in relaxed_pair_payload["coverage_penalty_table"]
            if entry["is_strict_complement"]
        )
        if len(relaxed_pair_payload["all_pairs"]) != 15:
            issues.append(f"Expected 15 relaxed candidate pairs, got {len(relaxed_pair_payload['all_pairs'])}.")
        if not strict_zero_penalty:
            issues.append("Strict complement pairs must have coverage_penalty=0 under relaxed routing.")
        if abs(payload_width := (
            moe_cfg.get("num_experts_per_tok", 2) * getattr(model.config, "moe_expert_intermediate_size", 0) / float(max(baseline_intermediate, 1))
        ) - 1.0) > 1e-6:
            issues.append(f"Relaxed top2 active width ratio must be 1.0, got {payload_width:.4f}.")
        relaxed_probe = probe.get("relaxed_pair_sanity", {})
        if not relaxed_probe.get("pass", False):
            issues.append("Relaxed routing probe sanity failed.")
        complement_payload = {
            "expert_composition_mapping": expert_mapping,
            "strict_complement_pairs": complement_pairs,
            "legal_complement_pairs": complement_pairs,
            "moe_output_scale": moe_cfg.get("moe_output_scale", 1.0),
            "coverage_penalty_lambda_schedule": {
                "mode": "fixed",
                "coverage_penalty_lambda_start": moe_cfg.get("coverage_penalty_lambda_start", moe_cfg.get("coverage_penalty_lambda", 0.0)),
                "coverage_penalty_lambda_end": moe_cfg.get("coverage_penalty_lambda_end", moe_cfg.get("coverage_penalty_lambda", 0.0)),
                "coverage_penalty_anneal_steps": moe_cfg.get("coverage_penalty_anneal_steps", 0),
            },
            **relaxed_pair_payload,
            "router_relaxed_pair_sanity": relaxed_probe,
        }
    elif moe_cfg.get("routing_mode") == "complement_pair_plus_free":
        expert_mapping = {
            str(expert_idx): list(value)
            for expert_idx, value in getattr(model.config, "moe_expert_group_assignments", {}).items()
        }
        complement_pairs = [list(pair) for pair in getattr(model.config, "moe_complement_pairs", [])]
        complement_sanity = validate_complement_pairs(expert_mapping, complement_pairs, [0, 1, 2, 3])
        if not complement_sanity["pass"]:
            issues.append("Base complement pairs do not cover Q0-Q3 exactly once.")
        pair_plus_free_probe = probe.get("pair_plus_free_sanity", {})
        if not pair_plus_free_probe.get("pass", False):
            issues.append("Pair-plus-free routing probe sanity failed.")
        if abs(payload_width := (
            moe_cfg.get("num_experts_per_tok", 3) * getattr(model.config, "moe_expert_intermediate_size", 0) / float(max(baseline_intermediate, 1))
        ) - 1.5) > 1e-6:
            issues.append(f"Pair-plus-free active width ratio must be 1.5, got {payload_width:.4f}.")
        complement_payload = {
            "expert_composition_mapping": expert_mapping,
            "strict_complement_pairs": complement_pairs,
            "legal_complement_pairs": complement_pairs,
            "complement_pair_coverage_sanity": complement_sanity,
            "free_expert_selection_rule": getattr(model.config, "moe_free_expert_selection_rule", None),
            "free_expert_scale": moe_cfg.get("free_expert_scale", 0.5),
            "moe_output_scale_base_pair": moe_cfg.get("moe_output_scale", 1.0),
            "router_pair_plus_free_sanity": pair_plus_free_probe,
        }

    learnable_scale_enabled = bool(moe_cfg.get("enable_learnable_moe_output_scale", False))
    output_scale_payload = collect_output_scale_payload(model, moe_layer_indices) if learnable_scale_enabled else {}
    if learnable_scale_enabled:
        if moe_cfg.get("scale_granularity") != "per_layer_per_pair":
            issues.append(f"Unsupported scale_granularity for learnable output scale: {moe_cfg.get('scale_granularity')}")
        if output_scale_payload["number_of_learnable_output_scale_parameters"] == 0:
            issues.append("Learnable output scale is enabled, but no trainable pair_log_scales were found.")
        expected_scale_params = len(moe_layer_indices) * len(getattr(model.config, "moe_complement_pairs", []) or [])
        if output_scale_payload["number_of_learnable_output_scale_parameters"] != expected_scale_params:
            issues.append(
                f"Unexpected learnable output scale parameter count: "
                f"{output_scale_payload['number_of_learnable_output_scale_parameters']} vs expected {expected_scale_params}."
            )
        if not output_scale_payload.get("scale_positive_by_construction", False):
            issues.append("Learnable output scale is not positive by construction.")
        init_scale = float(moe_cfg.get("initial_moe_output_scale", moe_cfg.get("moe_output_scale", 1.0)))
        if abs(init_scale - 2.0) > 1e-6:
            issues.append(f"Expected initial_moe_output_scale=2.0, got {init_scale}.")
        for layer_name, values in output_scale_payload.get("initial_scale_values", {}).items():
            if any(abs(value - init_scale) > 1e-6 for value in values):
                issues.append(f"Learnable output scales in {layer_name} were not initialized to {init_scale}.")

    total_params = sum(param.numel() for param in model.parameters())
    active_expert_params, active_total_params = estimate_active_param_budget(
        model,
        moe_layer_indices=moe_layer_indices,
        num_experts_per_tok=moe_cfg.get("num_experts_per_tok", 2),
    )
    payload: Dict[str, object] = {
        "freeze_mode": freeze_mode,
        "norm_scope": norm_scope,
        "strict_trainable_check": strict_trainable_check,
        "warnings": warnings,
        "optimizer_hparams": optimizer_hparams,
        "optimizer_group_summary": optimizer_group_summary,
        "experiment": config.get("experiment_name"),
        "config_path": str(args.config_path),
        "pretrained_path": args.pretrained_path,
        "init_method": moe_cfg.get("init_method", "copy_noise"),
        "layer_indices": moe_layer_indices,
        "num_moe_layers": len(moe_layer_indices),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "num_experts": moe_cfg.get("num_experts", 4),
        "top_k": moe_cfg.get("num_experts_per_tok", 2),
        "expert_intermediate_size": getattr(model.config, "moe_expert_intermediate_size", None),
        "router_aux_loss_coef": moe_cfg.get("router_aux_loss_coef", 0.01),
        "noise_alpha": moe_cfg.get("noise_alpha"),
        "noise_mode": moe_cfg.get("noise_mode", "legacy_global_std"),
        "active_expert_params_per_token": active_expert_params,
        "active_total_params_per_token_approx": active_total_params,
        "active_width_vs_baseline": float(
            moe_cfg.get("num_experts_per_tok", 2) * getattr(model.config, "moe_expert_intermediate_size", 0)
        )
        / float(max(baseline_intermediate, 1)),
        "router_probe": probe,
        "init_eval": init_eval_metrics,
        "expert_metrics_summary": ExpertMonitor(model, moe_layer_indices).compute_metrics()["summary"],
        "layers": layers,
        "ternary_zero_ratio_avg": zero_ratio_avg,
        **trainable_summary,
    }

    if moe_cfg.get("grouped_topk", False):
        experts_per_group = moe_cfg["num_experts"] // moe_cfg["num_virtual_groups"]
        experts = model.model.layers[moe_layer_indices[0]].mlp.experts
        params_per_expert = sum(param.numel() for param in experts[0].parameters())
        total_expert_params = 0
        for layer_idx in moe_layer_indices:
            total_expert_params += sum(param.numel() for expert in model.model.layers[layer_idx].mlp.experts for param in expert.parameters())
        active_expert_params = len(moe_layer_indices) * moe_cfg["num_experts_per_tok"] * params_per_expert
        payload.update(
            {
                "active_expert_params_per_token": active_expert_params,
                "active_total_params_per_token_approx": total_params - total_expert_params + active_expert_params,
                "num_virtual_groups": moe_cfg.get("num_virtual_groups", 1),
                "topk_per_group": moe_cfg.get("topk_per_group", 1),
                "moe_output_scale": moe_cfg.get("moe_output_scale", 1.0),
                "group_assignment_sanity_check": probe.get("group_assignment_sanity"),
                "expert_to_shard_id": group_assignment,
                "experts_per_group": experts_per_group,
            }
        )
    if complement_payload is not None:
        payload.update(complement_payload)
    if learnable_scale_enabled:
        payload.update(
            {
                "scale_granularity": moe_cfg.get("scale_granularity", "global"),
                "initial_moe_output_scale": moe_cfg.get("initial_moe_output_scale", moe_cfg.get("moe_output_scale", 1.0)),
                **output_scale_payload,
            }
        )

    if moe_cfg.get("init_method") == "complement_pair_6e" and float(moe_cfg.get("noise_alpha") or 0.0) == 0.0:
        dense_eval_args = build_eval_args(training_cfg.get("precision", "bf16"), training_cfg.get("max_eval_batches", 64))
        source_model = source_model.to(device)
        dense_eval = evaluate(source_model, dataloader, device, dense_eval_args)
        payload["dense_reference_eval"] = dense_eval
        if abs(float(init_eval_metrics["val_ppl"]) - float(dense_eval["val_ppl"])) > 1.0:
            issues.append(
                "Alpha=0 complement-pair init deviates strongly from dense reference under preflight eval; "
                "check fused gate/up split ordering and moe_output_scale."
            )
        source_model = source_model.cpu()

    payload["pass"] = len(issues) == 0
    payload["issues"] = issues
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names_path = args.output_path.parent / "trainable_param_names.txt"
    trainable_names_path.write_text(
        "\n".join(trainable_summary["trainable_parameter_names"]) + ("\n" if trainable_summary["trainable_parameter_names"] else ""),
        encoding="utf-8",
    )
    (args.output_path.parent / "trainable_param_summary.json").write_text(
        json.dumps(trainable_summary, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (args.output_path.parent / "optimizer_param_groups.json").write_text(
        json.dumps(optimizer_group_summary, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    parameter_budget = getattr(model.config, "moe_parameter_budget_verification", None)
    if parameter_budget is not None:
        (args.output_path.parent / "parameter_budget_verification.json").write_text(
            json.dumps(parameter_budget, ensure_ascii=False, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
    args.output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
