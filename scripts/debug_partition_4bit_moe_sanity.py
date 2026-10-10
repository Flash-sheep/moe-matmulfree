#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import apply_freeze_for_upcycling, upcycle_dense_to_moe
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
    summarize_trainable_parameters,
)
from scripts.run_sparse_upcycling import load_config, write_trainable_debug_outputs
from scripts.train_moe_lm import (
    ensure_cuda_device,
    extract_optional_loss_metrics,
    flatten_router_metrics,
    get_precision_dtype,
    precision_context,
    set_seed,
)


VARIANT_TO_CONFIG = {
    "4e_top1": REPO_ROOT / "experiments/sparse_upcycling/configs/partition4_4bit_random_top1_moe_only_5k.json",
    "8e_top2": REPO_ROOT / "experiments/sparse_upcycling/configs/partition8_4bit_random_top2_moe_only_5k.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run sanity for partition 4bit MoE variants.")
    parser.add_argument("--variant", choices=sorted(VARIANT_TO_CONFIG.keys()), required=True)
    parser.add_argument("--pretrained-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--max-batches", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def build_model_for_variant(config: Dict, pretrained_path: str):
    training = config.get("training", {})
    moe_cfg = config.get("moe", {})
    model = HGRNBitForCausalLM.from_pretrained(pretrained_path, torch_dtype=torch.bfloat16)
    model = upcycle_dense_to_moe(
        model=model,
        moe_layer_indices=moe_cfg.get("layer_indices", list(range(12, 24))),
        num_experts=moe_cfg.get("num_experts", 4),
        num_experts_per_tok=moe_cfg.get("num_experts_per_tok", 1),
        noise_scale=moe_cfg.get("noise_scale", 0.0),
        use_quantized_experts=moe_cfg.get("use_quantized_experts", True),
        moe_family=moe_cfg.get("moe_family", "standard"),
        expert_weight_bits=moe_cfg.get("expert_weight_bits", 2),
        router_aux_loss_coef=moe_cfg.get("router_aux_loss_coef", 0.01),
        router_jitter_noise=moe_cfg.get("router_jitter_noise", 0.0),
        router_bias=moe_cfg.get("router_bias", False),
        normalize_topk_prob=moe_cfg.get("normalize_topk_prob", True),
        expert_intermediate_factor=moe_cfg.get("expert_intermediate_factor", 1.0),
        expert_intermediate_size=moe_cfg.get("expert_intermediate_size"),
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
        enable_output_norm_logging=moe_cfg.get("enable_output_norm_logging", True),
    )
    return model, training, moe_cfg, config.get("freeze", {})


def main() -> None:
    args = parse_args()
    config_path = VARIANT_TO_CONFIG[args.variant]
    config = load_config(config_path)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    model, training, moe_cfg, freeze_cfg = build_model_for_variant(config, args.pretrained_path)

    freeze_mode = infer_freeze_mode(freeze_cfg, dense_baseline=False)
    norm_scope = infer_norm_scope(freeze_cfg, freeze_mode)
    strict_trainable_check = infer_strict_trainable_check(freeze_cfg, config)
    moe_layer_indices = list(moe_cfg.get("layer_indices", list(range(12, 24))))
    local_backbone_layer_indices = resolve_local_backbone_layer_indices(
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=freeze_cfg.get("local_backbone_layer_indices"),
    )
    optimizer_hparams = resolve_optimizer_hparams(config, training, freeze_cfg, freeze_mode)

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
        dense_baseline=False,
    )
    trainable_summary = summarize_trainable_parameters(
        model=model,
        freeze_mode=freeze_mode,
        moe_layer_indices=moe_layer_indices,
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
    write_trainable_debug_outputs(output_dir, trainable_summary, optimizer_group_summary)
    hard_check_issues = (
        run_strict_trainable_checks(
            trainable_summary=trainable_summary,
            optimizer_group_summary=optimizer_group_summary,
            freeze_mode=freeze_mode,
            freeze_embeddings=freeze_cfg.get("freeze_embeddings", True),
            freeze_lm_head=freeze_cfg.get("freeze_lm_head", True),
            local_backbone_layer_indices=local_backbone_layer_indices,
            require_moe_router=True,
        )
        if strict_trainable_check
        else []
    )

    parameter_budget = getattr(model.config, "moe_parameter_budget_verification", None)
    if parameter_budget is not None:
        (output_dir / "parameter_budget_verification.json").write_text(
            json.dumps(parameter_budget, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    device = ensure_cuda_device(training.get("device", "cuda"))
    model.to(device)
    model.train()
    precision_args = SimpleNamespace(precision=training.get("precision", "bf16"))
    precision_dtype, _ = get_precision_dtype(precision_args)

    first_moe_block = model.model.layers[moe_layer_indices[0]].mlp
    batch_metrics: List[Dict[str, object]] = []
    forward_ok = False
    backward_ok = False
    for batch_idx in range(int(args.max_batches)):
        input_ids = torch.randint(0, model.config.vocab_size, (1, int(args.seq_len)), device=device)
        model.zero_grad(set_to_none=True)
        with precision_context(precision_dtype):
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                output_router_logits=True,
                return_dict=True,
            )
            loss = outputs.loss
        forward_ok = True
        loss.backward()
        backward_ok = True
        flat_router = flatten_router_metrics(outputs.router_metrics)
        optional_loss_metrics = extract_optional_loss_metrics(outputs)
        batch_metrics.append(
            {
                "batch_idx": batch_idx,
                "loss": float(loss.detach().cpu().item()),
                "optional_losses": optional_loss_metrics,
                "router_metrics": flat_router,
            }
        )

    summary = {
        "variant": args.variant,
        "config_path": str(config_path),
        "pretrained_path": args.pretrained_path,
        "moe_family": getattr(model.config, "moe_family", None),
        "model_construction_ok": True,
        "forward_ok": forward_ok,
        "backward_ok": backward_ok,
        "num_experts": int(first_moe_block.num_experts),
        "expert_width": int(first_moe_block.expert_intermediate_size),
        "top_k": int(first_moe_block.top_k),
        "active_width": int(first_moe_block.active_width),
        "total_expert_width": int(first_moe_block.total_expert_width),
        "active_width_ratio_vs_dense": float(first_moe_block.active_width_ratio_vs_dense),
        "total_width_ratio_vs_dense": float(first_moe_block.total_width_ratio_vs_dense),
        "expert_weight_bits": int(first_moe_block.expert_weight_bits),
        "parameter_budget_verification": parameter_budget,
        "trainable_params": int(trainable_params),
        "frozen_params": int(frozen_params),
        "trainable_parameter_counts_by_module_type": trainable_summary["trainable_parameter_counts_by_module_type"],
        "optimizer_group_summary": optimizer_group_summary,
        "strict_trainable_check_enabled": bool(strict_trainable_check),
        "strict_trainable_check_issues": hard_check_issues,
        "batch_metrics": batch_metrics,
        "sanity_pass": bool(
            forward_ok
            and backward_ok
            and int(first_moe_block.active_width) == 704
            and int(first_moe_block.total_expert_width) == 2816
            and int(first_moe_block.expert_weight_bits) == 4
            and not hard_check_issues
        ),
    }
    (output_dir / "partition_4bit_sanity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
