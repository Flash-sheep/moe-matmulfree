#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import ExpertMonitor, StreamingTextDataset, apply_freeze_for_upcycling, upcycle_dense_to_moe
from scripts.complement_pair_diagnostics_lib import aggregate_router_usage
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, enrich_router_metrics, evaluate, get_precision_dtype, precision_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess sparse upcycling outputs.")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--val-data-source", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _distribution_entropy(values: List[float]) -> Tuple[float, float]:
    safe_values = [max(float(value), 1e-9) for value in values]
    total = sum(safe_values)
    if total <= 0.0:
        return 0.0, 0.0
    normalized = [value / total for value in safe_values]
    entropy = float(-sum(value * math.log(value) for value in normalized))
    max_entropy = math.log(len(normalized)) if normalized else 0.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    return entropy, normalized_entropy


def load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def build_loader(
    data_source: str,
    tokenizer_path: str,
    max_length: int,
    text_field: str,
    batch_size: int,
    max_samples: int | None,
) -> Tuple[DataLoader, Dict[str, object]]:
    dataset = StreamingTextDataset(
        data_source=data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        split="validation",
        text_field=text_field,
        max_samples=max_samples,
    )
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, pin_memory=True, collate_fn=collate_streaming_batch)
    return loader, dataset.get_manifest()


def compute_ternary_ratios(model, moe_layer_indices: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    ratios: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layer_idx in moe_layer_indices:
        layer_key = f"layer_{layer_idx}"
        ratios[layer_key] = {}
        moe = model.model.layers[layer_idx].mlp
        for expert_idx, expert in enumerate(moe.experts):
            expert_key = f"expert_{expert_idx}"
            ratios[layer_key][expert_key] = {}
            for name, param in expert.named_parameters():
                if "weight" not in name:
                    continue
                weight = param.data.detach().float()
                scale = 1.0 / weight.abs().mean().clamp_min(1e-6)
                ternary = (weight * scale).round().clamp(-1, 1)
                total = ternary.numel()
                ratios[layer_key][expert_key][name] = {
                    "neg": float((ternary == -1).sum().item() / total),
                    "zero": float((ternary == 0).sum().item() / total),
                    "pos": float((ternary == 1).sum().item() / total),
                }
    return ratios


def flatten_router_metrics(router_metrics) -> Dict[str, float]:
    if not router_metrics:
        return {}
    result: Dict[str, float] = {}
    stackable: Dict[str, List[torch.Tensor]] = {}
    for layer_metrics in router_metrics:
        for key, value in layer_metrics.items():
            if isinstance(value, torch.Tensor):
                stackable.setdefault(key, []).append(value.detach().float().cpu())
    for key, values in stackable.items():
        stacked = torch.stack([value if value.ndim > 0 else value.reshape(1) for value in values], dim=0)
        if key in {"tokens_per_expert", "router_prob_per_expert", "route_load"}:
            result[key] = stacked.mean(dim=0).tolist()
        else:
            result[key] = float(stacked.mean().item())
    return enrich_router_metrics(result)


@torch.no_grad()
def evaluate_pair_usage(
    model: HGRNBitForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    precision: str,
    max_eval_batches: int | None,
) -> Dict:
    model.eval()
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    total_loss = 0.0
    total_lm_loss = 0.0
    total_router_aux = 0.0
    total_batches = 0
    router_metrics_batches = []

    for batch_index, batch in enumerate(dataloader):
        if max_eval_batches is not None and batch_index >= max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        with precision_context(precision_dtype):
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                output_router_logits=True,
                return_dict=True,
            )
        total_loss += float(outputs.loss.detach().cpu())
        total_lm_loss += float(outputs.lm_loss.detach().cpu()) if outputs.lm_loss is not None else float(outputs.loss.detach().cpu())
        total_router_aux += float(outputs.router_aux_loss.detach().cpu()) if outputs.router_aux_loss is not None else 0.0
        total_batches += 1
        if outputs.router_metrics:
            router_metrics_batches.append(outputs.router_metrics)

    if total_batches == 0:
        raise RuntimeError("Pair-usage evaluation saw zero batches.")

    layer_indices = list(getattr(model.config, "moe_layer_indices", []) or [])
    usage = aggregate_router_usage(router_metrics_batches, layer_indices)
    avg_loss = total_loss / total_batches
    payload = {
        "val_loss": avg_loss,
        "val_lm_loss": total_lm_loss / total_batches,
        "val_router_aux_loss": total_router_aux / total_batches,
        "val_ppl": float(math.exp(min(avg_loss, 20.0))),
        "num_batches": total_batches,
        "pair_usage": usage,
        "router_entropy": usage.get("mean_router_entropy_across_layers"),
        "tokens_per_expert": usage.get("global_expert_token_fraction"),
    }
    model.train()
    return payload


def build_initial_upcycled_model(config: Dict, pretrained_path: str) -> HGRNBitForCausalLM:
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    base_model = HGRNBitForCausalLM.from_pretrained(pretrained_path, torch_dtype=torch.bfloat16)
    moe_cfg = config.get("moe", {})
    freeze_cfg = config.get("freeze", {})
    moe_layer_indices = list(moe_cfg.get("layer_indices", list(range(12, 24))))
    model = upcycle_dense_to_moe(
        model=base_model,
        moe_layer_indices=moe_layer_indices,
        num_experts=moe_cfg.get("num_experts", 8),
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
    freeze_mode = freeze_cfg.get("freeze_mode", "moe_only")
    norm_scope = freeze_cfg.get("norm_scope", "none")
    local_backbone_layer_indices = freeze_cfg.get("local_backbone_layer_indices")
    apply_freeze_for_upcycling(
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
    )
    return model


def extract_trainable_norm_payload(config: Dict, pretrained_path: str, final_model: HGRNBitForCausalLM) -> Dict[str, object]:
    initial_model = build_initial_upcycled_model(config, pretrained_path)
    trainable_names = []
    initial_norms = {}
    final_norms = {}
    norm_deltas = {}
    for name, param in initial_model.named_parameters():
        if param.requires_grad and "norm" in name.lower():
            trainable_names.append(name)
            initial_norms[name] = float(param.detach().float().norm().item())
    for name, param in final_model.named_parameters():
        if name in initial_norms:
            final_norms[name] = float(param.detach().float().norm().item())
            norm_deltas[name] = final_norms[name] - initial_norms[name]
    return {
        "trainable_norm_parameter_names": trainable_names,
        "number_of_trainable_norm_parameter_tensors": len(trainable_names),
        "initial_norms": initial_norms,
        "final_norms": final_norms,
        "norm_deltas": norm_deltas,
        "gradient_stats_available": False,
    }


def extract_learned_output_scales(model: HGRNBitForCausalLM, moe_layer_indices: List[int]) -> Dict[str, object]:
    layer_scales = {}
    values = []
    pair_names = []
    complement_pairs = list(getattr(model.config, "moe_complement_pairs", []) or [])
    for pair_idx, pair in enumerate(complement_pairs):
        pair_names.append(f"pair{pair_idx}:{tuple(pair)}")
    for layer_idx in moe_layer_indices:
        block = model.model.layers[layer_idx].mlp
        if not hasattr(block, "pair_log_scales") or block.pair_log_scales is None:
            continue
        scales = torch.exp(block.pair_log_scales.detach().float().cpu())
        values.extend(float(v) for v in scales.tolist())
        layer_scales[f"layer_{layer_idx}"] = {
            pair_names[idx] if idx < len(pair_names) else f"pair{idx}": float(value)
            for idx, value in enumerate(scales.tolist())
        }
    scale_tensor = torch.tensor(values, dtype=torch.float32) if values else torch.tensor([], dtype=torch.float32)
    deviations = [abs(v - 2.0) for v in values]
    return {
        "scale_granularity": getattr(model.config, "moe_output_scale_granularity", None),
        "initial_moe_output_scale": getattr(model.config, "moe_initial_output_scale", None),
        "pair_names": pair_names,
        "layer_pair_scales": layer_scales,
        "scale_mean": float(scale_tensor.mean().item()) if values else None,
        "scale_min": float(scale_tensor.min().item()) if values else None,
        "scale_max": float(scale_tensor.max().item()) if values else None,
        "scale_deviation_from_2.0_mean_abs": float(sum(deviations) / len(deviations)) if deviations else None,
        "has_scale_explosion": bool(values and max(values) > 4.0),
        "has_scale_collapse": bool(values and min(values) < 1.0),
    }


def build_loss_curve(
    log_records: List[Dict],
    eval1024: Dict | None = None,
    formal_checkpoint_source: str | None = None,
) -> Dict[str, List[Dict]]:
    train_curve = []
    eval_curve = []
    train_optional_fields = [
        "total_loss",
        "normalized_total_loss",
        "normalized_lm_loss",
        "normalized_router_aux_loss",
        "raw_accumulated_total_loss",
        "router_z_loss",
        "load_balancing_loss",
        "lm_loss",
        "router_aux_loss",
        "grad_norm",
        "grad_norm_last",
        "grad_norm_avg",
        "lr",
        "lr_moe",
        "lr_shared_expert",
        "lr_backbone",
        "lr_norm_or_bias",
        "lr_embed_lm_head",
        "tokens_seen",
        "elapsed_time",
        "elapsed_time_sec",
        "loss_logging_version",
        "router_entropy",
        "expert_usage",
        "expert_entropy",
        "normalized_expert_entropy",
        "expert_load_imbalance",
        "dead_expert_count",
        "pair_usage",
        "global_pair_fraction",
        "pair_entropy",
        "normalized_pair_entropy",
        "pair_load_imbalance",
        "strict_complement_pair_fraction",
        "non_complement_pair_fraction",
        "average_coverage_penalty",
        "repeated_quarter_frequency",
        "missing_quarter_frequency",
        "free_expert_usage",
        "free_expert_entropy",
        "normalized_free_expert_entropy",
        "free_expert_overlap_rate",
        "base_output_norm",
        "free_output_norm",
        "final_output_norm",
        "active_width_ratio",
        "free_expert_scale",
        "shared_output_norm",
        "sparse_output_norm",
        "sparse_to_shared_norm_ratio",
        "residual_scale",
        "active_width",
        "active_width_ratio",
        "shared_width",
        "sparse_expert_width",
        "num_sparse_experts",
        "active_sparse_experts",
        "iso_area_budget_width",
        "parameter_budget_delta",
        "router_params",
        "shared_expert_params",
        "sparse_expert_params",
    ]
    for record in log_records:
        if "train_loss" in record or "normalized_total_loss" in record:
            train_entry = {
                "step": record["step"],
                "train_loss": record.get("train_loss", record.get("normalized_total_loss")),
            }
            for field in train_optional_fields:
                if field in record:
                    train_entry[field] = record.get(field)
            train_curve.append(train_entry)
        if "proxy_val_loss" in record or record.get("val_eval_name") == "proxy_val":
            eval_curve.append(
                {
                    "step": record["step"],
                    "proxy_val_loss": record.get("proxy_val_loss", record.get("val_loss")),
                    "proxy_val_ppl": record.get("proxy_val_ppl", record.get("val_ppl")),
                    "proxy_val_lm_loss": record.get("proxy_val_lm_loss", record.get("val_lm_loss")),
                    "proxy_val_router_aux_loss": record.get("proxy_val_router_aux_loss", record.get("val_router_aux_loss")),
                    "proxy_val_router_entropy": record.get("proxy_val_router_entropy", record.get("val_router_entropy")),
                    "proxy_val_actual_num_batches": record.get("proxy_val_actual_num_batches", record.get("val_actual_num_batches")),
                    "proxy_val_actual_num_sequences": record.get("proxy_val_actual_num_sequences", record.get("val_actual_num_sequences")),
                    "proxy_val_actual_num_tokens": record.get("proxy_val_actual_num_tokens", record.get("val_actual_num_tokens")),
                    "proxy_val_scope": record.get("proxy_val_scope", record.get("val_eval_scope")),
                    "proxy_val_max_eval_batches": record.get("proxy_val_max_eval_batches"),
                    "proxy_val_max_val_samples": record.get("proxy_val_max_val_samples"),
                    "val_loss": record.get("proxy_val_loss", record.get("val_loss")),
                    "val_ppl": record.get("proxy_val_ppl", record.get("val_ppl")),
                    "val_lm_loss": record.get("proxy_val_lm_loss", record.get("val_lm_loss")),
                    "eval_name": record.get("val_eval_name", "proxy_val"),
                    "eval_scope": record.get("proxy_val_scope", record.get("val_eval_scope")),
                    "actual_num_batches": record.get("proxy_val_actual_num_batches", record.get("val_actual_num_batches")),
                    "actual_num_sequences": record.get("proxy_val_actual_num_sequences", record.get("val_actual_num_sequences")),
                    "actual_num_tokens": record.get("proxy_val_actual_num_tokens", record.get("val_actual_num_tokens")),
                }
            )
    formal_eval_curve = []
    if eval1024 is not None:
        formal_eval_curve.append(
            {
                "step": None,
                "val_loss": eval1024.get("val_loss"),
                "val_ppl": eval1024.get("val_ppl"),
                "val_lm_loss": eval1024.get("val_lm_loss"),
                "val_router_aux_loss": eval1024.get("val_router_aux_loss"),
                "val_router_entropy": eval1024.get("val_router_entropy"),
                "eval_name": eval1024.get("eval_name", "formal_eval_1024"),
                "eval_scope": eval1024.get("eval_scope", "1024seq"),
                "actual_num_batches": eval1024.get("actual_num_batches"),
                "actual_num_sequences": eval1024.get("actual_num_sequences"),
                "actual_num_tokens": eval1024.get("actual_num_tokens"),
                "checkpoint_source": formal_checkpoint_source,
                "formal_eval_1024_lm_loss": eval1024.get("val_lm_loss"),
                "formal_eval_1024_ppl": eval1024.get("val_ppl"),
            }
        )
    return {"train": train_curve, "eval": eval_curve, "formal_eval_1024": formal_eval_curve}


def build_relaxed_pair_usage_payload(model: HGRNBitForCausalLM, pair_usage_1024: Dict[str, object]) -> Dict[str, object]:
    usage = pair_usage_1024.get("pair_usage", {})
    global_pair_fraction = usage.get("global_pair_fraction", []) or []
    per_layer = usage.get("per_layer", []) or []
    pair_metadata = list(getattr(model.config, "moe_relaxed_pair_penalties", []) or [])
    strict_pair_indices = [idx for idx, entry in enumerate(pair_metadata) if entry.get("is_strict_complement")]
    non_strict_pair_indices = [idx for idx, entry in enumerate(pair_metadata) if not entry.get("is_strict_complement")]
    strict_fraction = sum(global_pair_fraction[idx] for idx in strict_pair_indices if idx < len(global_pair_fraction))
    non_strict_fraction = sum(global_pair_fraction[idx] for idx in non_strict_pair_indices if idx < len(global_pair_fraction))
    per_layer_pair_usage = {
        f"layer_{entry['layer_index']}": entry.get("pair_fraction", [])
        for entry in per_layer
    }
    return {
        "all_15_pair_fractions": [
            {
                "pair_index": idx,
                "pair": entry.get("pair"),
                "fraction": float(global_pair_fraction[idx]) if idx < len(global_pair_fraction) else 0.0,
                "coverage_penalty": entry.get("coverage_penalty"),
                "repeated_quarters": entry.get("repeated_quarters"),
                "missing_quarters": entry.get("missing_quarters"),
                "is_strict_complement": bool(entry.get("is_strict_complement", False)),
            }
            for idx, entry in enumerate(pair_metadata)
        ],
        "strict_complement_pair_fraction": float(strict_fraction),
        "non_complement_pair_fraction": float(non_strict_fraction),
        "average_coverage_penalty": usage.get("average_coverage_penalty"),
        "repeated_quarter_frequency": usage.get("repeated_quarter_frequency"),
        "missing_quarter_frequency": usage.get("missing_quarter_frequency"),
        "per_layer_pair_usage": per_layer_pair_usage,
    }


def build_free_expert_usage_payload(pair_usage_1024: Dict[str, object]) -> Dict[str, object]:
    usage = pair_usage_1024.get("pair_usage", {})
    per_layer = usage.get("per_layer", []) or []
    return {
        "base_pair_usage": usage.get("global_pair_fraction"),
        "free_expert_usage": usage.get("global_free_expert_fraction"),
        "free_expert_entropy": usage.get("global_free_expert_entropy", usage.get("free_expert_entropy")),
        "free_expert_entropy_normalized": usage.get("global_free_expert_entropy_normalized", usage.get("free_expert_entropy_normalized")),
        "free_expert_overlap_repetition_rate": usage.get("free_expert_overlap_rate"),
        "base_output_norm": usage.get("base_output_norm"),
        "free_output_norm": usage.get("free_output_norm"),
        "final_output_norm": usage.get("final_output_norm"),
        "active_width_ratio": usage.get("active_width_ratio"),
        "per_layer_free_expert_usage": {
            f"layer_{entry['layer_index']}": entry.get("free_expert_fraction", [])
            for entry in per_layer
            if "free_expert_fraction" in entry
        },
    }


def build_shared_residual_metrics_payload(
    config: Dict[str, object],
    init_verification: Dict[str, object],
    eval_results_1024: Dict[str, object],
    pair_usage_1024: Dict[str, object],
    checkpoint_path: Path,
) -> Dict[str, object]:
    moe_cfg = config.get("moe", {}) if isinstance(config.get("moe"), dict) else {}
    budget = init_verification.get("parameter_budget_verification", {})
    sparse_usage = eval_results_1024.get("val_tokens_per_expert") or pair_usage_1024.get("tokens_per_expert") or []
    sparse_usage = [float(value) for value in sparse_usage] if isinstance(sparse_usage, list) else []
    sparse_entropy, normalized_sparse_entropy = _distribution_entropy(sparse_usage) if sparse_usage else (None, None)
    sparse_load_imbalance = float(max(sparse_usage) - min(sparse_usage)) if sparse_usage else None
    dead_sparse_expert_count = int(sum(1 for value in sparse_usage if float(value) <= 1e-8)) if sparse_usage else None
    return {
        "moe_arch": moe_cfg.get("moe_arch", config.get("moe_arch", "shared_residual")),
        "eval_name": eval_results_1024.get("eval_name", "formal_eval_1024"),
        "eval_scope": eval_results_1024.get("eval_scope", "1024seq"),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_source": eval_results_1024.get("checkpoint_source", str(checkpoint_path)),
        "actual_num_sequences": eval_results_1024.get("actual_num_sequences"),
        "actual_num_batches": eval_results_1024.get("actual_num_batches"),
        "router_entropy": eval_results_1024.get("val_router_entropy", pair_usage_1024.get("router_entropy")),
        "sparse_expert_usage": sparse_usage or None,
        "sparse_expert_entropy": sparse_entropy,
        "normalized_sparse_expert_entropy": normalized_sparse_entropy,
        "sparse_load_imbalance": sparse_load_imbalance,
        "dead_sparse_expert_count": dead_sparse_expert_count,
        "shared_width": budget.get("resolved_shared_width"),
        "sparse_expert_width": budget.get("sparse_expert_width"),
        "num_sparse_experts": budget.get("num_sparse_experts"),
        "active_sparse_experts": moe_cfg.get("sparse_top_k", moe_cfg.get("moe_sparse_top_k")),
        "active_width": budget.get("active_width"),
        "active_width_ratio_vs_dense": budget.get("active_width_ratio_vs_dense"),
        "parameter_budget_delta": budget.get("delta_params"),
        "parameter_budget_delta_percent": budget.get("delta_percent"),
        "baseline_total_params": budget.get("baseline_total_params"),
        "new_total_params": budget.get("new_total_params"),
        "strict_total_param_fair_passed": budget.get("strict_total_param_fair_passed"),
        "residual_scale_init": moe_cfg.get("residual_scale_init", moe_cfg.get("moe_residual_scale_init")),
        "residual_scale_learnable": moe_cfg.get("residual_scale_learnable", moe_cfg.get("moe_residual_scale_learnable")),
        "residual_scale_max": moe_cfg.get("residual_scale_max", moe_cfg.get("moe_residual_scale_max")),
        "shared_output_norm": None,
        "sparse_output_norm": None,
        "final_output_norm": None,
        "sparse_to_shared_norm_ratio": None,
    }


def build_training_report(
    config: Dict,
    init_verification: Dict,
    log_records: List[Dict],
    eval64: Dict,
    eval1024: Dict,
    pair64: Dict,
    pair1024: Dict,
    extras: Dict[str, object],
    output_dir: Path,
    checkpoint_path: Path,
) -> Dict[str, object]:
    freeze_record = next((record for record in log_records if record.get("event") == "freeze"), {})
    best_checkpoint_record = next(
        (
            record
            for record in reversed(log_records)
            if record.get("event") in {"checkpoint_best_proxy", "checkpoint_best"}
        ),
        {},
    )
    last_checkpoint_record = next((record for record in reversed(log_records) if "checkpoint" in record), {})
    dataset_manifest_train_path = output_dir / "dataset_manifest_train.json"
    dataset_manifest_val_path = output_dir / "dataset_manifest_val.json"
    train_manifest = load_json(dataset_manifest_train_path) if dataset_manifest_train_path.exists() else None
    val_manifest = load_json(dataset_manifest_val_path) if dataset_manifest_val_path.exists() else None
    proxy_eval_record = next(
        (
            record
            for record in reversed(log_records)
            if record.get("val_eval_name") == "proxy_val" or "proxy_val_loss" in record
        ),
        {},
    )
    group_names = [
        entry.get("group_name")
        for entry in freeze_record.get("optimizer_group_summary", [])
        if entry.get("param_count", 0) > 0
    ]
    return {
        "experiment_name": config.get("experiment_name"),
        "description": config.get("description"),
        "base_checkpoint_path": init_verification.get("pretrained_path"),
        "optimizer_resumed": False,
        "scheduler_resumed": False,
        "loss_logging_version": "v2_normalized",
        "train_loss_is_normalized": True,
        "gradient_accumulation_steps": config.get("training", {}).get("gradient_accumulation_steps"),
        "log_interval": config.get("training", {}).get("log_interval"),
        "preflight": {
            "total_params": init_verification.get("total_params"),
            "trainable_params": init_verification.get("trainable_params"),
            "frozen_params": init_verification.get("frozen_params"),
            "ternary_zero_ratio_avg": init_verification.get("ternary_zero_ratio_avg"),
            "init_eval": init_verification.get("init_eval"),
        },
        "training": {
            "max_steps": config.get("training", {}).get("max_steps"),
            "estimated_tokens_trained": (
                int(config.get("training", {}).get("batch_size", 0))
                * int(config.get("training", {}).get("gradient_accumulation_steps", 0))
                * int(config.get("training", {}).get("max_length", 0))
                * int(config.get("training", {}).get("max_steps", 0))
            ),
            "freeze_record": freeze_record,
            "best_checkpoint_record": best_checkpoint_record,
        },
        "proxy_eval": {
            "enabled": proxy_eval_record != {},
            "eval_name": proxy_eval_record.get("val_eval_name", "proxy_val"),
            "eval_scope": proxy_eval_record.get("val_eval_scope"),
            "actual_num_sequences": proxy_eval_record.get("val_actual_num_sequences"),
            "actual_num_batches": proxy_eval_record.get("val_actual_num_batches"),
            "actual_num_tokens": proxy_eval_record.get("proxy_val_actual_num_tokens", proxy_eval_record.get("val_actual_num_tokens")),
            "batch_size": proxy_eval_record.get("proxy_val_batch_size"),
            "max_eval_batches": proxy_eval_record.get("proxy_val_max_eval_batches"),
            "max_val_samples": proxy_eval_record.get("proxy_val_max_val_samples"),
            "metric_for_checkpoint_best_proxy": "proxy_val_lm_loss",
        },
        "formal_eval_1024": {
            "result_path": str(output_dir / "eval_results_1024.json"),
            "checkpoint_source": str(checkpoint_path),
            "actual_num_sequences": eval1024.get("actual_num_sequences"),
            "actual_num_batches": eval1024.get("actual_num_batches"),
            "actual_num_tokens": eval1024.get("actual_num_tokens"),
        },
        "checkpointing": {
            "checkpoint_best_proxy_path": best_checkpoint_record.get("path", str(output_dir / "checkpoint_best_proxy")),
            "checkpoint_best_proxy_step": best_checkpoint_record.get("step"),
            "checkpoint_best_proxy_metric_name": "proxy_val_lm_loss",
            "checkpoint_best_proxy_metric": best_checkpoint_record.get(
                "best_proxy_val_lm_loss",
                best_checkpoint_record.get("best_proxy_val_loss", best_checkpoint_record.get("best_val_loss")),
            ),
            "checkpoint_best_alias_behavior": best_checkpoint_record.get("checkpoint_best_alias_behavior", "legacy_checkpoint_best"),
            "checkpoint_best_alias_name": "checkpoint_best",
            "checkpoint_best_alias_target": "checkpoint_best_proxy",
            "checkpoint_best_alias_path": best_checkpoint_record.get("checkpoint_best_alias", str(output_dir / "checkpoint_best")),
            "checkpoint_last_path": last_checkpoint_record.get("checkpoint"),
            "checkpoint_best_eval1024_path": None,
            "checkpoint_best_eval1024_metric": None,
            "checkpoint_best_eval1024_step": None,
        },
        "lr_logging": {
            "logged_group_lrs": True,
            "group_names": group_names,
        },
        "optimizer_groups": {entry.get("group_name"): entry for entry in freeze_record.get("optimizer_group_summary", [])},
        "moe_metrics_logging": {
            "router_entropy": True,
            "expert_usage": True,
            "pair_usage": True,
            "structure_specific_metrics": [
                "strict_complement_pair_fraction",
                "non_complement_pair_fraction",
                "free_expert_usage",
                "base_output_norm",
                "free_output_norm",
                "final_output_norm",
                "active_width_ratio",
                "shared_output_norm",
                "sparse_output_norm",
                "sparse_to_shared_norm_ratio",
                "residual_scale",
            ],
        },
        "dataset_manifests": {
            "train": str(dataset_manifest_train_path) if dataset_manifest_train_path.exists() else None,
            "val": str(dataset_manifest_val_path) if dataset_manifest_val_path.exists() else None,
            "train_payload": train_manifest,
            "val_payload": val_manifest,
        },
        "eval_results": {
            "eval_results_64": eval64,
            "eval_results_1024": eval1024,
            "pair_usage_64": {
                "val_ppl": pair64.get("val_ppl"),
                "router_entropy": pair64.get("router_entropy"),
                "global_pair_fraction": pair64.get("pair_usage", {}).get("global_pair_fraction"),
                "global_pair_entropy": pair64.get("pair_usage", {}).get("global_pair_entropy"),
            },
            "pair_usage_1024": {
                "val_ppl": pair1024.get("val_ppl"),
                "router_entropy": pair1024.get("router_entropy"),
                "global_pair_fraction": pair1024.get("pair_usage", {}).get("global_pair_fraction"),
                "global_pair_entropy": pair1024.get("pair_usage", {}).get("global_pair_entropy"),
            },
        },
        "final_ppl_1024": eval1024.get("val_ppl"),
        "extras": extras,
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    init_verification_path = args.output_dir / "init_verification.json"
    init_verification = load_json(init_verification_path) if init_verification_path.exists() else {}
    log_records = load_jsonl(args.output_dir / "train_log.jsonl")
    moe_layer_indices = list(config.get("moe", {}).get("layer_indices", list(range(12, 24))))
    training_cfg = config.get("training", {})
    device = ensure_cuda_device(args.device)
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        if (args.output_dir / "checkpoint_best_proxy").exists():
            checkpoint_path = args.output_dir / "checkpoint_best_proxy"
        else:
            checkpoint_path = args.output_dir / "checkpoint_best"
    if not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    model = HGRNBitForCausalLM.from_pretrained(str(checkpoint_path), torch_dtype=torch.bfloat16).to(device)
    text_field = training_cfg.get("text_field", "text")
    max_length = training_cfg.get("max_length", 2048)

    eval_loader_64, eval_manifest_64 = build_loader(
        data_source=args.val_data_source,
        tokenizer_path=args.tokenizer_path,
        max_length=max_length,
        text_field=text_field,
        batch_size=4,
        max_samples=None,
    )
    eval_args_64 = type(
        "EvalArgs",
        (),
        {
            "precision": args.precision,
            "max_eval_batches": 64,
            "use_moe": True,
            "eval_name": "formal_eval_64",
            "eval_scope": "64batch",
            "batch_size": 4,
            "max_val_samples": None,
            "data_source": args.val_data_source,
            "split": "validation",
            "checkpoint_source": str(checkpoint_path),
            "eval_seed": config.get("seed"),
            "eval_file_list": eval_manifest_64.get("first_20_files"),
            "eval_file_count": eval_manifest_64.get("file_count"),
        },
    )()
    eval_results_64 = evaluate(model, eval_loader_64, device, eval_args_64)
    write_json(args.output_dir / "eval_results_64.json", eval_results_64)

    eval_loader_1024, eval_manifest_1024 = build_loader(
        data_source=args.val_data_source,
        tokenizer_path=args.tokenizer_path,
        max_length=max_length,
        text_field=text_field,
        batch_size=4,
        max_samples=1024,
    )
    eval_args_1024 = type(
        "EvalArgs",
        (),
        {
            "precision": args.precision,
            "max_eval_batches": None,
            "use_moe": True,
            "eval_name": "formal_eval_1024",
            "eval_scope": "1024seq",
            "batch_size": 4,
            "max_val_samples": 1024,
            "data_source": args.val_data_source,
            "split": "validation",
            "checkpoint_source": str(checkpoint_path),
            "eval_seed": config.get("seed"),
            "eval_file_list": eval_manifest_1024.get("first_20_files"),
            "eval_file_count": eval_manifest_1024.get("file_count"),
        },
    )()
    eval_results_1024 = evaluate(model, eval_loader_1024, device, eval_args_1024)
    write_json(args.output_dir / "eval_results_1024.json", eval_results_1024)

    pair_usage_64 = evaluate_pair_usage(model, eval_loader_64, device, args.precision, max_eval_batches=64)
    pair_usage_64.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "data_source": args.val_data_source,
            "tokenizer_path": args.tokenizer_path,
            "mode": "pair_usage_only",
            "batch_size": 4,
            "max_eval_batches": 64,
            "max_samples": None,
            "max_length": max_length,
            "precision": args.precision,
            "eval_name": "formal_eval_64",
            "eval_scope": "64batch",
            "eval_file_count": eval_manifest_64.get("file_count"),
        }
    )
    write_json(args.output_dir / "pair_usage_64.json", pair_usage_64)

    pair_usage_1024 = evaluate_pair_usage(model, eval_loader_1024, device, args.precision, max_eval_batches=None)
    pair_usage_1024.update(
        {
            "checkpoint_path": str(checkpoint_path),
            "data_source": args.val_data_source,
            "tokenizer_path": args.tokenizer_path,
            "mode": "pair_usage_only",
            "batch_size": 4,
            "max_eval_batches": None,
            "max_samples": 1024,
            "max_length": max_length,
            "precision": args.precision,
            "eval_name": "formal_eval_1024",
            "eval_scope": "1024seq",
            "eval_file_count": eval_manifest_1024.get("file_count"),
        }
    )
    write_json(args.output_dir / "pair_usage_1024.json", pair_usage_1024)

    routing_mode = config.get("moe", {}).get("routing_mode", "standard")
    if routing_mode == "relaxed_complement_coverage":
        relaxed_pair_usage = build_relaxed_pair_usage_payload(model, pair_usage_1024)
        write_json(args.output_dir / "relaxed_pair_usage_1024.json", relaxed_pair_usage)
    elif routing_mode == "complement_pair_plus_free":
        free_expert_usage = build_free_expert_usage_payload(pair_usage_1024)
        write_json(args.output_dir / "free_expert_usage_1024.json", free_expert_usage)

    moe_arch = config.get("moe", {}).get("moe_arch", config.get("moe_arch", "standard"))
    if moe_arch == "shared_residual":
        shared_residual_metrics = build_shared_residual_metrics_payload(
            config=config,
            init_verification=init_verification,
            eval_results_1024=eval_results_1024,
            pair_usage_1024=pair_usage_1024,
            checkpoint_path=checkpoint_path,
        )
        write_json(args.output_dir / "shared_residual_metrics_1024.json", shared_residual_metrics)

    ternary_ratios = compute_ternary_ratios(model, moe_layer_indices)
    write_json(args.output_dir / "ternary_ratios.json", ternary_ratios)

    similarity_history_path = args.output_dir / "expert_metrics.json"
    similarity_history = load_json(similarity_history_path) if similarity_history_path.exists() else []
    if not similarity_history:
        similarity_history = [
            {"step": record["step"], "avg_expert_similarity": record["avg_expert_similarity"]}
            for record in log_records
            if "avg_expert_similarity" in record
        ]
        write_json(similarity_history_path, similarity_history)
    write_json(args.output_dir / "similarity_curve.json", similarity_history)

    loss_curve = build_loss_curve(
        log_records,
        eval1024=eval_results_1024,
        formal_checkpoint_source=str(checkpoint_path),
    )
    write_json(args.output_dir / "loss_curve.json", loss_curve)

    extras: Dict[str, object] = {}
    if config.get("freeze", {}).get("trainable_extra_patterns"):
        trainable_norm_payload = extract_trainable_norm_payload(config, args.pretrained_path, model)
        write_json(args.output_dir / "trainable_norm_params.json", trainable_norm_payload)
        extras["trainable_norm_params"] = trainable_norm_payload
    if config.get("moe", {}).get("enable_learnable_moe_output_scale", False):
        learned_scales_payload = extract_learned_output_scales(model, moe_layer_indices)
        write_json(args.output_dir / "learned_output_scales.json", learned_scales_payload)
        extras["learned_output_scales"] = learned_scales_payload

    final_monitor = ExpertMonitor(model, moe_layer_indices).compute_metrics()
    training_report = build_training_report(
        config=config,
        init_verification=init_verification,
        log_records=log_records,
        eval64=eval_results_64,
        eval1024=eval_results_1024,
        pair64=pair_usage_64,
        pair1024=pair_usage_1024,
        extras={
            **extras,
            "final_expert_metrics_summary": final_monitor.get("summary"),
        },
        output_dir=args.output_dir,
        checkpoint_path=checkpoint_path,
    )
    write_json(args.output_dir / "training_report.json", training_report)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
