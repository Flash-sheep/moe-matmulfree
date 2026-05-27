#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import StreamingTextDataset
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, get_precision_dtype, precision_context


DEFAULT_DATA_SOURCE = str(REPO_ROOT / "datasets" / "SlimPajama-6B" / "data")
DEFAULT_TOKENIZER_PATH = str(REPO_ROOT / "checkpoints" / "MMfreeLM-370M")


@dataclass
class EvalSpec:
    checkpoint_path: Path
    data_source: str
    tokenizer_path: str
    output_path: Path
    max_length: int = 2048
    batch_size: int = 2
    max_samples: Optional[int] = None
    max_eval_batches: Optional[int] = None
    text_field: str = "text"
    precision: str = "bf16"
    device: str = "cuda"


def resolve_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def write_json(path: Path, payload: Dict) -> Path:
    target = resolve_output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def build_validation_loader(spec: EvalSpec) -> DataLoader:
    dataset = StreamingTextDataset(
        data_source=spec.data_source,
        tokenizer_path=spec.tokenizer_path,
        max_length=spec.max_length,
        split="validation",
        text_field=spec.text_field,
        max_samples=spec.max_samples,
    )
    return DataLoader(dataset, batch_size=spec.batch_size, collate_fn=collate_streaming_batch)


def load_model(checkpoint_path: Path, device: str = "cuda") -> Tuple[HGRNBitForCausalLM, torch.device]:
    resolved_device = ensure_cuda_device(device)
    model = HGRNBitForCausalLM.from_pretrained(str(checkpoint_path), torch_dtype=torch.bfloat16).to(resolved_device)
    return model, resolved_device


def get_moe_layers(model: HGRNBitForCausalLM) -> List[Tuple[int, object]]:
    layer_indices = list(getattr(model.config, "moe_layer_indices", []) or [])
    result = []
    for layer_idx in layer_indices:
        moe_block = model.model.layers[layer_idx].mlp
        result.append((layer_idx, moe_block))
    return result


@contextmanager
def apply_eval_overrides(
    model: HGRNBitForCausalLM,
    pair_weight_mode: Optional[str] = None,
    output_scale_override: Optional[float] = None,
) -> Iterator[None]:
    saved: List[Tuple[object, Optional[str], Optional[float]]] = []
    try:
        for _, moe_block in get_moe_layers(model):
            saved.append(
                (
                    moe_block,
                    moe_block.router.eval_pair_weights_override,
                    moe_block.eval_output_scale_override,
                )
            )
            moe_block.router.eval_pair_weights_override = pair_weight_mode
            moe_block.eval_output_scale_override = output_scale_override
        yield
    finally:
        for moe_block, saved_pair_weight_mode, saved_output_scale in saved:
            moe_block.router.eval_pair_weights_override = saved_pair_weight_mode
            moe_block.eval_output_scale_override = saved_output_scale


def _entropy_from_fraction(fraction: torch.Tensor) -> float:
    safe = fraction.clamp_min(1e-9)
    return float((-(safe * safe.log()).sum()).item())


def _normalized_entropy_from_fraction(fraction: torch.Tensor) -> float:
    if fraction.numel() <= 1:
        return 0.0
    return _entropy_from_fraction(fraction) / math.log(fraction.numel())


def _tensor_to_list(value: torch.Tensor) -> List[float]:
    return [float(item) for item in value.tolist()]


def _compute_layer_usage_payload(
    layer_idx: int,
    pair_counts: torch.Tensor,
    expert_counts: torch.Tensor,
    active_tokens: float,
    router_entropy_sum: float,
    batch_count: int,
    extras: Optional[Dict[str, object]] = None,
) -> Dict:
    pair_counts = pair_counts.float()
    expert_counts = expert_counts.float()
    active_tokens_denom = max(active_tokens, 1.0)
    pair_fraction = pair_counts / active_tokens_denom
    pair_entropy = _entropy_from_fraction(pair_fraction)
    pair_entropy_norm = pair_entropy / math.log(pair_fraction.numel()) if pair_fraction.numel() > 1 else 0.0
    dominant_pair_fraction = float(pair_fraction.max().item()) if pair_fraction.numel() else 0.0
    pair_load_imbalance = float((pair_fraction.max() - pair_fraction.min()).item()) if pair_fraction.numel() else 0.0

    expert_token_fraction = expert_counts / active_tokens_denom
    expert_share_normalized = expert_counts / expert_counts.sum().clamp_min(1.0)
    expert_entropy = _entropy_from_fraction(expert_share_normalized)
    expert_entropy_norm = _normalized_entropy_from_fraction(expert_share_normalized)
    expert_load_imbalance = float(
        (expert_share_normalized.max() - expert_share_normalized.min()).item()
    ) if expert_share_normalized.numel() else 0.0

    payload = {
        "layer_index": layer_idx,
        "active_tokens": float(active_tokens),
        "pair0_count": float(pair_counts[0].item()) if pair_counts.numel() > 0 else 0.0,
        "pair1_count": float(pair_counts[1].item()) if pair_counts.numel() > 1 else 0.0,
        "pair2_count": float(pair_counts[2].item()) if pair_counts.numel() > 2 else 0.0,
        "pair0_fraction": float(pair_fraction[0].item()) if pair_fraction.numel() > 0 else 0.0,
        "pair1_fraction": float(pair_fraction[1].item()) if pair_fraction.numel() > 1 else 0.0,
        "pair2_fraction": float(pair_fraction[2].item()) if pair_fraction.numel() > 2 else 0.0,
        "pair_fraction": _tensor_to_list(pair_fraction),
        "pair_entropy": pair_entropy,
        "pair_entropy_normalized_by_log3": pair_entropy_norm,
        "dominant_pair_fraction": dominant_pair_fraction,
        "pair_load_imbalance": pair_load_imbalance,
        "expert_token_fraction": _tensor_to_list(expert_token_fraction),
        "expert_share_normalized": _tensor_to_list(expert_share_normalized),
        "expert_entropy": expert_entropy,
        "expert_entropy_normalized": expert_entropy_norm,
        "expert_load_imbalance": expert_load_imbalance,
        "router_entropy": router_entropy_sum / max(batch_count, 1),
    }
    if extras:
        payload.update(extras)
    return payload


def aggregate_router_usage(
    router_metrics_batches: Sequence[Tuple[Dict[str, torch.Tensor], ...]],
    layer_indices: Sequence[int],
) -> Dict:
    if not router_metrics_batches:
        return {
            "num_layers": 0,
            "per_layer": [],
            "global_pair_fraction": [],
            "global_pair_entropy": 0.0,
            "global_normalized_entropy": 0.0,
            "global_load_imbalance": 0.0,
            "global_expert_token_fraction": [],
            "global_expert_share_normalized": [],
            "expert_entropy": 0.0,
            "expert_entropy_normalized": 0.0,
            "expert_load_imbalance": 0.0,
            "notes": ["No router metrics were collected."],
        }

    pair_counts_by_layer: Dict[int, torch.Tensor] = {}
    expert_counts_by_layer: Dict[int, torch.Tensor] = {}
    active_tokens_by_layer: Dict[int, float] = {}
    router_entropy_sum_by_layer: Dict[int, float] = {}
    batch_count_by_layer: Dict[int, int] = {}
    scalar_metric_sums_by_layer: Dict[int, Dict[str, float]] = {}
    vector_metric_sums_by_layer: Dict[int, Dict[str, torch.Tensor]] = {}
    weighted_scalar_metric_sums: Dict[str, float] = {}
    weighted_scalar_metric_weights: Dict[str, float] = {}

    scalar_metric_names = {
        "strict_complement_pair_fraction",
        "non_complement_pair_fraction",
        "average_coverage_penalty",
        "repeated_quarter_frequency",
        "missing_quarter_frequency",
        "free_expert_entropy",
        "free_expert_entropy_normalized",
        "free_expert_overlap_rate",
        "base_output_norm",
        "free_output_norm",
        "final_output_norm",
        "active_width_ratio",
        "free_expert_scale",
    }
    vector_metric_names = {"free_expert_route_load", "free_expert_fraction"}

    for batch_metrics in router_metrics_batches:
        for layer_idx, layer_metrics in zip(layer_indices, batch_metrics):
            pair_counts = layer_metrics.get("pair_route_load")
            expert_counts = layer_metrics.get("route_load")
            active_tokens = float(layer_metrics.get("active_tokens", torch.tensor(0.0)).item())
            router_entropy = float(layer_metrics.get("router_entropy", torch.tensor(0.0)).item())

            if pair_counts is not None:
                pair_counts_cpu = pair_counts.detach().float().cpu()
                pair_counts_by_layer.setdefault(layer_idx, torch.zeros_like(pair_counts_cpu))
                pair_counts_by_layer[layer_idx] += pair_counts_cpu
            if expert_counts is not None:
                expert_counts_cpu = expert_counts.detach().float().cpu()
                expert_counts_by_layer.setdefault(layer_idx, torch.zeros_like(expert_counts_cpu))
                expert_counts_by_layer[layer_idx] += expert_counts_cpu
            for metric_name in scalar_metric_names:
                metric_value = layer_metrics.get(metric_name)
                if metric_value is None:
                    continue
                scalar_metric_sums_by_layer.setdefault(layer_idx, {})
                scalar_metric_sums_by_layer[layer_idx][metric_name] = (
                    scalar_metric_sums_by_layer[layer_idx].get(metric_name, 0.0) + float(metric_value.detach().float().cpu().item())
                )
                weighted_scalar_metric_sums[metric_name] = weighted_scalar_metric_sums.get(metric_name, 0.0) + (
                    float(metric_value.detach().float().cpu().item()) * max(active_tokens, 1.0)
                )
                weighted_scalar_metric_weights[metric_name] = weighted_scalar_metric_weights.get(metric_name, 0.0) + max(active_tokens, 1.0)
            for metric_name in vector_metric_names:
                metric_value = layer_metrics.get(metric_name)
                if metric_value is None:
                    continue
                metric_cpu = metric_value.detach().float().cpu()
                vector_metric_sums_by_layer.setdefault(layer_idx, {})
                if metric_name not in vector_metric_sums_by_layer[layer_idx]:
                    vector_metric_sums_by_layer[layer_idx][metric_name] = torch.zeros_like(metric_cpu)
                vector_metric_sums_by_layer[layer_idx][metric_name] += metric_cpu
            active_tokens_by_layer[layer_idx] = active_tokens_by_layer.get(layer_idx, 0.0) + active_tokens
            router_entropy_sum_by_layer[layer_idx] = router_entropy_sum_by_layer.get(layer_idx, 0.0) + router_entropy
            batch_count_by_layer[layer_idx] = batch_count_by_layer.get(layer_idx, 0) + 1

    per_layer = []
    global_pair_counts = None
    global_expert_counts = None
    global_active_tokens = 0.0
    global_router_entropy = 0.0
    global_router_batches = 0
    pair_imbalance_by_layer = {}

    for layer_idx in layer_indices:
        layer_pair_counts = pair_counts_by_layer.get(layer_idx)
        layer_expert_counts = expert_counts_by_layer.get(layer_idx)
        if layer_pair_counts is None or layer_expert_counts is None:
            continue
        layer_extras: Dict[str, object] = {}
        scalar_layer_metrics = scalar_metric_sums_by_layer.get(layer_idx, {})
        for metric_name, metric_sum in scalar_layer_metrics.items():
            layer_extras[metric_name] = metric_sum / max(batch_count_by_layer.get(layer_idx, 1), 1)
        vector_layer_metrics = vector_metric_sums_by_layer.get(layer_idx, {})
        for metric_name, metric_sum in vector_layer_metrics.items():
            layer_extras[metric_name] = _tensor_to_list(metric_sum / max(batch_count_by_layer.get(layer_idx, 1), 1))
        payload = _compute_layer_usage_payload(
            layer_idx=layer_idx,
            pair_counts=layer_pair_counts,
            expert_counts=layer_expert_counts,
            active_tokens=active_tokens_by_layer.get(layer_idx, 0.0),
            router_entropy_sum=router_entropy_sum_by_layer.get(layer_idx, 0.0),
            batch_count=batch_count_by_layer.get(layer_idx, 1),
            extras=layer_extras,
        )
        per_layer.append(payload)
        pair_imbalance_by_layer[str(layer_idx)] = payload["pair_load_imbalance"]

        global_pair_counts = layer_pair_counts if global_pair_counts is None else global_pair_counts + layer_pair_counts
        global_expert_counts = layer_expert_counts if global_expert_counts is None else global_expert_counts + layer_expert_counts
        global_active_tokens += active_tokens_by_layer.get(layer_idx, 0.0)
        global_router_entropy += router_entropy_sum_by_layer.get(layer_idx, 0.0)
        global_router_batches += batch_count_by_layer.get(layer_idx, 0)

    if global_pair_counts is None or global_expert_counts is None:
        return {
            "num_layers": 0,
            "per_layer": [],
            "notes": ["Router metrics did not include complement-pair statistics."],
        }

    global_pair_fraction = global_pair_counts / max(global_active_tokens, 1.0)
    global_pair_entropy = _entropy_from_fraction(global_pair_fraction)
    global_pair_entropy_norm = global_pair_entropy / math.log(global_pair_fraction.numel()) if global_pair_fraction.numel() > 1 else 0.0
    global_pair_imbalance = float((global_pair_fraction.max() - global_pair_fraction.min()).item())

    global_expert_token_fraction = global_expert_counts / max(global_active_tokens, 1.0)
    global_expert_share_normalized = global_expert_counts / global_expert_counts.sum().clamp_min(1.0)
    expert_entropy = _entropy_from_fraction(global_expert_share_normalized)
    expert_entropy_norm = _normalized_entropy_from_fraction(global_expert_share_normalized)
    expert_load_imbalance = float((global_expert_share_normalized.max() - global_expert_share_normalized.min()).item())

    global_payload = {
        "num_layers": len(per_layer),
        "per_layer": per_layer,
        "pair_load_imbalance_by_layer": pair_imbalance_by_layer,
        "global_pair_count": _tensor_to_list(global_pair_counts),
        "global_pair_fraction": _tensor_to_list(global_pair_fraction),
        "global_pair_entropy": global_pair_entropy,
        "global_normalized_entropy": global_pair_entropy_norm,
        "global_load_imbalance": global_pair_imbalance,
        "dominant_pair_fraction": float(global_pair_fraction.max().item()),
        "global_expert_count": _tensor_to_list(global_expert_counts),
        "global_expert_token_fraction": _tensor_to_list(global_expert_token_fraction),
        "global_expert_share_normalized": _tensor_to_list(global_expert_share_normalized),
        "expert_entropy": expert_entropy,
        "expert_entropy_normalized": expert_entropy_norm,
        "expert_load_imbalance": expert_load_imbalance,
        "mean_router_entropy_across_layers": global_router_entropy / max(global_router_batches, 1),
    }

    for metric_name, weighted_sum in weighted_scalar_metric_sums.items():
        global_payload[metric_name] = weighted_sum / max(weighted_scalar_metric_weights.get(metric_name, 1.0), 1.0)

    if any("free_expert_route_load" in metrics for metrics in vector_metric_sums_by_layer.values()):
        global_free_expert_counts = None
        for layer_metrics in vector_metric_sums_by_layer.values():
            metric_value = layer_metrics.get("free_expert_route_load")
            if metric_value is None:
                continue
            global_free_expert_counts = metric_value if global_free_expert_counts is None else global_free_expert_counts + metric_value
        if global_free_expert_counts is not None:
            global_free_expert_fraction = global_free_expert_counts / max(global_active_tokens, 1.0)
            safe_fraction = global_free_expert_fraction.clamp_min(1e-9)
            global_payload["global_free_expert_count"] = _tensor_to_list(global_free_expert_counts)
            global_payload["global_free_expert_fraction"] = _tensor_to_list(global_free_expert_fraction)
            global_payload["global_free_expert_entropy"] = _entropy_from_fraction(safe_fraction)
            global_payload["global_free_expert_entropy_normalized"] = _normalized_entropy_from_fraction(safe_fraction)

    return global_payload


def evaluate_with_router_diagnostics(
    model: HGRNBitForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    precision: str,
    max_eval_batches: Optional[int] = None,
) -> Dict:
    model.eval()
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    total_loss = 0.0
    total_lm_loss = 0.0
    total_router_aux = 0.0
    total_batches = 0
    total_samples = 0
    router_metrics_batches: List[Tuple[Dict[str, torch.Tensor], ...]] = []
    layer_indices = [layer_idx for layer_idx, _ in get_moe_layers(model)]

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_eval_batches is not None and batch_index >= max_eval_batches:
                break
            input_ids = batch["input_ids"].to(device)
            total_samples += int(input_ids.shape[0])
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
        raise RuntimeError("Evaluation loader produced zero batches.")

    avg_loss = total_loss / total_batches
    usage = aggregate_router_usage(router_metrics_batches, layer_indices)
    metrics = {
        "val_loss": avg_loss,
        "val_lm_loss": total_lm_loss / total_batches,
        "val_router_aux_loss": total_router_aux / total_batches,
        "val_ppl": float(math.exp(min(avg_loss, 20.0))),
        "num_batches": total_batches,
        "num_samples": total_samples,
        "pair_usage": usage,
    }
    if usage.get("mean_router_entropy_across_layers") is not None:
        metrics["router_entropy"] = usage["mean_router_entropy_across_layers"]
    if usage.get("global_expert_token_fraction") is not None:
        metrics["tokens_per_expert"] = usage["global_expert_token_fraction"]
    return metrics
