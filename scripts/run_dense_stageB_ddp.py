#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import signal
import shutil
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling.data_utils import StatefulPackedParquetDataset
from mmfreelm.upcycling.sparse_upcycling import _init_experts_random_ternary_matched
from scripts.run_dense_stageA_ddp import (
    append_jsonl,
    checksum_model,
    collect_initialization_diagnostics,
    compare_checksum_across_ranks,
    current_rng_state,
    gather_environment_report,
    make_gpu_snapshot,
    restore_rng_state,
    save_json,
    write_text,
)
from scripts.run_sparse_upcycling import (
    cleanup_distributed,
    distributed_barrier,
    distributed_reduce_float,
    get_precision_dtype,
    precision_context,
    setup_distributed,
    unwrap_model,
)
from scripts.train_moe_lm import set_seed


class FixedValidationDataset(Dataset):
    def __init__(self, array_path: str, limit: Optional[int] = None):
        self.data = np.load(array_path, mmap_mode="r")
        self.limit = min(int(limit), len(self.data)) if limit is not None else len(self.data)

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input_ids = torch.from_numpy(np.asarray(self.data[index, :-1], dtype=np.int64))
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "attention_mask": torch.ones_like(input_ids),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dense MatMul-free scratch Stage B DDP runner.")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--stop-after-tokens", type=int)
    parser.add_argument("--stop-after-steps", type=int)
    parser.add_argument("--checkpoint-name")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-activation", action="store_true")
    parser.add_argument("--test-mode", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_cfg = raw["model"]
    required_true = {
        "initialization.from_scratch": raw["initialization"]["from_scratch"],
        "model.train_all_parameters": raw["model"]["train_all_parameters"],
        "data.exact_resume": raw["data"]["exact_resume"],
        "data.deterministic_sharding": raw["data"]["deterministic_sharding"],
    }
    required_false = {
        "model.four_bit_enabled": raw["model"]["four_bit_enabled"],
        "data.rebuild_dataset": raw["data"]["rebuild_dataset"],
    }
    if not all(required_true.values()) or any(required_false.values()):
        raise ValueError(
            f"Unsafe Stage B config: required_true={required_true}, required_false={required_false}"
        )
    if raw["initialization"].get("pretrained_checkpoint") is not None:
        raise ValueError("Stage B must not load pretrained weights")
    if int(raw["data"].get("num_workers", -1)) != 0:
        raise ValueError("Stage B training requires num_workers=0 for exact consumed-cursor state")
    if bool(model_cfg.get("moe_enabled", False)):
        moe_cfg = raw.get("moe") or {}
        moe_family = str(moe_cfg.get("moe_family", "partition_capacity_4e"))
        if bool(model_cfg.get("four_bit_enabled", False)):
            raise ValueError("This scratch MoE runner does not permit four-bit mode.")
        if int(moe_cfg.get("expert_weight_bits", 0)) != 2:
            raise ValueError(
                "Partition scratch training requires ternary expert_weight_bits=2."
            )
        if not bool(moe_cfg.get("use_quantized_experts", False)):
            raise ValueError(
                "Partition scratch training requires use_quantized_experts=true "
                "for the ternary expert path."
            )
        if str(moe_cfg.get("init_method")) != "random_ternary_matched":
            raise ValueError("Scratch MoE training requires random_ternary_matched initialization.")
        if not bool(moe_cfg.get("all_layers", False)):
            raise ValueError("Scratch partition training must replace every FFN layer.")

        # Keep the historical 4E route unchanged, while allowing the two new
        # equal-width capacity variants through an explicit family name.
        expected_variants = {
            "partition_capacity_4e": (4, 2, 704),
            "partition_capacity_6e_top3": (6, 3, int(moe_cfg.get("expert_width", 0))),
            "partition_capacity_8e_top4": (8, 4, 352),
        }
        if moe_family not in expected_variants:
            raise ValueError(f"Unsupported scratch partition family `{moe_family}`.")
        expected_experts, expected_top_k, expected_width = expected_variants[moe_family]
        if int(moe_cfg.get("num_experts", 0)) != expected_experts:
            raise ValueError(f"{moe_family} requires num_experts={expected_experts}.")
        if int(moe_cfg.get("top_k", 0)) != expected_top_k:
            raise ValueError(f"{moe_family} requires top_k={expected_top_k}.")
        expert_width = int(moe_cfg.get("expert_width", 0))
        if expert_width <= 0:
            raise ValueError("moe.expert_width must be a positive integer.")
        if moe_family == "partition_capacity_4e" and expert_width != expected_width:
            raise ValueError("partition_capacity_4e requires expert_width=704 for the 370M model.")
        if moe_family == "partition_capacity_8e_top4" and expert_width != expected_width:
            raise ValueError("partition_capacity_8e_top4 requires expert_width=352 for the 370M model.")

        base_cfg = HGRNBitConfig.from_pretrained(raw["model"]["base_config_path"])
        dense_width = int(base_cfg.intermediate_size or 2816)
        if moe_family == "partition_capacity_6e_top3":
            expected_width = dense_width // expected_experts
            if expert_width != expected_width:
                raise ValueError(
                    f"{moe_family} requires floor({dense_width}/{expected_experts})="
                    f"{expected_width} expert channels per equal-width expert."
                )
        elif moe_family == "partition_capacity_8e_top4":
            expected_width = dense_width // expected_experts
            if expert_width != expected_width:
                raise ValueError(
                    f"{moe_family} requires {dense_width}/{expected_experts}="
                    f"{expected_width} expert channels per expert."
                )
        total_expert_width = int(moe_cfg.get("total_expert_width", 0))
        active_width = int(moe_cfg.get("active_width", 0))
        expected_total = expected_experts * expert_width
        expected_active = expected_top_k * expert_width
        if total_expert_width != expected_total:
            raise ValueError(
                f"{moe_family} declares total_expert_width={total_expert_width}, "
                f"but num_experts * expert_width={expected_total}."
            )
        if active_width != expected_active:
            raise ValueError(
                f"{moe_family} declares active_width={active_width}, "
                f"but top_k * expert_width={expected_active}."
            )
        if total_expert_width > dense_width:
            raise ValueError(
                f"{moe_family} exceeds dense intermediate width: "
                f"{total_expert_width} > {dense_width}."
            )
    return raw


def build_scratch_model_from_config(
    raw: Dict[str, object], seed: int, device: torch.device
) -> HGRNBitForCausalLM:
    """Build a fresh dense or ternary scratch-MoE model without loading weights."""
    cfg = HGRNBitConfig.from_pretrained(raw["model"]["base_config_path"])
    model_cfg = raw["model"]
    moe_cfg = raw.get("moe") or {}
    cfg.use_cache = False
    cfg.use_moe = bool(model_cfg.get("moe_enabled", False))
    if cfg.use_moe:
        cfg.moe_layer_indices = list(range(int(cfg.num_hidden_layers)))
        cfg.moe_num_experts = int(moe_cfg["num_experts"])
        cfg.moe_num_experts_per_tok = int(moe_cfg["top_k"])
        cfg.moe_use_quantized_experts = bool(moe_cfg.get("use_quantized_experts", True))
        cfg.moe_expert_weight_bits = int(moe_cfg.get("expert_weight_bits", 2))
        cfg.moe_expert_intermediate_factor = float(moe_cfg["expert_width"]) / float(cfg.intermediate_size or 2816)
        cfg.moe_expert_intermediate_size = int(moe_cfg["expert_width"])
        cfg.moe_family = str(moe_cfg.get("moe_family", "partition_capacity_4e"))
        cfg.moe_init_method = str(moe_cfg.get("init_method", "random_ternary_matched"))
        cfg.moe_partition_expert_width = int(moe_cfg["expert_width"])
        cfg.moe_partition_total_expert_width = int(moe_cfg["total_expert_width"])
        cfg.moe_partition_active_width = int(moe_cfg["active_width"])
        cfg.moe_partition_unused_dense_channels = int(moe_cfg.get("unused_dense_channels", 0))
        cfg.moe_router_aux_loss_coef = float(moe_cfg.get("router_aux_loss_coef", 0.0005))
        cfg.moe_router_jitter_noise = float(moe_cfg.get("router_jitter_noise", 0.0))
        cfg.moe_router_bias = bool(moe_cfg.get("router_bias", False))
        cfg.moe_normalize_topk_prob = bool(moe_cfg.get("normalize_topk_prob", True))
        cfg.moe_output_router_logits = False
        cfg.moe_output_scale = float(moe_cfg.get("output_scale", 1.0))
        cfg.moe_enable_output_norm_logging = bool(moe_cfg.get("enable_output_norm_logging", True))
    else:
        cfg.moe_layer_indices = []

    # Seed before construction so every parameter, including router weights, is fresh and reproducible.
    set_seed(seed)
    model = HGRNBitForCausalLM(cfg)
    if cfg.use_moe:
        for layer_idx, block in enumerate(model.model.layers):
            if not getattr(block, "use_moe", False):
                raise RuntimeError(f"Expected MoE block at layer {layer_idx}, found dense block.")
            down_proj_scale = float(moe_cfg.get("expert_down_proj_init_scale", 1.0))
            if not math.isfinite(down_proj_scale) or down_proj_scale <= 0.0:
                raise ValueError(
                    "moe.expert_down_proj_init_scale must be finite and positive, "
                    f"got {down_proj_scale}."
                )
            _init_experts_random_ternary_matched(
                block.mlp,
                down_proj_scale=down_proj_scale,
            )
    model.to(device)
    model.train()
    return model


def computed_config(raw: Dict[str, object]) -> Dict[str, object]:
    result = deepcopy(raw)
    training = result["training"]
    data = result["data"]
    distributed = result["distributed"]
    global_tokens_per_step = (
        int(training["micro_batch_size_per_rank"])
        * int(data["sequence_length"])
        * int(training["gradient_accumulation_steps"])
        * int(distributed["world_size"])
    )
    training["global_tokens_per_optimizer_step"] = global_tokens_per_step
    training["planned_total_optimizer_steps"] = math.ceil(
        int(training["planned_total_tokens"]) / global_tokens_per_step
    )
    training["stage_stop_optimizer_steps"] = math.ceil(
        int(training["stage_stop_tokens"]) / global_tokens_per_step
    )
    training["stage_actual_stop_tokens"] = (
        training["stage_stop_optimizer_steps"] * global_tokens_per_step
    )
    return result


def make_output_dir(path: Path, resume: bool, rank: int) -> None:
    if rank == 0:
        if resume:
            if not path.is_dir():
                raise FileNotFoundError(f"Resume output directory does not exist: {path}")
        else:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite Stage B output: {path}")
            path.mkdir(parents=True, exist_ok=False)
    if dist.is_initialized():
        dist.barrier()


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def batch_hash(batch: Dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in ("input_ids", "labels", "attention_mask"):
        digest.update(key.encode("ascii"))
        digest.update(batch[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def compact_worker_state(state: Dict[str, object]) -> Dict[str, object]:
    worker = state["workers"]["0"]
    return {
        "epoch": worker["epoch"],
        "assigned_shard_position": worker["assigned_shard_position"],
        "current_shard": worker["current_shard"],
        "row_group_index": worker["row_group_index"],
        "row_offset_in_group": worker["row_offset_in_group"],
        "documents_consumed": worker["documents_consumed"],
        "sequences_consumed": worker["sequences_consumed"],
        "tokens_emitted": worker["tokens_emitted"],
        "token_buffer_length": len(worker["token_buffer"]),
        "metadata_buffer_length": len(worker["metadata_buffer"]),
    }


def build_train_data(raw: Dict[str, object], rank: int, world_size: int):
    data = raw["data"]
    training = raw["training"]
    dataset = StatefulPackedParquetDataset(
        data_source=data["path"],
        tokenizer_path=data["tokenizer"],
        max_length=int(data["sequence_length"]),
        split=data["split"],
        text_field=data["text_field"],
        shard_rank=rank,
        shard_world_size=world_size,
        seed=int(data["data_seed"]),
        shuffle_shards=True,
        emit_state=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["micro_batch_size_per_rank"]),
        num_workers=0,
        drop_last=True,
    )
    return dataset, loader


def build_validation_loader(
    raw: Dict[str, object],
    rank: int,
    world_size: int,
    limit: int,
) -> DataLoader:
    validation = raw["validation"]
    dataset = FixedValidationDataset(validation["array"], limit=limit)
    indices = list(range(rank, len(dataset), world_size))
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=int(validation["batch_size_per_rank"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )


def build_optimizer_scheduler(model, raw: Dict[str, object]):
    training = raw["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["peak_learning_rate"]),
        betas=tuple(float(value) for value in training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = int(training["warmup_steps"])
    total_steps = int(training["planned_total_optimizer_steps"])
    midpoint = total_steps / 2

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = min(
            max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        midpoint_multiplier = 0.5 if step >= midpoint else 1.0
        return cosine * midpoint_multiplier

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def tensor_summary(tensor: torch.Tensor) -> Dict[str, object]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    safe = value[finite]
    if safe.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "rms": None,
            "min": None,
            "max": None,
            "non_finite_count": int((~finite).sum().item()),
        }
    return {
        "mean": float(safe.mean().item()),
        "std": float(safe.std(unbiased=False).item()),
        "rms": float(torch.sqrt(safe.square().mean()).item()),
        "min": float(safe.min().item()),
        "max": float(safe.max().item()),
        "non_finite_count": int((~finite).sum().item()),
    }


def activation_snapshot(
    model,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    global_tokens: int,
) -> Dict[str, object]:
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    rows: Dict[int, Dict[str, object]] = {
        index: {"layer_idx": index} for index in range(len(base_model.model.layers))
    }
    handles = []
    for layer_idx, block in enumerate(base_model.model.layers):
        def block_pre_hook(module, inputs, layer_idx=layer_idx):
            rows[layer_idx]["residual_input"] = tensor_summary(inputs[0])

        def attn_norm_hook(module, inputs, output, layer_idx=layer_idx):
            normalized = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["normalized_input"] = tensor_summary(normalized)

        def attn_hook(module, inputs, output, layer_idx=layer_idx):
            update = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["token_mixer_output"] = tensor_summary(update)

        def mlp_norm_hook(module, inputs, output, layer_idx=layer_idx):
            normalized = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["ffn_input"] = tensor_summary(normalized)

        def mlp_hook(module, inputs, output, layer_idx=layer_idx):
            # SparseMoEBlock returns (hidden, aux_loss, router_logits, metrics),
            # while dense MLP returns only the hidden tensor.
            hidden = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["ffn_output"] = tensor_summary(hidden)

        def block_hook(module, inputs, output, layer_idx=layer_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["residual_output"] = tensor_summary(hidden)

        handles.extend(
            [
                block.register_forward_pre_hook(block_pre_hook),
                block.attn_norm.register_forward_hook(attn_norm_hook),
                block.attn.register_forward_hook(attn_hook),
                block.mlp_norm.register_forward_hook(mlp_norm_hook),
                block.mlp.register_forward_hook(mlp_hook),
                block.register_forward_hook(block_hook),
            ]
        )
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    precision_dtype, _ = get_precision_dtype(
        type("Cfg", (), {"precision": precision})()
    )
    with torch.no_grad(), precision_context(precision_dtype):
        outputs = base_model(
            input_ids=input_ids,
            labels=labels,
            output_router_logits=False,
            return_dict=True,
        )
    for handle in handles:
        handle.remove()
    layer_rows = []
    for layer_idx in sorted(rows):
        row = rows[layer_idx]
        residual_input_rms = row["residual_input"]["rms"]
        row["token_mixer_update_to_residual_input_ratio"] = (
            row["token_mixer_output"]["rms"] / max(residual_input_rms, 1e-12)
        )
        row["ffn_update_to_residual_input_ratio"] = (
            row["ffn_output"]["rms"] / max(residual_input_rms, 1e-12)
        )
        layer_rows.append(row)
    logits = tensor_summary(outputs.logits)
    payload = {
        "global_tokens": int(global_tokens),
        "loss": float(outputs.loss.detach().cpu()),
        "logits": logits,
        "layers": layer_rows,
        "has_non_finite": (
            logits["non_finite_count"] > 0
            or any(
                row["residual_output"]["non_finite_count"] > 0
                for row in layer_rows
            )
        ),
    }
    if was_training:
        base_model.train()
    return payload


def evaluate_activation_alerts(
    snapshot: Dict[str, object],
    previous: Optional[Dict[str, object]],
    consecutive_growth_count: int,
    raw: Dict[str, object],
) -> Dict[str, object]:
    monitoring = raw["monitoring"]
    first_rms = snapshot["layers"][0]["residual_output"]["rms"]
    last_rms = snapshot["layers"][-1]["residual_output"]["rms"]
    ratio = last_rms / max(first_rms, 1e-12)
    growth = None
    if previous is not None:
        previous_last = previous["layers"][-1]["residual_output"]["rms"]
        growth = last_rms / max(previous_last, 1e-12)
    consecutive_growth_count = (
        consecutive_growth_count + 1
        if growth is not None
        and growth > float(monitoring["strong_warn_consecutive_growth_ratio"])
        else 0
    )
    warnings = []
    strong_warnings = []
    auto_stop_reasons = []
    if last_rms > float(monitoring["warn_last_residual_rms"]):
        warnings.append("last_layer_residual_rms")
    if growth is not None and growth > float(monitoring["warn_snapshot_growth_ratio"]):
        warnings.append("snapshot_growth")
    if ratio > float(monitoring["warn_last_first_ratio"]):
        warnings.append("last_first_residual_ratio")
    if last_rms > float(monitoring["strong_warn_last_residual_rms"]):
        strong_warnings.append("last_layer_residual_rms")
    if consecutive_growth_count >= 2:
        strong_warnings.append("consecutive_snapshot_growth")
    if snapshot["has_non_finite"]:
        auto_stop_reasons.append("non_finite_activation_or_logits")
    if last_rms > float(monitoring["auto_stop_last_residual_rms"]):
        auto_stop_reasons.append("last_layer_residual_rms")
    return {
        "first_layer_residual_rms": first_rms,
        "last_layer_residual_rms": last_rms,
        "last_first_ratio": ratio,
        "growth_vs_previous": growth,
        "consecutive_growth_count": consecutive_growth_count,
        "warnings": warnings,
        "strong_warnings": strong_warnings,
        "auto_stop_reasons": auto_stop_reasons,
    }


def append_activation_history(
    path: Path,
    snapshot: Dict[str, object],
    alert: Dict[str, object],
) -> None:
    middle = len(snapshot["layers"]) // 2
    selected = {
        "first": snapshot["layers"][0],
        "middle": snapshot["layers"][middle],
        "last": snapshot["layers"][-1],
    }
    row = {
        "global_tokens": snapshot["global_tokens"],
        "loss": snapshot["loss"],
        "logits_rms": snapshot["logits"]["rms"],
        "first_residual_rms": selected["first"]["residual_output"]["rms"],
        "middle_residual_rms": selected["middle"]["residual_output"]["rms"],
        "last_residual_rms": selected["last"]["residual_output"]["rms"],
        "first_normalized_rms": selected["first"]["normalized_input"]["rms"],
        "middle_normalized_rms": selected["middle"]["normalized_input"]["rms"],
        "last_normalized_rms": selected["last"]["normalized_input"]["rms"],
        "first_token_mixer_rms": selected["first"]["token_mixer_output"]["rms"],
        "middle_token_mixer_rms": selected["middle"]["token_mixer_output"]["rms"],
        "last_token_mixer_rms": selected["last"]["token_mixer_output"]["rms"],
        "first_ffn_rms": selected["first"]["ffn_output"]["rms"],
        "middle_ffn_rms": selected["middle"]["ffn_output"]["rms"],
        "last_ffn_rms": selected["last"]["ffn_output"]["rms"],
        "last_first_ratio": alert["last_first_ratio"],
        "warning_count": len(alert["warnings"]),
        "strong_warning_count": len(alert["strong_warnings"]),
        "auto_stop_count": len(alert["auto_stop_reasons"]),
    }
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(
    model,
    raw: Dict[str, object],
    dist_ctx,
    limit: int,
    kind: str,
    global_tokens: int,
) -> Dict[str, object]:
    loader = build_validation_loader(raw, dist_ctx.rank, dist_ctx.world_size, limit)
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    precision_dtype, _ = get_precision_dtype(
        type("Cfg", (), {"precision": raw["training"]["precision"]})()
    )
    local_loss_sum = 0.0
    local_tokens = 0
    local_sequences = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(dist_ctx.device)
        labels = batch["labels"].to(dist_ctx.device)
        with precision_context(precision_dtype):
            outputs = base_model(
                input_ids=input_ids,
                labels=labels,
                output_router_logits=False,
                return_dict=True,
            )
        valid_tokens = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
        local_loss_sum += float(outputs.loss.detach().cpu()) * valid_tokens
        local_tokens += valid_tokens
        local_sequences += int(input_ids.shape[0])
    stats = torch.tensor(
        [local_loss_sum, local_tokens, local_sequences],
        dtype=torch.float64,
        device=dist_ctx.device,
    )
    if dist_ctx.enabled:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss = float(stats[0].item() / max(stats[1].item(), 1))
    if was_training:
        base_model.train()
    return {
        "record_type": "validation",
        "validation_kind": kind,
        "global_tokens": int(global_tokens),
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "effective_tokens": int(stats[1].item()),
        "sequences": int(stats[2].item()),
    }


def save_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    dataset: StatefulPackedParquetDataset,
    dist_ctx,
    optimizer_step: int,
    global_tokens: int,
    last_data_hashes: List[str],
    raw: Dict[str, object],
) -> Dict[str, object]:
    ready_path = checkpoint_dir / "checkpoint_write_complete.json"
    if dist_ctx.rank == 0:
        if checkpoint_dir.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
    distributed_barrier(dist_ctx)
    dataset_state = dataset.state_dict()
    torch.save(
        dataset_state,
        checkpoint_dir / f"dataset_state_rank{dist_ctx.rank}.pt",
    )
    torch.save(
        current_rng_state(),
        checkpoint_dir / f"rng_state_rank{dist_ctx.rank}.pt",
    )
    (checkpoint_dir / f"data_boundary_rank{dist_ctx.rank}.json").write_text(
        json.dumps(
            {
                "optimizer_step": optimizer_step,
                "global_tokens": global_tokens,
                "last_data_hashes": last_data_hashes,
                "cursor": compact_worker_state(dataset_state),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    distributed_barrier(dist_ctx)
    if dist_ctx.rank == 0:
        trainer_state = {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "optimizer_step": optimizer_step,
            "scheduler_step": scheduler.last_epoch,
            "global_tokens": global_tokens,
            "world_size": dist_ctx.world_size,
        }
        torch.save(trainer_state, checkpoint_dir / "trainer_state.pt")
        unwrap_model(model).config.save_pretrained(checkpoint_dir)
        manifest = {
            "checkpoint_dir": str(checkpoint_dir),
            "optimizer_step": optimizer_step,
            "scheduler_step": scheduler.last_epoch,
            "global_tokens": global_tokens,
            "world_size": dist_ctx.world_size,
            "rank_specific_dataset_state": True,
            "packing_state_saved": True,
            "rng_state_saved_per_rank": True,
            "model_state_saved": True,
            "optimizer_state_saved": True,
            "scheduler_state_saved": True,
            "dataset_state_files": [
                f"dataset_state_rank{rank}.pt" for rank in range(dist_ctx.world_size)
            ],
            "rng_state_files": [
                f"rng_state_rank{rank}.pt" for rank in range(dist_ctx.world_size)
            ],
            "config": raw,
        }
        save_json(checkpoint_dir / "checkpoint_manifest.json", manifest)
        save_json(
            ready_path,
            {
                "optimizer_step": optimizer_step,
                "global_tokens": global_tokens,
                "trainer_state_size_bytes": (
                    checkpoint_dir / "trainer_state.pt"
                ).stat().st_size,
            },
        )
    else:
        # A multi-GB rank-0 torch.save to NFS can exceed NCCL's collective
        # timeout. Poll the shared completion marker without occupying NCCL,
        # then enter the final barrier only after rank 0 is ready.
        deadline = time.monotonic() + 3600
        while not ready_path.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for checkpoint writer: {ready_path}"
                )
            time.sleep(1)
    distributed_barrier(dist_ctx)
    return {
        "path": str(checkpoint_dir),
        "optimizer_step": optimizer_step,
        "global_tokens": global_tokens,
    }


def load_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    dataset: StatefulPackedParquetDataset,
    dist_ctx,
) -> Dict[str, object]:
    state = torch.load(
        checkpoint_dir / "trainer_state.pt",
        map_location=dist_ctx.device,
        weights_only=False,
    )
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    dataset_state = torch.load(
        checkpoint_dir / f"dataset_state_rank{dist_ctx.rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    dataset.load_state_dict(dataset_state)
    rng_state = torch.load(
        checkpoint_dir / f"rng_state_rank{dist_ctx.rank}.pt",
        map_location="cpu",
        weights_only=False,
    )
    restore_rng_state(rng_state)
    return state


def fixed_diagnostic_batch(raw: Dict[str, object]) -> Dict[str, torch.Tensor]:
    dataset = FixedValidationDataset(raw["validation"]["array"], limit=2)
    samples = [dataset[index] for index in range(2)]
    return {
        key: torch.stack([sample[key] for sample in samples])
        for key in ("input_ids", "labels", "attention_mask")
    }


def milestone_step_map(milestones: List[int], tokens_per_step: int) -> Dict[int, int]:
    return {
        0 if tokens == 0 else math.ceil(int(tokens) / tokens_per_step): int(tokens)
        for tokens in milestones
    }


def main() -> None:
    args = parse_args()
    raw = computed_config(load_config(args.config_path))
    dist_ctx = setup_distributed("cuda")
    if dist_ctx.world_size != int(raw["distributed"]["world_size"]):
        raise SystemExit(
            f"Expected world_size={raw['distributed']['world_size']}, got {dist_ctx.world_size}"
        )
    make_output_dir(
        args.output_dir,
        resume=args.resume_checkpoint is not None,
        rank=dist_ctx.rank,
    )
    tb_writer = None
    try:
        if dist_ctx.rank == 0 and args.resume_checkpoint is None:
            write_text(
                args.output_dir / "resolved_config.yaml",
                yaml.safe_dump(raw, sort_keys=False),
            )
            class ConfigProxy:
                pass

            cfg_proxy = ConfigProxy()
            cfg_proxy.raw = raw
            gather_environment_report(cfg_proxy, dist_ctx, args.output_dir)
            for source, target in (
                (
                    Path(raw["validation"]["manifest"]),
                    args.output_dir / "validation_manifest.json",
                ),
                (
                    Path(raw["validation"]["hashes"]),
                    args.output_dir / "validation_sample_hashes.txt",
                ),
                (
                    REPO_ROOT / "outputs/stageB_preflight_20260723/baseline_recipe_audit.json",
                    args.output_dir / "baseline_recipe_audit.json",
                ),
                (
                    REPO_ROOT / "outputs/stageB_preflight_20260723/baseline_recipe_audit.md",
                    args.output_dir / "baseline_recipe_audit.md",
                ),
            ):
                shutil.copy2(source, target)
        distributed_barrier(dist_ctx)

        dataset, train_loader = build_train_data(raw, dist_ctx.rank, dist_ctx.world_size)
        model = build_scratch_model_from_config(
            raw,
            int(raw["initialization"]["seed"]),
            dist_ctx.device,
        )
        # Full per-parameter statistics scan all 374M parameters and is only
        # needed once for a fresh formal run. Resume and equivalence tests use
        # the rank checksum below, avoiding minutes of redundant GPU work.
        if args.resume_checkpoint is None and not args.test_mode:
            collect_initialization_diagnostics(
                model,
                dist_ctx.device,
                True,
                dist_ctx.rank,
                args.output_dir,
                int(raw["initialization"]["seed"]),
                save_output=dist_ctx.rank == 0,
            )
        train_model = (
            DDP(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=False,
            )
            if dist_ctx.enabled
            else model
        )
        base_model = unwrap_model(train_model)
        optimizer, scheduler = build_optimizer_scheduler(base_model, raw)
        precision_dtype, scaler = get_precision_dtype(
            type("Cfg", (), {"precision": raw["training"]["precision"]})()
        )
        start_step = 0
        global_tokens = 0
        if args.resume_checkpoint is not None:
            previous_summary_path = args.output_dir / "stageB_process_summary.json"
            if previous_summary_path.exists():
                previous_summary = json.loads(
                    previous_summary_path.read_text(encoding="utf-8")
                )
                if previous_summary.get("status") not in {"completed", "graceful_interrupted"}:
                    raise RuntimeError(
                        "Refusing to resume an auto-stopped Stage B phase: "
                        f"{previous_summary_path}"
                    )
            restored = load_checkpoint(
                args.resume_checkpoint,
                train_model,
                optimizer,
                scheduler,
                scaler,
                dataset,
                dist_ctx,
            )
            start_step = int(restored["optimizer_step"])
            global_tokens = int(restored["global_tokens"])
        initial_checksum = compare_checksum_across_ranks(
            checksum_model(base_model),
            dist_ctx.device,
        )
        if initial_checksum["max_abs_diff"] != 0:
            raise RuntimeError(f"Rank parameters differ at start/resume: {initial_checksum}")

        tokens_per_step = int(raw["training"]["global_tokens_per_optimizer_step"])
        configured_stop_steps = int(raw["training"]["stage_stop_optimizer_steps"])
        if args.stop_after_steps is not None:
            stop_step = int(args.stop_after_steps)
        elif args.stop_after_tokens is not None:
            stop_step = math.ceil(int(args.stop_after_tokens) / tokens_per_step)
        else:
            stop_step = configured_stop_steps
        if stop_step <= start_step:
            raise ValueError(f"stop_step={stop_step} must be greater than start_step={start_step}")

        checkpoint_steps = {
            math.ceil(int(tokens) / tokens_per_step): int(tokens)
            for tokens in (raw.get("checkpoint") or {}).get("milestone_tokens", [])
            if int(tokens) > 0 and math.ceil(int(tokens) / tokens_per_step) < stop_step
        }
        periodic_checkpoint_records: List[Dict[str, object]] = []

        metrics_path = args.output_dir / "training_metrics.jsonl"
        tensorboard_dir = args.output_dir / str(raw["training"].get("tensorboard_subdir", "tensorboard"))
        if dist_ctx.rank == 0 and bool(raw["training"].get("enable_tensorboard", False)):
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))
        hash_paths = [
            args.output_dir / f"data_hashes_rank{rank}.jsonl"
            for rank in range(dist_ctx.world_size)
        ]
        diagnostic_batch = fixed_diagnostic_batch(raw)
        activation_dir = args.output_dir / "activation_snapshots"
        if dist_ctx.rank == 0:
            activation_dir.mkdir(exist_ok=True)
        distributed_barrier(dist_ctx)
        activation_steps = milestone_step_map(
            raw["monitoring"]["activation_snapshot_token_milestones"],
            tokens_per_step,
        )
        full_val_steps = milestone_step_map(
            raw["validation"]["full_token_milestones"],
            tokens_per_step,
        )
        proxy_val_steps = milestone_step_map(
            raw["validation"]["proxy_token_milestones"],
            tokens_per_step,
        )
        previous_snapshot = None
        consecutive_growth_count = 0
        alerts: List[Dict[str, object]] = []
        validation_records: List[Dict[str, object]] = []
        if start_step > 0 and dist_ctx.rank == 0:
            existing_snapshots = sorted(activation_dir.glob("tokens_*.json"))
            if existing_snapshots:
                previous_snapshot = json.loads(
                    existing_snapshots[-1].read_text(encoding="utf-8")
                )
                consecutive_growth_count = int(
                    previous_snapshot.get("alert", {}).get(
                        "consecutive_growth_count",
                        0,
                    )
                )

        if start_step == 0 and not args.skip_activation and dist_ctx.rank == 0:
            snapshot = activation_snapshot(
                base_model,
                diagnostic_batch,
                dist_ctx.device,
                raw["training"]["precision"],
                0,
            )
            alert = evaluate_activation_alerts(
                snapshot,
                None,
                0,
                raw,
            )
            snapshot["alert"] = alert
            save_json(activation_dir / "tokens_0000000000.json", snapshot)
            append_activation_history(
                args.output_dir / "activation_rms_history.csv",
                snapshot,
                alert,
            )
            previous_snapshot = snapshot
            alerts.append({"global_tokens": 0, **alert})
        distributed_barrier(dist_ctx)
        if start_step == 0 and not args.skip_validation:
            validation_record = evaluate(
                train_model,
                raw,
                dist_ctx,
                int(raw["validation"]["full_sequences"]),
                "full",
                0,
            )
            if dist_ctx.rank == 0:
                append_jsonl(metrics_path, validation_record)
                validation_records.append(validation_record)
        distributed_barrier(dist_ctx)

        train_iterator = iter(train_loader)
        loss_window = deque(maxlen=int(raw["training"]["smoothed_loss_window"]))
        peak_memory = {rank: 0.0 for rank in range(dist_ctx.world_size)}
        step_times = []
        auto_stop = False
        auto_stop_reasons: List[str] = []
        stop_requested = False

        def request_graceful_stop(signum, _frame):
            nonlocal stop_requested
            stop_requested = True
            if dist_ctx.rank == 0:
                print(
                    f"Received signal {signum}; will save a checkpoint after the current optimizer step.",
                    flush=True,
                )

        previous_signal_handlers = {
            signal.SIGINT: signal.getsignal(signal.SIGINT),
            signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        }
        signal.signal(signal.SIGINT, request_graceful_stop)
        signal.signal(signal.SIGTERM, request_graceful_stop)
        last_data_hashes: List[str] = []
        initial_train_loss = None
        final_train_loss = None
        for step in range(start_step + 1, stop_step + 1):
            step_start = time.time()
            train_model.train()
            optimizer.zero_grad(set_to_none=True)
            local_tokens = 0
            data_time = 0.0
            local_losses = []
            micro_hashes = []
            for micro_index in range(int(raw["training"]["gradient_accumulation_steps"])):
                data_start = time.time()
                batch = next(train_iterator)
                data_time += time.time() - data_start
                micro_hashes.append(batch_hash(batch))
                local_tokens += int(batch["input_ids"].numel())
                sync_context = (
                    train_model.no_sync()
                    if (
                        dist_ctx.enabled
                        and micro_index
                        < int(raw["training"]["gradient_accumulation_steps"]) - 1
                    )
                    else nullcontext()
                )
                with sync_context:
                    input_ids = batch["input_ids"].to(dist_ctx.device)
                    labels = batch["labels"].to(dist_ctx.device)
                    with precision_context(precision_dtype):
                        outputs = train_model(
                            input_ids=input_ids,
                            labels=labels,
                            output_router_logits=False,
                            return_dict=True,
                        )
                        loss = outputs.loss / int(
                            raw["training"]["gradient_accumulation_steps"]
                        )
                    scaler.scale(loss).backward()
                    local_losses.append(float(outputs.loss.detach().cpu()))
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                train_model.parameters(),
                float(raw["training"]["grad_clip"]),
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step_time = time.time() - step_start
            step_times.append(step_time)
            step_global_tokens = int(
                distributed_reduce_float(float(local_tokens), dist_ctx, average=False)
            )
            global_tokens += step_global_tokens
            mean_loss = distributed_reduce_float(
                sum(local_losses) / len(local_losses),
                dist_ctx,
                average=True,
            )
            grad_norm_value = distributed_reduce_float(
                float(grad_norm.detach().cpu()),
                dist_ctx,
                average=True,
            )
            loss_window.append(mean_loss)
            local_step_hash = hashlib.sha256(
                "\n".join(micro_hashes).encode("ascii")
            ).hexdigest()
            last_data_hashes = micro_hashes
            append_jsonl(
                hash_paths[dist_ctx.rank],
                {
                    "optimizer_step": step,
                    "global_tokens": global_tokens,
                    "micro_batch_hashes": micro_hashes,
                    "step_hash": local_step_hash,
                    "cursor": compact_worker_state(dataset.state_dict()),
                },
            )
            if dist_ctx.enabled:
                gathered_hashes = (
                    [None for _ in range(dist_ctx.world_size)]
                    if dist_ctx.rank == 0
                    else None
                )
                dist.gather_object(local_step_hash, gathered_hashes, dst=0)
            else:
                gathered_hashes = [local_step_hash]
            checksum_report = None
            if step % int(raw["training"]["checksum_interval_steps"]) == 0:
                checksum_report = compare_checksum_across_ranks(
                    checksum_model(base_model),
                    dist_ctx.device,
                )
                if checksum_report["max_abs_diff"] != 0:
                    auto_stop = True
                    auto_stop_reasons.append("rank_parameter_checksum_mismatch")
            finite_failure = not math.isfinite(mean_loss) or not math.isfinite(grad_norm_value)
            if finite_failure:
                auto_stop = True
                auto_stop_reasons.append("non_finite_loss_or_gradient")
            if dist_ctx.rank == 0:
                gpu_snapshot = make_gpu_snapshot()
                for row in gpu_snapshot.get("gpus", []):
                    peak_memory[int(row["index"])] = max(
                        peak_memory.get(int(row["index"]), 0.0),
                        float(row["memory_used_mib"]),
                    )
                record = {
                    "record_type": "train",
                    "optimizer_step": step,
                    "scheduler_step": scheduler.last_epoch,
                    "micro_steps_per_optimizer_step": int(
                        raw["training"]["gradient_accumulation_steps"]
                    ),
                    "train_loss": mean_loss,
                    "smoothed_train_loss": sum(loss_window) / len(loss_window),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm_value,
                    "parameter_norm": math.sqrt(
                        sum(
                            float(param.detach().float().square().sum().item())
                            for param in base_model.parameters()
                        )
                    ),
                    "local_tokens_last_step": local_tokens,
                    "global_tokens_last_step": step_global_tokens,
                    "global_tokens": global_tokens,
                    "actual_non_padding_tokens_last_step": step_global_tokens,
                    "step_time_sec": step_time,
                    "data_loading_time_sec": data_time,
                    "tokens_per_sec": step_global_tokens / max(step_time, 1e-9),
                    "rank_step_hashes": gathered_hashes,
                    "rank0_cursor": compact_worker_state(dataset.state_dict()),
                    "gpu_snapshot": gpu_snapshot,
                }
                if checksum_report is not None:
                    record["rank_checksum"] = checksum_report
                append_jsonl(metrics_path, record)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", mean_loss, step)
                    tb_writer.add_scalar("train/smoothed_loss", sum(loss_window) / len(loss_window), step)
                    tb_writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], step)
                    tb_writer.add_scalar("train/grad_norm", grad_norm_value, step)
                    tb_writer.add_scalar("train/global_tokens", global_tokens, step)
                    tb_writer.add_scalar("train/tokens_per_sec", record["tokens_per_sec"], step)
                if initial_train_loss is None:
                    initial_train_loss = mean_loss
                final_train_loss = mean_loss

            snapshot_due = step in activation_steps
            lightweight_due = (
                step % int(raw["monitoring"]["lightweight_interval_steps"]) == 0
            )
            if (
                not args.skip_activation
                and dist_ctx.rank == 0
                and (snapshot_due or lightweight_due)
            ):
                snapshot = activation_snapshot(
                    base_model,
                    diagnostic_batch,
                    dist_ctx.device,
                    raw["training"]["precision"],
                    global_tokens,
                )
                alert = evaluate_activation_alerts(
                    snapshot,
                    previous_snapshot,
                    consecutive_growth_count,
                    raw,
                )
                consecutive_growth_count = alert["consecutive_growth_count"]
                snapshot["alert"] = alert
                append_activation_history(
                    args.output_dir / "activation_rms_history.csv",
                    snapshot,
                    alert,
                )
                alerts.append({"global_tokens": global_tokens, **alert})
                previous_snapshot = snapshot
                if snapshot_due:
                    target_tokens = activation_steps[step]
                    save_json(
                        activation_dir / f"tokens_{target_tokens:010d}.json",
                        snapshot,
                    )
                if alert["auto_stop_reasons"]:
                    auto_stop = True
                    auto_stop_reasons.extend(alert["auto_stop_reasons"])
            stop_flag = torch.tensor(
                [int(auto_stop or stop_requested)],
                dtype=torch.int32,
                device=dist_ctx.device,
            )
            if dist_ctx.enabled:
                # A signal may reach a non-zero rank first. Reduce the flag so
                # every rank exits through the same checkpoint path.
                dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
            if bool(stop_flag.item()):
                if not auto_stop:
                    stop_requested = True
                break

            if not args.skip_validation and (
                step in full_val_steps or step in proxy_val_steps
            ):
                if step in full_val_steps:
                    kind = "full"
                    limit = int(raw["validation"]["full_sequences"])
                else:
                    kind = "proxy"
                    limit = int(raw["validation"]["proxy_sequences"])
                validation_record = evaluate(
                    train_model,
                    raw,
                    dist_ctx,
                    limit,
                    kind,
                    global_tokens,
                )
                if dist_ctx.rank == 0:
                    append_jsonl(metrics_path, validation_record)
                    validation_records.append(validation_record)
                    if tb_writer is not None:
                        tb_writer.add_scalar(
                            f"validation/{kind}_loss",
                            validation_record["loss"],
                            step,
                        )
                        tb_writer.add_scalar(
                            f"validation/{kind}_ppl",
                            validation_record["ppl"],
                            step,
                        )

            if not auto_stop and step in checkpoint_steps:
                checkpoint_tokens = checkpoint_steps[step]
                periodic_result = save_checkpoint(
                    args.output_dir / f"checkpoint_{checkpoint_tokens // 1_000_000_000}b",
                    train_model,
                    optimizer,
                    scheduler,
                    scaler,
                    dataset,
                    dist_ctx,
                    step,
                    global_tokens,
                    last_data_hashes,
                    raw,
                )
                if dist_ctx.rank == 0:
                    periodic_checkpoint_records.append(periodic_result)
                    append_jsonl(
                        metrics_path,
                        {
                            "record_type": "checkpoint",
                            "optimizer_step": step,
                            "global_tokens": global_tokens,
                            "checkpoint": periodic_result,
                        },
                    )

        final_step = step
        if auto_stop:
            checkpoint_name = "checkpoint_emergency"
        elif stop_requested:
            checkpoint_name = "checkpoint_signal_interrupt"
        elif args.checkpoint_name:
            checkpoint_name = args.checkpoint_name
        elif stop_step < configured_stop_steps:
            checkpoint_name = "checkpoint_controlled_interrupt"
        else:
            checkpoint_name = "checkpoint_final"
        checkpoint_result = save_checkpoint(
            args.output_dir / checkpoint_name,
            train_model,
            optimizer,
            scheduler,
            scaler,
            dataset,
            dist_ctx,
            final_step,
            global_tokens,
            last_data_hashes,
            raw,
        )
        final_checksum = compare_checksum_across_ranks(
            checksum_model(base_model),
            dist_ctx.device,
        )
        if dist_ctx.rank == 0:
            summary = {
                "status": (
                    "auto_stopped"
                    if auto_stop
                    else ("graceful_interrupted" if stop_requested else "completed")
                ),
                "test_mode": args.test_mode,
                "from_scratch": args.resume_checkpoint is None,
                "resumed_from": str(args.resume_checkpoint)
                if args.resume_checkpoint is not None
                else None,
                "pretrained_weights_loaded": False,
                "stage_a_checkpoint_loaded": False,
                "start_optimizer_step": start_step,
                "final_optimizer_step": final_step,
                "scheduler_step": scheduler.last_epoch,
                "global_tokens": global_tokens,
                "initial_train_loss_this_process": initial_train_loss,
                "final_train_loss_this_process": final_train_loss,
                "final_learning_rate": scheduler.get_last_lr()[0],
                "average_tokens_per_sec_this_process": (
                    (final_step - start_step) * tokens_per_step / max(sum(step_times), 1e-9)
                ),
                "peak_device_memory_used_mib": peak_memory,
                "initial_rank_checksum": initial_checksum,
                "final_rank_checksum": final_checksum,
                "activation_alerts": alerts,
                "auto_stop": auto_stop,
                "auto_stop_reasons": sorted(set(auto_stop_reasons)),
                "stop_requested": stop_requested,
                "validation_records_this_process": validation_records,
                "checkpoint": checkpoint_result,
                "periodic_checkpoints": periodic_checkpoint_records,
                "tensorboard_dir": str(tensorboard_dir) if tb_writer is not None else None,
            }
            save_json(args.output_dir / "stageB_process_summary.json", summary)
            save_json(
                args.output_dir / "checkpoint_manifest.json",
                {
                    "latest_checkpoint": checkpoint_result,
                    "periodic_checkpoints": periodic_checkpoint_records,
                    "exact_dataset_resume": True,
                    "rank_specific_dataset_state": True,
                    "packing_remainder_saved": True,
                },
            )
            report_lines = [
                "# Stage B Process Report",
                "",
                f"- status: {summary['status']}",
                f"- start_optimizer_step: {start_step}",
                f"- final_optimizer_step: {final_step}",
                f"- global_tokens: {global_tokens}",
                f"- final_learning_rate: {summary['final_learning_rate']}",
                f"- checkpoint: {checkpoint_result['path']}",
                f"- auto_stop: {auto_stop}",
                f"- auto_stop_reasons: {summary['auto_stop_reasons']}",
            ]
            write_text(
                args.output_dir / "stageB_process_report.md",
                "\n".join(report_lines) + "\n",
            )
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
            tb_writer = None
        for signum, handler in previous_signal_handlers.items():
            signal.signal(signum, handler)
        distributed_barrier(dist_ctx)
        cleanup_distributed(dist_ctx)
    except Exception as exc:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        for signum, handler in locals().get("previous_signal_handlers", {}).items():
            signal.signal(signum, handler)
        if dist_ctx.rank == 0:
            save_json(
                args.output_dir / "failure_report.json",
                {
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        cleanup_distributed(dist_ctx)
        raise


if __name__ == "__main__":
    main()
