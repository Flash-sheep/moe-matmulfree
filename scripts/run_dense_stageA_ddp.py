#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling.data_utils import StreamingTextDataset
from scripts.run_sparse_upcycling import (
    cleanup_distributed,
    distributed_barrier,
    distributed_reduce_float,
    get_precision_dtype,
    precision_context,
    setup_distributed,
    unwrap_model,
)
from scripts.train_moe_lm import extract_optional_loss_metrics, json_default, set_seed


@dataclass
class StageAConfig:
    raw: Dict[str, object]
    experiment_name: str
    experiment_purpose: str
    base_config_path: str
    dataset_path: str
    validation_path: str
    dataset_manifest: str
    validation_manifest: str
    tokenizer: str
    sequence_length: int
    text_field: str
    world_size: int
    devices: List[int]
    micro_batch_size_per_rank: int
    grad_accum: int
    precision: str
    learning_rate: float
    min_lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    scheduler_total_steps: int
    max_eval_batches: int
    proxy_max_val_samples: int
    log_interval: int
    eval_interval: int
    save_interval: int
    target_stage_a_tokens: int
    short_test_optimizer_steps: int
    ddp_equivalence_optimizer_steps: int
    ddp_equivalence_grad_accum_steps: int
    ddp_equivalence_per_rank_batch_size: int
    sharding_check_batches: int
    sharding_check_batch_size: int
    distributed_eval_reference_batches: int
    checkpoint_resume_test_steps: int
    checkpoint_resume_save_step: int
    activation_snapshot_token_milestones: List[int]
    output_root_dir: str
    initialization_seed: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dense matmul-free scratch Stage A DDP diagnostics/training.")
    p.add_argument("--config-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--skip-stage-a-train", action="store_true")
    return p.parse_args()


def load_stagea_config(path: Path) -> StageAConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return StageAConfig(
        raw=raw,
        experiment_name=raw["experiment"]["name"],
        experiment_purpose=raw["experiment"]["purpose"],
        base_config_path=raw["model"]["base_config_path"],
        dataset_path=raw["data"]["path"],
        validation_path=raw["data"]["validation_path"],
        dataset_manifest=raw["data"]["dataset_manifest"],
        validation_manifest=raw["data"]["validation_manifest"],
        tokenizer=raw["data"]["tokenizer"],
        sequence_length=int(raw["data"]["sequence_length"]),
        text_field=raw["data"]["text_field"],
        world_size=int(raw["distributed"]["world_size"]),
        devices=[int(x) for x in raw["distributed"]["devices"]],
        micro_batch_size_per_rank=int(raw["training"]["micro_batch_size_per_rank"]),
        grad_accum=int(raw["training"]["gradient_accumulation_steps"]),
        precision=str(raw["training"]["precision"]),
        learning_rate=float(raw["training"]["learning_rate"]),
        min_lr=float(raw["training"]["min_lr"]),
        weight_decay=float(raw["training"]["weight_decay"]),
        warmup_steps=int(raw["training"]["warmup_steps"]),
        grad_clip=float(raw["training"]["grad_clip"]),
        scheduler_total_steps=int(raw["training"]["scheduler_total_steps"]),
        max_eval_batches=int(raw["training"]["max_eval_batches"]),
        proxy_max_val_samples=int(raw["training"]["proxy_max_val_samples"]),
        log_interval=int(raw["training"]["log_interval"]),
        eval_interval=int(raw["training"]["eval_interval"]),
        save_interval=int(raw["training"]["save_interval"]),
        target_stage_a_tokens=int(raw["training"]["target_stage_a_tokens"]),
        short_test_optimizer_steps=int(raw["training"]["short_test_optimizer_steps"]),
        ddp_equivalence_optimizer_steps=int(raw["training"]["ddp_equivalence_optimizer_steps"]),
        ddp_equivalence_grad_accum_steps=int(raw["training"]["ddp_equivalence_grad_accum_steps"]),
        ddp_equivalence_per_rank_batch_size=int(raw["training"]["ddp_equivalence_per_rank_batch_size"]),
        sharding_check_batches=int(raw["diagnostics"]["sharding_check_batches"]),
        sharding_check_batch_size=int(raw["diagnostics"]["sharding_check_batch_size"]),
        distributed_eval_reference_batches=int(raw["diagnostics"]["distributed_eval_reference_batches"]),
        checkpoint_resume_test_steps=int(raw["diagnostics"]["checkpoint_resume_test_steps"]),
        checkpoint_resume_save_step=int(raw["diagnostics"]["checkpoint_resume_save_step"]),
        activation_snapshot_token_milestones=[int(x) for x in raw["diagnostics"]["activation_snapshot_token_milestones"]],
        output_root_dir=str(raw["output"]["root_dir"]),
        initialization_seed=int(raw["model"]["initialization_seed"]),
    )


def make_output_dir(cfg: StageAConfig, requested: Optional[Path]) -> Path:
    if requested is not None:
        if requested.exists():
            raise SystemExit(f"Refusing to overwrite existing output dir: {requested}")
        requested.mkdir(parents=True, exist_ok=False)
        return requested
    root = Path(cfg.output_root_dir)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = root / f"{cfg.experiment_name}_{stamp}"
    if candidate.exists():
        raise SystemExit(f"Unexpected existing output dir: {candidate}")
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def resolve_output_dir(cfg: StageAConfig, requested: Optional[Path], dist_ctx) -> Path:
    if dist_ctx.rank == 0:
        output_dir = make_output_dir(cfg, requested)
        payload = [str(output_dir)]
    else:
        payload = [None]
    if dist.is_initialized():
        dist.broadcast_object_list(payload, src=0)
    return Path(payload[0])


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def mark_stage(output_dir: Path, rank: int, stage: str, status: str, extra: Optional[Dict[str, object]] = None) -> None:
    if rank != 0:
        return
    payload: Dict[str, object] = {
        "stage": stage,
        "status": status,
        "timestamp": time.time(),
    }
    if extra:
        payload.update(extra)
    save_json(output_dir / "stage_progress.json", payload)


def collate_streaming_batch(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}


def build_loader(
    data_source: str,
    tokenizer_path: str,
    max_length: int,
    batch_size: int,
    text_field: str,
    split: str,
    max_samples: Optional[int],
    shard_rank: int,
    shard_world_size: int,
) -> Tuple[DataLoader, Dict[str, object], StreamingTextDataset]:
    dataset = StreamingTextDataset(
        data_source=data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        split=split,
        text_field=text_field,
        max_samples=max_samples,
        shard_rank=shard_rank,
        shard_world_size=shard_world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_streaming_batch,
    )
    return loader, dataset.get_manifest(), dataset


def repeat_dataloader(dataloader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in dataloader:
            yield batch


def batch_hashes(batch: Dict[str, torch.Tensor], prefix_tokens: int = 64) -> List[str]:
    hashes: List[str] = []
    input_ids = batch["input_ids"].cpu()
    for row in input_ids:
        prefix = row[:prefix_tokens].tolist()
        digest = hashlib.sha1(json.dumps(prefix).encode("utf-8")).hexdigest()
        hashes.append(digest)
    return hashes


def collect_fixed_batches(loader: DataLoader, num_batches: int) -> List[Dict[str, torch.Tensor]]:
    batches: List[Dict[str, torch.Tensor]] = []
    iterator = iter(loader)
    for _ in range(num_batches):
        batch = next(iterator)
        batches.append({key: value.clone() for key, value in batch.items()})
    return batches


def split_global_batches_for_rank(
    global_batches: Sequence[Dict[str, torch.Tensor]],
    per_rank_batch_size: int,
    rank: int,
    world_size: int,
) -> List[Dict[str, torch.Tensor]]:
    split_batches: List[Dict[str, torch.Tensor]] = []
    for batch in global_batches:
        start = rank * per_rank_batch_size
        end = start + per_rank_batch_size
        split_batches.append(
            {
                "input_ids": batch["input_ids"][start:end].clone(),
                "labels": batch["labels"][start:end].clone(),
            }
        )
    return split_batches


def build_scratch_model(base_config_path: str, seed: int, device: torch.device) -> HGRNBitForCausalLM:
    cfg = HGRNBitConfig.from_pretrained(base_config_path)
    cfg.use_moe = False
    cfg.moe_layer_indices = []
    cfg.use_cache = False
    set_seed(seed)
    model = HGRNBitForCausalLM(cfg)
    model.to(device)
    model.train()
    return model


def parameter_stats(tensor: torch.Tensor) -> Dict[str, object]:
    data = tensor.detach().float().cpu()
    finite = torch.isfinite(data)
    finite_data = data[finite]
    if finite_data.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "rms": None,
            "l2_norm": None,
            "min": None,
            "max": None,
            "zero_ratio": None,
            "non_finite_count": int((~finite).sum().item()),
        }
    return {
        "mean": float(finite_data.mean().item()),
        "std": float(finite_data.std(unbiased=False).item()),
        "rms": float(torch.sqrt((finite_data.square().mean())).item()),
        "l2_norm": float(torch.linalg.vector_norm(finite_data).item()),
        "min": float(finite_data.min().item()),
        "max": float(finite_data.max().item()),
        "zero_ratio": float((finite_data == 0).float().mean().item()),
        "non_finite_count": int((~finite).sum().item()),
    }


def checksum_model(model: HGRNBitForCausalLM) -> Dict[str, float]:
    payload = {"sum": 0.0, "abs_sum": 0.0, "sq_sum": 0.0}
    for param in model.parameters():
        data = param.detach().float()
        payload["sum"] += float(data.sum().item())
        payload["abs_sum"] += float(data.abs().sum().item())
        payload["sq_sum"] += float(data.square().sum().item())
    return payload


def compare_checksum_across_ranks(local_checksum: Dict[str, float], device: torch.device) -> Dict[str, object]:
    keys = ["sum", "abs_sum", "sq_sum"]
    tensor = torch.tensor([[float(local_checksum[k]) for k in keys]], device=device, dtype=torch.float64)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)] if world_size > 1 else [tensor.clone()]
    if world_size > 1:
        dist.all_gather(gathered, tensor)
    rows = [[float(x) for x in item.view(-1).tolist()] for item in gathered]
    max_abs_diff = 0.0
    for i in range(1, len(rows)):
        for j in range(len(keys)):
            max_abs_diff = max(max_abs_diff, abs(rows[i][j] - rows[0][j]))
    return {"per_rank": [{k: row[idx] for idx, k in enumerate(keys)} for row in rows], "max_abs_diff": max_abs_diff}


def tied_parameter_report(model: HGRNBitForCausalLM) -> Dict[str, object]:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    embeddings = input_embeddings.weight if input_embeddings is not None and hasattr(input_embeddings, "weight") else None
    lm_head = output_embeddings.weight if output_embeddings is not None and hasattr(output_embeddings, "weight") else None
    return {
        "config_tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
        "embedding_trainable": bool(embeddings.requires_grad) if embeddings is not None else None,
        "lm_head_trainable": bool(lm_head.requires_grad) if lm_head is not None else None,
        "same_storage": embeddings.data_ptr() == lm_head.data_ptr() if embeddings is not None and lm_head is not None else False,
        "same_object": embeddings is lm_head if embeddings is not None and lm_head is not None else False,
    }


def collect_initialization_diagnostics(
    model: HGRNBitForCausalLM,
    device: torch.device,
    dist_enabled: bool,
    rank: int,
    output_dir: Path,
    seed: int,
    save_output: bool = True,
) -> Dict[str, object]:
    base_model = unwrap_model(model)
    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    param_records: Dict[str, object] = {}
    for name, param in base_model.named_parameters():
        stats = parameter_stats(param)
        stats.update({"shape": list(param.shape), "dtype": str(param.dtype), "device": str(param.device), "trainable": bool(param.requires_grad)})
        param_records[name] = stats
    local_checksum = checksum_model(base_model)
    checksum_report = compare_checksum_across_ranks(local_checksum, device) if dist_enabled else {"per_rank": [local_checksum], "max_abs_diff": 0.0}
    payload = {
        "model_created_from_config": True,
        "pretrained_weights_loaded": False,
        "checkpoint_loaded": False,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "initialization_seed": seed,
        "total_parameter_count": total_params,
        "unique_parameter_count": total_params,
        "trainable_parameter_count": trainable_params,
        "frozen_parameter_count": frozen_params,
        "trainable_ratio": trainable_params / max(total_params, 1),
        "tied_parameter_relationships": tied_parameter_report(base_model),
        "parameter_checksums": checksum_report,
        "parameter_stats": param_records,
        "rank_gpu_binding": {
            "rank": rank,
            "device": str(device),
            "cuda_device_index": int(device.index) if device.index is not None else None,
        },
    }
    if save_output:
        save_json(output_dir / "initialization_diagnostics.json", payload)
    return payload


def rms_of(tensor: torch.Tensor) -> float:
    data = tensor.detach().float()
    return float(torch.sqrt(data.square().mean()).item())


def register_activation_hooks(model: HGRNBitForCausalLM):
    rows: Dict[int, Dict[str, object]] = {}
    handles = []
    base_model = unwrap_model(model)
    for layer_idx, block in enumerate(base_model.model.layers):
        rows[layer_idx] = {"layer_idx": layer_idx}

        def block_pre_hook(module, inputs, layer_idx=layer_idx):
            hidden = inputs[0]
            rows[layer_idx]["layer_input_rms"] = rms_of(hidden)

        def attn_hook(module, inputs, output, layer_idx=layer_idx):
            attn_out = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["token_mixer_output_rms"] = rms_of(attn_out)

        def mlp_pre_hook(module, inputs, layer_idx=layer_idx):
            hidden = inputs[0]
            rows[layer_idx]["ffn_input_rms"] = rms_of(hidden)

        def mlp_hook(module, inputs, output, layer_idx=layer_idx):
            rows[layer_idx]["ffn_output_rms"] = rms_of(output)

        def block_hook(module, inputs, output, layer_idx=layer_idx):
            hidden = output[0] if isinstance(output, tuple) else output
            rows[layer_idx]["layer_output_rms"] = rms_of(hidden)

        handles.append(block.register_forward_pre_hook(block_pre_hook))
        handles.append(block.attn.register_forward_hook(attn_hook))
        handles.append(block.mlp.register_forward_pre_hook(mlp_pre_hook))
        handles.append(block.mlp.register_forward_hook(mlp_hook))
        handles.append(block.register_forward_hook(block_hook))
    return rows, handles


@torch.no_grad()
def collect_activation_diagnostics(
    model: HGRNBitForCausalLM,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    output_json: Path,
    output_csv: Path,
) -> Dict[str, object]:
    rows, handles = register_activation_hooks(model)
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    with precision_context(precision_dtype):
        outputs = base_model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
    for handle in handles:
        handle.remove()
    for layer_idx, row in rows.items():
        row["ffn_output_to_layer_input_rms_ratio"] = row.get("ffn_output_rms", 0.0) / max(row.get("layer_input_rms", 1e-12), 1e-12)
        row["residual_output_rms"] = row.get("layer_output_rms")
    csv_rows = [rows[idx] for idx in sorted(rows)]
    payload = {
        "input_shape": list(input_ids.shape),
        "label_shape": list(labels.shape),
        "attention_mask_shape": None,
        "non_padding_token_count": int((input_ids != 0).sum().item()),
        "initial_loss": float(outputs.loss.detach().cpu()),
        "logits_mean": float(outputs.logits.detach().float().mean().item()),
        "logits_std": float(outputs.logits.detach().float().std(unbiased=False).item()),
        "logits_rms": rms_of(outputs.logits),
        "logits_max_abs": float(outputs.logits.detach().float().abs().max().item()),
        "hidden_state_mean": float(outputs.logits.detach().float().mean().item()),
        "hidden_state_std": float(outputs.logits.detach().float().std(unbiased=False).item()),
        "has_nan": bool(torch.isnan(outputs.logits.detach()).any().item() or torch.isnan(outputs.loss.detach()).any().item()),
        "has_inf": bool(torch.isinf(outputs.logits.detach()).any().item() or torch.isinf(outputs.loss.detach()).any().item()),
        "by_layer": csv_rows,
    }
    save_json(output_json, payload)
    save_csv(output_csv, csv_rows)
    if was_training:
        base_model.train()
    return payload


def collect_gradient_diagnostics(
    model: HGRNBitForCausalLM,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    precision: str,
    output_json: Path,
) -> Dict[str, object]:
    base_model = unwrap_model(model)
    base_model.train()
    for param in base_model.parameters():
        if param.grad is not None:
            param.grad = None
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    with precision_context(precision_dtype):
        outputs = base_model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
    outputs.loss.backward()
    grad_stats: Dict[str, object] = {}
    missing: List[str] = []
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
            grad_stats[name] = {"gradient_exists": False}
            continue
        stats = parameter_stats(param.grad)
        stats.update({"gradient_exists": True, "shape": list(param.grad.shape)})
        grad_stats[name] = stats
    payload = {
        "loss": float(outputs.loss.detach().cpu()),
        "missing_grad_parameters": missing,
        "parameter_grad_stats": grad_stats,
    }
    save_json(output_json, payload)
    for param in base_model.parameters():
        if param.grad is not None:
            param.grad = None
    return payload


def clone_state_dict_cpu(model: HGRNBitForCausalLM) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in unwrap_model(model).state_dict().items()}


def load_state_dict_cpu(model: HGRNBitForCausalLM, state: Dict[str, torch.Tensor]) -> None:
    unwrap_model(model).load_state_dict(state)


def build_optimizer_and_scheduler(model: HGRNBitForCausalLM, cfg: StageAConfig):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    min_lr_ratio = cfg.min_lr / cfg.learning_rate

    def lr_lambda(step: int):
        if step < cfg.warmup_steps:
            return step / max(cfg.warmup_steps, 1)
        progress = min(max((step - cfg.warmup_steps) / max(cfg.scheduler_total_steps - cfg.warmup_steps, 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def run_train_step_fixed_batches(
    model,
    optimizer,
    scheduler,
    scaler,
    batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
    grad_clip: float,
    ddp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for idx, batch in enumerate(batches):
        sync_ctx = nullcontext()
        if ddp_enabled and isinstance(model, DDP) and idx < len(batches) - 1:
            sync_ctx = model.no_sync()
        with sync_ctx:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with precision_context(precision_dtype):
                outputs = model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
                loss = outputs.loss / len(batches)
            scaler.scale(loss).backward()
            total_loss += float(outputs.loss.detach().cpu())
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    return total_loss / max(len(batches), 1), float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)


def compare_models(ref_state: Dict[str, torch.Tensor], test_state: Dict[str, torch.Tensor]) -> Dict[str, object]:
    max_abs = 0.0
    mean_abs_sum = 0.0
    total_count = 0
    worst_key = None
    for key in ref_state:
        ref = ref_state[key].float()
        test = test_state[key].float()
        diff = (ref - test).abs()
        current_max = float(diff.max().item())
        if current_max > max_abs:
            max_abs = current_max
            worst_key = key
        mean_abs_sum += float(diff.sum().item())
        total_count += diff.numel()
    return {
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs_sum / max(total_count, 1),
        "worst_parameter": worst_key,
    }


def run_ddp_equivalence_test(
    cfg: StageAConfig,
    dist_ctx,
    output_dir: Path,
) -> Dict[str, object]:
    world_size = dist_ctx.world_size
    rank = dist_ctx.rank
    per_rank_bs = cfg.ddp_equivalence_per_rank_batch_size
    global_bs = per_rank_bs * world_size
    global_batches_needed = cfg.ddp_equivalence_grad_accum_steps
    global_loader, _, _ = build_loader(
        data_source=cfg.dataset_path,
        tokenizer_path=cfg.tokenizer,
        max_length=cfg.sequence_length,
        batch_size=global_bs,
        text_field=cfg.text_field,
        split="train",
        max_samples=None,
        shard_rank=0,
        shard_world_size=1,
    )
    global_batches = collect_fixed_batches(global_loader, global_batches_needed) if rank == 0 else None
    obj_list = [global_batches] if rank == 0 else [None]
    if dist.is_initialized():
        dist.broadcast_object_list(obj_list, src=0)
    global_batches = obj_list[0]
    rank_batches = split_global_batches_for_rank(global_batches, per_rank_bs, rank, world_size)
    base_model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
    initial_state = clone_state_dict_cpu(base_model)

    ref_metrics = None
    ref_post_state = None
    if rank == 0:
        ref_model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
        load_state_dict_cpu(ref_model, initial_state)
        ref_optimizer, ref_scheduler = build_optimizer_and_scheduler(ref_model, cfg)
        _, ref_scaler = get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())
        ref_loss, ref_grad_norm = run_train_step_fixed_batches(
            ref_model,
            ref_optimizer,
            ref_scheduler,
            ref_scaler,
            global_batches,
            dist_ctx.device,
            cfg.precision,
            cfg.grad_clip,
            ddp_enabled=False,
        )
        ref_post_state = clone_state_dict_cpu(ref_model)
        ref_metrics = {
            "single_card_loss": ref_loss,
            "single_card_grad_norm": ref_grad_norm,
            "single_card_optimizer_step": ref_optimizer.state.get(next(iter(ref_optimizer.param_groups[0]["params"])), {}).get("step", 1),
            "single_card_scheduler_last_epoch": ref_scheduler.last_epoch,
        }

    ddp_model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
    load_state_dict_cpu(ddp_model, initial_state)
    ddp_optimizer, ddp_scheduler = build_optimizer_and_scheduler(ddp_model, cfg)
    _, ddp_scaler = get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())
    ddp_model = DDP(ddp_model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, find_unused_parameters=False)
    ddp_loss, ddp_grad_norm_local = run_train_step_fixed_batches(
        ddp_model,
        ddp_optimizer,
        ddp_scheduler,
        ddp_scaler,
        rank_batches,
        dist_ctx.device,
        cfg.precision,
        cfg.grad_clip,
        ddp_enabled=True,
    )
    ddp_loss = distributed_reduce_float(ddp_loss, dist_ctx, average=True)
    ddp_grad_norm = distributed_reduce_float(ddp_grad_norm_local, dist_ctx, average=True)
    rank_checksum_consistency = compare_checksum_across_ranks(checksum_model(ddp_model.module), dist_ctx.device)
    report = {}
    if rank == 0:
        ddp_post_state = clone_state_dict_cpu(ddp_model.module)
        ref_vs_ddp = compare_models(ref_post_state, ddp_post_state)
        report = {
            "world_size": world_size,
            "grad_accumulation_steps": cfg.ddp_equivalence_grad_accum_steps,
            "per_rank_batch_size": per_rank_bs,
            "global_batch_size_per_microstep": global_bs,
            "single_card_loss": ref_metrics["single_card_loss"],
            "ddp_loss": ddp_loss,
            "loss_abs_difference": abs(ref_metrics["single_card_loss"] - ddp_loss),
            "single_card_grad_norm": ref_metrics["single_card_grad_norm"],
            "ddp_grad_norm": ddp_grad_norm,
            "gradient_norm_abs_difference": abs(ref_metrics["single_card_grad_norm"] - ddp_grad_norm),
            "single_card_scheduler_last_epoch": ref_metrics["single_card_scheduler_last_epoch"],
            "ddp_scheduler_last_epoch": ddp_scheduler.last_epoch,
            "parameter_difference_after_step": ref_vs_ddp,
            "rank_checksum_consistency_after_step": rank_checksum_consistency,
            "tolerance": {"max_abs_difference": 5e-3, "mean_abs_difference": 1e-4, "loss_abs_difference": 5e-4},
            "pass": (
                ref_vs_ddp["max_abs_difference"] <= 5e-3
                and ref_vs_ddp["mean_abs_difference"] <= 1e-4
                and abs(ref_metrics["single_card_loss"] - ddp_loss) <= 5e-4
                and rank_checksum_consistency["max_abs_diff"] <= 1e-8
            ),
        }
        save_json(output_dir / "ddp_equivalence_report.json", report)
        md = [
            "# DDP Equivalence Report",
            "",
            f"- world_size: {world_size}",
            f"- single_card_loss: {ref_metrics['single_card_loss']:.6f}",
            f"- ddp_loss: {ddp_loss:.6f}",
            f"- loss_abs_difference: {abs(ref_metrics['single_card_loss'] - ddp_loss):.6e}",
            f"- max_abs_parameter_difference: {ref_vs_ddp['max_abs_difference']:.6e}",
            f"- mean_abs_parameter_difference: {ref_vs_ddp['mean_abs_difference']:.6e}",
            f"- pass: {report['pass']}",
        ]
        write_text(output_dir / "ddp_equivalence_report.md", "\n".join(md) + "\n")
    distributed_barrier(dist_ctx)
    return report


def run_data_sharding_report(cfg: StageAConfig, dist_ctx, output_dir: Path) -> Dict[str, object]:
    loader, _, _ = build_loader(
        data_source=cfg.dataset_path,
        tokenizer_path=cfg.tokenizer,
        max_length=cfg.sequence_length,
        batch_size=cfg.sharding_check_batch_size,
        text_field=cfg.text_field,
        split="train",
        max_samples=None,
        shard_rank=dist_ctx.rank,
        shard_world_size=dist_ctx.world_size,
    )
    iterator = iter(loader)
    local_hashes: List[str] = []
    for _ in range(cfg.sharding_check_batches):
        batch = next(iterator)
        local_hashes.extend(batch_hashes(batch))
    gathered = [None for _ in range(dist_ctx.world_size)] if dist_ctx.rank == 0 else None
    dist.gather_object(local_hashes, gathered, dst=0)
    report = {}
    if dist_ctx.rank == 0:
        rank_sets = [set(lst) for lst in gathered]
        overlaps = {}
        for i in range(len(rank_sets)):
            for j in range(i + 1, len(rank_sets)):
                overlaps[f"{i}_{j}"] = len(rank_sets[i].intersection(rank_sets[j]))
        report = {
            "world_size": dist_ctx.world_size,
            "batches_per_rank_checked": cfg.sharding_check_batches,
            "samples_per_rank_checked": [len(lst) for lst in gathered],
            "intra_rank_duplicates": [len(lst) - len(set(lst)) for lst in gathered],
            "inter_rank_intersections": overlaps,
            "distinct_data_across_ranks": all(v == 0 for v in overlaps.values()),
        }
        save_json(output_dir / "data_sharding_report.json", report)
    distributed_barrier(dist_ctx)
    return report


def evaluate_fixed_batches(
    model,
    batches: Sequence[Dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> Dict[str, float]:
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    total_loss = 0.0
    total_sequences = 0
    total_tokens = 0
    for batch in batches:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with precision_context(precision_dtype):
            outputs = base_model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
        total_loss += float(outputs.loss.detach().cpu()) * int(input_ids.shape[0])
        total_sequences += int(input_ids.shape[0])
        total_tokens += int(input_ids.numel())
    avg_loss = total_loss / max(total_sequences, 1)
    if was_training:
        base_model.train()
    return {"loss": avg_loss, "ppl": math.exp(min(avg_loss, 20.0)), "sequences": total_sequences, "tokens": total_tokens}


def run_distributed_eval_report(cfg: StageAConfig, dist_ctx, output_dir: Path) -> Dict[str, object]:
    global_bs = cfg.micro_batch_size_per_rank * dist_ctx.world_size
    global_loader, _, _ = build_loader(
        data_source=cfg.validation_path,
        tokenizer_path=cfg.tokenizer,
        max_length=cfg.sequence_length,
        batch_size=global_bs,
        text_field=cfg.text_field,
        split="validation",
        max_samples=None,
        shard_rank=0,
        shard_world_size=1,
    )
    fixed_global_batches = collect_fixed_batches(global_loader, cfg.distributed_eval_reference_batches) if dist_ctx.rank == 0 else None
    obj_list = [fixed_global_batches] if dist_ctx.rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    fixed_global_batches = obj_list[0]
    rank_batches = split_global_batches_for_rank(fixed_global_batches, cfg.micro_batch_size_per_rank, dist_ctx.rank, dist_ctx.world_size)
    model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
    ref = evaluate_fixed_batches(model, fixed_global_batches, dist_ctx.device, cfg.precision) if dist_ctx.rank == 0 else None
    local = evaluate_fixed_batches(model, rank_batches, dist_ctx.device, cfg.precision)
    loss_x_seq = distributed_reduce_float(local["loss"] * local["sequences"], dist_ctx, average=False)
    seqs = distributed_reduce_float(float(local["sequences"]), dist_ctx, average=False)
    tokens = distributed_reduce_float(float(local["tokens"]), dist_ctx, average=False)
    report = {}
    if dist_ctx.rank == 0:
        ddp_loss = loss_x_seq / max(seqs, 1.0)
        report = {
            "reference_batches": cfg.distributed_eval_reference_batches,
            "single_card_reference": ref,
            "ddp_aggregated": {"loss": ddp_loss, "ppl": math.exp(min(ddp_loss, 20.0)), "sequences": int(seqs), "tokens": int(tokens)},
            "loss_abs_difference": abs(ref["loss"] - ddp_loss),
            "pass": abs(ref["loss"] - ddp_loss) <= 5e-4,
        }
        save_json(output_dir / "distributed_eval_report.json", report)
    distributed_barrier(dist_ctx)
    return report


def run_short_training_smoke(
    cfg: StageAConfig,
    dist_ctx,
    output_dir: Path,
    train_loader: DataLoader,
    proxy_val_loader: Optional[DataLoader],
) -> Dict[str, object]:
    model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
    ddp_model = DDP(model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, find_unused_parameters=False)
    optimizer, scheduler = build_optimizer_and_scheduler(ddp_model.module, cfg)
    _, scaler = get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())
    train_iter = repeat_dataloader(train_loader)
    metrics: List[Dict[str, object]] = []
    rank_checksum_steps = {0, 1, 5, 10, cfg.short_test_optimizer_steps}
    initial_checksums = compare_checksum_across_ranks(checksum_model(ddp_model.module), dist_ctx.device)
    for step in range(1, cfg.short_test_optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: List[float] = []
        local_tokens = 0
        for micro_idx in range(cfg.grad_accum):
            batch = next(train_iter)
            local_tokens += int(batch["input_ids"].numel())
            sync_ctx = ddp_model.no_sync() if micro_idx < cfg.grad_accum - 1 else nullcontext()
            with sync_ctx:
                input_ids = batch["input_ids"].to(dist_ctx.device)
                labels = batch["labels"].to(dist_ctx.device)
                with precision_context(get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())[0]):
                    outputs = ddp_model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
                    loss = outputs.loss / cfg.grad_accum
                scaler.scale(loss).backward()
                micro_losses.append(float(outputs.loss.detach().cpu()))
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        mean_loss = distributed_reduce_float(sum(micro_losses) / max(len(micro_losses), 1), dist_ctx, average=True)
        grad_norm_value = distributed_reduce_float(float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm), dist_ctx, average=True)
        global_tokens = distributed_reduce_float(float(local_tokens), dist_ctx, average=False)
        step_checksum = compare_checksum_across_ranks(checksum_model(ddp_model.module), dist_ctx.device) if step in rank_checksum_steps else None
        if dist_ctx.rank == 0:
            record = {
                "optimizer_step": step,
                "scheduler_step": scheduler.last_epoch,
                "train_loss": mean_loss,
                "grad_norm": grad_norm_value,
                "global_tokens_seen_step": int(global_tokens),
                "has_non_finite_loss": not math.isfinite(mean_loss),
                "has_non_finite_grad_norm": not math.isfinite(grad_norm_value),
            }
            if step_checksum is not None:
                record["rank_checksum"] = step_checksum
            metrics.append(record)
    report: Dict[str, object] = {}
    if dist_ctx.rank == 0:
        activation_batch = next(iter(proxy_val_loader)) if proxy_val_loader is not None else next(iter(train_loader))
        collect_activation_diagnostics(ddp_model.module, activation_batch, dist_ctx.device, cfg.precision, output_dir / "activation_diagnostics_after_short_test.json", output_dir / "activation_by_layer_after_short_test.csv")
        collect_gradient_diagnostics(ddp_model.module, activation_batch, dist_ctx.device, cfg.precision, output_dir / "gradient_diagnostics_after_short_test.json")
        proxy_metrics = proxy_evaluate(ddp_model, cfg, dist_ctx, proxy_val_loader) if proxy_val_loader is not None else {}
        report = {
            "initial_rank_checksums": initial_checksums,
            "steps": metrics,
            "final_rank_checksums": None,
            "proxy_validation": proxy_metrics,
            "pass": all(
                math.isfinite(float(item["train_loss"])) and math.isfinite(float(item["grad_norm"]))
                for item in metrics
            ),
        }
    final_rank_checksums = compare_checksum_across_ranks(checksum_model(ddp_model.module), dist_ctx.device)
    if dist_ctx.rank == 0:
        report["final_rank_checksums"] = final_rank_checksums
        save_json(output_dir / "short_training_report.json", report)
    distributed_barrier(dist_ctx)
    return report


def make_gpu_snapshot() -> Dict[str, object]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.reserved,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        rows = []
        for line in out.strip().splitlines():
            idx, util, mem_used, mem_reserved, mem_total = [x.strip() for x in line.split(",")]
            rows.append(
                {
                    "index": int(idx),
                    "utilization_gpu_pct": float(util),
                    "memory_used_mib": float(mem_used),
                    "memory_reserved_mib": float(mem_reserved),
                    "memory_total_mib": float(mem_total),
                }
            )
        return {"success": True, "gpus": rows}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, step: int, global_tokens: int, rng_state: Dict[str, object]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    config_path = path / "config.json"
    config_path.write_text(
        unwrap_model(model).config.to_json_string(use_diff=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "global_tokens": global_tokens,
            "rng_state": rng_state,
        },
        path / "trainer_state.pt",
    )


def coordinated_rank0_checkpoint_save(
    dist_ctx,
    marker_root: Path,
    checkpoint_name: str,
    save_fn,
    poll_interval_sec: float = 1.0,
    timeout_sec: float = 7200.0,
) -> None:
    marker_root.mkdir(parents=True, exist_ok=True)
    ready_marker = marker_root / f".{checkpoint_name}.ready"
    error_marker = marker_root / f".{checkpoint_name}.error"
    if dist_ctx.rank == 0:
        ready_marker.unlink(missing_ok=True)
        error_marker.unlink(missing_ok=True)
        try:
            save_fn()
            ready_marker.write_text("ok\n", encoding="utf-8")
        except Exception as exc:
            error_marker.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            raise
    else:
        start = time.time()
        while True:
            if error_marker.exists():
                message = error_marker.read_text(encoding="utf-8").strip()
                raise RuntimeError(f"rank0 checkpoint save failed for {checkpoint_name}: {message}")
            if ready_marker.exists():
                break
            if time.time() - start > timeout_sec:
                raise TimeoutError(f"timed out waiting for rank0 checkpoint save: {checkpoint_name}")
            time.sleep(poll_interval_sec)
    distributed_barrier(dist_ctx)
    if dist_ctx.rank == 0:
        ready_marker.unlink(missing_ok=True)
        error_marker.unlink(missing_ok=True)
    distributed_barrier(dist_ctx)


def load_checkpoint(path: Path, model, optimizer, scheduler, scaler, device: torch.device) -> Dict[str, object]:
    state = torch.load(path / "trainer_state.pt", map_location=device, weights_only=False)
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return state


def current_rng_state() -> Dict[str, object]:
    return {
        "torch_cpu": torch.get_rng_state().tolist(),
        "torch_cuda": [state.tolist() for state in torch.cuda.get_rng_state_all()],
    }


def restore_rng_state(payload: Dict[str, object]) -> None:
    torch.set_rng_state(torch.tensor(payload["torch_cpu"], dtype=torch.uint8))
    cuda_states = [torch.tensor(state, dtype=torch.uint8) for state in payload["torch_cuda"]]
    torch.cuda.set_rng_state_all(cuda_states)


def run_checkpoint_resume_report(cfg: StageAConfig, dist_ctx, output_dir: Path) -> Dict[str, object]:
    global_bs = cfg.micro_batch_size_per_rank * dist_ctx.world_size
    global_loader, _, _ = build_loader(
        data_source=cfg.dataset_path,
        tokenizer_path=cfg.tokenizer,
        max_length=cfg.sequence_length,
        batch_size=global_bs,
        text_field=cfg.text_field,
        split="train",
        max_samples=None,
        shard_rank=0,
        shard_world_size=1,
    )
    global_batches = collect_fixed_batches(global_loader, cfg.checkpoint_resume_test_steps) if dist_ctx.rank == 0 else None
    obj_list = [global_batches] if dist_ctx.rank == 0 else [None]
    dist.broadcast_object_list(obj_list, src=0)
    global_batches = obj_list[0]
    rank_batches = split_global_batches_for_rank(global_batches, cfg.micro_batch_size_per_rank, dist_ctx.rank, dist_ctx.world_size)

    def build_bundle():
        model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
        optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
        _, scaler = get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())
        ddp = DDP(model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, find_unused_parameters=False)
        return ddp, optimizer, scheduler, scaler

    # uninterrupted control
    control_model, control_opt, control_sched, control_scaler = build_bundle()
    control_rng = current_rng_state()
    for step_idx in range(cfg.checkpoint_resume_test_steps):
        run_train_step_fixed_batches(
            control_model,
            control_opt,
            control_sched,
            control_scaler,
            [rank_batches[step_idx]],
            dist_ctx.device,
            cfg.precision,
            cfg.grad_clip,
            ddp_enabled=True,
        )
    control_state = clone_state_dict_cpu(control_model.module) if dist_ctx.rank == 0 else None

    # save and resume
    restore_rng_state(control_rng)
    resume_model, resume_opt, resume_sched, resume_scaler = build_bundle()
    if dist_ctx.rank == 0:
        save_path_obj: Optional[str] = tempfile.mkdtemp(prefix="dense_stageA_resume_", dir="/tmp")
    else:
        save_path_obj = None
    if dist.is_initialized():
        payload = [save_path_obj]
        dist.broadcast_object_list(payload, src=0)
        save_path_obj = payload[0]
    save_path = Path(save_path_obj)
    for step_idx in range(cfg.checkpoint_resume_save_step):
        run_train_step_fixed_batches(
            resume_model,
            resume_opt,
            resume_sched,
            resume_scaler,
            [rank_batches[step_idx]],
            dist_ctx.device,
            cfg.precision,
            cfg.grad_clip,
            ddp_enabled=True,
        )
    try:
        if dist_ctx.rank == 0:
            save_checkpoint(
                save_path,
                resume_model,
                resume_opt,
                resume_sched,
                resume_scaler,
                cfg.checkpoint_resume_save_step,
                cfg.checkpoint_resume_save_step
                * cfg.micro_batch_size_per_rank
                * cfg.sequence_length
                * dist_ctx.world_size,
                current_rng_state(),
            )
        distributed_barrier(dist_ctx)
        post_model, post_opt, post_sched, post_scaler = build_bundle()
        state = load_checkpoint(save_path, post_model, post_opt, post_sched, post_scaler, dist_ctx.device)
        restore_rng_state(state["rng_state"])
        for step_idx in range(cfg.checkpoint_resume_save_step, cfg.checkpoint_resume_test_steps):
            run_train_step_fixed_batches(
                post_model,
                post_opt,
                post_sched,
                post_scaler,
                [rank_batches[step_idx]],
                dist_ctx.device,
                cfg.precision,
                cfg.grad_clip,
                ddp_enabled=True,
            )
        rank_checksum_consistency = compare_checksum_across_ranks(checksum_model(post_model.module), dist_ctx.device)
        report = {}
        if dist_ctx.rank == 0:
            resumed_state = clone_state_dict_cpu(post_model.module)
            control_vs_resumed = compare_models(control_state, resumed_state)
            report = {
                "steps_total": cfg.checkpoint_resume_test_steps,
                "save_step": cfg.checkpoint_resume_save_step,
                "checkpoint_path": str(save_path),
                "restored_step": int(state["step"]),
                "restored_global_tokens": int(state["global_tokens"]),
                "scheduler_last_epoch_after_load": post_sched.last_epoch,
                "optimizer_state_loaded": True,
                "scheduler_state_loaded": True,
                "rng_state_loaded": True,
                "parameter_difference_vs_uninterrupted": control_vs_resumed,
                "rank_checksum_consistency_after_resume": rank_checksum_consistency,
                "dataset_progress_recovery": "not_exact_iterable_cursor_resume; fixed-batch resume test only",
                "pass": control_vs_resumed["max_abs_difference"] <= 5e-3
                and rank_checksum_consistency["max_abs_diff"] <= 1e-8,
            }
            save_json(output_dir / "checkpoint_resume_report.json", report)
        distributed_barrier(dist_ctx)
        return report
    finally:
        distributed_barrier(dist_ctx)
        if dist_ctx.rank == 0:
            shutil.rmtree(save_path, ignore_errors=True)
        distributed_barrier(dist_ctx)


def gather_environment_report(cfg: StageAConfig, dist_ctx, output_dir: Path) -> None:
    if dist_ctx.rank != 0:
        return
    git_commit = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    git_status = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--short"], text=True)
    git_diff = subprocess.check_output(["git", "-C", str(REPO_ROOT), "diff"], text=True)
    python_version = sys.version
    try:
        import torch.utils.collect_env as collect_env
        env_dump = collect_env.get_pretty_env_info()
    except Exception as exc:
        env_dump = f"collect_env_failed: {exc}"
    topology = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    lines = [
        f"git_commit: {git_commit}",
        "git_status:",
        git_status,
        f"python_version: {python_version}",
        f"torch_version: {torch.__version__}",
        f"cuda_version: {torch.version.cuda}",
        f"nccl_version: {torch.cuda.nccl.version() if torch.cuda.is_available() else None}",
        f"hostname: {socket.gethostname()}",
        f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}",
        f"distributed_backend: {cfg.raw['distributed']['backend']}",
        f"world_size: {dist_ctx.world_size}",
        f"gpu_count_visible: {torch.cuda.device_count()}",
        "gpu_topology:",
        topology,
        "torch_collect_env:",
        env_dump,
    ]
    write_text(output_dir / "environment_report.txt", "\n".join(lines) + "\n")
    write_text(output_dir / "git_diff.patch", git_diff)


def stagea_steps(cfg: StageAConfig) -> Tuple[int, int]:
    global_tokens_per_step = cfg.micro_batch_size_per_rank * cfg.sequence_length * cfg.grad_accum * cfg.world_size
    steps = math.ceil(cfg.target_stage_a_tokens / global_tokens_per_step)
    return steps, global_tokens_per_step


def proxy_evaluate(model, cfg: StageAConfig, dist_ctx, val_loader: Optional[DataLoader]) -> Dict[str, object]:
    if dist_ctx.rank != 0 or val_loader is None:
        return {}
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())
    total_loss = 0.0
    total_sequences = 0
    batches = 0
    for batch in val_loader:
        input_ids = batch["input_ids"].to(dist_ctx.device)
        labels = batch["labels"].to(dist_ctx.device)
        with precision_context(precision_dtype):
            outputs = base_model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
        total_loss += float(outputs.loss.detach().cpu()) * int(input_ids.shape[0])
        total_sequences += int(input_ids.shape[0])
        batches += 1
        if batches >= cfg.max_eval_batches or total_sequences >= cfg.proxy_max_val_samples:
            break
    avg_loss = total_loss / max(total_sequences, 1)
    metrics = {"proxy_val_loss": avg_loss, "proxy_val_ppl": math.exp(min(avg_loss, 20.0)), "proxy_val_sequences": total_sequences, "proxy_val_batches": batches}
    if was_training:
        base_model.train()
    return metrics


def run_stage_a_training(cfg: StageAConfig, dist_ctx, output_dir: Path, model: HGRNBitForCausalLM, train_loader: DataLoader, proxy_val_loader: Optional[DataLoader]) -> Dict[str, object]:
    target_steps, global_tokens_per_step = stagea_steps(cfg)
    ddp_model = DDP(model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank, find_unused_parameters=False)
    optimizer, scheduler = build_optimizer_and_scheduler(ddp_model.module, cfg)
    _, scaler = get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())
    train_iter = repeat_dataloader(train_loader)
    base_model = unwrap_model(ddp_model)
    metrics_path = output_dir / "training_metrics.jsonl"
    initial_train_loss = None
    last_train_loss = None
    initial_proxy_loss = None
    last_proxy_loss = None
    best_proxy = None
    best_proxy_step = None
    best_proxy_dir = output_dir / "checkpoint_best_proxy"
    checkpoint_mid_dir = output_dir / "checkpoint_mid"
    checkpoint_final_dir = output_dir / "checkpoint_final"
    checkpoint_mid_step = max(1, target_steps // 2)
    snapshot_steps = {math.ceil(tokens / global_tokens_per_step): tokens for tokens in cfg.activation_snapshot_token_milestones}
    peak_allocated = 0
    peak_reserved = 0
    step_durations: List[float] = []
    global_tokens_seen = 0
    for step in range(1, target_steps + 1):
        ddp_model.train()
        optimizer.zero_grad(set_to_none=True)
        step_start = time.time()
        local_tokens_seen = 0
        micro_data_time = 0.0
        micro_losses: List[float] = []
        for micro_idx in range(cfg.grad_accum):
            data_t0 = time.time()
            batch = next(train_iter)
            micro_data_time += time.time() - data_t0
            local_tokens_seen += int(batch["input_ids"].numel())
            sync_ctx = ddp_model.no_sync() if micro_idx < cfg.grad_accum - 1 else nullcontext()
            with sync_ctx:
                input_ids = batch["input_ids"].to(dist_ctx.device)
                labels = batch["labels"].to(dist_ctx.device)
                with precision_context(get_precision_dtype(type("Cfg", (), {"precision": cfg.precision})())[0]):
                    outputs = ddp_model(input_ids=input_ids, labels=labels, output_router_logits=False, return_dict=True)
                    loss = outputs.loss / cfg.grad_accum
                scaler.scale(loss).backward()
                micro_losses.append(float(outputs.loss.detach().cpu()))
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        step_time = time.time() - step_start
        step_durations.append(step_time)
        global_tokens_seen += int(distributed_reduce_float(float(local_tokens_seen), dist_ctx, average=False))
        mean_loss = distributed_reduce_float(sum(micro_losses) / max(len(micro_losses), 1), dist_ctx, average=True)
        grad_norm_value = distributed_reduce_float(float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm), dist_ctx, average=True)
        gpu_snapshot = make_gpu_snapshot() if dist_ctx.rank == 0 else None
        if dist_ctx.rank == 0:
            allocated = float(torch.cuda.max_memory_allocated(dist_ctx.device) / (1024 ** 2))
            reserved = float(torch.cuda.max_memory_reserved(dist_ctx.device) / (1024 ** 2))
            peak_allocated = max(peak_allocated, allocated)
            peak_reserved = max(peak_reserved, reserved)
            record = {
                "optimizer_step": step,
                "scheduler_step": scheduler.last_epoch,
                "micro_steps_per_optimizer_step": cfg.grad_accum,
                "train_loss": mean_loss,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": grad_norm_value,
                "parameter_norm": math.sqrt(sum(float(p.detach().float().square().sum().item()) for p in base_model.parameters())),
                "local_tokens_seen_last_step": local_tokens_seen,
                "global_tokens_seen": global_tokens_seen,
                "actual_non_padding_tokens_seen_last_step": local_tokens_seen,
                "step_time_sec": step_time,
                "data_loading_time_sec": micro_data_time,
                "tokens_per_sec": global_tokens_per_step / max(step_time, 1e-9),
                "gpu_snapshot": gpu_snapshot,
            }
            append_jsonl(metrics_path, record)
            last_train_loss = mean_loss
            if initial_train_loss is None:
                initial_train_loss = mean_loss
        if step in snapshot_steps and dist_ctx.rank == 0:
            batch = next(repeat_dataloader(proxy_val_loader if proxy_val_loader is not None else train_loader))
            collect_activation_diagnostics(ddp_model.module, batch, dist_ctx.device, cfg.precision, output_dir / f"activation_snapshot_{snapshot_steps[step]}.json", output_dir / f"activation_snapshot_{snapshot_steps[step]}.csv")
        if step in snapshot_steps:
            distributed_barrier(dist_ctx)
        if step % cfg.eval_interval == 0:
            metrics = proxy_evaluate(ddp_model, cfg, dist_ctx, proxy_val_loader)
            save_best_proxy = False
            if dist_ctx.rank == 0:
                metrics["optimizer_step"] = step
                append_jsonl(metrics_path, metrics)
                last_proxy_loss = metrics["proxy_val_loss"]
                if initial_proxy_loss is None:
                    initial_proxy_loss = metrics["proxy_val_loss"]
                if best_proxy is None or metrics["proxy_val_loss"] < best_proxy:
                    best_proxy = metrics["proxy_val_loss"]
                    best_proxy_step = step
                    if best_proxy_dir.exists():
                        shutil.rmtree(best_proxy_dir)
                    save_best_proxy = True
            save_best_payload = [save_best_proxy if dist_ctx.rank == 0 else None]
            if dist.is_initialized():
                dist.broadcast_object_list(save_best_payload, src=0)
            if save_best_payload[0]:
                coordinated_rank0_checkpoint_save(
                    dist_ctx,
                    output_dir,
                    "checkpoint_best_proxy",
                    lambda: save_checkpoint(
                        best_proxy_dir,
                        ddp_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        global_tokens_seen,
                        current_rng_state(),
                    ),
                )
            distributed_barrier(dist_ctx)
        if step == checkpoint_mid_step:
            coordinated_rank0_checkpoint_save(
                dist_ctx,
                output_dir,
                "checkpoint_mid",
                lambda: save_checkpoint(
                    checkpoint_mid_dir,
                    ddp_model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    global_tokens_seen,
                    current_rng_state(),
                ),
            )
        if step % cfg.log_interval == 0:
            distributed_barrier(dist_ctx)
    coordinated_rank0_checkpoint_save(
        dist_ctx,
        output_dir,
        "checkpoint_final",
        lambda: save_checkpoint(
            checkpoint_final_dir,
            ddp_model,
            optimizer,
            scheduler,
            scaler,
            target_steps,
            global_tokens_seen,
            current_rng_state(),
        ),
    )
    if dist_ctx.rank == 0:
        batch = next(repeat_dataloader(proxy_val_loader if proxy_val_loader is not None else train_loader))
        collect_activation_diagnostics(ddp_model.module, batch, dist_ctx.device, cfg.precision, output_dir / "activation_diagnostics_after_stageA.json", output_dir / "activation_by_layer_after_stageA.csv")
        collect_gradient_diagnostics(ddp_model.module, batch, dist_ctx.device, cfg.precision, output_dir / "gradient_diagnostics_after_stageA.json")
        summary = {
            "optimizer_steps": target_steps,
            "global_tokens_per_optimizer_step": global_tokens_per_step,
            "global_tokens_seen": global_tokens_seen,
            "initial_train_loss": initial_train_loss,
            "final_train_loss": last_train_loss,
            "initial_proxy_val_loss": initial_proxy_loss,
            "final_proxy_val_loss": last_proxy_loss,
            "best_proxy_val_loss": best_proxy,
            "best_proxy_step": best_proxy_step,
            "peak_memory_allocated_mib_rank0": peak_allocated,
            "peak_memory_reserved_mib_rank0": peak_reserved,
            "avg_tokens_per_sec": global_tokens_seen / max(sum(step_durations), 1e-9),
        }
        save_json(output_dir / "stageA_training_summary.json", summary)
    distributed_barrier(dist_ctx)
    return summary if dist_ctx.rank == 0 else {
        "optimizer_steps": target_steps,
        "global_tokens_per_optimizer_step": global_tokens_per_step,
        "global_tokens_seen": global_tokens_seen,
    }


def main() -> None:
    args = parse_args()
    cfg = load_stagea_config(args.config_path)
    dist_ctx = setup_distributed("cuda")
    if dist_ctx.world_size != cfg.world_size:
        cleanup_distributed(dist_ctx)
        raise SystemExit(f"Config world_size={cfg.world_size} but runtime WORLD_SIZE={dist_ctx.world_size}")
    output_dir = resolve_output_dir(cfg, args.output_dir, dist_ctx)
    try:
        if dist_ctx.rank == 0:
            save_json(output_dir / "config_snapshot.json", cfg.raw)
            write_text(output_dir / "resolved_config.yaml", yaml.safe_dump(cfg.raw, sort_keys=False, allow_unicode=True))
            gather_environment_report(cfg, dist_ctx, output_dir)
            mark_stage(output_dir, dist_ctx.rank, "bootstrap", "done")
        distributed_barrier(dist_ctx)

        mark_stage(output_dir, dist_ctx.rank, "build_loaders", "started")
        train_loader, train_manifest, _ = build_loader(
            data_source=cfg.dataset_path,
            tokenizer_path=cfg.tokenizer,
            max_length=cfg.sequence_length,
            batch_size=cfg.micro_batch_size_per_rank,
            text_field=cfg.text_field,
            split="train",
            max_samples=None,
            shard_rank=dist_ctx.rank,
            shard_world_size=dist_ctx.world_size,
        )
        val_loader, val_manifest, _ = build_loader(
            data_source=cfg.validation_path,
            tokenizer_path=cfg.tokenizer,
            max_length=cfg.sequence_length,
            batch_size=cfg.micro_batch_size_per_rank,
            text_field=cfg.text_field,
            split="validation",
            max_samples=cfg.proxy_max_val_samples,
            shard_rank=dist_ctx.rank,
            shard_world_size=dist_ctx.world_size,
        )
        proxy_val_loader = None
        if dist_ctx.rank == 0:
            proxy_val_loader, proxy_val_manifest, _ = build_loader(
                data_source=cfg.validation_path,
                tokenizer_path=cfg.tokenizer,
                max_length=cfg.sequence_length,
                batch_size=cfg.micro_batch_size_per_rank,
                text_field=cfg.text_field,
                split="validation",
                max_samples=cfg.proxy_max_val_samples,
                shard_rank=0,
                shard_world_size=1,
            )
        else:
            proxy_val_manifest = None
        if dist_ctx.rank == 0:
            save_json(output_dir / "train_dataset_manifest_runtime.json", train_manifest)
            save_json(output_dir / "val_dataset_manifest_runtime.json", val_manifest)
            if proxy_val_manifest is not None:
                save_json(output_dir / "proxy_val_dataset_manifest_runtime.json", proxy_val_manifest)
            mark_stage(output_dir, dist_ctx.rank, "build_loaders", "done")

        mark_stage(output_dir, dist_ctx.rank, "initial_diagnostics", "started")
        model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
        collect_initialization_diagnostics(
            model,
            dist_ctx.device,
            dist_ctx.enabled,
            dist_ctx.rank,
            output_dir,
            cfg.initialization_seed,
            save_output=dist_ctx.rank == 0,
        )
        if dist_ctx.rank == 0:
            initial_batch = next(iter(proxy_val_loader if proxy_val_loader is not None else val_loader))
            collect_activation_diagnostics(model, initial_batch, dist_ctx.device, cfg.precision, output_dir / "activation_diagnostics_initial.json", output_dir / "activation_by_layer_initial.csv")
            collect_gradient_diagnostics(model, initial_batch, dist_ctx.device, cfg.precision, output_dir / "gradient_diagnostics_initial.json")
            mark_stage(output_dir, dist_ctx.rank, "initial_diagnostics", "done")
        distributed_barrier(dist_ctx)

        mark_stage(output_dir, dist_ctx.rank, "ddp_equivalence", "started")
        ddp_equiv = run_ddp_equivalence_test(cfg, dist_ctx, output_dir)
        mark_stage(output_dir, dist_ctx.rank, "ddp_equivalence", "done")
        mark_stage(output_dir, dist_ctx.rank, "data_sharding", "started")
        sharding = run_data_sharding_report(cfg, dist_ctx, output_dir)
        mark_stage(output_dir, dist_ctx.rank, "data_sharding", "done")
        mark_stage(output_dir, dist_ctx.rank, "checkpoint_resume", "started")
        resume = run_checkpoint_resume_report(cfg, dist_ctx, output_dir)
        mark_stage(output_dir, dist_ctx.rank, "checkpoint_resume", "done")
        mark_stage(output_dir, dist_ctx.rank, "distributed_eval", "started")
        dist_eval = run_distributed_eval_report(cfg, dist_ctx, output_dir)
        mark_stage(output_dir, dist_ctx.rank, "distributed_eval", "done")
        mark_stage(output_dir, dist_ctx.rank, "short_training", "started")
        short_train = run_short_training_smoke(cfg, dist_ctx, output_dir, train_loader, proxy_val_loader)
        mark_stage(output_dir, dist_ctx.rank, "short_training", "done")

        verdict = "PASS"
        warnings: List[str] = []
        if dist_ctx.rank == 0:
            if not ddp_equiv.get("pass", False):
                verdict = "FAIL"
            if not sharding.get("distinct_data_across_ranks", False):
                verdict = "FAIL"
            if not resume.get("pass", False):
                warnings.append("checkpoint resume exact parameter continuity failed tolerance")
                if verdict != "FAIL":
                    verdict = "PASS WITH WARNINGS"
            if str(resume.get("dataset_progress_recovery", "")).startswith("not_exact"):
                warnings.append("streaming dataset exact cursor resume is not implemented; resume used fixed batches")
                if verdict != "FAIL":
                    verdict = "PASS WITH WARNINGS"
            if not dist_eval.get("pass", False):
                verdict = "FAIL"
            if not short_train.get("pass", False):
                verdict = "FAIL"
        verdict_obj = [verdict] if dist_ctx.rank == 0 else [None]
        if dist.is_initialized():
            dist.broadcast_object_list(verdict_obj, src=0)
        verdict = verdict_obj[0]
        if verdict == "FAIL":
            if dist_ctx.rank == 0:
                mark_stage(output_dir, dist_ctx.rank, "precheck_verdict", "failed")
            cleanup_distributed(dist_ctx)
            if dist_ctx.rank == 0:
                raise SystemExit("Stage A prechecks failed; see reports.")
            raise SystemExit(1)

        stagea_result = {}
        if not args.skip_stage_a_train:
            mark_stage(output_dir, dist_ctx.rank, "stageA_train", "started")
            model = build_scratch_model(cfg.base_config_path, cfg.initialization_seed, dist_ctx.device)
            stagea_result = run_stage_a_training(cfg, dist_ctx, output_dir, model, train_loader, proxy_val_loader)
            mark_stage(output_dir, dist_ctx.rank, "stageA_train", "done")

        if dist_ctx.rank == 0:
            summary = {
                "verdict": verdict if not warnings else "PASS WITH WARNINGS",
                "warnings": warnings,
                "ddp_equivalence_pass": ddp_equiv.get("pass"),
                "data_sharding_distinct": sharding.get("distinct_data_across_ranks"),
                "checkpoint_resume_pass": resume.get("pass"),
                "distributed_eval_pass": dist_eval.get("pass"),
                "short_training_pass": short_train.get("pass"),
                "stageA": stagea_result,
            }
            md = [
                "# Stage A Report",
                "",
                f"- verdict: {summary['verdict']}",
                f"- warnings: {warnings}",
                f"- ddp_equivalence_pass: {summary['ddp_equivalence_pass']}",
                f"- data_sharding_distinct: {summary['data_sharding_distinct']}",
                f"- checkpoint_resume_pass: {summary['checkpoint_resume_pass']}",
                f"- distributed_eval_pass: {summary['distributed_eval_pass']}",
                f"- short_training_pass: {summary['short_training_pass']}",
            ]
            if stagea_result:
                md.extend([
                    f"- optimizer_steps: {stagea_result.get('optimizer_steps')}",
                    f"- global_tokens_seen: {stagea_result.get('global_tokens_seen')}",
                    f"- initial_train_loss: {stagea_result.get('initial_train_loss')}",
                    f"- final_train_loss: {stagea_result.get('final_train_loss')}",
                    f"- initial_proxy_val_loss: {stagea_result.get('initial_proxy_val_loss')}",
                    f"- final_proxy_val_loss: {stagea_result.get('final_proxy_val_loss')}",
                    f"- best_proxy_val_loss: {stagea_result.get('best_proxy_val_loss')}",
                    f"- best_proxy_step: {stagea_result.get('best_proxy_step')}",
                    f"- avg_tokens_per_sec: {stagea_result.get('avg_tokens_per_sec')}",
                ])
            save_json(output_dir / "stageA_summary.json", summary)
            write_text(output_dir / "stageA_report.md", "\n".join(md) + "\n")
            mark_stage(output_dir, dist_ctx.rank, "complete", "done", {"verdict": summary["verdict"]})
        cleanup_distributed(dist_ctx)
    except Exception as exc:
        if dist_ctx.rank == 0:
            save_json(
                output_dir / "failure_report.json",
                {
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            mark_stage(output_dir, dist_ctx.rank, "failed", "error", {"exception_type": type(exc).__name__})
        cleanup_distributed(dist_ctx)
        raise


if __name__ == "__main__":
    main()
