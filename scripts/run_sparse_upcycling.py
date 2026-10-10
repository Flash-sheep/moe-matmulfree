#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM, HGRNBitMLP
from mmfreelm.upcycling import (
    ExpertMonitor,
    StreamingTextDataset,
    apply_freeze_for_upcycling,
    upcycle_dense_to_moe,
)
from mmfreelm.upcycling.param_groups import (
    build_optimizer_param_groups,
    optimizer_lr_map,
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
from scripts.train_moe_lm import (
    append_jsonl,
    ensure_cuda_device,
    evaluate,
    enrich_router_metrics,
    extract_optional_loss_metrics,
    flatten_router_metrics,
    get_precision_dtype,
    json_default,
    precision_context,
    resume_checkpoint,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse upcycling training entrypoint.")
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Temporarily disabled: resuming with a different max_steps changes the LR schedule semantics.",
    )
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--val-data-source", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--eval-interval", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--disable-best-checkpoint", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


class DistributedContext:
    def __init__(self, *, enabled: bool, rank: int, local_rank: int, world_size: int, device: torch.device) -> None:
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed(requested_device: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested via WORLD_SIZE>1, but CUDA is not available.")
        # Bind the process before constructing NCCL so early barriers do not
        # have to guess which GPU belongs to this rank.
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                device_id=torch.device(f"cuda:{local_rank}"),
            )
        device = torch.device(f"cuda:{local_rank}")
        return DistributedContext(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size, device=device)
    device = ensure_cuda_device(requested_device)
    return DistributedContext(enabled=False, rank=0, local_rank=0, world_size=1, device=device)


def cleanup_distributed(dist_ctx: DistributedContext) -> None:
    if dist_ctx.enabled and dist.is_initialized():
        dist.barrier(device_ids=[dist_ctx.local_rank])
        dist.destroy_process_group()


def distributed_barrier(dist_ctx: DistributedContext) -> None:
    if dist_ctx.enabled and dist.is_initialized():
        dist.barrier(device_ids=[dist_ctx.local_rank])


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def broadcast_bool(flag: bool, dist_ctx: DistributedContext) -> bool:
    if not dist_ctx.enabled:
        return flag
    tensor = torch.tensor([1 if flag else 0], device=dist_ctx.device, dtype=torch.int32)
    dist.broadcast(tensor, src=0)
    return bool(int(tensor.item()))


def distributed_reduce_float(value: float, dist_ctx: DistributedContext, average: bool = True) -> float:
    if not dist_ctx.enabled:
        return float(value)
    tensor = torch.tensor([float(value)], device=dist_ctx.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    reduced = float(tensor.item())
    if average:
        reduced /= max(dist_ctx.world_size, 1)
    return reduced


def distributed_reduce_metric_dict(metrics: Dict[str, object], dist_ctx: DistributedContext) -> Dict[str, object]:
    if not dist_ctx.enabled:
        return metrics
    reduced: Dict[str, object] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, list):
            tensor = torch.tensor(value, device=dist_ctx.device, dtype=torch.float64)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / max(dist_ctx.world_size, 1)
            reduced[key] = tensor.tolist()
        else:
            reduced[key] = distributed_reduce_float(float(value), dist_ctx, average=True)
    return reduced


def reinitialize_dense_ffn_modules(
    *,
    model: HGRNBitForCausalLM,
    reinit_cfg: Optional[Dict],
    output_dir: Path,
    write_summary: bool = True,
) -> Dict[str, object]:
    cfg = dict(reinit_cfg or {})
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        payload = {"enabled": False, "applied": False, "strategy": None, "layer_indices": [], "num_modules": 0}
        if write_summary:
            (output_dir / "dense_ffn_reinit_summary.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
                encoding="utf-8",
            )
        return payload

    strategy = str(cfg.get("strategy", "model_default"))
    scope = str(cfg.get("scope", "all_ffn"))
    requested_layer_indices = cfg.get("layer_indices")
    if requested_layer_indices is None or requested_layer_indices == "all":
        target_layer_indices = list(range(len(model.model.layers)))
    else:
        target_layer_indices = [int(idx) for idx in requested_layer_indices]

    module_records: List[Dict[str, object]] = []
    for layer_idx in target_layer_indices:
        if layer_idx < 0 or layer_idx >= len(model.model.layers):
            raise ValueError(f"dense_reinit layer index out of range: {layer_idx}")
        block = model.model.layers[layer_idx]
        if not isinstance(block.mlp, HGRNBitMLP):
            raise ValueError(
                f"dense_reinit expected dense HGRNBitMLP at layer {layer_idx}, "
                f"but found {block.mlp.__class__.__name__}"
            )
        gate_before = float(block.mlp.gate_proj.weight.detach().float().norm().item())
        down_before = float(block.mlp.down_proj.weight.detach().float().norm().item())
        if strategy != "model_default":
            raise ValueError(f"Unsupported dense_reinit strategy `{strategy}`.")
        model._init_weights(block.mlp.gate_proj)
        model._init_weights(block.mlp.down_proj)
        gate_after = float(block.mlp.gate_proj.weight.detach().float().norm().item())
        down_after = float(block.mlp.down_proj.weight.detach().float().norm().item())
        module_records.append(
            {
                "layer_idx": int(layer_idx),
                "module_name": f"model.layers.{layer_idx}.mlp",
                "gate_proj_weight_norm_before": gate_before,
                "gate_proj_weight_norm_after": gate_after,
                "down_proj_weight_norm_before": down_before,
                "down_proj_weight_norm_after": down_after,
                "gate_proj_dtype": str(block.mlp.gate_proj.weight.dtype),
                "down_proj_dtype": str(block.mlp.down_proj.weight.dtype),
            }
        )

    payload = {
        "enabled": True,
        "applied": True,
        "scope": scope,
        "strategy": strategy,
        "layer_indices": target_layer_indices,
        "num_modules": len(module_records),
        "module_records": module_records,
    }
    if write_summary:
        (output_dir / "dense_ffn_reinit_summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
    return payload


def collate_streaming_batch(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}


def repeat_dataloader(dataloader, dataset=None) -> Iterator[Dict[str, torch.Tensor]]:
    epoch = 0
    while True:
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        epoch += 1
        for batch in dataloader:
            yield batch


def build_streaming_loader(
    data_source: str,
    tokenizer_path: str,
    max_length: int,
    batch_size: int,
    text_field: str = "text",
    split: str = "train",
    max_samples=None,
    shard_rank: int = 0,
    shard_world_size: int = 1,
):
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


def load_best_val_loss(log_path: Path):
    if not log_path.exists():
        return None
    best_val_loss = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        current = None
        if "proxy_val_lm_loss" in record:
            current = float(record["proxy_val_lm_loss"])
        elif "val_lm_loss" in record:
            current = float(record["val_lm_loss"])
        elif "proxy_val_loss" in record:
            current = float(record["proxy_val_loss"])
        elif "val_loss" in record:
            current = float(record["val_loss"])
        if current is not None:
            best_val_loss = current if best_val_loss is None else min(best_val_loss, current)
        if record.get("event") in {"checkpoint_best_proxy", "checkpoint_best"}:
            metric_value = record.get(
                "best_proxy_val_lm_loss",
                record.get("best_proxy_val_loss", record.get("best_val_loss")),
            )
            if metric_value is None:
                continue
            current = float(metric_value)
            best_val_loss = current if best_val_loss is None else min(best_val_loss, current)
    return best_val_loss


def accumulate_metric_dict(sums: Dict[str, object], counts: Dict[str, int], metrics: Dict[str, object]) -> None:
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, list):
            tensor_value = torch.tensor(value, dtype=torch.float32)
            if key not in sums:
                sums[key] = tensor_value.clone()
            else:
                sums[key] = sums[key] + tensor_value
        else:
            numeric_value = float(value)
            sums[key] = float(sums.get(key, 0.0)) + numeric_value
        counts[key] = counts.get(key, 0) + 1


def average_metric_dict(sums: Dict[str, object], counts: Dict[str, int]) -> Dict[str, object]:
    averaged: Dict[str, object] = {}
    for key, value in sums.items():
        count = max(counts.get(key, 1), 1)
        if isinstance(value, torch.Tensor):
            averaged[key] = (value / count).tolist()
        else:
            averaged[key] = float(value) / count
    return averaged


def write_dataset_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def _tensorboard_log(writer: SummaryWriter, prefix: str, step: int, payload: Dict[str, object]) -> None:
    for key, value in payload.items():
        if key in {"step", "event"} or value is None:
            continue
        tag = f"{prefix}/{key}"
        if isinstance(value, bool):
            writer.add_scalar(tag, int(value), step)
        elif isinstance(value, (int, float)):
            writer.add_scalar(tag, float(value), step)
        elif isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
            for idx, item in enumerate(value):
                writer.add_scalar(f"{tag}/{idx}", float(item), step)


def update_checkpoint_best_alias(best_proxy_dir: Path, alias_dir: Path) -> str:
    if alias_dir.exists() or alias_dir.is_symlink():
        if alias_dir.is_symlink() or alias_dir.is_file():
            alias_dir.unlink()
        elif alias_dir.is_dir() and alias_dir.resolve() == best_proxy_dir.resolve():
            return "shared_directory"
        else:
            return "existing_directory_preserved"
    try:
        alias_dir.symlink_to(best_proxy_dir.name, target_is_directory=True)
        return "symlink_to_checkpoint_best_proxy"
    except OSError:
        if alias_dir.exists():
            return "existing_directory_preserved"
        shutil.copytree(best_proxy_dir, alias_dir)
        return "copied_from_checkpoint_best_proxy"


def write_trainable_debug_outputs(
    output_dir: Path,
    trainable_summary: Dict[str, object],
    optimizer_group_summary: List[Dict[str, object]],
):
    names_path = output_dir / "trainable_param_names.txt"
    names_path.write_text(
        "\n".join(trainable_summary["trainable_parameter_names"]) + ("\n" if trainable_summary["trainable_parameter_names"] else ""),
        encoding="utf-8",
    )
    (output_dir / "trainable_param_summary.json").write_text(
        json.dumps(trainable_summary, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "optimizer_param_groups.json").write_text(
        json.dumps(optimizer_group_summary, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def build_init_verification_payload(
    *,
    config: Dict,
    args: argparse.Namespace,
    model,
    trainable_params: int,
    frozen_params: int,
    trainable_summary: Dict[str, object],
    optimizer_group_summary: List[Dict[str, object]],
    freeze_mode: str,
    norm_scope: str,
    moe_layer_indices: List[int],
    local_backbone_layer_indices: List[int],
    optimizer_hparams: Dict[str, float],
    strict_trainable_check: bool,
    hard_check_issues: List[str],
    warnings: List[str],
    dense_ffn_reinit_summary: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    total_params = sum(param.numel() for param in model.parameters())
    parameter_budget = getattr(model.config, "moe_parameter_budget_verification", None)
    return {
        "experiment": config.get("experiment_name"),
        "config_path": str(args.config_path),
        "pretrained_path": args.pretrained_path,
        "output_dir": str(args.output_dir),
        "pass": len(hard_check_issues) == 0,
        "issues": hard_check_issues,
        "warnings": warnings,
        "preflight_only": bool(args.preflight_only),
        "freeze_mode": freeze_mode,
        "norm_scope": norm_scope,
        "strict_trainable_check": strict_trainable_check,
        "freeze_embeddings": bool(config.get("freeze", {}).get("freeze_embeddings", True)),
        "freeze_lm_head": bool(config.get("freeze", {}).get("freeze_lm_head", True)),
        "moe_layer_indices": moe_layer_indices,
        "local_backbone_layer_indices": local_backbone_layer_indices,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "optimizer_hparams": optimizer_hparams,
        "parameter_budget_verification": parameter_budget,
        "dense_ffn_reinit_summary": dense_ffn_reinit_summary,
        "trainable_parameter_counts_by_module_type": trainable_summary["trainable_parameter_counts_by_module_type"],
        "trainable_parameter_tensors_by_module_type": trainable_summary["trainable_parameter_tensors_by_module_type"],
        "first_200_trainable_parameter_names": trainable_summary["first_200_trainable_parameter_names"],
        "extra_pattern_enabled_parameter_names": trainable_summary["extra_pattern_enabled_parameter_names"],
        "extra_pattern_enabled_parameter_count": trainable_summary["extra_pattern_enabled_parameter_count"],
        "selected_norm_parameter_names": trainable_summary["selected_norm_parameter_names"],
        "selected_norm_parameter_count": trainable_summary["selected_norm_parameter_count"],
        "local_backbone_parameter_count": trainable_summary["local_backbone_parameter_count"],
        "optimizer_group_summary": optimizer_group_summary,
    }


def main() -> None:
    args = parse_args()
    if args.resume_from is not None:
        raise SystemExit(
            "--resume-from is temporarily disabled. "
            "The current implementation can change LR schedule semantics when resuming "
            "with a different max_steps, which makes 5k->20k continuation scientifically unreliable. "
            "Run the target schedule from scratch instead."
        )
    config = load_config(args.config_path)

    training = config.get("training", {})
    moe_cfg = config.get("moe", {})
    freeze_cfg = config.get("freeze", {})
    dense_baseline = bool(config.get("dense_baseline", False))
    freeze_mode = infer_freeze_mode(freeze_cfg, dense_baseline=dense_baseline)
    norm_scope = infer_norm_scope(freeze_cfg, freeze_mode)
    moe_layer_indices = moe_cfg.get("layer_indices", list(range(12, 24)))
    local_backbone_layer_indices = resolve_local_backbone_layer_indices(
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=freeze_cfg.get("local_backbone_layer_indices"),
    )
    strict_trainable_check = infer_strict_trainable_check(freeze_cfg, config)
    optimizer_hparams = resolve_optimizer_hparams(config, training, freeze_cfg, freeze_mode)

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    precision = args.precision or training.get("precision", "bf16")
    dist_ctx = setup_distributed(args.device or training.get("device", "cuda"))
    device = dist_ctx.device
    set_seed(seed)

    output_dir = args.output_dir
    checkpoint_best_proxy_dir = output_dir / "checkpoint_best_proxy"
    checkpoint_best_alias_dir = output_dir / "checkpoint_best"
    checkpoints_dir = output_dir / "checkpoints"
    output_dir_ready = True
    if dist_ctx.is_main:
        if (
            checkpoint_best_proxy_dir.exists()
            or checkpoint_best_alias_dir.exists()
            or (checkpoints_dir.exists() and any(checkpoints_dir.iterdir()))
        ):
            output_dir_ready = False
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_ready = broadcast_bool(output_dir_ready, dist_ctx)
    if not output_dir_ready:
        cleanup_distributed(dist_ctx)
        raise SystemExit(f"Refusing to overwrite existing checkpoints in {output_dir}")
    distributed_barrier(dist_ctx)
    log_path = output_dir / "train_log.jsonl"
    tensorboard_enabled = bool(training.get("enable_tensorboard", config.get("enable_tensorboard", False)))
    tensorboard_log_dir = output_dir / str(training.get("tensorboard_subdir", "tensorboard"))
    tb_writer = SummaryWriter(log_dir=str(tensorboard_log_dir)) if tensorboard_enabled and dist_ctx.is_main else None
    if dist_ctx.is_main:
        append_jsonl(
            log_path,
            {
                "event": "start",
                "config_path": str(args.config_path),
                "resume_from": str(args.resume_from) if args.resume_from is not None else None,
                "loss_logging_version": "v2_normalized",
                "checkpoint_best_alias_behavior": "planned_checkpoint_best_proxy_alias",
                "tensorboard_enabled": tensorboard_enabled,
                "tensorboard_log_dir": str(tensorboard_log_dir) if tensorboard_enabled else None,
                "ddp_enabled": dist_ctx.enabled,
                "ddp_rank": dist_ctx.rank,
                "ddp_local_rank": dist_ctx.local_rank,
                "ddp_world_size": dist_ctx.world_size,
            },
        )
        (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer_path = args.tokenizer_path or args.pretrained_path
    dense_ffn_reinit_summary: Optional[Dict[str, object]] = None
    if args.resume_from is not None:
        model = HGRNBitForCausalLM.from_pretrained(str(args.resume_from), torch_dtype=torch.bfloat16)
        moe_layer_indices = list(getattr(model.config, "moe_layer_indices", None) or moe_layer_indices)
    else:
        model = HGRNBitForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16)
        if dense_baseline:
            model.config.use_moe = False
            model.config.moe_layer_indices = []
            moe_layer_indices = []
            dense_ffn_reinit_summary = reinitialize_dense_ffn_modules(
                model=model,
                reinit_cfg=config.get("dense_reinit", {}),
                output_dir=output_dir,
                write_summary=dist_ctx.is_main,
            )
        else:
            model = upcycle_dense_to_moe(
                model=model,
                moe_layer_indices=moe_layer_indices,
                num_experts=moe_cfg.get("num_experts", 8),
                num_experts_per_tok=moe_cfg.get("num_experts_per_tok", 2),
                noise_scale=moe_cfg.get("noise_scale", 0.05),
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
                enable_output_norm_logging=moe_cfg.get("enable_output_norm_logging", False),
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
        model=model,
        freeze_mode=freeze_mode,
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=local_backbone_layer_indices,
        norm_scope=norm_scope,
        trainable_extra_patterns=freeze_cfg.get("trainable_extra_patterns", []),
    )
    optimizer_param_groups, optimizer_group_summary = build_optimizer_param_groups(
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
    hard_check_issues = (
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
        if strict_trainable_check
        else []
    )
    warnings: List[str] = []
    if trainable_summary["extra_pattern_enabled_parameter_count"] > 32:
        warnings.append(
            "trainable_extra_patterns enabled more than 32 additional parameters; verify that scope expansion is intended."
        )
    parameter_budget = getattr(model.config, "moe_parameter_budget_verification", None)
    if moe_cfg.get("moe_arch", "standard") == "shared_residual":
        if not parameter_budget:
            hard_check_issues.append("Missing parameter_budget_verification for shared_residual architecture.")
        else:
            enforce_baseline_fair = bool(
                parameter_budget.get("enforce_baseline_fair", moe_cfg.get("strict_total_param_fair", False))
            )
            enforce_active_width_below_dense = bool(
                parameter_budget.get("enforce_active_width_below_dense", moe_cfg.get("strict_total_param_fair", False))
            )
            if enforce_baseline_fair and not bool(parameter_budget.get("strict_total_param_fair_passed", False)):
                hard_check_issues.append("strict_total_param_fair_passed=false for shared_residual architecture.")
            if enforce_baseline_fair and int(parameter_budget.get("new_total_params", 0)) > int(parameter_budget.get("baseline_total_params", 0)):
                hard_check_issues.append("new_total_params exceeds baseline_total_params.")
            if moe_cfg.get("enable_sparse_residual", True) and enforce_active_width_below_dense:
                active_width = int(parameter_budget.get("active_width", 0))
                if active_width >= int(model.config.intermediate_size or 2816):
                    hard_check_issues.append(
                        f"shared_residual active_width must stay below dense width; got {active_width}."
                    )
            init_stats = getattr(model.config, "moe_shared_residual_init_stats", {})
            zero_values = []
            for layer_stats in init_stats.values():
                sparse_stats = layer_stats.get("sparse_init_stats", {})
                zero_ratio_avg = sparse_stats.get("zero_ratio_avg")
                if zero_ratio_avg is not None:
                    zero_values.append(float(zero_ratio_avg))
            if zero_values:
                sparse_zero_ratio_avg = sum(zero_values) / len(zero_values)
                sparse_init_mode = str(parameter_budget.get("sparse_init", moe_cfg.get("sparse_init", "")))
                if sparse_init_mode == "random_ternary_matched":
                    if sparse_zero_ratio_avg < 0.30 or sparse_zero_ratio_avg > 0.45:
                        hard_check_issues.append(
                            f"shared_residual sparse expert zero ratio out of target range: {sparse_zero_ratio_avg:.4f}"
                        )
                elif sparse_init_mode == "dense_discarded_channel_split":
                    if sparse_zero_ratio_avg <= 0.0 or sparse_zero_ratio_avg >= 0.95:
                        hard_check_issues.append(
                            "shared_residual dense_discarded_channel_split produced an implausible sparse expert zero ratio: "
                            f"{sparse_zero_ratio_avg:.4f}"
                        )
                    elif sparse_zero_ratio_avg > 0.60:
                        warnings.append(
                            "shared_residual dense_discarded_channel_split zero ratio is highly sparse "
                            f"({sparse_zero_ratio_avg:.4f}); allowed for now, but monitor residual contribution and routing."
                        )
    if moe_cfg.get("moe_family") == "partition_4bit_random":
        if not parameter_budget:
            hard_check_issues.append("Missing parameter_budget_verification for partition_4bit_random.")
        else:
            if int(parameter_budget.get("expert_weight_bits", 0)) != 4:
                hard_check_issues.append("partition_4bit_random expected expert_weight_bits=4.")
            if not bool(parameter_budget.get("parameter_count_fair_vs_dense_ffn", False)):
                hard_check_issues.append("partition_4bit_random is not parameter-count fair vs dense FFN.")
            if int(parameter_budget.get("total_expert_width", 0)) != int(model.config.intermediate_size or 2816):
                hard_check_issues.append("partition_4bit_random total_expert_width does not match dense FFN width.")
            if int(parameter_budget.get("active_width", 0)) <= 0:
                hard_check_issues.append("partition_4bit_random active_width must be positive.")
            if bool(parameter_budget.get("bit_storage_fair_vs_ternary_dense", True)):
                hard_check_issues.append("partition_4bit_random must report bit_storage_fair_vs_ternary_dense=false.")
    if dist_ctx.is_main:
        write_trainable_debug_outputs(output_dir, trainable_summary, optimizer_group_summary)
    init_verification = build_init_verification_payload(
        config=config,
        args=args,
        model=model,
        trainable_params=trainable_params,
        frozen_params=frozen_params,
        trainable_summary=trainable_summary,
        optimizer_group_summary=optimizer_group_summary,
        freeze_mode=freeze_mode,
        norm_scope=norm_scope,
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=local_backbone_layer_indices,
        optimizer_hparams=optimizer_hparams,
        strict_trainable_check=strict_trainable_check,
        hard_check_issues=hard_check_issues,
        warnings=warnings,
        dense_ffn_reinit_summary=dense_ffn_reinit_summary,
    )
    if dist_ctx.is_main:
        (output_dir / "init_verification.json").write_text(
            json.dumps(init_verification, ensure_ascii=False, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
        if parameter_budget is not None:
            (output_dir / "parameter_budget_verification.json").write_text(
                json.dumps(parameter_budget, ensure_ascii=False, indent=2, default=json_default) + "\n",
                encoding="utf-8",
            )
        append_jsonl(
            output_dir / "train_log.jsonl",
            {
                "event": "freeze",
                "freeze_mode": freeze_mode,
                "norm_scope": norm_scope,
                "local_backbone_layer_indices": local_backbone_layer_indices,
                "strict_trainable_check": strict_trainable_check,
                "trainable_params": trainable_params,
                "frozen_params": frozen_params,
                "moe_layer_indices": moe_layer_indices,
                "dense_baseline": dense_baseline,
                "optimizer_group_summary": optimizer_group_summary,
                "optimizer_hparams": optimizer_hparams,
                "parameter_budget_verification": parameter_budget,
                "loss_logging_version": "v2_normalized",
                "train_loss_is_normalized": True,
                "ddp_enabled": dist_ctx.enabled,
                "ddp_world_size": dist_ctx.world_size,
            },
        )

    batch_size = args.batch_size or training.get("batch_size", 4)
    max_steps = args.max_steps or training.get("max_steps", 50000)
    grad_accum = args.gradient_accumulation_steps or training.get("gradient_accumulation_steps", 8)
    max_length = training.get("max_length", 2048)
    learning_rate = optimizer_hparams["learning_rate"]
    min_lr = training.get("min_lr")
    min_lr_ratio = training.get("min_lr_ratio")
    weight_decay = training.get("weight_decay", 0.01)
    warmup_steps = training.get("warmup_steps", 1000)
    grad_clip = training.get("grad_clip", 1.0)
    text_field = training.get("text_field", "text")
    log_interval = args.log_interval or training.get("log_interval", 50)
    eval_interval = args.eval_interval or training.get("eval_interval", 500)
    save_interval = args.save_interval or training.get("save_interval", 2000)
    monitor_interval = config.get("monitor_interval", 500)
    max_val_samples = training.get("max_val_samples", 1000)

    train_loader, train_manifest, train_dataset = build_streaming_loader(
        data_source=args.data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        batch_size=batch_size,
        text_field=text_field,
        split="train",
        shard_rank=dist_ctx.rank,
        shard_world_size=dist_ctx.world_size,
    )
    train_manifest_path = output_dir / "dataset_manifest_train.json"
    if dist_ctx.is_main:
        write_dataset_manifest(train_manifest_path, train_manifest)

    val_loader = None
    val_manifest = None
    val_manifest_path = None
    if args.val_data_source:
        val_loader, val_manifest, _ = build_streaming_loader(
            data_source=args.val_data_source,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            batch_size=batch_size,
            text_field=text_field,
            split="validation",
            max_samples=max_val_samples,
        )
        val_manifest_path = output_dir / "dataset_manifest_val.json"
        if dist_ctx.is_main:
            write_dataset_manifest(val_manifest_path, val_manifest)
    if dist_ctx.is_main:
        for split_name, manifest in (("train", train_manifest), ("validation", val_manifest)):
            if manifest and not manifest.get("split_filter_applied", True):
                append_jsonl(
                    log_path,
                    {
                        "event": "dataset_split_filter_warning",
                        "split": split_name,
                        "split_filter_applied": False,
                        "reason": manifest.get("reason"),
                        "file_count": manifest.get("file_count"),
                        "first_20_files": manifest.get("first_20_files"),
                    },
                )
        append_jsonl(
            log_path,
            {
                "event": "dataset_manifests",
                "train_dataset_manifest_path": str(train_manifest_path),
                "val_dataset_manifest_path": str(val_manifest_path) if val_manifest_path is not None else None,
                "train_dataset_manifest": train_manifest,
                "val_dataset_manifest": val_manifest,
            },
        )
    has_hard_check_issues = broadcast_bool(dist_ctx.is_main and bool(hard_check_issues), dist_ctx)
    if has_hard_check_issues:
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        cleanup_distributed(dist_ctx)
        if dist_ctx.is_main:
            raise SystemExit(
                "Strict trainable checks failed:\n- " + "\n- ".join(hard_check_issues)
            )
        raise SystemExit("Strict trainable checks failed on rank 0.")
    if args.preflight_only:
        if dist_ctx.is_main:
            append_jsonl(
                log_path,
                {
                    "event": "preflight_only_exit",
                    "freeze_mode": freeze_mode,
                    "trainable_params": trainable_params,
                    "frozen_params": frozen_params,
                    "train_dataset_manifest_path": str(train_manifest_path),
                    "val_dataset_manifest_path": str(val_manifest_path) if val_manifest_path is not None else None,
                },
            )
        if tb_writer is not None:
            tb_writer.flush()
            tb_writer.close()
        cleanup_distributed(dist_ctx)
        return

    model.to(device)

    optimizer = torch.optim.AdamW(optimizer_param_groups, betas=(0.9, 0.95))

    if min_lr is not None:
        min_lr_ratio = float(min_lr) / max(float(learning_rate), 1e-12)
    elif min_lr_ratio is None:
        min_lr_ratio = 0.0
    min_lr_ratio = max(0.0, min(float(min_lr_ratio), 1.0))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = min(max((step - warmup_steps) / max(max_steps - warmup_steps, 1), 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    precision_dtype, scaler = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    use_moe = bool(getattr(model.config, "use_moe", False))
    if dist_ctx.enabled:
        model = DDP(
            model,
            device_ids=[dist_ctx.local_rank],
            output_device=dist_ctx.local_rank,
            find_unused_parameters=use_moe,
        )
    model.train()
    base_model = unwrap_model(model)
    monitor = ExpertMonitor(base_model, moe_layer_indices) if dist_ctx.is_main else None

    train_iter: Iterator[Dict[str, torch.Tensor]] = repeat_dataloader(train_loader, train_dataset)
    window_total_loss_sum = 0.0
    window_lm_loss_sum = 0.0
    window_router_aux_loss_sum = 0.0
    window_router_z_loss_sum = 0.0
    window_load_balancing_loss_sum = 0.0
    window_raw_accumulated_total_loss_sum = 0.0
    window_grad_norm_sum = 0.0
    window_num_steps = 0
    window_router_z_loss_steps = 0
    window_load_balancing_loss_steps = 0
    window_router_metric_sums: Dict[str, object] = {}
    window_router_metric_counts: Dict[str, int] = {}
    tokens_seen = 0
    start_time = time.time()
    global_step = 0
    best_val_loss = load_best_val_loss(log_path) if dist_ctx.is_main else None
    if args.resume_from is not None:
        global_step = resume_checkpoint(args.resume_from, base_model, optimizer, scheduler, scaler, device)
        if dist_ctx.is_main:
            append_jsonl(
                log_path,
                {
                    "event": "resume",
                    "resume_from": str(args.resume_from),
                    "global_step": global_step,
                    "best_val_loss": best_val_loss,
                },
            )

    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        last_outputs = None
        step_total_loss_sum = 0.0
        step_lm_loss_sum = 0.0
        step_router_aux_loss_sum = 0.0
        step_router_z_loss_sum = 0.0
        step_load_balancing_loss_sum = 0.0
        step_microbatch_count = 0
        step_router_z_loss_count = 0
        step_load_balancing_loss_count = 0
        step_router_metric_sums: Dict[str, object] = {}
        step_router_metric_counts: Dict[str, int] = {}
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            with precision_context(precision_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_router_logits=use_moe,
                    return_dict=True,
                )
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            last_outputs = outputs
            optional_losses = extract_optional_loss_metrics(outputs)
            step_total_loss_sum += float(optional_losses["total_loss"] or 0.0)
            step_lm_loss_sum += float(optional_losses["lm_loss"] or 0.0)
            step_router_aux_loss_sum += float(optional_losses["router_aux_loss"] or 0.0)
            if optional_losses["router_z_loss"] is not None:
                step_router_z_loss_sum += float(optional_losses["router_z_loss"])
                step_router_z_loss_count += 1
            if optional_losses["load_balancing_loss"] is not None:
                step_load_balancing_loss_sum += float(optional_losses["load_balancing_loss"])
                step_load_balancing_loss_count += 1
            step_microbatch_count += 1
            tokens_seen += int(input_ids.numel())
            step_router_metrics = enrich_router_metrics(flatten_router_metrics(outputs.router_metrics))
            accumulate_metric_dict(step_router_metric_sums, step_router_metric_counts, step_router_metrics)

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        step_total_loss_avg = step_total_loss_sum / max(step_microbatch_count, 1)
        step_lm_loss_avg = step_lm_loss_sum / max(step_microbatch_count, 1)
        step_router_aux_loss_avg = step_router_aux_loss_sum / max(step_microbatch_count, 1)
        step_router_z_loss_avg = step_router_z_loss_sum / max(step_router_z_loss_count, 1) if step_router_z_loss_count > 0 else None
        step_load_balancing_loss_avg = (
            step_load_balancing_loss_sum / max(step_load_balancing_loss_count, 1)
            if step_load_balancing_loss_count > 0
            else None
        )
        step_router_metrics_avg = average_metric_dict(step_router_metric_sums, step_router_metric_counts)
        step_total_loss_avg = distributed_reduce_float(step_total_loss_avg, dist_ctx, average=True)
        step_lm_loss_avg = distributed_reduce_float(step_lm_loss_avg, dist_ctx, average=True)
        step_router_aux_loss_avg = distributed_reduce_float(step_router_aux_loss_avg, dist_ctx, average=True)
        if step_router_z_loss_avg is not None:
            step_router_z_loss_avg = distributed_reduce_float(step_router_z_loss_avg, dist_ctx, average=True)
        if step_load_balancing_loss_avg is not None:
            step_load_balancing_loss_avg = distributed_reduce_float(step_load_balancing_loss_avg, dist_ctx, average=True)
        step_router_metrics_avg = distributed_reduce_metric_dict(step_router_metrics_avg, dist_ctx)
        grad_norm_value = float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        grad_norm_value = distributed_reduce_float(grad_norm_value, dist_ctx, average=True)
        tokens_seen += 0
        global_tokens_seen = distributed_reduce_float(float(tokens_seen), dist_ctx, average=False)
        window_total_loss_sum += step_total_loss_avg
        window_lm_loss_sum += step_lm_loss_avg
        window_router_aux_loss_sum += step_router_aux_loss_avg
        if step_router_z_loss_avg is not None:
            window_router_z_loss_sum += step_router_z_loss_avg
            window_router_z_loss_steps += 1
        if step_load_balancing_loss_avg is not None:
            window_load_balancing_loss_sum += step_load_balancing_loss_avg
            window_load_balancing_loss_steps += 1
        window_raw_accumulated_total_loss_sum += step_total_loss_sum
        window_grad_norm_sum += grad_norm_value
        window_num_steps += 1
        accumulate_metric_dict(window_router_metric_sums, window_router_metric_counts, step_router_metrics_avg)

        if global_step % log_interval == 0 and last_outputs is not None and dist_ctx.is_main:
            lr_fields = optimizer_lr_map(optimizer)
            normalized_total_loss = window_total_loss_sum / max(window_num_steps, 1)
            normalized_lm_loss = window_lm_loss_sum / max(window_num_steps, 1)
            normalized_router_aux_loss = window_router_aux_loss_sum / max(window_num_steps, 1)
            raw_accumulated_total_loss = window_raw_accumulated_total_loss_sum / max(window_num_steps, 1)
            averaged_router_metrics = average_metric_dict(window_router_metric_sums, window_router_metric_counts)
            elapsed_time_sec = time.time() - start_time
            train_record = {
                "step": global_step,
                "train_loss": normalized_total_loss,
                "total_loss": normalized_total_loss,
                "normalized_total_loss": normalized_total_loss,
                "normalized_lm_loss": normalized_lm_loss,
                "normalized_router_aux_loss": normalized_router_aux_loss,
                "raw_accumulated_total_loss": raw_accumulated_total_loss,
                "raw_step_microbatch_count": step_microbatch_count,
                "lm_loss": normalized_lm_loss,
                "router_aux_loss": normalized_router_aux_loss,
                "grad_norm": grad_norm_value,
                "grad_norm_last": grad_norm_value,
                "grad_norm_avg": window_grad_norm_sum / max(window_num_steps, 1),
                "tokens_seen": int(global_tokens_seen),
                "elapsed_time": elapsed_time_sec,
                "elapsed_time_sec": elapsed_time_sec,
                "lr": lr_fields["lr_moe"] if lr_fields["lr_moe"] is not None else scheduler.get_last_lr()[0],
                **lr_fields,
                **averaged_router_metrics,
                "loss_logging_version": "v2_normalized",
                "train_loss_is_normalized": True,
            }
            if window_router_z_loss_steps > 0:
                train_record["router_z_loss"] = window_router_z_loss_sum / max(window_router_z_loss_steps, 1)
            if window_load_balancing_loss_steps > 0:
                train_record["load_balancing_loss"] = (
                    window_load_balancing_loss_sum / max(window_load_balancing_loss_steps, 1)
                )
            append_jsonl(log_path, train_record)
            if tb_writer is not None:
                _tensorboard_log(tb_writer, "train", global_step, train_record)
            window_total_loss_sum = 0.0
            window_lm_loss_sum = 0.0
            window_router_aux_loss_sum = 0.0
            window_router_z_loss_sum = 0.0
            window_load_balancing_loss_sum = 0.0
            window_raw_accumulated_total_loss_sum = 0.0
            window_grad_norm_sum = 0.0
            window_num_steps = 0
            window_router_z_loss_steps = 0
            window_load_balancing_loss_steps = 0
            window_router_metric_sums = {}
            window_router_metric_counts = {}

        if val_loader is not None and global_step % eval_interval == 0:
            eval_args = type(
                "EvalArgs",
                (),
                {
                    "precision": precision,
                    "max_eval_batches": training.get("max_eval_batches", None),
                    "use_moe": use_moe,
                    "eval_name": "proxy_val",
                    "eval_scope": None,
                    "batch_size": batch_size,
                    "max_val_samples": max_val_samples,
                    "data_source": args.val_data_source,
                    "split": "validation",
                    "checkpoint_source": str(output_dir / "checkpoint_latest_in_memory"),
                    "eval_seed": seed,
                    "eval_file_list": None if val_manifest is None else val_manifest.get("first_20_files"),
                    "eval_file_count": None if val_manifest is None else val_manifest.get("file_count"),
                },
            )()
            if dist_ctx.is_main:
                eval_metrics = evaluate(base_model, val_loader, device, eval_args)
                proxy_eval_record = {
                    "step": global_step,
                    "proxy_val_loss": eval_metrics.get("val_loss"),
                    "proxy_val_ppl": eval_metrics.get("val_ppl"),
                    "proxy_val_lm_loss": eval_metrics.get("val_lm_loss"),
                    "proxy_val_router_aux_loss": eval_metrics.get("val_router_aux_loss"),
                    "proxy_val_router_entropy": eval_metrics.get("val_router_entropy"),
                    "proxy_val_tokens_per_expert": eval_metrics.get("val_tokens_per_expert"),
                    "proxy_val_actual_num_batches": eval_metrics.get("actual_num_batches"),
                    "proxy_val_actual_num_sequences": eval_metrics.get("actual_num_sequences"),
                    "proxy_val_actual_num_tokens": eval_metrics.get("actual_num_tokens"),
                    "proxy_val_scope": eval_metrics.get("eval_scope"),
                    "proxy_val_max_eval_batches": eval_metrics.get("max_eval_batches"),
                    "proxy_val_max_val_samples": eval_metrics.get("max_val_samples"),
                    "proxy_val_batch_size": eval_metrics.get("batch_size"),
                    "proxy_val_data_source": eval_metrics.get("data_source"),
                    "proxy_val_split": eval_metrics.get("split"),
                    "proxy_val_checkpoint_source": eval_metrics.get("checkpoint_source"),
                    "proxy_val_eval_seed": eval_metrics.get("eval_seed"),
                    "proxy_val_eval_file_count": eval_metrics.get("eval_file_count"),
                    "val_loss": eval_metrics.get("val_loss"),
                    "val_ppl": eval_metrics.get("val_ppl"),
                    "val_lm_loss": eval_metrics.get("val_lm_loss"),
                    "val_router_aux_loss": eval_metrics.get("val_router_aux_loss"),
                    "val_router_entropy": eval_metrics.get("val_router_entropy"),
                    "val_tokens_per_expert": eval_metrics.get("val_tokens_per_expert"),
                    "val_eval_name": eval_metrics.get("eval_name"),
                    "val_eval_scope": eval_metrics.get("eval_scope"),
                    "val_actual_num_batches": eval_metrics.get("actual_num_batches"),
                    "val_actual_num_sequences": eval_metrics.get("actual_num_sequences"),
                    "val_actual_num_tokens": eval_metrics.get("actual_num_tokens"),
                    "val_data_source": eval_metrics.get("data_source"),
                    "val_split": eval_metrics.get("split"),
                }
                append_jsonl(log_path, proxy_eval_record)
                if tb_writer is not None:
                    _tensorboard_log(tb_writer, "proxy_val", global_step, proxy_eval_record)
                current_proxy_metric = float(eval_metrics.get("val_lm_loss", eval_metrics["val_loss"]))
                if not args.disable_best_checkpoint and (
                    best_val_loss is None or current_proxy_metric < best_val_loss
                ):
                    best_val_loss = current_proxy_metric
                    checkpoint_best_proxy_dir.mkdir(parents=True, exist_ok=True)
                    base_model.save_pretrained(checkpoint_best_proxy_dir)
                    alias_behavior = update_checkpoint_best_alias(checkpoint_best_proxy_dir, checkpoint_best_alias_dir)
                    append_jsonl(
                        log_path,
                        {
                            "step": global_step,
                            "event": "checkpoint_best_proxy",
                            "best_proxy_val_lm_loss": best_val_loss,
                            "best_proxy_val_loss": eval_metrics.get("val_loss"),
                            "best_val_loss": eval_metrics.get("val_loss"),
                            "path": str(checkpoint_best_proxy_dir),
                            "checkpoint_best_alias": str(checkpoint_best_alias_dir),
                            "checkpoint_best_alias_behavior": alias_behavior,
                            "best_checkpoint_selection_metric": "proxy_val_lm_loss",
                            "best_checkpoint_eval_name": eval_metrics.get("eval_name"),
                            "best_checkpoint_eval_scope": eval_metrics.get("eval_scope"),
                        },
                    )
            distributed_barrier(dist_ctx)

        if use_moe and global_step % monitor_interval == 0 and dist_ctx.is_main and monitor is not None:
            metrics = monitor.compute_metrics()
            monitor.log(global_step, metrics)
            append_jsonl(
                log_path,
                {
                    "step": global_step,
                    "avg_expert_similarity": metrics["summary"]["avg_expert_similarity"],
                    "expert_collapse": monitor.check_expert_collapse(),
                },
            )
            if tb_writer is not None:
                _tensorboard_log(
                    tb_writer,
                    "monitor",
                    global_step,
                    {
                        "avg_expert_similarity": metrics["summary"]["avg_expert_similarity"],
                        "expert_collapse": monitor.check_expert_collapse(),
                    },
                )

        if global_step % save_interval == 0:
            if dist_ctx.is_main:
                checkpoint_path = save_checkpoint(
                    checkpoint_dir=output_dir / "checkpoints",
                    step=global_step,
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    args=args,
                )
                append_jsonl(log_path, {"step": global_step, "checkpoint": str(checkpoint_path)})
            distributed_barrier(dist_ctx)

    if dist_ctx.is_main and monitor is not None:
        monitor.save(str(output_dir / "expert_metrics.json"))
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()
    cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
