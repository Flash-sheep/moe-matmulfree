#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterator

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import (
    ExpertMonitor,
    StreamingTextDataset,
    apply_freeze_for_upcycling,
    upcycle_dense_to_moe,
)
from scripts.train_moe_lm import (
    append_jsonl,
    ensure_cuda_device,
    evaluate,
    flatten_router_metrics,
    get_precision_dtype,
    json_default,
    precision_context,
    save_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse upcycling training entrypoint.")
    parser.add_argument("--pretrained-path", type=str, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--val-data-source", type=str)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--device", type=str)
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collate_streaming_batch(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)
    labels = torch.stack([item["labels"] for item in batch], dim=0)
    return {"input_ids": input_ids, "labels": labels}


def repeat_dataloader(dataloader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
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
):
    dataset = StreamingTextDataset(
        data_source=data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        split=split,
        text_field=text_field,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_streaming_batch,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config_path)

    training = config.get("training", {})
    moe_cfg = config.get("moe", {})
    freeze_cfg = config.get("freeze", {})

    seed = args.seed if args.seed is not None else config.get("seed", 42)
    precision = args.precision or training.get("precision", "bf16")
    device = ensure_cuda_device(args.device or training.get("device", "cuda"))
    set_seed(seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    append_jsonl(output_dir / "train_log.jsonl", {"event": "start", "config_path": str(args.config_path)})
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer_path = args.tokenizer_path or args.pretrained_path
    model = HGRNBitForCausalLM.from_pretrained(args.pretrained_path, torch_dtype=torch.bfloat16)

    moe_layer_indices = moe_cfg.get("layer_indices", list(range(12, 24)))
    model = upcycle_dense_to_moe(
        model=model,
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
    )
    append_jsonl(
        output_dir / "train_log.jsonl",
        {
            "event": "freeze",
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "moe_layer_indices": moe_layer_indices,
        },
    )

    model.to(device)
    model.train()

    batch_size = args.batch_size or training.get("batch_size", 4)
    max_steps = args.max_steps or training.get("max_steps", 50000)
    grad_accum = args.gradient_accumulation_steps or training.get("gradient_accumulation_steps", 8)
    max_length = training.get("max_length", 2048)
    learning_rate = training.get("learning_rate", 5e-4)
    min_lr = training.get("min_lr")
    min_lr_ratio = training.get("min_lr_ratio")
    weight_decay = training.get("weight_decay", 0.01)
    warmup_steps = training.get("warmup_steps", 1000)
    grad_clip = training.get("grad_clip", 1.0)
    text_field = training.get("text_field", "text")
    log_interval = training.get("log_interval", 50)
    eval_interval = training.get("eval_interval", 500)
    save_interval = training.get("save_interval", 2000)
    monitor_interval = config.get("monitor_interval", 500)
    max_val_samples = training.get("max_val_samples", 1000)

    train_loader = build_streaming_loader(
        data_source=args.data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        batch_size=batch_size,
        text_field=text_field,
        split="train",
    )
    val_loader = None
    if args.val_data_source:
        val_loader = build_streaming_loader(
            data_source=args.val_data_source,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            batch_size=batch_size,
            text_field=text_field,
            split="validation",
            max_samples=max_val_samples,
        )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

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
    monitor = ExpertMonitor(model, moe_layer_indices)

    train_iter: Iterator[Dict[str, torch.Tensor]] = repeat_dataloader(train_loader)
    running_loss = 0.0
    global_step = 0
    best_val_loss = None

    while global_step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        last_outputs = None
        for _ in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            with precision_context(precision_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_router_logits=True,
                    return_dict=True,
                )
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            last_outputs = outputs
            running_loss += float(outputs.loss.detach().cpu())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            grad_clip,
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        if global_step % log_interval == 0 and last_outputs is not None:
            router_metrics = flatten_router_metrics(last_outputs.router_metrics)
            append_jsonl(
                output_dir / "train_log.jsonl",
                {
                    "step": global_step,
                    "train_loss": running_loss / log_interval,
                    "lm_loss": float(last_outputs.lm_loss.detach().cpu()) if last_outputs.lm_loss is not None else None,
                    "router_aux_loss": float(last_outputs.router_aux_loss.detach().cpu()) if last_outputs.router_aux_loss is not None else None,
                    "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
                    "lr": scheduler.get_last_lr()[0],
                    **router_metrics,
                },
            )
            running_loss = 0.0

        if val_loader is not None and global_step % eval_interval == 0:
            eval_args = type(
                "EvalArgs",
                (),
                {
                    "precision": precision,
                    "max_eval_batches": training.get("max_eval_batches", None),
                    "use_moe": True,
                },
            )()
            eval_metrics = evaluate(model, val_loader, device, eval_args)
            append_jsonl(output_dir / "train_log.jsonl", {"step": global_step, **eval_metrics})
            current_val_loss = float(eval_metrics["val_loss"])
            if best_val_loss is None or current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_dir = output_dir / "checkpoint_best"
                best_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_dir)
                append_jsonl(
                    output_dir / "train_log.jsonl",
                    {
                        "step": global_step,
                        "event": "checkpoint_best",
                        "best_val_loss": best_val_loss,
                        "path": str(best_dir),
                    },
                )

        if global_step % monitor_interval == 0:
            metrics = monitor.compute_metrics()
            monitor.log(global_step, metrics)
            append_jsonl(
                output_dir / "train_log.jsonl",
                {
                    "step": global_step,
                    "avg_expert_similarity": metrics["summary"]["avg_expert_similarity"],
                    "expert_collapse": monitor.check_expert_collapse(),
                },
            )

        if global_step % save_interval == 0:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=output_dir / "checkpoints",
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
            )
            append_jsonl(output_dir / "train_log.jsonl", {"step": global_step, "checkpoint": str(checkpoint_path)})

    monitor.save(str(output_dir / "expert_metrics.json"))


if __name__ == "__main__":
    main()
