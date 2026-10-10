#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterator, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.upcycling import StreamingTextDataset
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import (
    append_jsonl,
    ensure_cuda_device,
    get_precision_dtype,
    json_default,
    precision_context,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained Hugging Face causal LM on the current SlimPajama setup and evaluate with the same 1024-seq PPL protocol."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--val-data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-scheduler-type", choices=["linear", "cosine", "constant"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--eval-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--max-eval-batches", type=int, default=64)
    parser.add_argument("--max-val-samples", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-best-checkpoint", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def repeat_dataloader(dataloader: DataLoader) -> Iterator[Dict[str, torch.Tensor]]:
    while True:
        for batch in dataloader:
            yield batch


def build_streaming_loader(
    data_source: str,
    tokenizer_path: str,
    max_length: int,
    batch_size: int,
    text_field: str,
    split: str,
    max_samples: Optional[int] = None,
):
    dataset = StreamingTextDataset(
        data_source=data_source,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        split=split,
        text_field=text_field,
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_streaming_batch,
    )
    return loader, dataset.get_manifest()


def evaluate_causal_lm(
    model,
    val_loader: DataLoader,
    device: torch.device,
    args: SimpleNamespace,
) -> Dict[str, float]:
    model.eval()
    precision_dtype, _ = get_precision_dtype(args)
    total_loss = 0.0
    total_batches = 0
    total_sequences = 0
    total_tokens = 0
    for batch_index, batch in enumerate(val_loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        total_sequences += int(input_ids.shape[0])
        total_tokens += int(input_ids.numel())
        with torch.no_grad():
            with precision_context(precision_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
        total_loss += float(outputs.loss.detach().cpu())
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Validation loader produced zero batches.")

    avg_loss = total_loss / total_batches
    eval_name = getattr(args, "eval_name", "eval")
    eval_scope = getattr(args, "eval_scope", None)
    if eval_scope is None:
        eval_scope = f"proxy_{total_sequences}seq" if eval_name == "proxy_val" else f"{total_sequences}seq"
    model.train()
    return {
        "eval_name": eval_name,
        "eval_scope": eval_scope,
        "val_loss": avg_loss,
        "val_lm_loss": avg_loss,
        "val_ppl": float(math.exp(min(avg_loss, 20.0))),
        "batch_size": getattr(args, "batch_size", None),
        "max_eval_batches": getattr(args, "max_eval_batches", None),
        "max_val_samples": getattr(args, "max_val_samples", None),
        "actual_num_batches": total_batches,
        "actual_num_sequences": total_sequences,
        "actual_num_tokens": total_tokens,
        "data_source": getattr(args, "data_source", None),
        "split": getattr(args, "split", None),
        "checkpoint_source": getattr(args, "checkpoint_source", None),
        "eval_seed": getattr(args, "eval_seed", None),
        "eval_file_list": getattr(args, "eval_file_list", None),
        "eval_file_count": getattr(args, "eval_file_count", None),
    }


def save_model_checkpoint(model, tokenizer, checkpoint_dir: Path, step: int) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "checkpoint_meta.json").write_text(
        json.dumps({"step": step}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = ensure_cuda_device(args.device)
    tokenizer_path = args.tokenizer_path or args.model_path

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    run_args_path = args.output_dir / "run_args.json"
    run_args_path.write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_loader, train_manifest = build_streaming_loader(
        data_source=args.data_source,
        tokenizer_path=tokenizer_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        text_field=args.text_field,
        split="train",
        max_samples=None,
    )
    val_loader, val_manifest = build_streaming_loader(
        data_source=args.val_data_source,
        tokenizer_path=tokenizer_path,
        max_length=args.max_length,
        batch_size=args.eval_batch_size,
        text_field=args.text_field,
        split="validation",
        max_samples=args.max_val_samples,
    )
    (args.output_dir / "dataset_manifest_train.json").write_text(
        json.dumps(train_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "dataset_manifest_val.json").write_text(
        json.dumps(val_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    precision_dtype, scaler = get_precision_dtype(args)
    model_kwargs = {"trust_remote_code": args.trust_remote_code}
    if precision_dtype is not None:
        model_kwargs["torch_dtype"] = precision_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs).to(device)
    if getattr(model.config, "use_cache", None) is not None:
        model.config.use_cache = False
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    train_iter = repeat_dataloader(train_loader)
    best_proxy_val = None
    tokens_seen = 0
    start_time = time.time()

    checkpoint_last_dir = args.output_dir / "checkpoint_last"
    checkpoint_best_proxy_dir = args.output_dir / "checkpoint_best_proxy"

    append_jsonl(log_path, {
        "event": "start",
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": args.precision,
        "data_source": args.data_source,
        "val_data_source": args.val_data_source,
    })

    for global_step in range(1, args.max_steps + 1):
        step_loss_sum = 0.0
        step_microbatch_count = 0
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.gradient_accumulation_steps):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            with precision_context(precision_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    use_cache=False,
                    return_dict=True,
                )
                raw_loss = outputs.loss
                loss = raw_loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            step_loss_sum += float(raw_loss.detach().cpu())
            step_microbatch_count += 1
            tokens_seen += int(input_ids.numel())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step_loss_avg = step_loss_sum / max(step_microbatch_count, 1)
        lr = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - start_time

        if global_step == 1 or global_step % args.log_interval == 0:
            append_jsonl(log_path, {
                "step": global_step,
                "train_loss": step_loss_avg,
                "total_loss": step_loss_avg,
                "normalized_total_loss": step_loss_avg,
                "normalized_lm_loss": step_loss_avg,
                "lm_loss": step_loss_avg,
                "raw_accumulated_total_loss": step_loss_sum,
                "grad_norm": float(grad_norm),
                "grad_norm_last": float(grad_norm),
                "lr": lr,
                "elapsed_time": elapsed,
                "tokens_seen": tokens_seen,
            })

        if global_step % args.eval_interval == 0 or global_step == args.max_steps:
            eval_args = SimpleNamespace(
                precision=args.precision,
                max_eval_batches=args.max_eval_batches,
                max_val_samples=args.max_val_samples,
                batch_size=args.eval_batch_size,
                data_source=args.val_data_source,
                split="validation",
                checkpoint_source=str(checkpoint_last_dir if checkpoint_last_dir.exists() else args.model_path),
                eval_seed=args.seed,
                eval_name="proxy_val",
                eval_scope=None,
                eval_file_list=val_manifest.get("first_20_files"),
                eval_file_count=val_manifest.get("file_count"),
            )
            eval_metrics = evaluate_causal_lm(model, val_loader, device, eval_args)
            record = {
                "step": global_step,
                "proxy_val_loss": eval_metrics["val_loss"],
                "proxy_val_lm_loss": eval_metrics["val_lm_loss"],
                "proxy_val_ppl": eval_metrics["val_ppl"],
                "proxy_val_actual_num_sequences": eval_metrics["actual_num_sequences"],
                "proxy_val_actual_num_batches": eval_metrics["actual_num_batches"],
                "proxy_val_actual_num_tokens": eval_metrics["actual_num_tokens"],
                "proxy_val_scope": eval_metrics["eval_scope"],
                "proxy_val_max_eval_batches": eval_metrics["max_eval_batches"],
                "proxy_val_max_val_samples": eval_metrics["max_val_samples"],
                "val_loss": eval_metrics["val_loss"],
                "val_lm_loss": eval_metrics["val_lm_loss"],
                "val_ppl": eval_metrics["val_ppl"],
                "val_eval_name": "proxy_val",
                "val_actual_num_sequences": eval_metrics["actual_num_sequences"],
            }
            append_jsonl(log_path, record)

            current_proxy = float(eval_metrics["val_lm_loss"])
            if not args.disable_best_checkpoint and (best_proxy_val is None or current_proxy < best_proxy_val):
                best_proxy_val = current_proxy
                save_model_checkpoint(model, tokenizer, checkpoint_best_proxy_dir, global_step)
                append_jsonl(log_path, {
                    "event": "checkpoint_best_proxy",
                    "step": global_step,
                    "best_proxy_val_lm_loss": current_proxy,
                    "path": str(checkpoint_best_proxy_dir),
                })

        if global_step % args.save_interval == 0 or global_step == args.max_steps:
            save_model_checkpoint(model, tokenizer, checkpoint_last_dir, global_step)
            append_jsonl(log_path, {
                "event": "checkpoint_last",
                "step": global_step,
                "path": str(checkpoint_last_dir),
            })

    eval64_args = SimpleNamespace(
        precision=args.precision,
        max_eval_batches=args.max_eval_batches,
        max_val_samples=None,
        batch_size=args.eval_batch_size,
        data_source=args.val_data_source,
        split="validation",
        checkpoint_source=str(checkpoint_best_proxy_dir if checkpoint_best_proxy_dir.exists() else checkpoint_last_dir),
        eval_seed=args.seed,
        eval_name="formal_eval_64",
        eval_scope="64batch_proxy_style",
        eval_file_list=val_manifest.get("first_20_files"),
        eval_file_count=val_manifest.get("file_count"),
    )
    eval1024_args = SimpleNamespace(
        precision=args.precision,
        max_eval_batches=None,
        max_val_samples=1024,
        batch_size=args.eval_batch_size,
        data_source=args.val_data_source,
        split="validation",
        checkpoint_source=str(checkpoint_best_proxy_dir if checkpoint_best_proxy_dir.exists() else checkpoint_last_dir),
        eval_seed=args.seed,
        eval_name="formal_eval_1024",
        eval_scope="1024seq",
        eval_file_list=val_manifest.get("first_20_files"),
        eval_file_count=val_manifest.get("file_count"),
    )

    best_eval_model_path = checkpoint_best_proxy_dir if checkpoint_best_proxy_dir.exists() else checkpoint_last_dir
    best_eval_model = AutoModelForCausalLM.from_pretrained(
        best_eval_model_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=precision_dtype if precision_dtype is not None else None,
    ).to(device)
    if getattr(best_eval_model.config, "use_cache", None) is not None:
        best_eval_model.config.use_cache = False

    eval64 = evaluate_causal_lm(best_eval_model, val_loader, device, eval64_args)
    (args.output_dir / "eval_results_64.json").write_text(
        json.dumps(eval64, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    eval1024 = evaluate_causal_lm(best_eval_model, val_loader, device, eval1024_args)
    (args.output_dir / "eval_results_1024.json").write_text(
        json.dumps(eval1024, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "model_path": args.model_path,
        "tokenizer_path": tokenizer_path,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "precision": args.precision,
        "data_source": args.data_source,
        "val_data_source": args.val_data_source,
        "checkpoint_best_proxy_path": str(checkpoint_best_proxy_dir) if checkpoint_best_proxy_dir.exists() else None,
        "checkpoint_last_path": str(checkpoint_last_dir) if checkpoint_last_dir.exists() else None,
        "best_proxy_val_lm_loss": best_proxy_val,
        "formal_eval_64": eval64,
        "formal_eval_1024": eval1024,
        "dataset_manifests": {
            "train": str(args.output_dir / "dataset_manifest_train.json"),
            "val": str(args.output_dir / "dataset_manifest_val.json"),
        },
    }
    (args.output_dir / "training_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
