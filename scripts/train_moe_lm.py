#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
from itertools import cycle
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train dense/MoE/Ternary-expert HGRNBit language models.")
    parser.add_argument("--config-file", type=Path, help="Optional JSON config file.")
    parser.add_argument("--train-data", type=Path, required=True, help="Training data path: .txt, .jsonl, directory, or .pt")
    parser.add_argument("--val-data", type=Path, required=True, help="Validation data path: .txt, .jsonl, directory, or .pt")
    parser.add_argument("--text-field", default="text", help="Field name for JSONL/JSON text data.")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer path or Hugging Face name.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for logs and checkpoints.")
    parser.add_argument("--resume-from", type=Path, help="Checkpoint directory to resume from.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--disable-fused-cross-entropy", action="store_true")

    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-hidden-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--hidden-ratio", type=int, default=4)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--max-position-embeddings", type=int, default=2048)
    parser.add_argument("--rms-norm-eps", type=float, default=1e-6)
    parser.add_argument("--initializer-range", type=float, default=0.02)

    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-num-experts-per-tok", type=int, default=2)
    parser.add_argument("--moe-router-aux-loss-coef", type=float, default=1e-2)
    parser.add_argument("--moe-router-jitter-noise", type=float, default=0.0)
    parser.add_argument("--moe-router-bias", action="store_true")
    parser.add_argument("--moe-normalize-topk-prob", action="store_true", default=True)
    parser.add_argument("--moe-no-normalize-topk-prob", action="store_false", dest="moe_normalize_topk_prob")
    parser.add_argument("--moe-output-router-logits", action="store_true")
    parser.add_argument("--moe-use-quantized-experts", action="store_true")

    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr-scheduler-type", default="cosine", choices=["linear", "cosine", "constant"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--add-eos-token", action="store_true")
    parser.add_argument("--disable-shuffle", action="store_true")
    return parser.parse_args()


def merge_config_file(args: argparse.Namespace) -> argparse.Namespace:
    if args.config_file is None:
        return args
    config_data = json.loads(args.config_file.read_text(encoding="utf-8"))
    for key, value in config_data.items():
        normalized_key = key.replace("-", "_")
        if hasattr(args, normalized_key):
            setattr(args, normalized_key, value)
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_cuda_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise RuntimeError(
            "This repository currently relies on Triton CUDA kernels in the model path. "
            "Use a CUDA device for real training."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the current environment.")
    return resolved


def load_text_samples(path: Path, text_field: str) -> List[str]:
    if path.is_dir():
        samples: List[str] = []
        for child in sorted(path.rglob("*")):
            if child.is_file():
                samples.extend(load_text_samples(child, text_field))
        return samples

    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        return [path.read_text(encoding="utf-8")]

    if suffix in {".jsonl", ".json"}:
        records: List[str] = []
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if text_field not in obj:
                    raise KeyError(f"Missing text field `{text_field}` in {path}")
                records.append(str(obj[text_field]))
        else:
            obj = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                for item in obj:
                    if text_field not in item:
                        raise KeyError(f"Missing text field `{text_field}` in {path}")
                    records.append(str(item[text_field]))
            elif isinstance(obj, dict):
                if text_field not in obj:
                    raise KeyError(f"Missing text field `{text_field}` in {path}")
                records.append(str(obj[text_field]))
            else:
                raise TypeError(f"Unsupported JSON structure in {path}")
        return records

    raise ValueError(f"Unsupported text dataset format: {path}")


def load_token_ids(path: Path) -> List[int]:
    if path.suffix.lower() != ".pt":
        raise ValueError(f"Unsupported tokenized dataset format: {path}")
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload.view(-1).tolist()
    if isinstance(payload, list):
        if payload and isinstance(payload[0], int):
            return payload
        if payload and isinstance(payload[0], list):
            flattened: List[int] = []
            for row in payload:
                flattened.extend(int(token) for token in row)
            return flattened
    if isinstance(payload, dict):
        if "input_ids" in payload:
            return load_token_ids_from_object(payload["input_ids"])
    raise TypeError(f"Unsupported tokenized payload in {path}")


def load_token_ids_from_object(obj) -> List[int]:
    if isinstance(obj, torch.Tensor):
        return obj.view(-1).tolist()
    if isinstance(obj, list):
        if obj and isinstance(obj[0], int):
            return obj
        flattened: List[int] = []
        for item in obj:
            flattened.extend(load_token_ids_from_object(item))
        return flattened
    raise TypeError("Unsupported token object for `input_ids`.")


def tokenize_samples(samples: Sequence[str], tokenizer, add_eos_token: bool) -> List[int]:
    token_ids: List[int] = []
    eos_token_id = tokenizer.eos_token_id
    for sample in samples:
        encoded = tokenizer(sample, add_special_tokens=False)["input_ids"]
        token_ids.extend(encoded)
        if add_eos_token and eos_token_id is not None:
            token_ids.append(eos_token_id)
    return token_ids


def load_data_stream(path: Path, tokenizer, text_field: str, add_eos_token: bool) -> List[int]:
    if path.suffix.lower() == ".pt":
        return load_token_ids(path)
    samples = load_text_samples(path, text_field=text_field)
    return tokenize_samples(samples, tokenizer=tokenizer, add_eos_token=add_eos_token)


class PackedTokenDataset(Dataset):
    def __init__(self, token_ids: Sequence[int], seq_len: int) -> None:
        if len(token_ids) < seq_len + 1:
            raise ValueError("Not enough tokens to build a sequence dataset.")
        usable_tokens = (len(token_ids) // seq_len) * seq_len
        token_tensor = torch.tensor(token_ids[:usable_tokens], dtype=torch.long)
        self.blocks = token_tensor.view(-1, seq_len)

    def __len__(self) -> int:
        return self.blocks.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.blocks[index]


def collate_input_ids(batch: Sequence[torch.Tensor]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack(batch, dim=0)
    return {"input_ids": input_ids}


def build_dataloaders(args: argparse.Namespace, tokenizer) -> Tuple[DataLoader, DataLoader]:
    train_tokens = load_data_stream(args.train_data, tokenizer, args.text_field, args.add_eos_token)
    val_tokens = load_data_stream(args.val_data, tokenizer, args.text_field, args.add_eos_token)

    train_dataset = PackedTokenDataset(train_tokens, seq_len=args.seq_len)
    val_dataset = PackedTokenDataset(val_tokens, seq_len=args.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=not args.disable_shuffle,
        num_workers=args.num_workers,
        collate_fn=collate_input_ids,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_input_ids,
        drop_last=False,
    )
    return train_loader, val_loader


def build_model_config(args: argparse.Namespace, tokenizer) -> HGRNBitConfig:
    vocab_size = args.vocab_size or len(tokenizer)
    return HGRNBitConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_heads=args.num_heads,
        hidden_ratio=args.hidden_ratio,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        rms_norm_eps=args.rms_norm_eps,
        use_moe=args.use_moe,
        moe_num_experts=args.moe_num_experts,
        moe_num_experts_per_tok=args.moe_num_experts_per_tok,
        moe_router_aux_loss_coef=args.moe_router_aux_loss_coef,
        moe_router_jitter_noise=args.moe_router_jitter_noise,
        moe_router_bias=args.moe_router_bias,
        moe_normalize_topk_prob=args.moe_normalize_topk_prob,
        moe_output_router_logits=args.moe_output_router_logits or args.use_moe,
        moe_use_quantized_experts=args.moe_use_quantized_experts,
        initializer_range=args.initializer_range,
        fuse_cross_entropy=not args.disable_fused_cross_entropy,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )


def get_precision_dtype(args: argparse.Namespace):
    if args.precision == "fp16":
        return torch.float16, GradScaler(enabled=True)
    if args.precision == "bf16":
        return torch.bfloat16, GradScaler(enabled=False)
    return None, GradScaler(enabled=False)


def precision_context(precision_dtype):
    if precision_dtype is None:
        return contextlib.nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=precision_dtype)
    return torch.cuda.amp.autocast(dtype=precision_dtype)


def flatten_router_metrics(router_metrics: Optional[Tuple[Dict[str, torch.Tensor], ...]]) -> Dict[str, float]:
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
            mean_values = stacked.mean(dim=0).tolist()
            result[key] = mean_values
        else:
            result[key] = float(stacked.mean().item())
    return result


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.item())
        return value.tolist()
    return value


def append_jsonl(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=json_default) + "\n")


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: HGRNBitForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    args: argparse.Namespace,
) -> Path:
    target = checkpoint_dir / f"step_{step:07d}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "args": vars(args),
        },
        target / "trainer_state.pt",
    )
    return target


def resume_checkpoint(
    checkpoint_dir: Path,
    model: HGRNBitForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> int:
    state_path = checkpoint_dir / "trainer_state.pt"
    state = torch.load(state_path, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    return int(state["step"])


@torch.no_grad()
def evaluate(
    model: HGRNBitForCausalLM,
    val_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    precision_dtype, _ = get_precision_dtype(args)
    total_loss = 0.0
    total_lm_loss = 0.0
    total_router_aux = 0.0
    total_batches = 0
    router_entropy_values: List[float] = []
    tokens_per_expert_values: List[List[float]] = []

    for batch_index, batch in enumerate(val_loader):
        if args.max_eval_batches is not None and batch_index >= args.max_eval_batches:
            break
        input_ids = batch["input_ids"].to(device)
        with precision_context(precision_dtype):
            outputs = model(
                input_ids=input_ids,
                labels=input_ids,
                output_router_logits=args.use_moe,
                return_dict=True,
            )
        total_loss += float(outputs.loss.detach().cpu())
        total_lm_loss += float(outputs.lm_loss.detach().cpu()) if outputs.lm_loss is not None else float(outputs.loss.detach().cpu())
        total_router_aux += float(outputs.router_aux_loss.detach().cpu()) if outputs.router_aux_loss is not None else 0.0
        total_batches += 1
        flat_router = flatten_router_metrics(outputs.router_metrics)
        if "router_entropy" in flat_router:
            router_entropy_values.append(float(flat_router["router_entropy"]))
        if "tokens_per_expert" in flat_router:
            tokens_per_expert_values.append(flat_router["tokens_per_expert"])

    if total_batches == 0:
        raise RuntimeError("Validation loader produced zero batches.")

    avg_loss = total_loss / total_batches
    metrics: Dict[str, float] = {
        "val_loss": avg_loss,
        "val_ppl": float(math.exp(min(avg_loss, 20.0))),
        "val_lm_loss": total_lm_loss / total_batches,
        "val_router_aux_loss": total_router_aux / total_batches,
    }
    if router_entropy_values:
        metrics["val_router_entropy"] = sum(router_entropy_values) / len(router_entropy_values)
    if tokens_per_expert_values:
        expert_count = len(tokens_per_expert_values[0])
        averaged = []
        for expert_idx in range(expert_count):
            averaged.append(sum(values[expert_idx] for values in tokens_per_expert_values) / len(tokens_per_expert_values))
        metrics["val_tokens_per_expert"] = averaged
    model.train()
    return metrics


def main() -> None:
    args = merge_config_file(parse_args())
    set_seed(args.seed)
    device = ensure_cuda_device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    args_snapshot_path = args.output_dir / "run_args.json"
    args_snapshot_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define either `pad_token_id` or `eos_token_id`.")
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = build_dataloaders(args, tokenizer)
    config = build_model_config(args, tokenizer)
    model = HGRNBitForCausalLM(config).to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    precision_dtype, scaler = get_precision_dtype(args)

    global_step = 0
    if args.resume_from is not None:
        global_step = resume_checkpoint(args.resume_from, model, optimizer, scheduler, scaler, device)

    train_iter: Iterator[Dict[str, torch.Tensor]] = cycle(train_loader)
    running_loss = 0.0

    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        last_outputs = None
        for _ in range(args.gradient_accumulation_steps):
            batch = next(train_iter)
            input_ids = batch["input_ids"].to(device)
            with precision_context(precision_dtype):
                outputs = model(
                    input_ids=input_ids,
                    labels=input_ids,
                    output_router_logits=args.use_moe,
                    return_dict=True,
                )
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(outputs.loss.detach().cpu())
            last_outputs = outputs

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        global_step += 1

        flat_router = flatten_router_metrics(last_outputs.router_metrics if last_outputs is not None else None)
        train_record: Dict[str, object] = {
            "step": global_step,
            "train_loss": running_loss / max(args.log_steps, 1),
            "lm_loss": float(last_outputs.lm_loss.detach().cpu()) if last_outputs and last_outputs.lm_loss is not None else None,
            "router_aux_loss": float(last_outputs.router_aux_loss.detach().cpu()) if last_outputs and last_outputs.router_aux_loss is not None else None,
            "grad_norm": float(grad_norm.detach().cpu()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "lr": scheduler.get_last_lr()[0],
        }
        train_record.update(flat_router)

        if global_step % args.log_steps == 0:
            append_jsonl(log_path, train_record)
            print(json.dumps(train_record, ensure_ascii=False, default=json_default))
            running_loss = 0.0

        if global_step % args.eval_steps == 0 or global_step == args.max_steps:
            eval_metrics = evaluate(model, val_loader, device, args)
            eval_record = {"step": global_step, **eval_metrics}
            append_jsonl(log_path, eval_record)
            print(json.dumps(eval_record, ensure_ascii=False, default=json_default))

        if global_step % args.save_steps == 0 or global_step == args.max_steps:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=args.output_dir / "checkpoints",
                step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                args=args,
            )
            append_jsonl(log_path, {"step": global_step, "checkpoint": str(checkpoint_path)})


if __name__ == "__main__":
    main()
