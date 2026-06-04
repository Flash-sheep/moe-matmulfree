#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import StreamingTextDataset, upcycle_dense_to_moe
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, evaluate, precision_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate exact full-shared no-residual identity against dense baseline.")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-indices", type=str, default="12-23")
    parser.add_argument("--shared-width", type=int, default=2816)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_layer_indices(spec: str) -> list[int]:
    result: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(int(part))
    if not result:
        raise ValueError("No layer indices parsed from --layer-indices.")
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_loader(args: argparse.Namespace) -> tuple[DataLoader, dict]:
    dataset = StreamingTextDataset(
        data_source=args.data_source,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        split="validation",
        text_field=args.text_field,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_streaming_batch)
    return loader, dataset.get_manifest()


def clone_dense_model(checkpoint_path: str, device: torch.device) -> HGRNBitForCausalLM:
    model = HGRNBitForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    model.eval()
    return model.to(device)


def clone_dense_model_cpu(checkpoint_path: str) -> HGRNBitForCausalLM:
    model = HGRNBitForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    model.eval()
    return model


def build_eval_args(args: argparse.Namespace, manifest: dict, checkpoint_source: str, eval_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        precision=args.precision,
        max_eval_batches=None,
        use_moe=False,
        batch_size=args.batch_size,
        max_val_samples=args.max_samples,
        data_source=args.data_source,
        split="validation",
        checkpoint_source=checkpoint_source,
        eval_seed=args.seed,
        eval_file_list=manifest.get("first_20_files"),
        eval_file_count=manifest.get("file_count"),
        eval_name=eval_name,
        eval_scope=f"{args.max_samples}seq",
    )


def compare_first_batch(
    dense_model: HGRNBitForCausalLM,
    shared_model: HGRNBitForCausalLM,
    dataloader: DataLoader,
    args: argparse.Namespace,
) -> dict:
    precision_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(args.device_obj)
    with torch.no_grad():
        with precision_context(precision_dtype):
            dense_outputs = dense_model(input_ids=input_ids, labels=input_ids, return_dict=True)
            shared_outputs = shared_model(input_ids=input_ids, labels=input_ids, return_dict=True)
    logit_diff = (dense_outputs.logits.float() - shared_outputs.logits.float()).abs()
    return {
        "batch_shape": list(input_ids.shape),
        "dense_loss": float(dense_outputs.loss.detach().cpu()),
        "shared_loss": float(shared_outputs.loss.detach().cpu()),
        "loss_delta": float((shared_outputs.loss - dense_outputs.loss).detach().cpu()),
        "logits_max_abs_diff": float(logit_diff.max().item()),
        "logits_mean_abs_diff": float(logit_diff.mean().item()),
        "logits_rms_diff": float(logit_diff.square().mean().sqrt().item()),
    }


def main() -> None:
    args = parse_args()
    args.layer_indices = parse_layer_indices(args.layer_indices)
    args.device_obj = ensure_cuda_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataloader, manifest = build_loader(args)
    write_json(
        args.output_dir / "run_spec.json",
        {
            "checkpoint_path": args.checkpoint_path,
            "data_source": args.data_source,
            "tokenizer_path": args.tokenizer_path,
            "layer_indices": args.layer_indices,
            "shared_width": args.shared_width,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "precision": args.precision,
            "seed": args.seed,
            "identity_semantics": {
                "moe_arch": "shared_residual",
                "enable_sparse_residual": False,
                "shared_init_requested": "dense_full_copy",
                "shared_init_effective": "dense_prefix",
                "equivalence_reason": "shared_width equals full dense intermediate size, so dense_prefix copies all channels in order",
                "skip_param_budget_resolver": True,
            },
        },
    )
    write_json(args.output_dir / "dataset_manifest_val.json", manifest)

    dense_model = clone_dense_model(args.checkpoint_path, args.device_obj)
    shared_model = clone_dense_model_cpu(args.checkpoint_path)
    upcycle_dense_to_moe(
        shared_model,
        moe_layer_indices=args.layer_indices,
        num_experts=0,
        num_experts_per_tok=0,
        use_quantized_experts=True,
        moe_arch="shared_residual",
        enable_sparse_residual=False,
        nominal_shared_width=args.shared_width,
        auto_resolve_shared_width=False,
        strict_total_param_fair=False,
        shared_init="dense_prefix",
        sparse_init="random_ternary_matched",
        sparse_expert_width=0,
        sparse_top_k=0,
        residual_scale_init=0.0,
        residual_scale_learnable=False,
        residual_scale_max=0.5,
        skip_param_budget_resolver=True,
    )
    shared_model = shared_model.to(args.device_obj)
    shared_model.eval()

    diff_payload = compare_first_batch(dense_model, shared_model, dataloader, args)
    write_json(args.output_dir / "first_batch_identity_diff.json", diff_payload)

    dense_metrics = evaluate(
        dense_model,
        dataloader,
        args.device_obj,
        build_eval_args(args, manifest, args.checkpoint_path, "dense_baseline_identity_ref"),
    )
    shared_metrics = evaluate(
        shared_model,
        dataloader,
        args.device_obj,
        build_eval_args(args, manifest, args.checkpoint_path, "full_shared_no_residual_identity"),
    )
    write_json(args.output_dir / "dense_eval_results_1024.json", dense_metrics)
    write_json(args.output_dir / "full_shared_no_residual_eval_results_1024.json", shared_metrics)
    summary = {
        "dense_val_ppl": float(dense_metrics["val_ppl"]),
        "shared_val_ppl": float(shared_metrics["val_ppl"]),
        "delta_ppl": float(shared_metrics["val_ppl"] - dense_metrics["val_ppl"]),
        "dense_val_lm_loss": float(dense_metrics["val_lm_loss"]),
        "shared_val_lm_loss": float(shared_metrics["val_lm_loss"]),
        "delta_lm_loss": float(shared_metrics["val_lm_loss"] - dense_metrics["val_lm_loss"]),
        "first_batch_identity_diff": diff_payload,
    }
    write_json(args.output_dir / "identity_summary.json", summary)


if __name__ == "__main__":
    main()
