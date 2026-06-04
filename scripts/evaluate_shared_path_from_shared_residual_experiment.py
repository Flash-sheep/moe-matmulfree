#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import StreamingTextDataset, upcycle_dense_to_moe
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an untrained shared-only path from a shared-residual experiment spec and run formal eval."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--source-output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_loader(args: argparse.Namespace) -> tuple[DataLoader, dict[str, Any]]:
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


def build_eval_args(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    checkpoint_source: str,
    eval_name: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        precision=args.precision,
        max_eval_batches=None,
        use_moe=True,
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


def clone_dense_model(checkpoint_path: str, device: torch.device) -> HGRNBitForCausalLM:
    model = HGRNBitForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    model.eval()
    return model.to(device)


def extract_resolved_shared_width(source_output_dir: Path) -> int:
    budget_path = source_output_dir / "parameter_budget_verification.json"
    if budget_path.exists():
        payload = load_json(budget_path)
        width = payload.get("resolved_shared_width")
        if width is not None:
            return int(width)
    report_path = source_output_dir / "training_report.json"
    if report_path.exists():
        payload = load_json(report_path)
        width = (
            payload.get("parameter_budget_verification", {}) or {}
        ).get("resolved_shared_width")
        if width is not None:
            return int(width)
    raise FileNotFoundError(f"Could not find resolved_shared_width in {source_output_dir}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = ensure_cuda_device(args.device)

    cfg = load_json(args.config_path)
    moe_cfg = cfg.get("moe", {})
    if moe_cfg.get("moe_arch") != "shared_residual":
        raise ValueError("This script expects a shared_residual config.")
    shared_init = str(moe_cfg.get("shared_init", "dense_top_channel"))
    if shared_init != "dense_top_channel":
        raise ValueError(f"Expected shared_init=dense_top_channel, got {shared_init}")

    resolved_shared_width = extract_resolved_shared_width(args.source_output_dir)
    layer_indices = list(moe_cfg.get("layer_indices", []))
    if not layer_indices:
        raise ValueError("Missing moe.layer_indices in config.")

    dataloader, manifest = build_loader(args)
    dense_model = clone_dense_model(args.checkpoint_path, device)

    shared_only_model = HGRNBitForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16)
    shared_only_model.eval()
    upcycle_dense_to_moe(
        shared_only_model,
        moe_layer_indices=layer_indices,
        num_experts=0,
        num_experts_per_tok=0,
        use_quantized_experts=bool(moe_cfg.get("use_quantized_experts", True)),
        moe_arch="shared_residual",
        enable_sparse_residual=False,
        nominal_shared_width=resolved_shared_width,
        auto_resolve_shared_width=False,
        min_shared_width=resolved_shared_width,
        shared_width_step=1,
        strict_total_param_fair=False,
        shared_init=shared_init,
        sparse_init="random_ternary_matched",
        sparse_expert_width=0,
        sparse_top_k=0,
        residual_scale_init=0.0,
        residual_scale_learnable=False,
        residual_scale_max=float(moe_cfg.get("residual_scale_max", 0.5)),
        router_aux_loss_coef=0.0,
        router_bias=False,
        router_jitter_noise=0.0,
        normalize_topk_prob=True,
        skip_param_budget_resolver=True,
    )
    shared_only_model = shared_only_model.to(device)
    shared_only_model.eval()

    dense_metrics = evaluate(
        dense_model,
        dataloader,
        device,
        build_eval_args(args, manifest, args.checkpoint_path, "dense_baseline_reference"),
    )
    shared_metrics = evaluate(
        shared_only_model,
        dataloader,
        device,
        build_eval_args(
            args,
            manifest,
            args.checkpoint_path,
            f"shared_only_from_dense_ref_{cfg.get('experiment_name', 'unknown')}",
        ),
    )

    source_eval_path = args.source_output_dir / "eval_results_1024.json"
    source_eval = load_json(source_eval_path) if source_eval_path.exists() else None
    parameter_budget = (
        getattr(shared_only_model.config, "moe_parameter_budget_verification", None) or {}
    )

    write_json(args.output_dir / "dataset_manifest_val.json", manifest)
    write_json(args.output_dir / "dense_eval_results_1024.json", dense_metrics)
    write_json(args.output_dir / "shared_only_from_dense_eval_results_1024.json", shared_metrics)
    write_json(
        args.output_dir / "run_spec.json",
        {
            "config_path": str(args.config_path),
            "source_output_dir": str(args.source_output_dir),
            "checkpoint_path": args.checkpoint_path,
            "data_source": args.data_source,
            "tokenizer_path": args.tokenizer_path,
            "resolved_shared_width": resolved_shared_width,
            "shared_init": shared_init,
            "layer_indices": layer_indices,
            "evaluation_semantics": {
                "built_from_trained_checkpoint": False,
                "built_from_dense_checkpoint": True,
                "enable_sparse_residual": False,
                "router_removed": True,
                "sparse_experts_removed": True,
                "shared_width_preserved_from_source_experiment": True,
            },
        },
    )
    write_json(args.output_dir / "parameter_budget_verification.json", parameter_budget)
    write_json(
        args.output_dir / "comparison_summary.json",
        {
            "source_experiment_name": cfg.get("experiment_name"),
            "source_experiment_eval_1024": source_eval,
            "dense_baseline_eval_1024": dense_metrics,
            "shared_only_from_dense_eval_1024": shared_metrics,
            "delta_vs_dense": {
                "ppl": float(shared_metrics["val_ppl"] - dense_metrics["val_ppl"]),
                "lm_loss": float(shared_metrics["val_lm_loss"] - dense_metrics["val_lm_loss"]),
            },
            "delta_vs_source_trained_shared_residual": None
            if source_eval is None
            else {
                "ppl": float(shared_metrics["val_ppl"] - source_eval["val_ppl"]),
                "lm_loss": float(shared_metrics["val_lm_loss"] - source_eval["val_lm_loss"]),
            },
        },
    )


if __name__ == "__main__":
    main()
