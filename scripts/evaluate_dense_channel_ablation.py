#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.modules.moe import compute_dense_channel_importance
from mmfreelm.upcycling import StreamingTextDataset
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-only dense FFN channel oracle ablation.")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-indices", type=str, default="12-23")
    parser.add_argument("--remove-count", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_layer_indices(spec: str) -> List[int]:
    result: List[int] = []
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


def build_loader(args: argparse.Namespace) -> tuple[DataLoader, Dict[str, object]]:
    dataset = StreamingTextDataset(
        data_source=args.data_source,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        split="validation",
        text_field=args.text_field,
        max_samples=args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True, collate_fn=collate_streaming_batch)
    return loader, dataset.get_manifest()


def clone_dense_model(checkpoint_path: str, device: torch.device) -> HGRNBitForCausalLM:
    model = HGRNBitForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.bfloat16)
    model.config.use_moe = False
    model.config.moe_layer_indices = []
    model.eval()
    return model.to(device)


def choose_channel_indices(importance: torch.Tensor, remove_count: int, strategy: str, seed: int, layer_idx: int) -> torch.Tensor:
    intermediate_size = int(importance.numel())
    if not (0 < remove_count < intermediate_size):
        raise ValueError(f"remove_count must be in (0, {intermediate_size}), got {remove_count}.")
    if strategy == "lowest":
        return torch.topk(importance, k=remove_count, largest=False).indices
    if strategy == "highest":
        return torch.topk(importance, k=remove_count, largest=True).indices
    if strategy == "random":
        rng = random.Random(seed + layer_idx)
        all_indices = list(range(intermediate_size))
        chosen = rng.sample(all_indices, remove_count)
        return torch.tensor(chosen, dtype=torch.long)
    raise ValueError(f"Unsupported strategy: {strategy}")


def apply_channel_ablation(model: HGRNBitForCausalLM, layer_indices: List[int], remove_count: int, strategy: str, seed: int) -> Dict[str, object]:
    layer_payload: Dict[str, object] = {}
    with torch.no_grad():
        for layer_idx in layer_indices:
            mlp = model.model.layers[layer_idx].mlp
            gate_weight = mlp.gate_proj.weight
            down_weight = mlp.down_proj.weight
            intermediate_size = down_weight.shape[1]
            importance = compute_dense_channel_importance(mlp).detach().float().cpu()
            chosen = choose_channel_indices(importance, remove_count=remove_count, strategy=strategy, seed=seed, layer_idx=layer_idx)
            chosen = torch.sort(chosen.detach().long().cpu()).values
            gate_rows = chosen
            up_rows = chosen + int(intermediate_size)
            gate_weight[gate_rows] = 0
            gate_weight[up_rows] = 0
            down_weight[:, chosen.to(down_weight.device)] = 0
            chosen_importance = importance.index_select(0, chosen)
            layer_payload[f"layer_{layer_idx}"] = {
                "removed_channel_indices": [int(v) for v in chosen.tolist()],
                "removed_channel_count": int(chosen.numel()),
                "removed_importance_mean": float(chosen_importance.mean().item()),
                "removed_importance_min": float(chosen_importance.min().item()),
                "removed_importance_max": float(chosen_importance.max().item()),
                "global_importance_mean": float(importance.mean().item()),
                "global_importance_min": float(importance.min().item()),
                "global_importance_max": float(importance.max().item()),
            }
    return layer_payload


def run_single_eval(
    model: HGRNBitForCausalLM,
    dataloader: DataLoader,
    manifest: Dict[str, object],
    args: argparse.Namespace,
    eval_name: str,
) -> Dict[str, object]:
    eval_args = SimpleNamespace(
        precision=args.precision,
        max_eval_batches=None,
        use_moe=False,
        batch_size=args.batch_size,
        max_val_samples=args.max_samples,
        data_source=args.data_source,
        split="validation",
        checkpoint_source=args.checkpoint_path,
        eval_seed=args.seed,
        eval_file_list=manifest.get("all_files"),
        eval_file_count=manifest.get("file_count"),
        eval_name=eval_name,
        eval_scope=f"{args.max_samples}seq",
    )
    return evaluate(model, dataloader, args.device_obj, eval_args)


def main() -> None:
    args = parse_args()
    args.device_obj = ensure_cuda_device(args.device)
    layer_indices = parse_layer_indices(args.layer_indices)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataloader, manifest = build_loader(args)
    write_json(
        args.output_dir / "run_spec.json",
        {
            "checkpoint_path": args.checkpoint_path,
            "data_source": args.data_source,
            "tokenizer_path": args.tokenizer_path,
            "layer_indices": layer_indices,
            "remove_count": args.remove_count,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_samples": args.max_samples,
            "precision": args.precision,
            "seed": args.seed,
        },
    )
    write_json(args.output_dir / "dataset_manifest_val.json", manifest)

    results: Dict[str, Dict[str, object]] = {}

    baseline_model = clone_dense_model(args.checkpoint_path, args.device_obj)
    baseline_metrics = run_single_eval(baseline_model, dataloader, manifest, args, eval_name="dense_baseline")
    results["baseline"] = {
        "metrics": baseline_metrics,
        "ablation": {
            "strategy": "none",
            "removed_channel_count_per_layer": 0,
            "layer_details": {},
        },
    }
    write_json(args.output_dir / "baseline_eval_results_1024.json", baseline_metrics)
    del baseline_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for strategy in ("lowest", "random", "highest"):
        model = clone_dense_model(args.checkpoint_path, args.device_obj)
        layer_details = apply_channel_ablation(
            model,
            layer_indices=layer_indices,
            remove_count=args.remove_count,
            strategy=strategy,
            seed=args.seed,
        )
        metrics = run_single_eval(model, dataloader, manifest, args, eval_name=f"remove_{args.remove_count}_{strategy}")
        payload = {
            "metrics": metrics,
            "ablation": {
                "strategy": strategy,
                "removed_channel_count_per_layer": args.remove_count,
                "layer_details": layer_details,
            },
        }
        results[strategy] = payload
        write_json(args.output_dir / f"remove_{args.remove_count}_{strategy}_eval_results_1024.json", payload)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_rows = []
    baseline_ppl = float(results["baseline"]["metrics"]["val_ppl"])
    for key in ("baseline", "lowest", "random", "highest"):
        metrics = results[key]["metrics"]
        ppl = float(metrics["val_ppl"])
        summary_rows.append(
            {
                "name": key,
                "val_ppl": ppl,
                "val_lm_loss": float(metrics["val_lm_loss"]),
                "delta_ppl_vs_baseline": ppl - baseline_ppl,
            }
        )
    summary = {
        "baseline_checkpoint": args.checkpoint_path,
        "remove_count_per_layer": args.remove_count,
        "layer_indices": layer_indices,
        "results": summary_rows,
    }
    write_json(args.output_dir / "ablation_summary.json", summary)


if __name__ == "__main__":
    main()
