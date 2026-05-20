#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import ExpertMonitor, StreamingTextDataset
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, evaluate, flatten_router_metrics, get_precision_dtype, precision_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a summary report for sparse upcycling runs.")
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--val-data-source", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--max-eval-batches", type=int, default=16)
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_records(records: List[Dict], predicate) -> List[Dict]:
    return [record for record in records if predicate(record)]


def compute_ternary_ratios(model, moe_layer_indices: List[int]) -> Dict[str, Dict[str, Dict[str, float]]]:
    ratios: Dict[str, Dict[str, Dict[str, float]]] = {}
    for layer_idx in moe_layer_indices:
        layer_key = f"layer_{layer_idx}"
        ratios[layer_key] = {}
        moe = model.model.layers[layer_idx].mlp
        for expert_idx, expert in enumerate(moe.experts):
            expert_key = f"expert_{expert_idx}"
            ratios[layer_key][expert_key] = {}
            for name, param in expert.named_parameters():
                if "weight" not in name:
                    continue
                weight = param.data.detach().float()
                scale = 1.0 / weight.abs().mean().clamp_min(1e-6)
                ternary = (weight * scale).round().clamp(-1, 1)
                total = ternary.numel()
                ratios[layer_key][expert_key][name] = {
                    "neg": float((ternary == -1).sum().item() / total),
                    "zero": float((ternary == 0).sum().item() / total),
                    "pos": float((ternary == 1).sum().item() / total),
                }
    return ratios


@torch.no_grad()
def collect_router_metrics(model, dataloader, device, precision: str) -> Tuple[Dict[str, float], Dict[str, Dict[str, List[float]]]]:
    eval_args = type("EvalArgs", (), {"precision": precision, "max_eval_batches": None, "use_moe": True})()
    aggregate_metrics = evaluate(model, dataloader, device, eval_args)

    per_layer = defaultdict(lambda: {"router_entropy": [], "tokens_per_expert": []})
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    for batch_index, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        with precision_context(precision_dtype):
            outputs = model(input_ids=input_ids, labels=input_ids, output_router_logits=True, return_dict=True)
        if outputs.router_metrics:
            for layer_offset, layer_metrics in enumerate(outputs.router_metrics):
                layer_name = f"layer_{model.config.moe_layer_indices[layer_offset]}"
                entropy = layer_metrics["router_entropy"].detach().float().cpu().item()
                tokens = layer_metrics["tokens_per_expert"].detach().float().cpu().tolist()
                per_layer[layer_name]["router_entropy"].append(entropy)
                per_layer[layer_name]["tokens_per_expert"].append(tokens)
    normalized_per_layer: Dict[str, Dict[str, List[float]]] = {}
    for layer_name, values in per_layer.items():
        entropies = values["router_entropy"]
        token_lists = values["tokens_per_expert"]
        averaged_tokens = []
        if token_lists:
            expert_count = len(token_lists[0])
            for expert_idx in range(expert_count):
                averaged_tokens.append(sum(tokens[expert_idx] for tokens in token_lists) / len(token_lists))
        normalized_per_layer[layer_name] = {
            "router_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
            "tokens_per_expert": averaged_tokens,
        }
    return aggregate_metrics, normalized_per_layer


def main() -> None:
    args = parse_args()
    config = load_json(args.config_path)
    baseline = load_json(args.baseline_path)
    log_records = load_jsonl(args.output_dir / "train_log.jsonl")

    device = ensure_cuda_device(args.device)
    model = HGRNBitForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)

    dataset = StreamingTextDataset(
        data_source=args.val_data_source,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        split="validation",
        text_field=config.get("training", {}).get("text_field", "text"),
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_streaming_batch)

    aggregate_eval, per_layer_router = collect_router_metrics(model, dataloader, device, args.precision)
    monitor = ExpertMonitor(model, config["moe"]["layer_indices"])
    expert_metrics = monitor.compute_metrics()
    ternary_ratios = compute_ternary_ratios(model, config["moe"]["layer_indices"])

    train_records = find_records(log_records, lambda record: "train_loss" in record)
    eval_records = find_records(log_records, lambda record: "val_loss" in record)
    monitor_records = find_records(log_records, lambda record: "avg_expert_similarity" in record)
    checkpoint_best_records = find_records(log_records, lambda record: record.get("event") == "checkpoint_best")
    freeze_record = next((record for record in log_records if record.get("event") == "freeze"), {})

    first_train_loss = float(train_records[0]["train_loss"]) if train_records else None
    final_train_loss = float(train_records[-1]["train_loss"]) if train_records else None
    initial_similarity = float(monitor_records[0]["avg_expert_similarity"]) if monitor_records else None
    final_similarity = float(monitor_records[-1]["avg_expert_similarity"]) if monitor_records else None
    similarity_trend = "stable"
    if initial_similarity is not None and final_similarity is not None:
        if final_similarity < initial_similarity - 1e-4:
            similarity_trend = "下降"
        elif final_similarity > initial_similarity + 1e-4:
            similarity_trend = "上升"

    final_eval = aggregate_eval
    dense_val_ppl = float(baseline["val_ppl"])
    moe_val_ppl = float(final_eval["val_ppl"])
    ppl_gap_percent = ((moe_val_ppl - dense_val_ppl) / dense_val_ppl) * 100.0

    tokens_seen = (
        config["training"]["max_steps"]
        * config["training"]["batch_size"]
        * config["training"]["gradient_accumulation_steps"]
        * config["training"]["max_length"]
    )

    summary_ternary = {"neg": 0.0, "zero": 0.0, "pos": 0.0, "count": 0}
    for layer_values in ternary_ratios.values():
        for expert_values in layer_values.values():
            for ratio_values in expert_values.values():
                summary_ternary["neg"] += ratio_values["neg"]
                summary_ternary["zero"] += ratio_values["zero"]
                summary_ternary["pos"] += ratio_values["pos"]
                summary_ternary["count"] += 1
    if summary_ternary["count"]:
        for key in ("neg", "zero", "pos"):
            summary_ternary[key] /= summary_ternary["count"]

    report = {
        "experiment": config.get("experiment_name", "first_real_upcycling"),
        "timing": {
            "elapsed_seconds": args.elapsed_seconds,
        },
        "config_summary": {
            "num_experts": config["moe"]["num_experts"],
            "num_experts_per_tok": config["moe"]["num_experts_per_tok"],
            "moe_layers": config["moe"]["layer_indices"],
            "use_quantized_experts": config["moe"]["use_quantized_experts"],
            "max_steps": config["training"]["max_steps"],
            "effective_batch_size": config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"],
            "learning_rate": config["training"]["learning_rate"],
            "total_tokens_seen": tokens_seen,
        },
        "model_stats": {
            "dense_params": 374108160,
            "moe_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": freeze_record.get("trainable_params"),
            "frozen_params": freeze_record.get("frozen_params"),
        },
        "baseline": {
            "dense_val_loss": baseline["val_loss"],
            "dense_val_ppl": baseline["val_ppl"],
        },
        "final_results": {
            "moe_val_loss": final_eval["val_loss"],
            "moe_val_ppl": final_eval["val_ppl"],
            "ppl_gap_percent": ppl_gap_percent,
        },
        "expert_analysis": {
            "initial_similarity": initial_similarity,
            "final_similarity": final_similarity,
            "similarity_trend": similarity_trend,
            "expert_collapse_detected": bool(monitor_records[-1]["expert_collapse"]) if monitor_records else False,
            "router_entropy_per_layer": {k: v["router_entropy"] for k, v in per_layer_router.items()},
            "tokens_per_expert_per_layer": {k: v["tokens_per_expert"] for k, v in per_layer_router.items()},
            "ternary_weight_ratios": ternary_ratios,
        },
        "training_curve": {
            "initial_train_loss": first_train_loss,
            "final_train_loss": final_train_loss,
            "loss_converged": bool(first_train_loss is not None and final_train_loss is not None and final_train_loss < first_train_loss),
            "training_stable": all(math.isfinite(float(record["train_loss"])) for record in train_records),
        },
        "hardware_params_extracted": {
            "ternary_ratio": {k: summary_ternary[k] for k in ("neg", "zero", "pos")},
            "activation_bitwidth": "INT8 activations, ternary weights",
            "expert_activation_frequency": {k: v["tokens_per_expert"] for k, v in per_layer_router.items()},
            "router_entropy": {k: v["router_entropy"] for k, v in per_layer_router.items()},
        },
        "artifacts": {
            "checkpoint_best": str(args.checkpoint_path),
            "checkpoint_best_records": checkpoint_best_records,
        },
        "issues": [],
        "next_steps": [],
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
