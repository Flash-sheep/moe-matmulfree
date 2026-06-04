#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.modules.activations import swiglu
from mmfreelm.upcycling import StreamingTextDataset, upcycle_dense_to_moe
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, precision_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layerwise identity debug for full shared no-residual conversion.")
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


def compare_tensors(dense_tensor: torch.Tensor, shared_tensor: torch.Tensor) -> Dict[str, Any]:
    dense = dense_tensor.detach().float()
    shared = shared_tensor.detach().float()
    diff = dense - shared
    dense_norm = dense.norm().item()
    shared_norm = shared.norm().item()
    flat_dense = dense.reshape(-1)
    flat_shared = shared.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_dense, flat_shared, dim=0).item()
    return {
        "shape_dense": list(dense.shape),
        "shape_shared": list(shared.shape),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rms_diff": float(diff.square().mean().sqrt().item()),
        "relative_l2_error": float(diff.norm().item() / max(dense_norm, 1e-12)),
        "cosine_similarity": float(cosine),
        "norm_dense": float(dense_norm),
        "norm_shared": float(shared_norm),
        "norm_ratio_shared_over_dense": float(shared_norm / max(dense_norm, 1e-12)),
    }


def summarize_state_dict_keys(dense_mlp, shared_expert) -> Dict[str, Any]:
    dense_keys = sorted(dense_mlp.state_dict().keys())
    shared_keys = sorted(shared_expert.state_dict().keys())
    dense_set = set(dense_keys)
    shared_set = set(shared_keys)
    common_keys = sorted(dense_set & shared_set)
    comparisons = {}
    for key in common_keys:
        dense_value = dense_mlp.state_dict()[key].detach().float()
        shared_value = shared_expert.state_dict()[key].detach().float()
        if dense_value.shape != shared_value.shape:
            comparisons[key] = {
                "shape_dense": list(dense_value.shape),
                "shape_shared": list(shared_value.shape),
                "shape_match": False,
            }
            continue
        diff = (dense_value - shared_value).abs()
        comparisons[key] = {
            "shape_dense": list(dense_value.shape),
            "shape_shared": list(shared_value.shape),
            "shape_match": True,
            "max_abs_diff": float(diff.max().item()),
            "mean_abs_diff": float(diff.mean().item()),
        }
    return {
        "dense_keys": dense_keys,
        "shared_keys": shared_keys,
        "missing_in_shared": sorted(dense_set - shared_set),
        "extra_in_shared": sorted(shared_set - dense_set),
        "common_key_comparisons": comparisons,
    }


def install_layer_hooks(model: HGRNBitForCausalLM, layer_indices: List[int], records: Dict[int, Dict[str, torch.Tensor]]):
    handles = []
    for layer_idx in layer_indices:
        layer = model.model.layers[layer_idx]
        layer_record = records.setdefault(layer_idx, {})

        def block_prehook(module, inputs, *, record=layer_record):
            record["block_input"] = inputs[0].detach().cpu()

        def mlp_norm_hook(module, inputs, outputs, *, record=layer_record):
            record["mlp_input"] = outputs[0].detach().cpu()

        def mlp_hook(module, inputs, outputs, *, record=layer_record):
            if isinstance(outputs, tuple):
                record["mlp_output"] = outputs[0].detach().cpu()
            else:
                record["mlp_output"] = outputs.detach().cpu()

        def block_hook(module, inputs, outputs, *, record=layer_record):
            record["block_output"] = outputs[0].detach().cpu()

        handles.append(layer.register_forward_pre_hook(block_prehook))
        handles.append(layer.mlp_norm.register_forward_hook(mlp_norm_hook))
        handles.append(layer.mlp.register_forward_hook(mlp_hook))
        handles.append(layer.register_forward_hook(block_hook))
    return handles


def compute_layerwise_debug(
    dense_model: HGRNBitForCausalLM,
    shared_model: HGRNBitForCausalLM,
    input_ids: torch.Tensor,
    layer_indices: List[int],
    precision_dtype: torch.dtype | None,
) -> Dict[str, Any]:
    dense_records: Dict[int, Dict[str, torch.Tensor]] = {}
    shared_records: Dict[int, Dict[str, torch.Tensor]] = {}
    dense_handles = install_layer_hooks(dense_model, layer_indices, dense_records)
    shared_handles = install_layer_hooks(shared_model, layer_indices, shared_records)
    try:
        with torch.no_grad():
            with precision_context(precision_dtype):
                dense_outputs = dense_model(input_ids=input_ids, labels=input_ids, return_dict=True)
                shared_outputs = shared_model(input_ids=input_ids, labels=input_ids, return_dict=True)
    finally:
        for handle in dense_handles + shared_handles:
            handle.remove()

    report: Dict[str, Any] = {
        "first_batch_loss": {
            "dense": float(dense_outputs.loss.detach().cpu()),
            "shared": float(shared_outputs.loss.detach().cpu()),
            "delta": float((shared_outputs.loss - dense_outputs.loss).detach().cpu()),
        },
        "first_batch_logits": compare_tensors(dense_outputs.logits, shared_outputs.logits),
        "layers": {},
        "first_divergence": None,
    }

    ordered_stages = [
        "block_input",
        "mlp_input",
        "gate_proj_output",
        "gate_part",
        "up_part",
        "swiglu_output",
        "down_proj_output",
        "mlp_output",
        "block_output",
    ]

    for layer_idx in layer_indices:
        dense_layer = dense_model.model.layers[layer_idx]
        shared_layer = shared_model.model.layers[layer_idx]
        dense_mlp = dense_layer.mlp
        shared_mlp = shared_layer.mlp.shared_expert
        dense_layer_record = dense_records[layer_idx]
        shared_layer_record = shared_records[layer_idx]

        dense_mlp_input = dense_layer_record["mlp_input"].to(input_ids.device)
        shared_mlp_input = shared_layer_record["mlp_input"].to(input_ids.device)
        dense_mlp_input_flat = dense_mlp_input.reshape(-1, dense_mlp_input.shape[-1]).to(dense_mlp.gate_proj.weight.dtype)
        shared_mlp_input_flat = shared_mlp_input.reshape(-1, shared_mlp_input.shape[-1]).to(shared_mlp.gate_proj.weight.dtype)

        with torch.no_grad():
            dense_gate_proj = dense_mlp.gate_proj(dense_mlp_input_flat)
            shared_gate_proj = shared_mlp.gate_proj(shared_mlp_input_flat)
            dense_gate, dense_up = dense_gate_proj.chunk(2, dim=-1)
            shared_gate, shared_up = shared_gate_proj.chunk(2, dim=-1)
            dense_swiglu = swiglu(dense_gate, dense_up)
            shared_swiglu = swiglu(shared_gate, shared_up)
            dense_down = dense_mlp.down_proj(dense_swiglu)
            shared_down = shared_mlp.down_proj(shared_swiglu)

        layer_report = {
            "module_types": {
                "dense_mlp": type(dense_mlp).__name__,
                "shared_mlp_wrapper": type(shared_layer.mlp).__name__,
                "shared_expert": type(shared_mlp).__name__,
                "dense_gate_proj": type(dense_mlp.gate_proj).__name__,
                "shared_gate_proj": type(shared_mlp.gate_proj).__name__,
                "dense_down_proj": type(dense_mlp.down_proj).__name__,
                "shared_down_proj": type(shared_mlp.down_proj).__name__,
            },
            "state_dict_summary": summarize_state_dict_keys(dense_mlp, shared_mlp),
            "stage_diffs": {
                "block_input": compare_tensors(dense_layer_record["block_input"], shared_layer_record["block_input"]),
                "mlp_input": compare_tensors(dense_mlp_input, shared_mlp_input),
                "gate_proj_output": compare_tensors(dense_gate_proj, shared_gate_proj),
                "gate_part": compare_tensors(dense_gate, shared_gate),
                "up_part": compare_tensors(dense_up, shared_up),
                "swiglu_output": compare_tensors(dense_swiglu, shared_swiglu),
                "down_proj_output": compare_tensors(dense_down, shared_down),
                "mlp_output": compare_tensors(
                    dense_layer_record["mlp_output"].reshape(-1, dense_layer_record["mlp_output"].shape[-1]),
                    shared_layer_record["mlp_output"].reshape(-1, shared_layer_record["mlp_output"].shape[-1]),
                ),
                "block_output": compare_tensors(dense_layer_record["block_output"], shared_layer_record["block_output"]),
            },
        }
        report["layers"][str(layer_idx)] = layer_report

    threshold = 1e-5
    for layer_idx in layer_indices:
        for stage in ordered_stages:
            stage_report = report["layers"][str(layer_idx)]["stage_diffs"][stage]
            if stage_report["mean_abs_diff"] > threshold:
                report["first_divergence"] = {
                    "layer_idx": int(layer_idx),
                    "stage": stage,
                    "mean_abs_diff": float(stage_report["mean_abs_diff"]),
                    "max_abs_diff": float(stage_report["max_abs_diff"]),
                    "relative_l2_error": float(stage_report["relative_l2_error"]),
                }
                return report
    return report


def main() -> None:
    args = parse_args()
    args.layer_indices = parse_layer_indices(args.layer_indices)
    args.device_obj = ensure_cuda_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataloader, manifest = build_loader(args)
    batch = next(iter(dataloader))
    input_ids = batch["input_ids"].to(args.device_obj)
    precision_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]

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

    report = compute_layerwise_debug(
        dense_model=dense_model,
        shared_model=shared_model,
        input_ids=input_ids,
        layer_indices=args.layer_indices,
        precision_dtype=precision_dtype,
    )
    write_json(
        args.output_dir / "run_spec.json",
        {
            "checkpoint_path": args.checkpoint_path,
            "data_source": args.data_source,
            "tokenizer_path": args.tokenizer_path,
            "layer_indices": args.layer_indices,
            "shared_width": args.shared_width,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "seed": args.seed,
        },
    )
    write_json(args.output_dir / "dataset_manifest_val.json", manifest)
    write_json(args.output_dir / "layerwise_debug_report.json", report)


if __name__ == "__main__":
    main()
