#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import StreamingTextDataset, upcycle_dense_to_moe
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, precision_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate-proj focused identity diagnostics for full shared no-residual.")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-idx", type=int, default=12)
    parser.add_argument("--shared-width", type=int, default=2816)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def clone_dense_model(checkpoint_path: str, device: torch.device, *, torch_dtype: Optional[torch.dtype]) -> HGRNBitForCausalLM:
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = HGRNBitForCausalLM.from_pretrained(checkpoint_path, **kwargs)
    model.eval()
    return model.to(device)


def build_shared_model(
    checkpoint_path: str,
    device: torch.device,
    *,
    shared_width: int,
    layer_idx: int,
    torch_dtype: Optional[torch.dtype],
) -> HGRNBitForCausalLM:
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = HGRNBitForCausalLM.from_pretrained(checkpoint_path, **kwargs)
    model.eval()
    upcycle_dense_to_moe(
        model,
        moe_layer_indices=list(range(layer_idx, 24)) if layer_idx == 12 else [layer_idx],
        num_experts=0,
        num_experts_per_tok=0,
        use_quantized_experts=True,
        moe_arch="shared_residual",
        enable_sparse_residual=False,
        nominal_shared_width=shared_width,
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
    return model.to(device).eval()


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> Dict[str, Any]:
    a = a.detach().float()
    b = b.detach().float()
    diff = a - b
    a_norm = a.norm().item()
    b_norm = b.norm().item()
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    cosine = torch.nn.functional.cosine_similarity(flat_a, flat_b, dim=0).item()
    return {
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs_diff": float(diff.abs().max().item()),
        "mean_abs_diff": float(diff.abs().mean().item()),
        "rms_diff": float(diff.square().mean().sqrt().item()),
        "relative_l2_error": float(diff.norm().item() / max(a_norm, 1e-12)),
        "cosine_similarity": float(cosine),
        "norm_a": float(a_norm),
        "norm_b": float(b_norm),
        "norm_ratio_b_over_a": float(b_norm / max(a_norm, 1e-12)),
    }


def primitive_repr(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return [primitive_repr(v) for v in value]
    if isinstance(value, list):
        return [primitive_repr(v) for v in value]
    return repr(value)


def module_attr_dump(module: torch.nn.Module) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "class_name": type(module).__name__,
        "training": bool(module.training),
        "parameters": {},
        "buffers": {},
        "public_attrs": {},
    }
    for name, param in module.named_parameters(recurse=False):
        payload["parameters"][name] = {
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "device": str(param.device),
            "requires_grad": bool(param.requires_grad),
        }
    for name, buf in module.named_buffers(recurse=False):
        payload["buffers"][name] = {
            "shape": list(buf.shape),
            "dtype": str(buf.dtype),
            "device": str(buf.device),
        }
    interesting = [
        "in_features",
        "out_features",
        "bias",
        "eps",
        "_is_residual_projection",
    ]
    for attr in interesting:
        if hasattr(module, attr):
            payload["public_attrs"][attr] = primitive_repr(getattr(module, attr))
    if hasattr(module, "norm"):
        norm = module.norm
        payload["norm"] = {
            "class_name": type(norm).__name__,
            "training": bool(norm.training),
            "eps": primitive_repr(getattr(norm, "eps", None)),
            "parameters": {
                name: {
                    "shape": list(param.shape),
                    "dtype": str(param.dtype),
                    "device": str(param.device),
                    "requires_grad": bool(param.requires_grad),
                }
                for name, param in norm.named_parameters(recurse=False)
            },
        }
    return payload


def diff_attr_dump(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    diff: Dict[str, Any] = {}
    for key in sorted(set(a.keys()) | set(b.keys())):
        if a.get(key) != b.get(key):
            diff[key] = {"a": a.get(key), "b": b.get(key)}
    return diff


def capture_mlp_input(
    model: HGRNBitForCausalLM,
    *,
    layer_idx: int,
    input_ids: torch.Tensor,
    precision_dtype: Optional[torch.dtype],
) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(module, inputs, outputs):
        captured["mlp_input"] = outputs[0].detach()
        return None

    handle = model.model.layers[layer_idx].mlp_norm.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            with precision_context(precision_dtype):
                model(input_ids=input_ids, labels=input_ids, return_dict=True)
    finally:
        handle.remove()
    return captured["mlp_input"]


def repeat_call_determinism(module: torch.nn.Module, x: torch.Tensor, *, precision_dtype: Optional[torch.dtype]) -> Dict[str, Any]:
    with torch.no_grad():
        with precision_context(precision_dtype):
            out1 = module(x)
            out2 = module(x)
    return compare_tensors(out1, out2)


def diagnose_config(
    checkpoint_path: str,
    *,
    device: torch.device,
    input_ids: torch.Tensor,
    layer_idx: int,
    shared_width: int,
    torch_dtype: Optional[torch.dtype],
    precision_dtype: Optional[torch.dtype],
) -> Dict[str, Any]:
    dense_model = clone_dense_model(checkpoint_path, device, torch_dtype=torch_dtype)
    shared_model = build_shared_model(checkpoint_path, device, shared_width=shared_width, layer_idx=layer_idx, torch_dtype=torch_dtype)

    dense_mlp_input = capture_mlp_input(
        dense_model,
        layer_idx=layer_idx,
        input_ids=input_ids,
        precision_dtype=precision_dtype,
    )
    shared_mlp_input = capture_mlp_input(
        shared_model,
        layer_idx=layer_idx,
        input_ids=input_ids,
        precision_dtype=precision_dtype,
    )
    dense_layer = dense_model.model.layers[layer_idx]
    shared_layer = shared_model.model.layers[layer_idx]
    dense_mlp = dense_layer.mlp
    shared_expert = shared_layer.mlp.shared_expert

    dense_gate = dense_mlp.gate_proj
    shared_gate = shared_expert.gate_proj

    common_input = dense_mlp_input.reshape(-1, dense_mlp_input.shape[-1])
    dense_input = common_input.to(dense_gate.weight.dtype)
    shared_input = common_input.to(shared_gate.weight.dtype)

    with torch.no_grad():
        with precision_context(precision_dtype):
            dense_gate_out = dense_gate(dense_input)
            shared_gate_out = shared_gate(shared_input)

    report: Dict[str, Any] = {
        "mlp_input_dense_vs_shared": compare_tensors(dense_mlp_input, shared_mlp_input),
        "direct_gate_proj_dense_vs_shared": compare_tensors(dense_gate_out, shared_gate_out),
        "dense_gate_repeat": repeat_call_determinism(dense_gate, dense_input, precision_dtype=precision_dtype),
        "shared_gate_repeat": repeat_call_determinism(shared_gate, shared_input, precision_dtype=precision_dtype),
        "gate_proj_attr_dense": module_attr_dump(dense_gate),
        "gate_proj_attr_shared": module_attr_dump(shared_gate),
        "gate_proj_attr_diff": diff_attr_dump(module_attr_dump(dense_gate), module_attr_dump(shared_gate)),
        "dtypes": {
            "dense_input": str(dense_input.dtype),
            "shared_input": str(shared_input.dtype),
            "dense_weight": str(dense_gate.weight.dtype),
            "shared_weight": str(shared_gate.weight.dtype),
            "dense_output": str(dense_gate_out.dtype),
            "shared_output": str(shared_gate_out.dtype),
        },
    }

    # Cross replacement on copied modules only.
    shared_expert_dense_gate = copy.deepcopy(shared_expert).to(device)
    shared_expert_dense_gate.gate_proj = dense_gate
    dense_mlp_shared_gate = copy.deepcopy(dense_mlp).to(device)
    dense_mlp_shared_gate.gate_proj = shared_gate
    with torch.no_grad():
        with precision_context(precision_dtype):
            base_shared_output = shared_expert(shared_input)
            cross_shared_output = shared_expert_dense_gate(shared_input.to(dense_gate.weight.dtype))
            base_dense_output = dense_mlp(dense_input)
            cross_dense_output = dense_mlp_shared_gate(dense_input.to(shared_gate.weight.dtype))
    report["cross_replace"] = {
        "shared_base_vs_dense_gate_swapped": compare_tensors(base_shared_output, cross_shared_output),
        "dense_base_vs_shared_gate_swapped": compare_tensors(base_dense_output, cross_dense_output),
        "dense_base_vs_shared_base": compare_tensors(base_dense_output, base_shared_output),
        "dense_base_vs_shared_with_dense_gate": compare_tensors(base_dense_output, cross_shared_output),
    }
    return report


def main() -> None:
    args = parse_args()
    args.device_obj = ensure_cuda_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataloader, manifest = build_loader(args)
    batch = next(iter(dataloader))
    base_input_ids = batch["input_ids"].to(args.device_obj)

    bf16_precision_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]
    report = {
        "layer_idx": int(args.layer_idx),
        "shared_width": int(args.shared_width),
        "diagnostics": {
            "bf16_autocast_on": diagnose_config(
                args.checkpoint_path,
                device=args.device_obj,
                input_ids=base_input_ids,
                layer_idx=args.layer_idx,
                shared_width=args.shared_width,
                torch_dtype=torch.bfloat16,
                precision_dtype=bf16_precision_dtype,
            ),
            "bf16_autocast_off": diagnose_config(
                args.checkpoint_path,
                device=args.device_obj,
                input_ids=base_input_ids,
                layer_idx=args.layer_idx,
                shared_width=args.shared_width,
                torch_dtype=torch.bfloat16,
                precision_dtype=None,
            ),
            "fp32_model_fp32_input": diagnose_config(
                args.checkpoint_path,
                device=args.device_obj,
                input_ids=base_input_ids,
                layer_idx=args.layer_idx,
                shared_width=args.shared_width,
                torch_dtype=torch.float32,
                precision_dtype=None,
            ),
            "bf16_model_fp32_input": diagnose_config(
                args.checkpoint_path,
                device=args.device_obj,
                input_ids=base_input_ids,
                layer_idx=args.layer_idx,
                shared_width=args.shared_width,
                torch_dtype=torch.bfloat16,
                precision_dtype=None,
            ),
        },
    }

    write_json(
        args.output_dir / "run_spec.json",
        {
            "checkpoint_path": args.checkpoint_path,
            "data_source": args.data_source,
            "tokenizer_path": args.tokenizer_path,
            "layer_idx": args.layer_idx,
            "shared_width": args.shared_width,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "seed": args.seed,
        },
    )
    write_json(args.output_dir / "dataset_manifest_val.json", manifest)
    write_json(args.output_dir / "gateproj_diagnostics.json", report)


if __name__ == "__main__":
    main()
