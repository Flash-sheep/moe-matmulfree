#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling.expert_monitor import ExpertMonitor
from mmfreelm.upcycling.sparse_upcycling import upcycle_dense_to_moe


CHECKPOINT_PATH = str(REPO_ROOT / "checkpoints" / "MMfreeLM-370M")
OUTPUT_PATH = REPO_ROOT / "outputs" / "svd_init_verification.json"
MOE_LAYERS = list(range(12, 24))
NUM_EXPERTS = 4
TOP_K = 2
EXPERT_INTERMEDIATE_FACTOR = 0.5


def count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model(init_method: str) -> HGRNBitForCausalLM:
    model = HGRNBitForCausalLM.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
    return upcycle_dense_to_moe(
        model=model,
        moe_layer_indices=MOE_LAYERS,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        noise_scale=0.15,
        use_quantized_experts=True,
        expert_intermediate_factor=EXPERT_INTERMEDIATE_FACTOR,
        init_method=init_method,
    )


def compute_energy_balance(model: HGRNBitForCausalLM) -> tuple[dict[str, list[float]], dict[str, float]]:
    per_layer_expert_norms: dict[str, list[float]] = {}
    per_layer_std_over_mean: dict[str, float] = {}
    for idx in [12, 18, 23]:
        moe = model.model.layers[idx].mlp
        norms = []
        for expert in moe.experts:
            total = 0.0
            for p in expert.parameters():
                total += p.data.detach().float().norm().item() ** 2
            norms.append(total ** 0.5)
        t = torch.tensor(norms, dtype=torch.float32)
        per_layer_expert_norms[f"layer_{idx}"] = [float(x) for x in norms]
        per_layer_std_over_mean[f"layer_{idx}"] = float((t.std(unbiased=False) / t.mean()).item())
    return per_layer_expert_norms, per_layer_std_over_mean


def compute_active_param_ratio(model: HGRNBitForCausalLM) -> float:
    baseline_mlp_params = 0
    active_moe_params = 0
    for layer_idx in MOE_LAYERS:
        dense_mlp = model.model.layers[layer_idx].mlp
        expert = dense_mlp.experts[0]
        expert_params = expert.gate_proj.weight.numel() + expert.down_proj.weight.numel()
        active_moe_params += TOP_K * expert_params
    base_model = HGRNBitForCausalLM.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
    for layer_idx in MOE_LAYERS:
        mlp = base_model.model.layers[layer_idx].mlp
        baseline_mlp_params += mlp.gate_proj.weight.numel() + mlp.down_proj.weight.numel()
    return active_moe_params / baseline_mlp_params


def run_forward_backward(model: HGRNBitForCausalLM) -> tuple[bool, bool, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 128), device=device)
    if device.type == "cuda":
        autocast_context = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    with autocast_context:
        outputs = model(input_ids=input_ids, labels=input_ids, output_router_logits=True, return_dict=True)
        loss = outputs.loss
    loss.backward()
    forward_ok = tuple(outputs.logits.shape) == (1, 128, model.config.vocab_size)
    backward_ok = all((p.grad is None) or torch.isfinite(p.grad).all().item() for p in model.parameters())
    return forward_ok, backward_ok, float(loss.item())


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_model = HGRNBitForCausalLM.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    del baseline_model

    model = build_model("svd_orthogonal")
    metrics = ExpertMonitor(model, MOE_LAYERS).compute_metrics()

    old_model = build_model("copy_noise")
    old_metrics = ExpertMonitor(old_model, MOE_LAYERS).compute_metrics()
    del old_model

    per_layer_expert_norms, per_layer_std_over_mean = compute_energy_balance(model)
    moe_total_params, moe_trainable_params = count_params(model)

    expert = model.model.layers[12].mlp.experts[0]
    expert_module_types = sorted({type(module).__name__ for _, module in expert.named_modules()})
    expert_shapes = {
        name: list(param.shape)
        for name, param in expert.named_parameters()
        if "weight" in name
    }
    expert_uses_bitlinear = any(
        type(module).__name__ in {"FusedBitLinear", "BitLinear", "BitLinear_Fuse"}
        for _, module in expert.named_modules()
    )

    forward_ok, backward_ok, loss_value = run_forward_backward(model)

    report = {
        "implementation": {
            "init_method": "svd_orthogonal",
            "assignment": "interleaved",
            "expert_intermediate_factor": EXPERT_INTERMEDIATE_FACTOR,
            "num_experts": NUM_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "moe_layers": MOE_LAYERS,
            "files_modified": [
                "mmfreelm/upcycling/svd_init.py",
                "mmfreelm/modules/moe.py",
                "mmfreelm/upcycling/sparse_upcycling.py",
                "mmfreelm/models/hgrn_bit/configuration_hgrn_bit.py",
                "mmfreelm/models/hgrn_bit/modeling_hgrn_bit.py",
                "mmfreelm/upcycling/__init__.py",
                "scripts/run_sparse_upcycling.py",
                "scripts/verify_svd_init.py",
            ],
        },
        "verification_1_orthogonality": {
            "avg_expert_similarity": float(metrics["summary"]["avg_expert_similarity"]),
            "per_layer_similarity": {k: v["expert_weight_similarity"] for k, v in metrics.items() if k.startswith("layer_")},
            "comparison_with_old_method": {
                "copy_noise_0.15": float(old_metrics["summary"]["avg_expert_similarity"]),
                "svd_orthogonal": float(metrics["summary"]["avg_expert_similarity"]),
            },
            "pass": bool(metrics["summary"]["avg_expert_similarity"] < 0.7),
        },
        "verification_2_energy_balance": {
            "per_layer_expert_norms": per_layer_expert_norms,
            "per_layer_std_over_mean": per_layer_std_over_mean,
            "pass": bool(all(value < 0.1 for value in per_layer_std_over_mean.values())),
        },
        "verification_3_param_count": {
            "non_moe_baseline_params": int(baseline_params),
            "moe_total_params": int(moe_total_params),
            "moe_trainable_params": int(moe_trainable_params),
            "param_increase_percent": float((moe_total_params - baseline_params) / baseline_params * 100.0),
            "active_params_vs_baseline": float(compute_active_param_ratio(model)),
        },
        "verification_4_bitlinear": {
            "expert_uses_bitlinear": bool(expert_uses_bitlinear),
            "expert_module_types": expert_module_types,
        },
        "verification_5_dimensions": {
            "expert_gate_proj_shape": expert_shapes.get("gate_proj.weight"),
            "expert_down_proj_shape": expert_shapes.get("down_proj.weight"),
            "expected_gate_proj_shape": [2816, 1024],
            "expected_down_proj_shape": [1024, 1408],
        },
        "verification_6_forward_backward": {
            "forward_ok": bool(forward_ok),
            "backward_ok": bool(backward_ok),
            "loss_value": float(loss_value),
        },
        "overall_pass": False,
        "ready_for_training": False,
        "issues": [],
    }

    if not report["verification_1_orthogonality"]["pass"]:
        report["issues"].append("Initial expert similarity did not fall below 0.7.")
    if not report["verification_2_energy_balance"]["pass"]:
        report["issues"].append("Expert norm balance failed std/mean < 0.1 for at least one checked layer.")
    if not report["verification_4_bitlinear"]["expert_uses_bitlinear"]:
        report["issues"].append("Expert modules no longer use BitLinear/FusedBitLinear.")
    if report["verification_5_dimensions"]["expert_gate_proj_shape"] != [2816, 1024]:
        report["issues"].append("Expert gate projection shape does not match the reduced intermediate size.")
    if report["verification_5_dimensions"]["expert_down_proj_shape"] != [1024, 1408]:
        report["issues"].append("Expert down projection shape does not match the reduced intermediate size.")
    if not forward_ok or not backward_ok or not math.isfinite(loss_value):
        report["issues"].append("Forward/backward verification failed.")

    report["overall_pass"] = not report["issues"]
    report["ready_for_training"] = report["overall_pass"]
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
