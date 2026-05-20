#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling.expert_monitor import ExpertMonitor
from mmfreelm.upcycling.sparse_upcycling import upcycle_dense_to_moe


CHECKPOINT_PATH = "/home/yjl/yjl-r760/matmulfreellm_assets/checkpoints/MMfreeLM-370M"
OUTPUT_PATH = REPO_ROOT / "outputs" / "partition_init_verification.json"
MOE_LAYERS = list(range(12, 24))


def compute_ternary_stats(weight: torch.Tensor) -> dict[str, float]:
    w = weight.detach().float().cpu()
    scale = 1.0 / w.abs().mean().clamp(min=1e-8)
    w_t = (w * scale).round().clamp(-1, 1)
    total = w_t.numel()
    return {
        "neg1_ratio": float((w_t == -1).sum().item() / total),
        "zero_ratio": float((w_t == 0).sum().item() / total),
        "pos1_ratio": float((w_t == 1).sum().item() / total),
    }


def run_forward_backward(model: HGRNBitForCausalLM) -> float:
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
    return float(loss.item())


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline = HGRNBitForCausalLM.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    baseline_layer = baseline.model.layers[12].mlp
    baseline_zero_ratios = {
        "gate_proj.weight": compute_ternary_stats(baseline_layer.gate_proj.weight)["zero_ratio"],
        "down_proj.weight": compute_ternary_stats(baseline_layer.down_proj.weight)["zero_ratio"],
    }

    model = HGRNBitForCausalLM.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
    model = upcycle_dense_to_moe(
        model=model,
        moe_layer_indices=MOE_LAYERS,
        num_experts=4,
        num_experts_per_tok=2,
        noise_scale=0.0,
        use_quantized_experts=True,
        expert_intermediate_factor=0.5,
        init_method="partition",
    )

    metrics = ExpertMonitor(model, MOE_LAYERS).compute_metrics()
    total_params = sum(p.numel() for p in model.parameters())
    increase_percent = (total_params - baseline_params) / baseline_params * 100.0

    sample_layers = [12, 18, 23]
    ternary_sample = {}
    zero_values = []
    for idx in sample_layers:
        layer_payload = {}
        for e_idx, expert in enumerate(model.model.layers[idx].mlp.experts):
            layer_payload[f"expert_{e_idx}.gate_proj.weight"] = compute_ternary_stats(expert.gate_proj.weight)
            layer_payload[f"expert_{e_idx}.down_proj.weight"] = compute_ternary_stats(expert.down_proj.weight)
            zero_values.append(layer_payload[f"expert_{e_idx}.gate_proj.weight"]["zero_ratio"])
            zero_values.append(layer_payload[f"expert_{e_idx}.down_proj.weight"]["zero_ratio"])
        ternary_sample[f"layer_{idx}"] = layer_payload
    zero_ratio_avg = float(sum(zero_values) / len(zero_values))

    expert = model.model.layers[12].mlp.experts[0]
    gate_shape = list(expert.gate_proj.weight.shape)
    down_shape = list(expert.down_proj.weight.shape)
    module_types = sorted({type(module).__name__ for _, module in expert.named_modules()})

    initial_loss = run_forward_backward(model)

    report = {
        "init_method": "partition_interleaved",
        "expert_intermediate_factor_requested": 0.5,
        "expert_intermediate_size_effective": int(model.config.moe_expert_intermediate_size),
        "num_experts": 4,
        "ternary_distribution": {
            "zero_ratio_avg": zero_ratio_avg,
            "matches_original": bool(0.30 <= zero_ratio_avg <= 0.40),
            "baseline_sample_zero_ratio": baseline_zero_ratios,
            "per_layer_sample": ternary_sample,
        },
        "orthogonality": {
            "avg_expert_similarity": float(metrics["summary"]["avg_expert_similarity"]),
            "comparison": {
                "svd": 0.005123514543002885,
                "copy_noise": 0.9791644174191688,
                "partition": float(metrics["summary"]["avg_expert_similarity"]),
            },
        },
        "param_count": {
            "total": int(total_params),
            "increase_percent": float(increase_percent),
        },
        "dimensions": {
            "gate_proj_shape": gate_shape,
            "down_proj_shape": down_shape,
            "expert_module_types": module_types,
        },
        "forward_backward": {
            "initial_loss": initial_loss,
            "ok": True,
        },
        "pass": bool(0.30 <= zero_ratio_avg <= 0.40 and metrics["summary"]["avg_expert_similarity"] < 0.5),
        "issues": [],
    }

    if not (0.30 <= zero_ratio_avg <= 0.40):
        report["issues"].append("Ternary zero ratio did not land in the 30-40% target band.")
    if not metrics["summary"]["avg_expert_similarity"] < 0.5:
        report["issues"].append("Expert similarity is not below 0.5.")

    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
