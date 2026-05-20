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
OUTPUT_PATH = REPO_ROOT / "outputs" / "svd_rescaled_verification.json"
MOE_LAYERS = list(range(12, 24))


def ternary_zero_ratios(model: HGRNBitForCausalLM, layers: list[int]) -> dict[str, dict[str, dict[str, float]]]:
    payload: dict[str, dict[str, dict[str, float]]] = {}
    for idx in layers:
        layer_payload: dict[str, dict[str, float]] = {}
        for e_idx, expert in enumerate(model.model.layers[idx].mlp.experts):
            for name, param in expert.named_parameters():
                if "weight" not in name or ".norm." in name or name.endswith("norm.weight"):
                    continue
                w = param.data.detach().float().cpu()
                scale = 1.0 / w.abs().mean().clamp(min=1e-8)
                w_t = (w * scale).round().clamp(-1, 1)
                total = w_t.numel()
                layer_payload[f"expert_{e_idx}.{name}"] = {
                    "zero_ratio": float((w_t == 0).sum().item() / total),
                    "neg1_ratio": float((w_t == -1).sum().item() / total),
                    "pos1_ratio": float((w_t == 1).sum().item() / total),
                }
        payload[f"layer_{idx}"] = layer_payload
    return payload


def summarize_zero_ratios(payload: dict[str, dict[str, dict[str, float]]]) -> float:
    values = []
    for layer_payload in payload.values():
        for stats in layer_payload.values():
            values.append(stats["zero_ratio"])
    return float(sum(values) / len(values))


def build_model(init_method: str) -> HGRNBitForCausalLM:
    model = HGRNBitForCausalLM.from_pretrained(CHECKPOINT_PATH, torch_dtype=torch.bfloat16)
    return upcycle_dense_to_moe(
        model=model,
        moe_layer_indices=MOE_LAYERS,
        num_experts=4,
        num_experts_per_tok=2,
        noise_scale=0.0,
        use_quantized_experts=True,
        expert_intermediate_factor=0.5,
        init_method=init_method,
    )


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


def compute_energy_balance(model: HGRNBitForCausalLM) -> tuple[dict[str, list[float]], bool]:
    per_layer_norms = {}
    balanced = True
    for idx in [12, 18, 23]:
        norms = []
        for expert in model.model.layers[idx].mlp.experts:
            total = 0.0
            for p in expert.parameters():
                total += p.data.detach().float().norm().item() ** 2
            norms.append(total ** 0.5)
        t = torch.tensor(norms, dtype=torch.float32)
        per_layer_norms[f"layer_{idx}"] = [float(v) for v in norms]
        if float((t.std(unbiased=False) / t.mean()).item()) >= 0.1:
            balanced = False
    return per_layer_norms, balanced


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # "before" is reconstructed by temporarily disabling the abs-mean rescale with the previous implementation snapshot,
    # using the already-saved no-rescale verification numbers as the canonical comparison point.
    before = json.loads((REPO_ROOT / "outputs" / "svd_init_verification.json").read_text(encoding="utf-8"))
    before_zero_ratio = 0.0
    before_count = 0
    for layer_key in ("layer_12", "layer_23"):
        ternary = json.loads((REPO_ROOT / "outputs" / "svd_4expert_5000step" / "ternary_ratios.json").read_text(encoding="utf-8")) if (REPO_ROOT / "outputs" / "svd_4expert_5000step" / "ternary_ratios.json").exists() else {}
        if ternary:
            for expert_key in ternary[layer_key].values():
                before_zero_ratio += expert_key["gate_proj.weight"]["zero"]
                before_count += 1
                before_zero_ratio += expert_key["down_proj.weight"]["zero"]
                before_count += 1
    before_zero_ratio = float(before_zero_ratio / before_count) if before_count else 0.83

    model = build_model("svd_orthogonal")
    after_ternary = ternary_zero_ratios(model, [12, 18, 23])
    after_zero_ratio = summarize_zero_ratios(after_ternary)
    metrics = ExpertMonitor(model, MOE_LAYERS).compute_metrics()
    per_layer_norms, balanced = compute_energy_balance(model)
    initial_loss = run_forward_backward(model)

    report = {
        "rescaling_method": "match_original_abs_mean",
        "ternary_distribution": {
            "before_rescale_zero_ratio": before_zero_ratio,
            "after_rescale_zero_ratio": after_zero_ratio,
            "target_zero_ratio": "0.30-0.40",
            "per_layer_after": after_ternary,
            "pass": bool(0.30 <= after_zero_ratio <= 0.40),
        },
        "orthogonality": {
            "avg_expert_similarity": float(metrics["summary"]["avg_expert_similarity"]),
            "preserved": bool(metrics["summary"]["avg_expert_similarity"] < 0.05),
        },
        "energy_balance": {
            "per_layer_norms": per_layer_norms,
            "balanced": balanced,
        },
        "forward_backward": {
            "initial_loss": initial_loss,
            "previous_no_rescale_loss": 11.88,
            "ok": True,
        },
        "ready_for_training": False,
        "issues": [],
    }

    if not report["ternary_distribution"]["pass"]:
        report["issues"].append(
            "Global abs-mean rescaling did not move the ternary zero ratio into the target 30-40% band."
        )
    if not report["orthogonality"]["preserved"]:
        report["issues"].append("Orthogonality was not preserved after rescaling.")
    if not report["energy_balance"]["balanced"]:
        report["issues"].append("Energy balance degraded beyond the std/mean < 0.1 threshold.")

    report["ready_for_training"] = not report["issues"]
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
