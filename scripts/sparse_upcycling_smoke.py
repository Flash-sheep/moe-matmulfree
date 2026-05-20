#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM, HGRNBitMLP
from mmfreelm.modules.moe import SparseMoEBlock
from mmfreelm.upcycling.expert_monitor import ExpertMonitor
from mmfreelm.upcycling.freeze import apply_freeze_for_upcycling
from mmfreelm.upcycling.sparse_upcycling import upcycle_dense_to_moe


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def module_type_names(module: torch.nn.Module) -> List[str]:
    return [type(child).__name__ for _, child in module.named_modules()]


def is_finite_scalar(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main() -> None:
    torch.manual_seed(42)

    issues: List[str] = []
    log_dir = REPO_ROOT / "experiments" / "sparse_upcycling" / "logs"
    checkpoint_dir = REPO_ROOT / "experiments" / "sparse_upcycling" / "tmp" / "test_dense_checkpoint"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "upcycling_smoke.json"

    config = HGRNBitConfig(
        vocab_size=32000,
        hidden_size=256,
        num_hidden_layers=8,
        num_heads=1,
        use_moe=False,
        fuse_cross_entropy=False,
    )
    dense_model = HGRNBitForCausalLM(config)
    dense_model_params = count_params(dense_model)
    dense_model.save_pretrained(checkpoint_dir)

    model = HGRNBitForCausalLM.from_pretrained(checkpoint_dir)
    model = upcycle_dense_to_moe(
        model=model,
        moe_layer_indices=[4, 5, 6, 7],
        num_experts=4,
        num_experts_per_tok=2,
        noise_scale=0.05,
        use_quantized_experts=True,
    )
    upcycled_model_params = count_params(model)

    dense_layers_ok = all(isinstance(model.model.layers[idx].mlp, HGRNBitMLP) for idx in range(4))
    moe_layers_ok = all(isinstance(model.model.layers[idx].mlp, SparseMoEBlock) for idx in range(4, 8))
    if not dense_layers_ok:
        issues.append("Layer 0-3 MLPs were not preserved as dense HGRNBitMLP modules.")
    if not moe_layers_ok:
        issues.append("Layer 4-7 MLPs were not replaced with SparseMoEBlock modules.")

    expert_module_types: Dict[str, List[str]] = {}
    router_module_types: Dict[str, List[str]] = {}
    expert_uses_bitlinear = True
    expert_uses_plain_linear = False

    for idx in [4, 5, 6, 7]:
        moe = model.model.layers[idx].mlp
        expert = moe.experts[0]
        expert_types = module_type_names(expert)
        expert_module_types[f"layer_{idx}"] = expert_types
        if not any(name in ("FusedBitLinear", "BitLinear", "BitLinear_Fuse") for name in expert_types):
            expert_uses_bitlinear = False
        if "Linear" in expert_types:
            expert_uses_plain_linear = True
        router_module_types[f"layer_{idx}"] = module_type_names(moe.router)

    if not expert_uses_bitlinear:
        issues.append("MoE experts do not expose BitLinear/FusedBitLinear modules.")
    if expert_uses_plain_linear:
        issues.append("MoE experts contain torch.nn.Linear, which is a blocking issue for this smoke test.")

    trainable_params, frozen_params = apply_freeze_for_upcycling(
        model=model,
        moe_layer_indices=[4, 5, 6, 7],
    )

    non_moe_requires_grad = model.model.layers[0].mlp.gate_proj.weight.requires_grad
    embeddings_requires_grad = model.model.embeddings.weight.requires_grad
    router_requires_grad = model.model.layers[4].mlp.router.gate.weight.requires_grad
    expert_requires_grad = model.model.layers[4].mlp.experts[0].gate_proj.weight.requires_grad
    token_mixer_requires_grad = model.model.layers[4].attn.i_proj.weight.requires_grad
    lm_head_requires_grad = model.lm_head.weight.requires_grad

    freeze_checks = {
        "non_moe_mlp_frozen": not non_moe_requires_grad,
        "embeddings_frozen": not embeddings_requires_grad,
        "router_trainable": router_requires_grad,
        "expert_trainable": expert_requires_grad,
        "token_mixer_frozen": not token_mixer_requires_grad,
        "lm_head_frozen": not lm_head_requires_grad,
    }
    if not all(freeze_checks.values()):
        issues.append(f"Freeze policy mismatch: {freeze_checks}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4)

    input_ids = torch.randint(0, 32000, (2, 128), device=device)
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels, output_router_logits=True, return_dict=True)
    loss = outputs.loss
    loss.backward()

    frozen_params_no_grad = True
    trainable_params_have_grad = False
    trainable_params_with_grad = 0
    for name, param in model.named_parameters():
        if not param.requires_grad and param.grad is not None:
            frozen_params_no_grad = False
            issues.append(f"Frozen parameter received gradient: {name}")
            break
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            trainable_params_have_grad = True
            trainable_params_with_grad += 1

    if not trainable_params_have_grad:
        issues.append("No trainable parameter received gradient during the single-step smoke test.")

    optimizer.step()

    router_aux_loss_value = float(outputs.router_aux_loss.detach().cpu()) if outputs.router_aux_loss is not None else None
    lm_loss_value = float(outputs.lm_loss.detach().cpu()) if outputs.lm_loss is not None else None
    single_step_loss = float(loss.detach().cpu())

    if not is_finite_scalar(single_step_loss):
        issues.append("Total loss is not finite.")
    if lm_loss_value is None or not is_finite_scalar(lm_loss_value):
        issues.append("LM loss is missing or not finite.")
    if router_aux_loss_value is None or not is_finite_scalar(router_aux_loss_value) or router_aux_loss_value == 0.0:
        issues.append("Router aux loss is missing, non-finite, or zero.")

    monitor = ExpertMonitor(model, [4, 5, 6, 7])
    metrics = monitor.compute_metrics()
    initial_expert_similarity = float(metrics["summary"]["avg_expert_similarity"])
    if not is_finite_scalar(initial_expert_similarity):
        issues.append("ExpertMonitor did not produce a finite average expert similarity.")

    status = "pass" if not issues else "fail"
    report = {
        "status": status,
        "dense_model_params": dense_model_params,
        "upcycled_model_params": upcycled_model_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "expert_uses_bitlinear": expert_uses_bitlinear,
        "expert_contains_torch_linear": expert_uses_plain_linear,
        "expert_module_types": expert_module_types,
        "router_module_types": router_module_types,
        "dense_layers_preserved": dense_layers_ok,
        "moe_layers_replaced": moe_layers_ok,
        "freeze_checks": freeze_checks,
        "device": str(device),
        "single_step_loss": single_step_loss,
        "lm_loss": lm_loss_value,
        "router_aux_loss": router_aux_loss_value,
        "frozen_params_no_grad": frozen_params_no_grad,
        "trainable_params_have_grad": trainable_params_have_grad,
        "trainable_params_with_grad": trainable_params_with_grad,
        "initial_expert_similarity": initial_expert_similarity,
        "monitor_summary": metrics["summary"],
        "issues": issues,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
