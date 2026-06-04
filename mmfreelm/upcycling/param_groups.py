# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from mmfreelm.upcycling.trainable_scope import (
    backbone_parameter_name,
    is_bias_parameter,
    is_embedding_parameter,
    is_lm_head_parameter,
    is_moe_parameter,
    is_moe_residual_scale_parameter,
    is_moe_router_parameter,
    is_moe_shared_expert_parameter,
    is_norm_parameter,
    local_backbone_parameter_name,
    parameter_layer_idx,
)


GROUP_ORDER = ("moe", "shared_expert", "backbone", "norm_or_bias", "embed_lm_head")


def resolve_optimizer_hparams(config: Dict, training_cfg: Dict, freeze_cfg: Dict, freeze_mode: str) -> Dict[str, float]:
    learning_rate = float(training_cfg.get("learning_rate", 5e-4))
    moe_lr = float(
        config.get(
            "moe_lr",
            training_cfg.get("moe_lr", freeze_cfg.get("moe_lr", learning_rate)),
        )
    )
    default_backbone_lr = learning_rate * 0.1
    shared_expert_lr = float(
        config.get(
            "shared_expert_lr",
            training_cfg.get("shared_expert_lr", freeze_cfg.get("shared_expert_lr", default_backbone_lr)),
        )
    )
    backbone_lr = float(
        config.get(
            "backbone_lr",
            training_cfg.get("backbone_lr", freeze_cfg.get("backbone_lr", default_backbone_lr)),
        )
    )
    norm_lr = float(
        config.get(
            "norm_lr",
            training_cfg.get("norm_lr", freeze_cfg.get("norm_lr", backbone_lr)),
        )
    )
    default_embed_lr = backbone_lr * 0.1
    embed_lr = config.get("embed_lr", training_cfg.get("embed_lr", freeze_cfg.get("embed_lr", default_embed_lr)))
    return {
        "learning_rate": learning_rate,
        "moe_lr": moe_lr,
        "shared_expert_lr": shared_expert_lr,
        "backbone_lr": backbone_lr,
        "norm_lr": norm_lr,
        "embed_lr": None if embed_lr is None else float(embed_lr),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "freeze_mode": freeze_mode,
    }


def _determine_group_name(
    name: str,
    freeze_mode: str,
    local_backbone_layer_indices: Sequence[int],
) -> str:
    if is_embedding_parameter(name) or is_lm_head_parameter(name):
        return "embed_lm_head"
    if is_norm_parameter(name) or is_bias_parameter(name):
        return "norm_or_bias"
    if is_moe_shared_expert_parameter(name):
        return "shared_expert"
    if is_moe_router_parameter(name) or is_moe_residual_scale_parameter(name):
        return "moe"
    if is_moe_parameter(name):
        return "moe"
    if freeze_mode == "local_backbone_ft" and local_backbone_parameter_name(name, local_backbone_layer_indices):
        return "backbone"
    if backbone_parameter_name(name):
        return "backbone"
    return "backbone"


def build_optimizer_param_groups(
    model,
    freeze_mode: str,
    moe_lr: float,
    shared_expert_lr: float,
    backbone_lr: float,
    norm_lr: float,
    embed_lr: Optional[float],
    weight_decay: float,
    local_backbone_layer_indices: Sequence[int],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    group_specs = {
        "moe": {
            "lr": float(moe_lr),
            "weight_decay": float(weight_decay),
            "params": [],
            "names": [],
        },
        "shared_expert": {
            "lr": float(shared_expert_lr),
            "weight_decay": float(weight_decay),
            "params": [],
            "names": [],
        },
        "backbone": {
            "lr": float(backbone_lr),
            "weight_decay": float(weight_decay),
            "params": [],
            "names": [],
        },
        "norm_or_bias": {
            "lr": float(norm_lr),
            "weight_decay": 0.0,
            "params": [],
            "names": [],
        },
        "embed_lm_head": {
            "lr": float(embed_lr if embed_lr is not None else backbone_lr),
            "weight_decay": float(weight_decay),
            "params": [],
            "names": [],
        },
    }

    requires_grad_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        requires_grad_names.append(name)
        group_name = _determine_group_name(
            name=name,
            freeze_mode=freeze_mode,
            local_backbone_layer_indices=local_backbone_layer_indices,
        )
        group_specs[group_name]["params"].append(param)
        group_specs[group_name]["names"].append(name)

    optimizer_groups: List[Dict[str, object]] = []
    optimizer_group_summary: List[Dict[str, object]] = []
    covered_names = set()
    for group_name in GROUP_ORDER:
        spec = group_specs[group_name]
        names = list(spec["names"])
        duplicate_names = covered_names.intersection(names)
        if duplicate_names:
            raise ValueError(
                f"Optimizer param group overlap detected for group `{group_name}`: {sorted(duplicate_names)[:10]}"
            )
        covered_names.update(names)
        param_count = sum(param.numel() for param in spec["params"])
        summary = {
            "group_name": group_name,
            "lr": float(spec["lr"]),
            "weight_decay": float(spec["weight_decay"]),
            "param_count": int(param_count),
            "num_tensors": len(spec["params"]),
            "first_20_param_names": names[:20],
            "parameter_names": names,
        }
        optimizer_group_summary.append(summary)
        if not spec["params"]:
            continue
        optimizer_groups.append(
            {
                "name": group_name,
                "params": spec["params"],
                "lr": spec["lr"],
                "weight_decay": spec["weight_decay"],
            }
        )

    missing = sorted(set(requires_grad_names) - covered_names)
    if missing:
        raise ValueError(f"Optimizer param groups failed to cover all trainable params: {missing[:20]}")
    return optimizer_groups, optimizer_group_summary


def optimizer_lr_map(optimizer) -> Dict[str, Optional[float]]:
    lr_map: Dict[str, Optional[float]] = {
        "lr_moe": None,
        "lr_shared_expert": None,
        "lr_backbone": None,
        "lr_norm_or_bias": None,
        "lr_embed_lm_head": None,
    }
    for group in optimizer.param_groups:
        group_name = group.get("name")
        if group_name == "moe":
            lr_map["lr_moe"] = float(group["lr"])
        elif group_name == "shared_expert":
            lr_map["lr_shared_expert"] = float(group["lr"])
        elif group_name == "backbone":
            lr_map["lr_backbone"] = float(group["lr"])
        elif group_name == "norm_or_bias":
            lr_map["lr_norm_or_bias"] = float(group["lr"])
        elif group_name == "embed_lm_head":
            lr_map["lr_embed_lm_head"] = float(group["lr"])
    return lr_map


def run_strict_trainable_checks(
    trainable_summary: Dict[str, object],
    optimizer_group_summary: List[Dict[str, object]],
    freeze_mode: str,
    freeze_embeddings: bool,
    freeze_lm_head: bool,
    local_backbone_layer_indices: Sequence[int],
    require_moe_router: bool = True,
) -> List[str]:
    issues: List[str] = []
    counts = trainable_summary["trainable_parameter_counts_by_module_type"]
    trainable_names = trainable_summary["trainable_parameter_names"]
    embedding_count = int(trainable_summary["trainable_embedding_parameter_count"])
    lm_head_count = int(trainable_summary["trainable_lm_head_parameter_count"])
    backbone_count = int(trainable_summary["trainable_backbone_parameter_count"])
    local_backbone_count = int(trainable_summary["local_backbone_parameter_count"])
    selected_norm_count = int(trainable_summary["selected_norm_parameter_count"])

    group_by_name = {entry["group_name"]: entry for entry in optimizer_group_summary}

    if require_moe_router and counts["moe_router"] <= 0:
        issues.append("No trainable MoE router parameters found.")
    if counts["moe_experts"] + counts.get("moe_shared_expert", 0) <= 0:
        issues.append("No trainable MoE expert parameters found.")

    if freeze_embeddings and embedding_count != 0:
        issues.append("Embedding parameters are trainable despite freeze_embeddings=true.")
    if freeze_lm_head and lm_head_count != 0:
        issues.append("lm_head parameters are trainable despite freeze_lm_head=true.")

    if freeze_mode == "moe_only":
        if backbone_count != 0:
            issues.append(f"moe_only expected backbone trainable params = 0, got {backbone_count}.")
        if embedding_count != 0 or lm_head_count != 0:
            issues.append("moe_only expected embeddings/lm_head to remain frozen.")
    elif freeze_mode == "moe_plus_norm":
        if selected_norm_count <= 0:
            issues.append("moe_plus_norm expected selected norm params > 0.")
        out_of_scope_backbone = max(backbone_count - selected_norm_count, 0)
        if out_of_scope_backbone != 0:
            issues.append(
                f"moe_plus_norm expected only selected norms outside MoE to train, got extra backbone params {out_of_scope_backbone}."
            )
    elif freeze_mode == "local_backbone_ft":
        if local_backbone_count <= 0:
            issues.append("local_backbone_ft expected local backbone params > 0.")
        if selected_norm_count <= 0:
            issues.append("local_backbone_ft expected local norm params > 0.")
        non_local_names = []
        target_layer_set = set(local_backbone_layer_indices)
        for name in trainable_names:
            layer_idx = parameter_layer_idx(name)
            if layer_idx is None:
                continue
            if layer_idx not in target_layer_set and not is_moe_parameter(name):
                non_local_names.append(name)
        if non_local_names:
            issues.append(
                "local_backbone_ft unexpectedly trains params outside local_backbone_layer_indices: "
                f"{non_local_names[:20]}"
            )
    elif freeze_mode == "partial_full_ft":
        if backbone_count <= 0:
            issues.append("partial_full_ft expected backbone trainable params > 0.")
        backbone_group = group_by_name.get("backbone")
        if not backbone_group or int(backbone_group["param_count"]) <= 0:
            issues.append("partial_full_ft expected optimizer backbone group param_count > 0.")
        moe_only_baseline = counts["moe_router"] + counts["moe_experts"] + counts["moe_pair_scales"]
        if int(sum(counts.values())) <= moe_only_baseline:
            issues.append("partial_full_ft trainable params are not larger than MoE-only baseline.")
    elif freeze_mode == "full_ft":
        total_trainable = int(sum(counts.values()))
        if total_trainable <= 0:
            issues.append("full_ft expected trainable params > 0.")
        if embedding_count <= 0:
            issues.append("full_ft expected embedding params to be trainable.")
        if lm_head_count <= 0:
            issues.append("full_ft expected lm_head params to be trainable.")

    for required_group in ("moe", "norm_or_bias", "backbone", "embed_lm_head"):
        if required_group not in group_by_name:
            continue
        if group_by_name[required_group]["param_count"] < 0:
            issues.append(f"Optimizer group `{required_group}` has invalid negative param_count.")

    return issues
