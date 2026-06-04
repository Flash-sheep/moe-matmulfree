# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


MODULE_TYPES = (
    "moe_router",
    "moe_experts",
    "moe_shared_expert",
    "moe_residual_scale",
    "moe_pair_scales",
    "token_mixer",
    "attention_or_sequence_mixer",
    "norm",
    "non_moe_mlp",
    "embedding",
    "lm_head",
    "other",
)


_LAYER_NAME_PATTERN = re.compile(r"model\.layers\.(\d+)\.")


def infer_freeze_mode(freeze_cfg: Dict, dense_baseline: bool = False) -> str:
    freeze_mode = freeze_cfg.get("freeze_mode")
    if freeze_mode:
        return str(freeze_mode)
    if dense_baseline:
        return "full_ft"
    return "moe_only"


def resolve_local_backbone_layer_indices(
    moe_layer_indices: Sequence[int],
    local_backbone_layer_indices: Optional[Sequence[int]] = None,
) -> List[int]:
    if local_backbone_layer_indices is None:
        return list(moe_layer_indices)
    return list(local_backbone_layer_indices)


def infer_norm_scope(freeze_cfg: Dict, freeze_mode: str) -> str:
    norm_scope = freeze_cfg.get("norm_scope")
    if norm_scope:
        return str(norm_scope)
    if not freeze_cfg.get("freeze_rmsnorm", True):
        return "all_norm"
    if freeze_mode == "moe_plus_norm":
        return "moe_mlp_norm"
    if freeze_mode == "local_backbone_ft":
        return "moe_layers_all_norm"
    if freeze_mode in {"partial_full_ft", "full_ft"}:
        return "all_norm"
    return "none"


def infer_strict_trainable_check(freeze_cfg: Dict, config: Dict) -> bool:
    if "strict_trainable_check" in freeze_cfg:
        return bool(freeze_cfg["strict_trainable_check"])
    return bool(config.get("strict_trainable_check", False))


def parameter_layer_idx(name: str) -> Optional[int]:
    match = _LAYER_NAME_PATTERN.search(name)
    if not match:
        return None
    return int(match.group(1))


def is_embedding_parameter(name: str) -> bool:
    return name.startswith("model.embeddings.")


def is_lm_head_parameter(name: str) -> bool:
    return name.startswith("lm_head.")


def is_moe_router_parameter(name: str) -> bool:
    return ".mlp.router." in name


def is_moe_shared_expert_parameter(name: str) -> bool:
    return ".mlp.shared_expert." in name


def is_moe_pair_scale_parameter(name: str) -> bool:
    return name.endswith("pair_log_scales")


def is_moe_residual_scale_parameter(name: str) -> bool:
    return ".mlp.raw_residual_scale" in name


def is_moe_expert_parameter(name: str) -> bool:
    return ".mlp.experts." in name or ".mlp.sparse_experts." in name


def is_moe_parameter(name: str) -> bool:
    return (
        is_moe_router_parameter(name)
        or is_moe_expert_parameter(name)
        or is_moe_shared_expert_parameter(name)
        or is_moe_pair_scale_parameter(name)
        or is_moe_residual_scale_parameter(name)
    )


def is_bias_parameter(name: str) -> bool:
    return name.endswith(".bias")


def is_norm_parameter(name: str) -> bool:
    return "norm" in name.lower() and (name.endswith(".weight") or name.endswith(".bias"))


def is_attention_or_sequence_mixer_parameter(name: str) -> bool:
    return ".attn." in name and not is_norm_parameter(name)


def is_token_mixer_parameter(name: str) -> bool:
    lowered = name.lower()
    if any(token in lowered for token in (".token_mixer.", ".mlgru.", ".mixer.")):
        return not is_norm_parameter(name)
    return False


def is_non_moe_mlp_parameter(name: str) -> bool:
    if ".mlp." not in name:
        return False
    if is_moe_parameter(name):
        return False
    if ".mlp_norm" in name:
        return False
    return True


def is_backbone_parameter(name: str) -> bool:
    if is_embedding_parameter(name) or is_lm_head_parameter(name) or is_moe_parameter(name):
        return False
    if name == "model.lower_bounds":
        return True
    if name.startswith("model.norm."):
        return True
    if name.startswith("model.layers."):
        return True
    return False


def classify_parameter_module_type(name: str) -> str:
    if is_embedding_parameter(name):
        return "embedding"
    if is_lm_head_parameter(name):
        return "lm_head"
    if is_moe_router_parameter(name):
        return "moe_router"
    if is_moe_shared_expert_parameter(name):
        return "moe_shared_expert"
    if is_moe_residual_scale_parameter(name):
        return "moe_residual_scale"
    if is_moe_pair_scale_parameter(name):
        return "moe_pair_scales"
    if is_moe_expert_parameter(name):
        return "moe_experts"
    if is_token_mixer_parameter(name):
        return "token_mixer"
    if is_attention_or_sequence_mixer_parameter(name):
        return "attention_or_sequence_mixer"
    if is_norm_parameter(name):
        return "norm"
    if is_non_moe_mlp_parameter(name):
        return "non_moe_mlp"
    return "other"


def selected_norm_parameter_name(
    name: str,
    norm_scope: str,
    moe_layer_indices: Sequence[int],
    local_backbone_layer_indices: Sequence[int],
) -> bool:
    if not is_norm_parameter(name):
        return False
    if is_moe_expert_parameter(name):
        return False
    if norm_scope == "none":
        return False
    if norm_scope == "all_norm":
        return True
    layer_idx = parameter_layer_idx(name)
    target_layers = set(local_backbone_layer_indices or moe_layer_indices)
    if norm_scope == "moe_mlp_norm":
        return (
            layer_idx in set(moe_layer_indices)
            and name == f"model.layers.{layer_idx}.mlp_norm.weight"
        )
    if norm_scope == "moe_layers_all_norm":
        return layer_idx in target_layers
    raise ValueError(f"Unsupported norm_scope `{norm_scope}`.")


def local_backbone_parameter_name(name: str, local_backbone_layer_indices: Sequence[int]) -> bool:
    layer_idx = parameter_layer_idx(name)
    if layer_idx is None or layer_idx not in set(local_backbone_layer_indices):
        return False
    module_type = classify_parameter_module_type(name)
    return module_type in {"token_mixer", "attention_or_sequence_mixer"}


def backbone_parameter_name(name: str) -> bool:
    module_type = classify_parameter_module_type(name)
    return module_type in {
        "token_mixer",
        "attention_or_sequence_mixer",
        "norm",
        "non_moe_mlp",
        "other",
    }


def should_train_parameter(
    name: str,
    freeze_mode: str,
    moe_layer_indices: Sequence[int],
    local_backbone_layer_indices: Sequence[int],
    norm_scope: str,
    freeze_embeddings: bool,
    freeze_lm_head: bool,
) -> bool:
    if freeze_mode == "full_ft":
        return True

    trainable = False
    if is_moe_parameter(name):
        trainable = True
    elif freeze_mode == "moe_plus_norm":
        trainable = selected_norm_parameter_name(name, norm_scope, moe_layer_indices, local_backbone_layer_indices)
    elif freeze_mode == "local_backbone_ft":
        trainable = local_backbone_parameter_name(name, local_backbone_layer_indices) or selected_norm_parameter_name(
            name,
            norm_scope,
            moe_layer_indices,
            local_backbone_layer_indices,
        )
    elif freeze_mode == "partial_full_ft":
        trainable = backbone_parameter_name(name)
    elif freeze_mode != "moe_only":
        raise ValueError(f"Unsupported freeze_mode `{freeze_mode}`.")

    if is_embedding_parameter(name):
        return trainable or not freeze_embeddings
    if is_lm_head_parameter(name):
        return trainable or not freeze_lm_head
    return trainable


def apply_trainable_patterns(
    trainable_names: Iterable[str],
    all_parameter_names: Iterable[str],
    trainable_extra_patterns: Optional[Sequence[str]],
) -> Tuple[Set[str], List[str]]:
    trainable_name_set = set(trainable_names)
    matched_names: List[str] = []
    patterns = list(trainable_extra_patterns or [])
    if not patterns:
        return trainable_name_set, matched_names
    for name in all_parameter_names:
        if any(pattern in name for pattern in patterns):
            if name not in trainable_name_set:
                matched_names.append(name)
            trainable_name_set.add(name)
    return trainable_name_set, matched_names


def summarize_trainable_parameters(
    model,
    freeze_mode: str,
    moe_layer_indices: Sequence[int],
    local_backbone_layer_indices: Sequence[int],
    norm_scope: str,
    trainable_extra_patterns: Optional[Sequence[str]] = None,
    first_n: int = 200,
) -> Dict[str, object]:
    trainable_parameter_names: List[str] = []
    by_module_type: Dict[str, List[str]] = {module_type: [] for module_type in MODULE_TYPES}
    extra_pattern_enabled_names: List[str] = []
    selected_norm_names: List[str] = []
    local_backbone_names: List[str] = []
    extra_patterns = list(trainable_extra_patterns or [])

    count_by_name = {name: param.numel() for name, param in model.named_parameters()}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_parameter_names.append(name)
        by_module_type[classify_parameter_module_type(name)].append(name)
        if selected_norm_parameter_name(name, norm_scope, moe_layer_indices, local_backbone_layer_indices):
            selected_norm_names.append(name)
        if local_backbone_parameter_name(name, local_backbone_layer_indices):
            local_backbone_names.append(name)
        if extra_patterns and any(pattern in name for pattern in extra_patterns):
            base_selected = should_train_parameter(
                name=name,
                freeze_mode=freeze_mode,
                moe_layer_indices=moe_layer_indices,
                local_backbone_layer_indices=local_backbone_layer_indices,
                norm_scope=norm_scope,
                freeze_embeddings=True,
                freeze_lm_head=True,
            )
            if not base_selected:
                extra_pattern_enabled_names.append(name)

    module_type_counts = {
        module_type: sum(count_by_name[name] for name in names)
        for module_type, names in by_module_type.items()
    }
    module_type_num_tensors = {
        module_type: len(names)
        for module_type, names in by_module_type.items()
    }
    return {
        "freeze_mode": freeze_mode,
        "norm_scope": norm_scope,
        "moe_layer_indices": list(moe_layer_indices),
        "local_backbone_layer_indices": list(local_backbone_layer_indices),
        "trainable_parameter_names": trainable_parameter_names,
        "first_200_trainable_parameter_names": trainable_parameter_names[:first_n],
        "trainable_parameter_names_by_module_type": by_module_type,
        "trainable_parameter_counts_by_module_type": module_type_counts,
        "trainable_parameter_tensors_by_module_type": module_type_num_tensors,
        "extra_pattern_enabled_parameter_names": extra_pattern_enabled_names,
        "extra_pattern_enabled_parameter_count": len(extra_pattern_enabled_names),
        "selected_norm_parameter_names": selected_norm_names,
        "selected_norm_parameter_count": sum(count_by_name[name] for name in selected_norm_names),
        "local_backbone_parameter_names": local_backbone_names,
        "local_backbone_parameter_count": sum(count_by_name[name] for name in local_backbone_names),
        "trainable_backbone_parameter_count": sum(
            module_type_counts[module_type]
            for module_type in ("token_mixer", "attention_or_sequence_mixer", "norm", "non_moe_mlp", "other")
        ),
        "trainable_embedding_parameter_count": module_type_counts["embedding"],
        "trainable_lm_head_parameter_count": module_type_counts["lm_head"],
    }
