# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional, Sequence

from mmfreelm.upcycling.trainable_scope import (
    infer_freeze_mode,
    infer_norm_scope,
    resolve_local_backbone_layer_indices,
    should_train_parameter,
)


def apply_freeze_for_upcycling(
    model,
    moe_layer_indices: List[int],
    freeze_embeddings: bool = True,
    freeze_lm_head: bool = True,
    freeze_token_mixer: bool = True,
    freeze_non_moe_mlp: bool = True,
    freeze_rmsnorm: bool = True,
    trainable_extra_patterns: Optional[List[str]] = None,
    freeze_mode: str = "moe_only",
    local_backbone_layer_indices: Optional[Sequence[int]] = None,
    norm_scope: Optional[str] = None,
    dense_baseline: bool = False,
):
    del freeze_token_mixer, freeze_non_moe_mlp, freeze_rmsnorm

    resolved_freeze_mode = infer_freeze_mode({"freeze_mode": freeze_mode}, dense_baseline=dense_baseline)
    resolved_local_backbone_layers = resolve_local_backbone_layer_indices(
        moe_layer_indices=moe_layer_indices,
        local_backbone_layer_indices=local_backbone_layer_indices,
    )
    resolved_norm_scope = norm_scope or infer_norm_scope({}, resolved_freeze_mode)
    trainable_patterns = list(trainable_extra_patterns or [])

    for name, param in model.named_parameters():
        param.requires_grad = should_train_parameter(
            name=name,
            freeze_mode=resolved_freeze_mode,
            moe_layer_indices=moe_layer_indices,
            local_backbone_layer_indices=resolved_local_backbone_layers,
            norm_scope=resolved_norm_scope,
            freeze_embeddings=freeze_embeddings,
            freeze_lm_head=freeze_lm_head,
        )

    if trainable_patterns:
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in trainable_patterns):
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, frozen_params
