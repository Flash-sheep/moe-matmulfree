# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Optional


def apply_freeze_for_upcycling(
    model,
    moe_layer_indices: List[int],
    freeze_embeddings: bool = True,
    freeze_lm_head: bool = True,
    freeze_token_mixer: bool = True,
    freeze_non_moe_mlp: bool = True,
    freeze_rmsnorm: bool = True,
    trainable_extra_patterns: Optional[List[str]] = None,
):
    del freeze_embeddings, freeze_lm_head, freeze_token_mixer, freeze_non_moe_mlp
    trainable_patterns = trainable_extra_patterns or []

    for param in model.parameters():
        param.requires_grad = False

    for layer_idx in moe_layer_indices:
        block = model.model.layers[layer_idx]
        for param in block.mlp.parameters():
            param.requires_grad = True

    if not freeze_rmsnorm:
        for name, param in model.named_parameters():
            if "norm" in name.lower() and "weight" in name:
                param.requires_grad = True

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in trainable_patterns):
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, frozen_params
