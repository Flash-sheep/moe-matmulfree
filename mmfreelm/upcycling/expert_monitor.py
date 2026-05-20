# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Dict, List

import torch
import torch.nn.functional as F


class ExpertMonitor:
    def __init__(self, model, moe_layer_indices: List[int]):
        self.model = model
        self.moe_layer_indices = moe_layer_indices
        self.history: List[Dict] = []

    def compute_metrics(self) -> Dict:
        metrics: Dict[str, Dict] = {}
        similarities = []

        for layer_idx in self.moe_layer_indices:
            block = self.model.model.layers[layer_idx]
            moe = block.mlp
            layer_metrics = {
                "expert_weight_similarity": self._compute_expert_similarity(moe),
                "expert_weight_norms": self._compute_expert_norms(moe),
                "router_weight_stats": self._compute_router_stats(moe),
            }
            similarities.append(layer_metrics["expert_weight_similarity"]["mean"])
            metrics[f"layer_{layer_idx}"] = layer_metrics

        metrics["summary"] = {
            "avg_expert_similarity": sum(similarities) / len(similarities) if similarities else 0.0,
            "num_moe_layers": len(self.moe_layer_indices),
        }
        return metrics

    def _compute_expert_similarity(self, moe) -> Dict[str, float]:
        expert_vectors = []
        for expert in moe.experts:
            params = [p.data.detach().float().reshape(-1) for p in expert.parameters()]
            expert_vectors.append(torch.cat(params))

        if len(expert_vectors) < 2:
            return {"mean": 1.0, "min": 1.0, "max": 1.0}

        similarities = []
        for i in range(len(expert_vectors)):
            for j in range(i + 1, len(expert_vectors)):
                sim = F.cosine_similarity(expert_vectors[i].unsqueeze(0), expert_vectors[j].unsqueeze(0)).item()
                similarities.append(sim)
        return {
            "mean": sum(similarities) / len(similarities),
            "min": min(similarities),
            "max": max(similarities),
        }

    def _compute_expert_norms(self, moe) -> List[float]:
        norms = []
        for expert in moe.experts:
            total = 0.0
            for p in expert.parameters():
                total += p.data.detach().float().norm().item() ** 2
            norms.append(total ** 0.5)
        return norms

    def _compute_router_stats(self, moe) -> Dict[str, float]:
        router = moe.router.gate
        w = router.weight.data.detach().float()
        return {
            "weight_mean": w.mean().item(),
            "weight_std": w.std().item(),
            "weight_norm": w.norm().item(),
        }

    def log(self, step: int, metrics: Dict):
        entry = {"step": step, "avg_expert_similarity": metrics["summary"]["avg_expert_similarity"]}
        for key, layer_metrics in metrics.items():
            if key.startswith("layer_"):
                entry[f"{key}_sim"] = layer_metrics["expert_weight_similarity"]["mean"]
        self.history.append(entry)

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.history, handle, indent=2)

    def check_expert_collapse(self, threshold: float = 0.99) -> bool:
        if not self.history:
            return False
        sim = self.history[-1].get("avg_expert_similarity", 0.0)
        return sim > threshold
