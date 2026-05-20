# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List

import torch


def _split_fused_gate_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[0] % 2 != 0:
        raise ValueError(f"Expected fused gate_proj weight with even first dimension, got {tuple(weight.shape)}")
    half = weight.shape[0] // 2
    return weight[:half], weight[half:]


def _build_rank_assignment(rank: int, num_experts: int, assignment: str) -> List[torch.Tensor]:
    if assignment not in {"interleaved", "contiguous"}:
        raise ValueError(f"Unsupported assignment `{assignment}`.")
    if assignment == "interleaved":
        return [torch.arange(expert_idx, rank, num_experts, dtype=torch.long) for expert_idx in range(num_experts)]

    group_size = (rank + num_experts - 1) // num_experts
    assignments = []
    for expert_idx in range(num_experts):
        start = expert_idx * group_size
        end = min(rank, start + group_size)
        assignments.append(torch.arange(start, end, dtype=torch.long))
    return assignments


def _build_partition_assignment(size: int, num_experts: int, assignment: str) -> List[torch.Tensor]:
    if assignment not in {"interleaved", "contiguous"}:
        raise ValueError(f"Unsupported assignment `{assignment}`.")
    if assignment == "interleaved":
        return [torch.arange(expert_idx, size, num_experts, dtype=torch.long) for expert_idx in range(num_experts)]

    group_size = size // num_experts
    assignments = []
    for expert_idx in range(num_experts):
        start = expert_idx * group_size
        end = start + group_size
        assignments.append(torch.arange(start, end, dtype=torch.long))
    return assignments


def _match_original_abs_mean(expert_weight: torch.Tensor, original_weight: torch.Tensor) -> torch.Tensor:
    expert_abs_mean = expert_weight.abs().mean()
    original_abs_mean = original_weight.detach().float().cpu().abs().mean()
    if expert_abs_mean <= 0:
        return expert_weight
    return expert_weight * (original_abs_mean / expert_abs_mean)


def _svd_rows(weight: torch.Tensor, indices: torch.Tensor, output_rows: int) -> torch.Tensor:
    weight = weight.detach().float().cpu()
    _, singular_values, vh = torch.linalg.svd(weight, full_matrices=False)
    take = indices[: min(indices.numel(), output_rows)]
    result = torch.zeros(output_rows, weight.shape[1], dtype=torch.float32)
    if take.numel() == 0:
        return result
    rows = singular_values[take].unsqueeze(1) * vh[take]
    result[: rows.shape[0]] = rows
    return result


def _svd_cols(weight: torch.Tensor, indices: torch.Tensor, output_cols: int) -> torch.Tensor:
    weight = weight.detach().float().cpu()
    u, singular_values, _ = torch.linalg.svd(weight, full_matrices=False)
    take = indices[: min(indices.numel(), output_cols)]
    result = torch.zeros(weight.shape[0], output_cols, dtype=torch.float32)
    if take.numel() == 0:
        return result
    cols = u[:, take] * singular_values[take].unsqueeze(0)
    result[:, : cols.shape[1]] = cols
    return result


def svd_orthogonal_init(
    original_mlp,
    num_experts: int,
    expert_intermediate: int,
    assignment: str = "interleaved",
) -> List[Dict[str, torch.Tensor]]:
    """
    Build expert initialization states from a dense MLP using disjoint SVD components.

    The fused gate projection is first split into gate/up halves. Each expert receives a
    disjoint subset of singular directions, which keeps the initial expert weights close
    to Frobenius-orthogonal while allowing a narrower expert intermediate width.
    """
    gate_fused = original_mlp.gate_proj.weight.detach()
    down_weight = original_mlp.down_proj.weight.detach()
    gate_weight, up_weight = _split_fused_gate_weight(gate_fused)

    gate_rank = min(gate_weight.shape)
    up_rank = min(up_weight.shape)
    down_rank = min(down_weight.shape)
    gate_assignments = _build_rank_assignment(gate_rank, num_experts, assignment)
    up_assignments = _build_rank_assignment(up_rank, num_experts, assignment)
    down_assignments = _build_rank_assignment(down_rank, num_experts, assignment)

    expert_states: List[Dict[str, torch.Tensor]] = []
    for expert_idx in range(num_experts):
        gate_rows = _svd_rows(gate_weight, gate_assignments[expert_idx], expert_intermediate)
        up_rows = _svd_rows(up_weight, up_assignments[expert_idx], expert_intermediate)
        down_cols = _svd_cols(down_weight, down_assignments[expert_idx], expert_intermediate)
        gate_rows = _match_original_abs_mean(gate_rows, gate_weight)
        up_rows = _match_original_abs_mean(up_rows, up_weight)
        down_cols = _match_original_abs_mean(down_cols, down_weight)
        expert_states.append(
            {
                "gate_proj.weight": torch.cat([gate_rows, up_rows], dim=0),
                "down_proj.weight": down_cols,
            }
        )
    return expert_states


def partition_init(
    original_mlp,
    num_experts: int,
    expert_intermediate: int,
    assignment: str = "interleaved",
) -> List[Dict[str, torch.Tensor]]:
    """
    Split the dense MLP weights into non-overlapping expert subspaces by rows/columns.

    The fused gate projection is split into gate/up halves. Each expert receives a disjoint
    subset of rows from both halves and the matching columns from down_proj.
    """
    source_state = original_mlp.state_dict()
    gate_fused = source_state["gate_proj.weight"].detach().float().cpu()
    down_weight = source_state["down_proj.weight"].detach().float().cpu()
    gate_weight, up_weight = _split_fused_gate_weight(gate_fused)
    base_intermediate = gate_weight.shape[0]
    assignments = _build_partition_assignment(base_intermediate, num_experts, assignment)

    expert_states: List[Dict[str, torch.Tensor]] = []
    for expert_idx, indices in enumerate(assignments):
        take = indices[: min(indices.numel(), expert_intermediate)]
        if take.numel() < expert_intermediate:
            raise ValueError(
                f"Partition init requested expert_intermediate={expert_intermediate}, but only "
                f"{take.numel()} non-overlapping rows are available for expert {expert_idx}. "
                "Use a smaller expert size or fewer experts."
            )
        expert_state: Dict[str, torch.Tensor] = {
            "gate_proj.weight": torch.cat(
                [
                    gate_weight.index_select(0, take),
                    up_weight.index_select(0, take),
                ],
                dim=0,
            ),
            "down_proj.weight": down_weight.index_select(1, take),
        }
        for key, value in source_state.items():
            if key in expert_state:
                continue
            source_value = value.detach().clone().float().cpu()
            if key == "down_proj.norm.weight":
                expert_state[key] = source_value.index_select(0, take)
            else:
                expert_state[key] = source_value
        expert_states.append(expert_state)
    return expert_states
