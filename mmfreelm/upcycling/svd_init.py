# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _split_fused_gate_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2 or weight.shape[0] % 2 != 0:
        raise ValueError(f"Expected fused gate_proj weight with even first dimension, got {tuple(weight.shape)}")
    half = weight.shape[0] // 2
    return weight[:half], weight[half:]


def _apply_relative_std_noise(weight: torch.Tensor, noise_alpha: float) -> torch.Tensor:
    if noise_alpha <= 0:
        return weight
    std = float(weight.std().item())
    if std <= 0:
        return weight
    return weight + torch.randn_like(weight) * (noise_alpha * std)


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


def virtual_group_partition_copy_noise_init(
    original_mlp,
    num_experts: int,
    expert_intermediate: int,
    num_virtual_groups: int,
    noise_alpha: float,
) -> tuple[List[Dict[str, torch.Tensor]], Dict[int, int]]:
    """
    Split the dense MLP into contiguous intermediate shards, then clone each shard within a
    virtual group while adding small relative-std noise to break symmetry.
    """
    if num_virtual_groups < 1:
        raise ValueError("`num_virtual_groups` must be at least 1.")
    if num_experts % num_virtual_groups != 0:
        raise ValueError("`num_experts` must be divisible by `num_virtual_groups`.")

    source_state = original_mlp.state_dict()
    gate_fused = source_state["gate_proj.weight"].detach().float().cpu()
    down_weight = source_state["down_proj.weight"].detach().float().cpu()
    gate_weight, up_weight = _split_fused_gate_weight(gate_fused)
    base_intermediate = gate_weight.shape[0]
    shard_size = base_intermediate // num_virtual_groups
    if shard_size < expert_intermediate:
        raise ValueError(
            f"Requested expert_intermediate={expert_intermediate}, but virtual-group shard size is {shard_size}."
        )

    experts_per_group = num_experts // num_virtual_groups
    expert_states: List[Dict[str, torch.Tensor]] = []
    expert_to_group: Dict[int, int] = {}

    for group_idx in range(num_virtual_groups):
        start = group_idx * shard_size
        take = torch.arange(start, start + expert_intermediate, dtype=torch.long)
        shard_state: Dict[str, torch.Tensor] = {
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
            if key in shard_state:
                continue
            source_value = value.detach().clone().float().cpu()
            if key == "down_proj.norm.weight":
                shard_state[key] = source_value.index_select(0, take)
            else:
                shard_state[key] = source_value

        for replica_idx in range(experts_per_group):
            expert_idx = group_idx * experts_per_group + replica_idx
            expert_to_group[expert_idx] = group_idx
            expert_state: Dict[str, torch.Tensor] = {}
            for key, value in shard_state.items():
                cloned = value.clone()
                if cloned.is_floating_point():
                    std = float(cloned.std().item())
                    if noise_alpha > 0 and std > 0:
                        cloned = cloned + torch.randn_like(cloned) * (noise_alpha * std)
                expert_state[key] = cloned
            expert_states.append(expert_state)

    return expert_states, expert_to_group


def complement_pair_6e_init(
    original_mlp,
    expert_intermediate: int,
    noise_alpha: float,
) -> tuple[List[Dict[str, torch.Tensor]], Dict[int, Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Build 6 half-width experts from four quarter shards of the dense intermediate space.

    Expert layout:
      0 -> Q0 + Q1
      1 -> Q0 + Q2
      2 -> Q0 + Q3
      3 -> Q1 + Q2
      4 -> Q1 + Q3
      5 -> Q2 + Q3

    Legal complement pairs:
      (0, 5), (1, 4), (2, 3)
    """
    source_state = original_mlp.state_dict()
    gate_fused = source_state["gate_proj.weight"].detach().float().cpu()
    down_weight = source_state["down_proj.weight"].detach().float().cpu()
    gate_weight, up_weight = _split_fused_gate_weight(gate_fused)
    base_intermediate = gate_weight.shape[0]
    if base_intermediate % 4 != 0:
        raise ValueError(
            f"Complement-pair init requires intermediate_size divisible by 4, got {base_intermediate}."
        )
    quarter_size = base_intermediate // 4
    expected_expert_intermediate = quarter_size * 2
    if expert_intermediate != expected_expert_intermediate:
        raise ValueError(
            f"Complement-pair init requires expert_intermediate={expected_expert_intermediate}, "
            f"got {expert_intermediate}."
        )

    quarter_indices = [
        torch.arange(q_idx * quarter_size, (q_idx + 1) * quarter_size, dtype=torch.long)
        for q_idx in range(4)
    ]
    expert_quarters: Dict[int, Tuple[int, int]] = {
        0: (0, 1),
        1: (0, 2),
        2: (0, 3),
        3: (1, 2),
        4: (1, 3),
        5: (2, 3),
    }
    legal_complement_pairs: List[Tuple[int, int]] = [(0, 5), (1, 4), (2, 3)]

    expert_states: List[Dict[str, torch.Tensor]] = []
    for expert_idx in range(6):
        quarter_a, quarter_b = expert_quarters[expert_idx]
        take = torch.cat([quarter_indices[quarter_a], quarter_indices[quarter_b]], dim=0)
        expert_state: Dict[str, torch.Tensor] = {
            "gate_proj.weight": torch.cat(
                [
                    _apply_relative_std_noise(gate_weight.index_select(0, take), noise_alpha),
                    _apply_relative_std_noise(up_weight.index_select(0, take), noise_alpha),
                ],
                dim=0,
            ),
            "down_proj.weight": _apply_relative_std_noise(down_weight.index_select(1, take), noise_alpha),
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

    return expert_states, expert_quarters, legal_complement_pairs


def complement_copy_12e_init(
    original_mlp,
    expert_intermediate: int,
    noise_alpha: float,
) -> tuple[List[Dict[str, torch.Tensor]], Dict[int, Tuple[int, int]], Dict[int, int], List[Tuple[int, int]]]:
    """
    Build 12 half-width experts from six half-combinations, with each half-combination copied twice.

    Expert layout:
      0,1   -> Q0 + Q1
      2,3   -> Q0 + Q2
      4,5   -> Q0 + Q3
      6,7   -> Q1 + Q2
      8,9   -> Q1 + Q3
      10,11 -> Q2 + Q3

    Legal complement-copy paths:
      family 0: (0,10), (0,11), (1,10), (1,11)
      family 1: (2,8),  (2,9),  (3,8),  (3,9)
      family 2: (4,6),  (4,7),  (5,6),  (5,7)
    """
    source_state = original_mlp.state_dict()
    gate_fused = source_state["gate_proj.weight"].detach().float().cpu()
    down_weight = source_state["down_proj.weight"].detach().float().cpu()
    gate_weight, up_weight = _split_fused_gate_weight(gate_fused)
    base_intermediate = gate_weight.shape[0]
    if base_intermediate % 4 != 0:
        raise ValueError(
            f"Complement-copy init requires intermediate_size divisible by 4, got {base_intermediate}."
        )
    quarter_size = base_intermediate // 4
    expected_expert_intermediate = quarter_size * 2
    if expert_intermediate != expected_expert_intermediate:
        raise ValueError(
            f"Complement-copy init requires expert_intermediate={expected_expert_intermediate}, "
            f"got {expert_intermediate}."
        )

    quarter_indices = [
        torch.arange(q_idx * quarter_size, (q_idx + 1) * quarter_size, dtype=torch.long)
        for q_idx in range(4)
    ]
    combo_quarters: Dict[int, Tuple[int, int]] = {
        0: (0, 1),
        1: (0, 2),
        2: (0, 3),
        3: (1, 2),
        4: (1, 3),
        5: (2, 3),
    }
    expert_quarters: Dict[int, Tuple[int, int]] = {
        0: combo_quarters[0],
        1: combo_quarters[0],
        2: combo_quarters[1],
        3: combo_quarters[1],
        4: combo_quarters[2],
        5: combo_quarters[2],
        6: combo_quarters[3],
        7: combo_quarters[3],
        8: combo_quarters[4],
        9: combo_quarters[4],
        10: combo_quarters[5],
        11: combo_quarters[5],
    }
    copy_group_mapping: Dict[int, int] = {
        0: 0, 1: 0,
        2: 1, 3: 1,
        4: 2, 5: 2,
        6: 3, 7: 3,
        8: 4, 9: 4,
        10: 5, 11: 5,
    }
    legal_paths: List[Tuple[int, int]] = [
        (0, 10), (0, 11), (1, 10), (1, 11),
        (2, 8), (2, 9), (3, 8), (3, 9),
        (4, 6), (4, 7), (5, 6), (5, 7),
    ]

    expert_states: List[Dict[str, torch.Tensor]] = []
    for expert_idx in range(12):
        quarter_a, quarter_b = expert_quarters[expert_idx]
        take = torch.cat([quarter_indices[quarter_a], quarter_indices[quarter_b]], dim=0)
        expert_state: Dict[str, torch.Tensor] = {
            "gate_proj.weight": torch.cat(
                [
                    _apply_relative_std_noise(gate_weight.index_select(0, take), noise_alpha),
                    _apply_relative_std_noise(up_weight.index_select(0, take), noise_alpha),
                ],
                dim=0,
            ),
            "down_proj.weight": _apply_relative_std_noise(down_weight.index_select(1, take), noise_alpha),
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

    return expert_states, expert_quarters, copy_group_mapping, legal_paths
