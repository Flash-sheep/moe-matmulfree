#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare uninterrupted and checkpoint-resumed Stage B runs."
    )
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--resumed-dir", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=str, required=True)
    parser.add_argument("--resumed-checkpoint", type=str, required=True)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--model-atol", type=float, default=1e-3)
    parser.add_argument("--optimizer-atol", type=float, default=1e-3)
    parser.add_argument("--loss-atol", type=float, default=5e-2)
    parser.add_argument("--grad-norm-atol", type=float, default=5e-1)
    parser.add_argument("--parameter-norm-atol", type=float, default=1e-1)
    parser.add_argument("--compare-step-start", type=int, default=1)
    parser.add_argument("--compare-step-end", type=int)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[Dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def tensor_difference(
    left: torch.Tensor,
    right: torch.Tensor,
    chunk_size: int = 1_000_000,
) -> Dict[str, object]:
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "compatible": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
            "left_dtype": str(left.dtype),
            "right_dtype": str(right.dtype),
        }
    left_flat = left.detach().cpu().reshape(-1)
    right_flat = right.detach().cpu().reshape(-1)
    max_abs = 0.0
    abs_sum = 0.0
    ref_sum = 0.0
    count = left_flat.numel()
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        left_chunk = left_flat[start:stop].float()
        right_chunk = right_flat[start:stop].float()
        difference = (left_chunk - right_chunk).abs()
        max_abs = max(max_abs, float(difference.max().item()))
        abs_sum += float(difference.double().sum().item())
        ref_sum += float(left_chunk.abs().double().sum().item())
    mean_abs = abs_sum / max(count, 1)
    return {
        "compatible": True,
        "numel": count,
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "relative_mean_difference": abs_sum / max(ref_sum, 1e-30),
    }


def tensor_items(value: object, prefix: str = "") -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, dict):
        for key in sorted(value, key=lambda item: str(item)):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from tensor_items(value[key], child)
    elif isinstance(value, (list, tuple)):
        for index, child_value in enumerate(value):
            child = f"{prefix}[{index}]"
            yield from tensor_items(child_value, child)


def compare_tensor_trees(left: object, right: object) -> Dict[str, object]:
    left_items = dict(tensor_items(left))
    right_items = dict(tensor_items(right))
    key_match = left_items.keys() == right_items.keys()
    worst_name = None
    max_abs = 0.0
    weighted_abs_sum = 0.0
    total_numel = 0
    incompatible = []
    for name in sorted(left_items.keys() & right_items.keys()):
        result = tensor_difference(left_items[name], right_items[name])
        if not result["compatible"]:
            incompatible.append({"name": name, **result})
            continue
        if result["max_abs_difference"] > max_abs:
            max_abs = result["max_abs_difference"]
            worst_name = name
        weighted_abs_sum += result["mean_abs_difference"] * result["numel"]
        total_numel += result["numel"]
    return {
        "tensor_keys_match": key_match,
        "left_tensor_count": len(left_items),
        "right_tensor_count": len(right_items),
        "incompatible_tensors": incompatible,
        "max_abs_difference": max_abs,
        "worst_tensor": worst_name,
        "mean_abs_difference": weighted_abs_sum / max(total_numel, 1),
        "total_numel_compared": total_numel,
    }


def compare_hash_logs(
    control_dir: Path,
    resumed_dir: Path,
    world_size: int,
    step_start: int,
    step_end: int | None,
) -> Dict[str, object]:
    per_rank = []
    all_match = True
    for rank in range(world_size):
        control = load_jsonl(control_dir / f"data_hashes_rank{rank}.jsonl")
        resumed = load_jsonl(resumed_dir / f"data_hashes_rank{rank}.jsonl")
        control_by_step = {
            int(row["optimizer_step"]): row["step_hash"] for row in control
        }
        resumed_by_step = {
            int(row["optimizer_step"]): row["step_hash"] for row in resumed
        }
        selected_steps = [
            step
            for step in sorted(control_by_step)
            if step >= step_start
            and (step_end is None or step <= step_end)
        ]
        missing_steps = [
            step for step in selected_steps if step not in resumed_by_step
        ]
        mismatch_indices = [
            step
            for step in selected_steps
            if step in resumed_by_step
            and control_by_step[step] != resumed_by_step[step]
        ]
        exact = not missing_steps and not mismatch_indices
        all_match = all_match and exact
        per_rank.append(
            {
                "rank": rank,
                "control_steps_compared": len(selected_steps),
                "resumed_steps_available": len(resumed_by_step),
                "exact_match": exact,
                "mismatch_count": len(mismatch_indices) + len(missing_steps),
                "first_mismatch_step": mismatch_indices[0] if mismatch_indices else None,
                "resume_boundary_step_51_match": (
                    51 in control_by_step
                    and 51 in resumed_by_step
                    and control_by_step[51] == resumed_by_step[51]
                ),
            }
        )
    return {"all_ranks_exact": all_match, "per_rank": per_rank}


def compare_metric_logs(
    control_dir: Path,
    resumed_dir: Path,
    step_start: int,
    step_end: int | None,
) -> Dict[str, object]:
    control = [
        row
        for row in load_jsonl(control_dir / "training_metrics.jsonl")
        if row.get("record_type") == "train"
    ]
    resumed = [
        row
        for row in load_jsonl(resumed_dir / "training_metrics.jsonl")
        if row.get("record_type") == "train"
    ]
    control_by_step = {int(row["optimizer_step"]): row for row in control}
    resumed_by_step = {int(row["optimizer_step"]): row for row in resumed}
    selected_steps = [
        step
        for step in sorted(control_by_step)
        if step >= step_start
        and (step_end is None or step <= step_end)
    ]
    keys = (
        "train_loss",
        "learning_rate",
        "grad_norm",
        "parameter_norm",
        "global_tokens",
        "scheduler_step",
    )
    maxima = {key: 0.0 for key in keys}
    common_steps = [step for step in selected_steps if step in resumed_by_step]
    for step in common_steps:
        left = control_by_step[step]
        right = resumed_by_step[step]
        for key in keys:
            maxima[key] = max(maxima[key], abs(float(left[key]) - float(right[key])))
    return {
        "steps_compared": len(common_steps),
        "selected_step_range": [step_start, step_end],
        "step_count_match": bool(selected_steps)
        and len(common_steps) == len(selected_steps),
        "max_abs_differences": maxima,
        "resume_boundary": {
            "control_step_51": control_by_step.get(51),
            "resumed_step_51": resumed_by_step.get(51),
        },
    }


def non_tensor_projection(value: object) -> object:
    if torch.is_tensor(value):
        return {
            "tensor_shape": list(value.shape),
            "tensor_dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {
            str(key): non_tensor_projection(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [non_tensor_projection(child) for child in value]
    return value


def main() -> None:
    args = parse_args()
    control_checkpoint = args.control_dir / args.control_checkpoint
    resumed_checkpoint = args.resumed_dir / args.resumed_checkpoint
    hash_report = compare_hash_logs(
        args.control_dir,
        args.resumed_dir,
        args.world_size,
        args.compare_step_start,
        args.compare_step_end,
    )
    metric_report = compare_metric_logs(
        args.control_dir,
        args.resumed_dir,
        args.compare_step_start,
        args.compare_step_end,
    )

    control_state = torch.load(
        control_checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    resumed_state = torch.load(
        resumed_checkpoint / "trainer_state.pt",
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model_report = compare_tensor_trees(
        control_state["model"],
        resumed_state["model"],
    )
    optimizer_report = compare_tensor_trees(
        control_state["optimizer"],
        resumed_state["optimizer"],
    )
    scheduler_match = (
        non_tensor_projection(control_state["scheduler"])
        == non_tensor_projection(resumed_state["scheduler"])
    )
    scalar_state = {
        key: {
            "control": control_state[key],
            "resumed": resumed_state[key],
            "match": control_state[key] == resumed_state[key],
        }
        for key in ("optimizer_step", "scheduler_step", "global_tokens", "world_size")
    }
    dataset_reports = []
    for rank in range(args.world_size):
        left = torch.load(
            control_checkpoint / f"dataset_state_rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        right = torch.load(
            resumed_checkpoint / f"dataset_state_rank{rank}.pt",
            map_location="cpu",
            weights_only=False,
        )
        dataset_reports.append(
            {
                "rank": rank,
                "exact_match": left == right,
            }
        )

    tolerance_pass = (
        model_report["max_abs_difference"] <= args.model_atol
        and optimizer_report["max_abs_difference"] <= args.optimizer_atol
    )
    metric_differences = metric_report["max_abs_differences"]
    metric_tolerance_pass = (
        metric_differences["train_loss"] <= args.loss_atol
        and metric_differences["grad_norm"] <= args.grad_norm_atol
        and metric_differences["parameter_norm"] <= args.parameter_norm_atol
        and metric_differences["learning_rate"] == 0
        and metric_differences["global_tokens"] == 0
        and metric_differences["scheduler_step"] == 0
    )
    passed = (
        hash_report["all_ranks_exact"]
        and metric_report["step_count_match"]
        and metric_tolerance_pass
        and tolerance_pass
        and scheduler_match
        and all(item["match"] for item in scalar_state.values())
        and all(item["exact_match"] for item in dataset_reports)
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "tolerances": {
            "model": args.model_atol,
            "optimizer": args.optimizer_atol,
            "train_loss": args.loss_atol,
            "grad_norm": args.grad_norm_atol,
            "parameter_norm": args.parameter_norm_atol,
            "learning_rate": 0,
            "global_tokens": 0,
            "scheduler_step": 0,
        },
        "data_hashes": hash_report,
        "training_metrics": metric_report,
        "model_parameters": model_report,
        "optimizer_state": optimizer_report,
        "scheduler_state_exact": scheduler_match,
        "scalar_trainer_state": scalar_state,
        "dataset_state": dataset_reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage B Training Resume Equivalence",
        "",
        f"- status: {report['status']}",
        f"- data hashes exact on all ranks: {hash_report['all_ranks_exact']}",
        f"- model max absolute difference: {model_report['max_abs_difference']}",
        f"- model mean absolute difference: {model_report['mean_abs_difference']}",
        f"- optimizer max absolute difference: {optimizer_report['max_abs_difference']}",
        f"- scheduler state exact: {scheduler_match}",
        f"- dataset states exact: {all(item['exact_match'] for item in dataset_reports)}",
        f"- model tolerance: {args.model_atol}",
        f"- optimizer tolerance: {args.optimizer_atol}",
        f"- loss tolerance: {args.loss_atol}",
    ]
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
