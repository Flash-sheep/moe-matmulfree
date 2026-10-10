#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.upcycling.data_utils import StatefulPackedParquetDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-resume tests for stateful parquet packing.")
    parser.add_argument("--data-source", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "ddp", "multiworker", "all"), default="all")
    parser.add_argument("--split", default="train")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-batches", type=int, default=200)
    parser.add_argument("--checkpoint-batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def batch_signature(batch: Dict[str, object]) -> Dict[str, object]:
    return {
        "input_ids": tensor_hash(batch["input_ids"]),
        "labels": tensor_hash(batch["labels"]),
        "attention_mask": tensor_hash(batch["attention_mask"]),
        "shape": list(batch["input_ids"].shape),
    }


def make_dataset(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    emit_state: bool,
) -> StatefulPackedParquetDataset:
    return StatefulPackedParquetDataset(
        data_source=args.data_source,
        tokenizer_path=args.tokenizer_path,
        max_length=args.sequence_length,
        split=args.split,
        shard_rank=rank,
        shard_world_size=world_size,
        seed=args.seed,
        shuffle_shards=True,
        emit_state=emit_state,
    )


def update_emitted_worker_states(
    batch: Dict[str, object],
    states: Dict[str, Dict[str, object]],
) -> None:
    worker_ids = batch.get("__worker_id__")
    state_json = batch.get("__state_json__")
    if worker_ids is None or state_json is None:
        return
    worker_values = worker_ids.tolist() if isinstance(worker_ids, torch.Tensor) else worker_ids
    for worker_id, payload in zip(worker_values, state_json):
        states[str(int(worker_id))] = json.loads(payload)


def collect(
    dataset: StatefulPackedParquetDataset,
    args: argparse.Namespace,
    num_batches: int,
    num_workers: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "drop_last": True,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 1
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(**loader_kwargs)
    iterator = iter(loader)
    signatures: List[Dict[str, object]] = []
    emitted_states: Dict[str, Dict[str, object]] = {}
    for _ in range(num_batches):
        batch = next(iterator)
        signatures.append(batch_signature(batch))
        update_emitted_worker_states(batch, emitted_states)
    if num_workers == 0:
        state = dataset.state_dict()
    else:
        state = dataset.state_dict()
        state["workers"] = deepcopy(emitted_states)
    del iterator
    del loader
    gc.collect()
    return signatures, state


def compare_signatures(
    control: List[Dict[str, object]],
    resumed: List[Dict[str, object]],
) -> Dict[str, object]:
    mismatch_indices = [
        index
        for index, (left, right) in enumerate(zip(control, resumed))
        if left != right
    ]
    return {
        "control_batches": len(control),
        "resumed_batches": len(resumed),
        "mismatch_count": len(mismatch_indices),
        "first_mismatch_index": mismatch_indices[0] if mismatch_indices else None,
        "input_ids_exact": all(a["input_ids"] == b["input_ids"] for a, b in zip(control, resumed)),
        "labels_exact": all(a["labels"] == b["labels"] for a, b in zip(control, resumed)),
        "attention_mask_exact": all(
            a["attention_mask"] == b["attention_mask"] for a, b in zip(control, resumed)
        ),
        "pass": len(control) == len(resumed) and not mismatch_indices,
    }


def run_continuity(
    args: argparse.Namespace,
    rank: int,
    world_size: int,
    num_workers: int,
) -> Dict[str, object]:
    emit_state = num_workers > 0
    control_dataset = make_dataset(args, rank, world_size, emit_state)
    control, _ = collect(control_dataset, args, args.num_batches, num_workers)

    before_dataset = make_dataset(args, rank, world_size, emit_state)
    before, checkpoint_state = collect(
        before_dataset,
        args,
        args.checkpoint_batch,
        num_workers,
    )
    resumed_dataset = make_dataset(args, rank, world_size, emit_state)
    resumed_dataset.load_state_dict(checkpoint_state)
    after, final_state = collect(
        resumed_dataset,
        args,
        args.num_batches - args.checkpoint_batch,
        num_workers,
    )
    combined = before + after
    comparison = compare_signatures(control, combined)
    worker_states = checkpoint_state["workers"]
    packing_remainders = {
        key: {
            "token_buffer_length": len(state["token_buffer"]),
            "metadata_buffer_length": len(state["metadata_buffer"]),
            "assigned_shard_position": state["assigned_shard_position"],
            "row_group_index": state["row_group_index"],
            "row_offset_in_group": state["row_offset_in_group"],
            "sequences_consumed": state["sequences_consumed"],
        }
        for key, state in worker_states.items()
    }
    resume_boundary_exact = (
        control[args.checkpoint_batch] == after[0]
        if args.checkpoint_batch < len(control) and after
        else False
    )
    return {
        "rank": rank,
        "world_size": world_size,
        "num_workers": num_workers,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "checkpoint_batch": args.checkpoint_batch,
        "comparison": comparison,
        "resume_boundary_exact": resume_boundary_exact,
        "packing_remainders": packing_remainders,
        "packing_remainder_nonempty": any(
            item["token_buffer_length"] > 0 for item in packing_remainders.values()
        ),
        "control_batch_hashes": [
            hashlib.sha256(json.dumps(item, sort_keys=True).encode("utf-8")).hexdigest()
            for item in control
        ],
        "final_dataset_state_summary": {
            key: {
                "sequences_consumed": state["sequences_consumed"],
                "tokens_emitted": state["tokens_emitted"],
                "documents_consumed": state["documents_consumed"],
            }
            for key, state in final_state["workers"].items()
        },
        "pass": comparison["pass"] and resume_boundary_exact,
    }


def write_report(output_dir: Path, report: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "streaming_resume_test_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Streaming Exact Resume Test",
        "",
        f"- mode: {report['mode']}",
        f"- pass: {report['pass']}",
    ]
    for result in report["results"]:
        lines.extend(
            [
                "",
                f"## rank={result['rank']} workers={result['num_workers']}",
                "",
                f"- batch hashes exact: {result['comparison']['pass']}",
                f"- input_ids exact: {result['comparison']['input_ids_exact']}",
                f"- labels exact: {result['comparison']['labels_exact']}",
                f"- attention_mask exact: {result['comparison']['attention_mask_exact']}",
                f"- resume boundary exact: {result['resume_boundary_exact']}",
                f"- packing remainder nonempty: {result['packing_remainder_nonempty']}",
            ]
        )
    (output_dir / "streaming_resume_test_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.checkpoint_batch <= 0 or args.checkpoint_batch >= args.num_batches:
        raise SystemExit("--checkpoint-batch must be between 0 and --num-batches")
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    results: List[Dict[str, object]] = []
    if args.mode in {"single", "all"} and not distributed:
        results.append(run_continuity(args, 0, 1, 0))
    if args.mode in {"multiworker", "all"} and not distributed:
        results.append(run_continuity(args, 0, 1, args.num_workers))
    if args.mode == "ddp":
        local_result = run_continuity(args, rank, world_size, 0)
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(local_result, gathered, dst=0)
        if rank == 0:
            results.extend(gathered)
            rank_hash_sets = [
                set(result["control_batch_hashes"])
                for result in results
            ]
            intersections = {
                f"{left}_{right}": len(rank_hash_sets[left] & rank_hash_sets[right])
                for left in range(world_size)
                for right in range(left + 1, world_size)
            }
            for result in results:
                result["inter_rank_hash_intersections"] = intersections
                result["inter_rank_disjoint"] = all(value == 0 for value in intersections.values())
                result["pass"] = result["pass"] and result["inter_rank_disjoint"]

    if not distributed or rank == 0:
        report = {
            "mode": args.mode,
            "data_source": args.data_source,
            "sequence_length": args.sequence_length,
            "seed": args.seed,
            "results": results,
            "pass": bool(results) and all(result["pass"] for result in results),
        }
        write_report(args.output_dir, report)
        if not report["pass"]:
            raise SystemExit(1)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
