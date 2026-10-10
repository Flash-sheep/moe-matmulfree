#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dense_stageA_ddp import build_scratch_model
from scripts.run_sparse_upcycling import get_precision_dtype, precision_context


class ValidationDataset(Dataset):
    def __init__(self, path: Path, limit: int):
        self.array = np.load(path, mmap_mode="r")
        self.limit = min(limit, len(self.array))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        input_ids = torch.from_numpy(
            np.asarray(self.array[index, :-1], dtype=np.int64)
        )
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare single-rank and two-rank Stage B validation."
    )
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--sequences", type=int, default=128)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser.parse_args()


@torch.no_grad()
def accumulate(model, loader, device, precision):
    loss_sum = 0.0
    token_count = 0
    sequence_count = 0
    dtype, _ = get_precision_dtype(
        type("Cfg", (), {"precision": precision})()
    )
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with precision_context(dtype):
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                output_router_logits=False,
                return_dict=True,
            )
        effective_tokens = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
        loss_sum += float(outputs.loss.detach().cpu()) * effective_tokens
        token_count += effective_tokens
        sequence_count += int(input_ids.shape[0])
    return loss_sum, token_count, sequence_count


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise SystemExit("This diagnostic must run with torchrun --nproc_per_node=2")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    raw = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))
    dataset = ValidationDataset(Path(raw["validation"]["array"]), args.sequences)
    model = build_scratch_model(
        raw["model"]["base_config_path"],
        int(raw["initialization"]["seed"]),
        device,
    )
    ddp_model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    ddp_model.eval()
    distributed_indices = list(range(rank, len(dataset), world_size))
    distributed_loader = DataLoader(
        Subset(dataset, distributed_indices),
        batch_size=int(raw["validation"]["batch_size_per_rank"]),
        shuffle=False,
        num_workers=0,
    )
    local = accumulate(
        ddp_model.module,
        distributed_loader,
        device,
        raw["training"]["precision"],
    )
    aggregate = torch.tensor(local, dtype=torch.float64, device=device)
    dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
    distributed_loss = float(aggregate[0].item() / aggregate[1].item())

    reference_payload = [None]
    if rank == 0:
        full_loader = DataLoader(
            dataset,
            batch_size=int(raw["validation"]["batch_size_per_rank"]),
            shuffle=False,
            num_workers=0,
        )
        reference = accumulate(
            ddp_model.module,
            full_loader,
            device,
            raw["training"]["precision"],
        )
        reference_payload[0] = {
            "loss": reference[0] / reference[1],
            "effective_tokens": reference[1],
            "sequences": reference[2],
        }
    dist.broadcast_object_list(reference_payload, src=0)
    reference = reference_payload[0]
    difference = abs(distributed_loss - reference["loss"])
    report = {
        "status": "PASS" if difference <= args.atol else "FAIL",
        "world_size": world_size,
        "sequences": len(dataset),
        "distributed_effective_tokens": int(aggregate[1].item()),
        "distributed_sequences": int(aggregate[2].item()),
        "distributed_loss": distributed_loss,
        "single_rank_reference_loss": reference["loss"],
        "max_abs_loss_difference": difference,
        "tolerance": args.atol,
        "weighted_by_effective_tokens": True,
        "rank_partitions_disjoint": True,
        "optimizer_or_scheduler_updated": False,
        "model_training_state_after_eval": bool(ddp_model.training),
    }
    if rank == 0:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
