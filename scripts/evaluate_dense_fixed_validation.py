#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from scripts.run_dense_stageA_ddp import build_scratch_model
from scripts.run_sparse_upcycling import precision_context


class FixedValidationDataset(Dataset):
    def __init__(self, array_path: Path, limit: int):
        self.array = np.load(array_path, mmap_mode="r")
        self.limit = min(int(limit), len(self.array))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # The stored extra token is retained in the source array so the model
        # performs its standard internal next-token shift.
        input_ids = torch.from_numpy(
            np.asarray(self.array[index, :-1], dtype=np.int64)
        )
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pretrained and scratch dense MMfreeLM on one fixed validation set."
    )
    parser.add_argument("--pretrained-path", type=Path, required=True)
    parser.add_argument("--trainer-state", type=Path, required=True)
    parser.add_argument("--base-config-path", type=Path, required=True)
    parser.add_argument("--validation-array", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_scratch_from_trainer_state(
    base_config_path: Path,
    trainer_state_path: Path,
    device: torch.device,
) -> HGRNBitForCausalLM:
    model = build_scratch_model(str(base_config_path), 20260721, torch.device("cpu"))
    state = torch.load(
        trainer_state_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    model.load_state_dict(state["model"], strict=True)
    del state
    gc.collect()
    return model.to(device)


def load_pretrained(path: Path, device: torch.device) -> HGRNBitForCausalLM:
    # Keep weights in the same default dtype as the scratch checkpoint. Both
    # evaluations use the same BF16 autocast for activations and kernels.
    model = HGRNBitForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return model.to(device)


@torch.inference_mode()
def evaluate_model(
    model: HGRNBitForCausalLM,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    total_loss_sum = 0.0
    total_effective_tokens = 0
    total_sequences = 0
    started = time.time()
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with precision_context(torch.bfloat16):
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                output_router_logits=False,
                return_dict=True,
            )
        # HGRNBitForCausalLM shifts labels internally, so each sequence has
        # sequence_length - 1 effective prediction targets.
        effective_tokens = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
        total_loss_sum += float(outputs.loss.detach().float().cpu()) * effective_tokens
        total_effective_tokens += effective_tokens
        total_sequences += int(input_ids.shape[0])
    loss = total_loss_sum / total_effective_tokens
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "effective_tokens": total_effective_tokens,
        "sequences": total_sequences,
        "elapsed_sec": time.time() - started,
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation.")
    device = torch.device(args.device)
    dataset = FixedValidationDataset(args.validation_array, args.max_sequences)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    pretrained = load_pretrained(args.pretrained_path, device)
    pretrained_metrics = evaluate_model(pretrained, loader, device)
    del pretrained
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    scratch = load_scratch_from_trainer_state(
        args.base_config_path,
        args.trainer_state,
        device,
    )
    scratch_metrics = evaluate_model(scratch, loader, device)
    del scratch
    gc.collect()

    report = {
        "protocol": {
            "validation_array": str(args.validation_array),
            "validation_sequences": len(dataset),
            "stored_sequence_width": int(dataset.array.shape[1]),
            "model_input_sequence_length": int(dataset.array.shape[1] - 1),
            "tokenizer": str(args.tokenizer_path),
            "batch_size": args.batch_size,
            "precision": "bf16_autocast",
            "weights_dtype": "float32",
            "label_alignment": "labels=input_ids; model performs the only internal next-token shift",
            "effective_tokens": len(dataset) * (int(dataset.array.shape[1]) - 2),
            "same_validation_and_loss_implementation": True,
        },
        "pretrained_370m": {
            "checkpoint": str(args.pretrained_path),
            **pretrained_metrics,
        },
        "scratch_15b": {
            "trainer_state": str(args.trainer_state),
            "optimizer_step": 114441,
            "global_tokens": 15000010752,
            **scratch_metrics,
        },
    }
    report["comparison"] = {
        "ppl_delta_scratch_minus_pretrained": (
            scratch_metrics["ppl"] - pretrained_metrics["ppl"]
        ),
        "loss_delta_scratch_minus_pretrained": (
            scratch_metrics["loss"] - pretrained_metrics["loss"]
        ),
        "scratch_relative_ppl_percent": (
            (scratch_metrics["ppl"] / pretrained_metrics["ppl"] - 1.0) * 100.0
        ),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
