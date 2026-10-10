#!/usr/bin/env python3
"""Compare a dense and a MoE Stage-B checkpoint on one fixed validation set.

Both checkpoints produced by ``run_dense_stageB_ddp.py`` keep the model state
inside ``trainer_state.pt``.  This evaluator reconstructs each architecture
from its saved/base config and applies the same fixed-array loss protocol to
both models.  It does not read the online training dataset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    """Read the deterministic validation array without materializing it."""

    def __init__(self, array_path: Path, limit: int):
        self.array = np.load(array_path, mmap_mode="r")
        self.limit = min(int(limit), len(self.array))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # Keep the extra stored token out of the input. The model performs its
        # standard internal next-token shift against labels=input_ids.
        input_ids = torch.from_numpy(
            np.asarray(self.array[index, :-1], dtype=np.int64)
        )
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-validation comparison for dense and partition MoE checkpoints."
    )
    parser.add_argument("--dense-base-config", type=Path, required=True)
    parser.add_argument("--dense-trainer-state", type=Path, required=True)
    parser.add_argument("--moe-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--validation-array", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-sequences", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_trainer_state(path: Path) -> Dict[str, object]:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def load_dense_model(
    base_config_path: Path,
    trainer_state_path: Path,
) -> HGRNBitForCausalLM:
    model = build_scratch_model(
        str(base_config_path),
        seed=20260721,
        device=torch.device("cpu"),
    )
    state = load_trainer_state(trainer_state_path)
    model.load_state_dict(state["model"], strict=True)
    del state
    gc.collect()
    return model


def load_moe_model(checkpoint_dir: Path) -> HGRNBitForCausalLM:
    config = HGRNBitConfig.from_pretrained(str(checkpoint_dir))
    model = HGRNBitForCausalLM(config)
    state = load_trainer_state(checkpoint_dir / "trainer_state.pt")
    model.load_state_dict(state["model"], strict=True)
    del state
    gc.collect()
    return model


def model_parameter_info(model: HGRNBitForCausalLM) -> Dict[str, object]:
    params = list(model.parameters())
    seen = set()
    unique_count = 0
    for param in params:
        if id(param) not in seen:
            seen.add(id(param))
            unique_count += param.numel()
    return {
        "parameter_count": int(sum(param.numel() for param in params)),
        "unique_parameter_count": int(unique_count),
        "weights_dtype": str(next(model.parameters()).dtype),
        "use_moe": bool(getattr(model.config, "use_moe", False)),
        "moe_family": getattr(model.config, "moe_family", None),
        "moe_layer_count": len(getattr(model.config, "moe_layer_indices", []) or []),
        "moe_num_experts": getattr(model.config, "moe_num_experts", None),
        "moe_top_k": getattr(model.config, "moe_num_experts_per_tok", None),
        "moe_expert_width": getattr(model.config, "moe_expert_intermediate_size", None),
    }


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
        # HGRNBitForCausalLM shifts labels internally. Each sequence therefore
        # contributes sequence_length - 1 prediction targets.
        effective_tokens = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
        total_loss_sum += float(outputs.loss.detach().float().cpu()) * effective_tokens
        total_effective_tokens += effective_tokens
        total_sequences += int(input_ids.shape[0])
    loss = total_loss_sum / max(total_effective_tokens, 1)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "effective_tokens": total_effective_tokens,
        "sequences": total_sequences,
        "elapsed_sec": time.time() - started,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_markdown(path: Path, report: Dict[str, object]) -> None:
    dense = report["dense_scratch_15b"]
    moe = report["partition_capacity_4e_15b"]
    comparison = report["comparison"]
    lines = [
        "# Stage-B Fixed Validation Comparison",
        "",
        "| model | loss | PPL | sequences | effective tokens |",
        "|---|---:|---:|---:|---:|",
        f"| dense scratch 15B | {dense['loss']:.8f} | {dense['ppl']:.8f} | {dense['sequences']} | {dense['effective_tokens']} |",
        f"| partition 4E top2 15B | {moe['loss']:.8f} | {moe['ppl']:.8f} | {moe['sequences']} | {moe['effective_tokens']} |",
        "",
        f"- PPL delta (partition - dense): `{comparison['ppl_delta_partition_minus_dense']:.8f}`",
        f"- Loss delta (partition - dense): `{comparison['loss_delta_partition_minus_dense']:.8f}`",
        f"- Partition relative PPL: `{comparison['partition_relative_ppl_percent']:.4f}%`",
        "",
        "Both models use the same fixed validation array, labels=input_ids, internal next-token shift, effective-token weighted loss, and BF16 autocast.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    dense = load_dense_model(args.dense_base_config, args.dense_trainer_state)
    dense_info = model_parameter_info(dense)
    dense_metrics = evaluate_model(dense.to(device), loader, device)
    del dense
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    moe = load_moe_model(args.moe_checkpoint_dir)
    moe_info = model_parameter_info(moe)
    moe_metrics = evaluate_model(moe.to(device), loader, device)
    del moe
    gc.collect()

    partition_relative_ppl = (
        (moe_metrics["ppl"] / dense_metrics["ppl"] - 1.0) * 100.0
    )
    report = {
        "protocol": {
            "validation_array": str(args.validation_array),
            "validation_array_sha256": sha256_file(args.validation_array),
            "validation_manifest": str(args.validation_manifest)
            if args.validation_manifest
            else None,
            "validation_sequences": len(dataset),
            "stored_sequence_width": int(dataset.array.shape[1]),
            "model_input_sequence_length": int(dataset.array.shape[1] - 1),
            "tokenizer": str(args.tokenizer_path) if args.tokenizer_path else None,
            "batch_size": args.batch_size,
            "precision": "bf16_autocast",
            "label_alignment": "labels=input_ids; model performs internal next-token shift",
            "effective_tokens": len(dataset) * (int(dataset.array.shape[1]) - 2),
            "same_validation_and_loss_implementation": True,
        },
        "dense_scratch_15b": {
            "checkpoint": str(args.dense_trainer_state),
            "architecture_config": str(args.dense_base_config),
            "optimizer_step": 114441,
            "global_tokens": 15000010752,
            "model_info": dense_info,
            **dense_metrics,
        },
        "partition_capacity_4e_15b": {
            "checkpoint_dir": str(args.moe_checkpoint_dir),
            "trainer_state": str(args.moe_checkpoint_dir / "trainer_state.pt"),
            "optimizer_step": 114441,
            "global_tokens": 15000010752,
            "model_info": moe_info,
            **moe_metrics,
        },
        "comparison": {
            "ppl_delta_partition_minus_dense": moe_metrics["ppl"] - dense_metrics["ppl"],
            "loss_delta_partition_minus_dense": moe_metrics["loss"] - dense_metrics["loss"],
            "partition_relative_ppl_percent": partition_relative_ppl,
        },
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_path.with_suffix(".md"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
