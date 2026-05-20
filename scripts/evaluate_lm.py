#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.upcycling import ExpertMonitor, StreamingTextDataset
from scripts.run_sparse_upcycling import collate_streaming_batch
from scripts.train_moe_lm import ensure_cuda_device, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained LM checkpoint.")
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--data-source", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = ensure_cuda_device(args.device)
    tokenizer_path = args.tokenizer_path or args.checkpoint_path

    dataset = StreamingTextDataset(
        data_source=args.data_source,
        tokenizer_path=tokenizer_path,
        max_length=args.max_length,
        split="validation",
        text_field=args.text_field,
        max_samples=args.max_samples,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_streaming_batch)

    model = HGRNBitForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    eval_args = type(
        "EvalArgs",
        (),
        {
            "precision": args.precision,
            "max_eval_batches": None,
            "use_moe": bool(getattr(model.config, "use_moe", False)),
        },
    )()
    metrics = evaluate(model, dataloader, device, eval_args)

    moe_indices = getattr(model.config, "moe_layer_indices", []) or []
    if moe_indices:
        monitor = ExpertMonitor(model, moe_indices)
        metrics["expert_metrics"] = monitor.compute_metrics()

    payload = json.dumps(metrics, ensure_ascii=False, indent=2)
    if args.output_path is not None:
        args.output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
