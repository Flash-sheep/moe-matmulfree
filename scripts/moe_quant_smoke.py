#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.configuration_hgrn_bit import HGRNBitConfig
from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-check HGRNBit MoE variants.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--hidden-ratio", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--quantized-experts", action="store_true")
    parser.add_argument("--output-json", type=Path)
    return parser


def tensor_to_list(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    return value


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)

    config = HGRNBitConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_ratio=args.hidden_ratio,
        use_moe=args.use_moe,
        moe_num_experts=args.num_experts,
        moe_num_experts_per_tok=args.top_k,
        moe_output_router_logits=args.use_moe,
        moe_use_quantized_experts=args.quantized_experts,
    )
    model = HGRNBitForCausalLM(config).to(args.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history = []
    for step in range(args.steps):
        input_ids = torch.randint(
            low=0,
            high=args.vocab_size,
            size=(args.batch_size, args.seq_len),
            device=args.device,
        )
        outputs = model(
            input_ids=input_ids,
            labels=input_ids,
            output_router_logits=args.use_moe,
            return_dict=True,
        )
        loss = outputs.loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        record = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "lm_loss": tensor_to_list(outputs.lm_loss),
            "router_aux_loss": tensor_to_list(outputs.router_aux_loss),
        }
        if outputs.router_metrics:
            first_layer_metrics = outputs.router_metrics[0]
            record["router_entropy"] = tensor_to_list(first_layer_metrics["router_entropy"])
            record["tokens_per_expert"] = tensor_to_list(first_layer_metrics["tokens_per_expert"])
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
