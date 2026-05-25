#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.complement_pair_diagnostics_lib import (
    DEFAULT_DATA_SOURCE,
    DEFAULT_TOKENIZER_PATH,
    EvalSpec,
    apply_eval_overrides,
    build_validation_loader,
    evaluate_with_router_diagnostics,
    load_model,
    write_json,
)


MODE_CONFIG = {
    "normal_uniform_scaled": {
        "pair_weight_mode": "uniform",
        "output_scale": 2.0,
        "active_path_fair": True,
        "notes": "Matches current checkpoint evaluation path: hard complement pair, uniform 0.5/0.5, scale 2.0.",
    },
    "learned_pair_weights_scaled": {
        "pair_weight_mode": "router",
        "output_scale": 2.0,
        "active_path_fair": True,
        "notes": "Hard complement pair, learned softmax weights within selected pair, scale 2.0.",
    },
    "learned_pair_weights_no_scale": {
        "pair_weight_mode": "router",
        "output_scale": 1.0,
        "active_path_fair": True,
        "notes": "Hard complement pair, learned softmax weights within selected pair, no scale compensation.",
    },
    "uniform_no_scale": {
        "pair_weight_mode": "uniform",
        "output_scale": 1.0,
        "active_path_fair": True,
        "notes": "Hard complement pair, uniform 0.5/0.5, no scale compensation.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval-only ablations for complement-pair MoE combine rules.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-source", type=str, default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", default=list(MODE_CONFIG.keys()))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = EvalSpec(
        checkpoint_path=args.checkpoint,
        data_source=args.data_source,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        max_eval_batches=args.eval_batches,
        text_field=args.text_field,
        precision=args.precision,
        device=args.device,
    )
    model, device = load_model(spec.checkpoint_path, device=spec.device)
    dataloader = build_validation_loader(spec)
    results: List[Dict] = []
    skipped_modes: List[Dict] = []

    for mode in args.modes:
        if mode == "pair_softmax_mixture_scaled":
            skipped_modes.append(
                {
                    "mode": mode,
                    "status": "skipped",
                    "notes": "Not implemented in this diagnostic pass; would require non-fair multi-pair activation.",
                }
            )
            continue
        if mode not in MODE_CONFIG:
            raise ValueError(f"Unsupported mode `{mode}`.")
        cfg = MODE_CONFIG[mode]
        with apply_eval_overrides(
            model=model,
            pair_weight_mode=cfg["pair_weight_mode"],
            output_scale_override=cfg["output_scale"],
        ):
            metrics = evaluate_with_router_diagnostics(
                model=model,
                dataloader=dataloader,
                device=device,
                precision=spec.precision,
                max_eval_batches=spec.max_eval_batches,
            )
        results.append(
            {
                "mode": mode,
                "val_loss": metrics["val_loss"],
                "val_lm_loss": metrics["val_lm_loss"],
                "val_router_aux_loss": metrics["val_router_aux_loss"],
                "val_ppl": metrics["val_ppl"],
                "router_entropy": metrics.get("router_entropy"),
                "pair_usage": metrics["pair_usage"],
                "expert_usage": {
                    "tokens_per_expert": metrics.get("tokens_per_expert"),
                    "global_expert_share_normalized": metrics["pair_usage"].get("global_expert_share_normalized"),
                    "expert_entropy": metrics["pair_usage"].get("expert_entropy"),
                    "expert_entropy_normalized": metrics["pair_usage"].get("expert_entropy_normalized"),
                    "expert_load_imbalance": metrics["pair_usage"].get("expert_load_imbalance"),
                },
                "tokens_per_expert": metrics.get("tokens_per_expert"),
                "moe_output_scale": cfg["output_scale"],
                "active_path_fair": cfg["active_path_fair"],
                "pair_weight_mode": cfg["pair_weight_mode"],
                "notes": cfg["notes"],
            }
        )

    payload = {
        "checkpoint_path": str(spec.checkpoint_path),
        "data_source": spec.data_source,
        "tokenizer_path": spec.tokenizer_path,
        "batch_size": spec.batch_size,
        "max_eval_batches": spec.max_eval_batches,
        "max_samples": spec.max_samples,
        "max_length": spec.max_length,
        "precision": spec.precision,
        "results": results,
        "skipped_modes": skipped_modes,
    }
    written = write_json(spec.output_path, payload)
    print(written)


if __name__ == "__main__":
    main()
