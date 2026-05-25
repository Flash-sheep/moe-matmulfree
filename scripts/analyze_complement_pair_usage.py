#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.complement_pair_diagnostics_lib import (
    DEFAULT_DATA_SOURCE,
    DEFAULT_TOKENIZER_PATH,
    EvalSpec,
    build_validation_loader,
    evaluate_with_router_diagnostics,
    load_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze complement-pair usage on a trained checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-source", type=str, default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--tokenizer-path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output", type=Path, required=True)
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
    metrics = evaluate_with_router_diagnostics(
        model=model,
        dataloader=dataloader,
        device=device,
        precision=spec.precision,
        max_eval_batches=spec.max_eval_batches,
    )
    payload = {
        "checkpoint_path": str(spec.checkpoint_path),
        "data_source": spec.data_source,
        "tokenizer_path": spec.tokenizer_path,
        "mode": "pair_usage_only",
        "batch_size": spec.batch_size,
        "max_eval_batches": spec.max_eval_batches,
        "max_samples": spec.max_samples,
        "max_length": spec.max_length,
        "precision": spec.precision,
        "diagnostic_notes": [
            "Token-position and token-loss bucket diagnostics were skipped in this pass.",
            "Expert token fractions sum to top_k=2 under complement-pair routing.",
            "Expert entropy/load imbalance are computed from normalized expert assignment shares.",
        ],
        **metrics,
    }
    written = write_json(spec.output_path, payload)
    print(written)


if __name__ == "__main__":
    main()
