#!/usr/bin/env python3
"""Compare initialization scales for the all-layer ternary partition MoE.

This is intentionally a diagnostic-only runner.  It constructs a fresh model for
each candidate, runs activation and gradient probes, and optionally performs a
small number of optimizer steps.  It never loads a checkpoint or writes model
weights, so it cannot modify an existing training experiment.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import sys
import time
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_dense_stageB_ddp import (  # noqa: E402
    activation_snapshot,
    build_scratch_model_from_config,
)
from scripts.run_dense_stageA_ddp import build_scratch_model  # noqa: E402
from scripts.run_sparse_upcycling import get_precision_dtype, precision_context  # noqa: E402


DEPTH_RESIDUAL_SCALE = 1.0 / math.sqrt(2.0 * 24.0)

VARIANTS: Dict[str, Dict[str, float | str]] = {
    "baseline_output1": {
        "description": "Current run: no MoE output damping and no expert down projection rescale.",
        "output_scale": 1.0,
        "down_proj_scale": 1.0,
    },
    "output_scale_0p5": {
        "description": "Keep expert weights unchanged and damp the complete MoE branch by 0.5.",
        "output_scale": 0.5,
        "down_proj_scale": 1.0,
    },
    "output_scale_0p25": {
        "description": "Keep expert weights unchanged and damp the complete MoE branch by 0.25.",
        "output_scale": 0.25,
        "down_proj_scale": 1.0,
    },
    "output_scale_0p05": {
        "description": "Damp the complete MoE branch to approximately the dense scratch FFN output regime.",
        "output_scale": 0.05,
        "down_proj_scale": 1.0,
    },
    "output_scale_0p01": {
        "description": "Damp the complete MoE branch to closely match the dense scratch FFN output regime.",
        "output_scale": 0.01,
        "down_proj_scale": 1.0,
    },
    "downproj_scale_0p01": {
        "description": "Encode the small residual scale in each expert down projection instead of the MoE output.",
        "output_scale": 1.0,
        "down_proj_scale": 0.01,
    },
    "downproj_scale_0p0075": {
        "description": "Approximate dense FFN branch amplitude using a 0.0075 expert down projection scale.",
        "output_scale": 1.0,
        "down_proj_scale": 0.0075,
    },
    "downproj_depth_rescaled": {
        "description": "Restore GPT-2-style residual projection scale 1/sqrt(2*num_layers) on each expert down projection.",
        "output_scale": 1.0,
        "down_proj_scale": DEPTH_RESIDUAL_SCALE,
    },
    "downproj_depth_rescaled_output0p5": {
        "description": "Depth-rescaled expert down projection plus a conservative 0.5 MoE branch scale.",
        "output_scale": 0.5,
        "down_proj_scale": DEPTH_RESIDUAL_SCALE,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=REPO_ROOT / "configs/pretrain/partition_capacity_4e_all_layers_random_scratch_relaxed30b_15b.yaml",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=sorted(VARIANTS),
        default=sorted(VARIANTS),
    )
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=0)
    parser.add_argument("--include-dense-reference", action="store_true")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor_summary(tensor: torch.Tensor) -> Dict[str, float | int | bool]:
    value = tensor.detach().float()
    finite = torch.isfinite(value)
    safe = value[finite]
    if safe.numel() == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "non_finite_count": int((~finite).sum().item()),
            "finite": False,
        }
    return {
        "mean": float(safe.mean().item()),
        "std": float(safe.std(unbiased=False).item()),
        "rms": float(torch.sqrt(safe.square().mean()).item()),
        "min": float(safe.min().item()),
        "max": float(safe.max().item()),
        "non_finite_count": int((~finite).sum().item()),
        "finite": bool(torch.isfinite(safe).all().item()),
    }


def make_fixed_batch(raw: Dict[str, object], args: argparse.Namespace, device: torch.device) -> Dict[str, torch.Tensor]:
    """Use one deterministic synthetic batch so every candidate sees identical tokens."""
    vocab_size = int(raw["model"].get("vocab_size", 0) or 0)
    if vocab_size <= 0:
        base_config = json.loads((Path(raw["model"]["base_config_path"]) / "config.json").read_text())
        vocab_size = int(base_config["vocab_size"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 1000)
    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(args.batch_size, args.seq_len),
        generator=generator,
        dtype=torch.long,
    )
    input_ids = input_ids.to(device)
    return {
        "input_ids": input_ids,
        "labels": input_ids.clone(),
        "attention_mask": torch.ones_like(input_ids),
    }


def apply_down_projection_scale(model: torch.nn.Module, scale: float) -> None:
    if scale == 1.0:
        return
    with torch.no_grad():
        for block in model.model.layers:
            if not getattr(block, "use_moe", False):
                raise RuntimeError("The initialization smoke expects MoE in every model layer.")
            for expert in block.mlp.experts:
                expert.down_proj.weight.mul_(scale)


def activation_summary(snapshot: Dict[str, object]) -> Dict[str, object]:
    layers = snapshot["layers"]
    residual = [float(row["residual_output"]["rms"]) for row in layers]
    ffn = [float(row["ffn_output"]["rms"]) for row in layers]
    ratios = [float(row["ffn_update_to_residual_input_ratio"]) for row in layers]
    return {
        "loss": float(snapshot["loss"]),
        "logits": snapshot["logits"],
        "has_non_finite": bool(snapshot["has_non_finite"]),
        "num_layers": len(layers),
        "first_residual_rms": residual[0],
        "middle_residual_rms": residual[len(residual) // 2],
        "last_residual_rms": residual[-1],
        "max_residual_rms": max(residual),
        "last_first_residual_ratio": residual[-1] / max(residual[0], 1e-12),
        "first_ffn_rms": ffn[0],
        "middle_ffn_rms": ffn[len(ffn) // 2],
        "last_ffn_rms": ffn[-1],
        "max_ffn_to_residual_input_ratio": max(ratios),
        "layers": layers,
    }


def gradient_probe(model: torch.nn.Module, batch: Dict[str, torch.Tensor], precision: str) -> Dict[str, object]:
    model.train()
    model.zero_grad(set_to_none=True)
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    with precision_context(precision_dtype):
        outputs = model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
            output_router_logits=False,
            return_dict=True,
        )
        loss = outputs.loss
    loss.backward()
    grad_squares = 0.0
    grad_records = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            grad_records.append({"name": name, "exists": False})
            continue
        grad = parameter.grad.detach().float()
        finite = torch.isfinite(grad)
        safe = grad[finite]
        grad_squares += float(grad.square().sum().item())
        grad_records.append(
            {
                "name": name,
                "exists": True,
                "rms": float(torch.sqrt(safe.square().mean()).item()) if safe.numel() else float("nan"),
                "max_abs": float(safe.abs().max().item()) if safe.numel() else float("nan"),
                "non_finite_count": int((~finite).sum().item()),
            }
        )
    expert_records = [row for row in grad_records if ".mlp.experts." in row["name"]]
    router_records = [row for row in grad_records if ".mlp.router." in row["name"]]
    return {
        "loss": float(loss.detach().cpu().item()),
        "gradient_norm": math.sqrt(grad_squares),
        "parameter_count": len(grad_records),
        "grad_exists_count": sum(bool(row["exists"]) for row in grad_records),
        "grad_missing_count": sum(not bool(row["exists"]) for row in grad_records),
        "non_finite_gradient_count": sum(int(row.get("non_finite_count", 0)) for row in grad_records),
        "expert_grad_exists_count": sum(bool(row["exists"]) for row in expert_records),
        "router_grad_exists_count": sum(bool(row["exists"]) for row in router_records),
        "records": grad_records,
    }


def short_train(
    model: torch.nn.Module,
    raw: Dict[str, object],
    batch: Dict[str, torch.Tensor],
    precision: str,
    steps: int,
) -> Dict[str, object]:
    if steps <= 0:
        return {"enabled": False, "steps": 0, "losses": []}
    training = raw["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("peak_learning_rate", 0.004)),
        betas=tuple(float(value) for value in training.get("betas", [0.9, 0.95])),
        eps=float(training.get("epsilon", 1e-8)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": precision})())
    losses: List[float] = []
    grad_norms: List[float] = []
    finite = True
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with precision_context(precision_dtype):
            outputs = model(
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                output_router_logits=False,
                return_dict=True,
            )
            loss = outputs.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("grad_clip", 1.0)))
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
        grad_value = float(grad_norm.detach().cpu().item())
        losses.append(loss_value)
        grad_norms.append(grad_value)
        finite = finite and math.isfinite(loss_value) and math.isfinite(grad_value)
    return {
        "enabled": True,
        "steps": steps,
        "losses": losses,
        "gradient_norms": grad_norms,
        "finite": finite,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "gradient_norm_max": max(grad_norms) if grad_norms else None,
    }


def write_markdown(path: Path, payload: Dict[str, object]) -> None:
    lines = [
        "# Partition Initialization Smoke Test",
        "",
        f"- config: `{payload['config_path']}`",
        f"- device: `{payload['device']}`",
        f"- seed: `{payload['seed']}`",
        f"- batch: `{payload['batch_size']} x {payload['seq_len']}`",
        f"- train steps per candidate: `{payload['train_steps']}`",
        "",
        "| candidate | output scale | down-proj scale | initial loss | first residual RMS | last residual RMS | last/first | max FFN/residual | finite | short smoke |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for name, result in payload["results"].items():
        initial = result["initial_activation"]
        short = result["short_train"]
        lines.append(
            "| {name} | {out:.4g} | {down:.4g} | {loss:.4f} | {first:.4f} | {last:.4f} | {ratio:.4f} | {ffn:.4f} | {finite} | {short} |".format(
                name=name,
                out=result["output_scale"],
                down=result["down_proj_scale"],
                loss=initial["loss"],
                first=initial["first_residual_rms"],
                last=initial["last_residual_rms"],
                ratio=initial["last_first_residual_ratio"],
                ffn=initial["max_ffn_to_residual_input_ratio"],
                finite=not initial["has_non_finite"],
                short=("pass" if short.get("finite", True) else "fail") if short["enabled"] else "not-run",
            )
        )
    lines.extend(
        [
            "",
            "The script is diagnostic-only. It does not load checkpoints and does not write model weights.",
            f"Residual projection reference scale: `1/sqrt(2*24) = {DEPTH_RESIDUAL_SCALE:.8f}`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite smoke output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=args.overwrite)
    raw = yaml.safe_load(args.config_path.read_text(encoding="utf-8"))
    if bool(raw["initialization"].get("pretrained_checkpoint")):
        raise ValueError("Smoke test config must not specify a pretrained checkpoint.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    batch = make_fixed_batch(raw, args, device)
    payload: Dict[str, object] = {
        "config_path": str(args.config_path),
        "device": str(device),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "precision": args.precision,
        "train_steps": args.train_steps,
        "include_dense_reference": args.include_dense_reference,
        "depth_residual_scale": DEPTH_RESIDUAL_SCALE,
        "results": {},
    }
    for variant_name in args.variants:
        variant = VARIANTS[variant_name]
        started = time.time()
        candidate_raw = deepcopy(raw)
        candidate_raw["moe"]["output_scale"] = float(variant["output_scale"])
        candidate_raw["moe"]["expert_down_proj_init_scale"] = float(variant["down_proj_scale"])
        # The shared initializer prints per-layer ternary statistics. Keep the
        # smoke output machine-readable while retaining the statistics in the
        # model construction path used by the real runner.
        with redirect_stdout(io.StringIO()):
            model = build_scratch_model_from_config(candidate_raw, seed=args.seed, device=device)
        initial = activation_summary(
            activation_snapshot(model, batch, device, args.precision, global_tokens=0)
        )
        gradient = gradient_probe(model, batch, args.precision)
        short = short_train(model, candidate_raw, batch, args.precision, args.train_steps)
        post_train = None
        if short["enabled"]:
            post_train = activation_summary(
                activation_snapshot(model, batch, device, args.precision, global_tokens=args.train_steps)
            )
        result = {
            "description": variant["description"],
            "output_scale": float(variant["output_scale"]),
            "down_proj_scale": float(variant["down_proj_scale"]),
            "initial_activation": initial,
            "initial_gradient": gradient,
            "short_train": short,
            "post_train_activation": post_train,
            "elapsed_sec": time.time() - started,
        }
        payload["results"][variant_name] = result
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if args.include_dense_reference:
        started = time.time()
        with redirect_stdout(io.StringIO()):
            dense_model = build_scratch_model(
                raw["model"]["base_config_path"],
                seed=args.seed,
                device=device,
            )
        dense_initial = activation_summary(
            activation_snapshot(dense_model, batch, device, args.precision, global_tokens=0)
        )
        dense_gradient = gradient_probe(dense_model, batch, args.precision)
        dense_short = short_train(dense_model, raw, batch, args.precision, args.train_steps)
        dense_post = None
        if dense_short["enabled"]:
            dense_post = activation_summary(
                activation_snapshot(dense_model, batch, device, args.precision, global_tokens=args.train_steps)
            )
        payload["dense_reference"] = {
            "description": "Original dense scratch model with its standard model initialization.",
            "initial_activation": dense_initial,
            "initial_gradient": dense_gradient,
            "short_train": dense_short,
            "post_train_activation": dense_post,
            "elapsed_sec": time.time() - started,
        }
        del dense_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload["created_at"] = time.time()
    (args.output_dir / "initialization_smoke_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_dir / "initialization_smoke_results.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
