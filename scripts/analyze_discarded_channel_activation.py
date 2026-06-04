#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from mmfreelm.modules.activations import swiglu
from mmfreelm.modules.moe import compute_dense_channel_importance, select_dense_channel_indices
from scripts.eval_moe_by_domain import (
    build_domain_sequences,
    collate_domain_batch,
    read_validation_by_domain,
)
from scripts.train_moe_lm import ensure_cuda_device, precision_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze discarded dense FFN channel activation coverage.")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--val-data-source", type=str, default="datasets/SlimPajama-6B/data")
    parser.add_argument("--tokenizer-path", type=str, default="checkpoints/MMfreeLM-370M")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layer-indices", type=str, default="12-23")
    parser.add_argument("--shared-widths", type=str, default="2288,2560")
    parser.add_argument("--selection-mode", type=str, default="dense_top_channel")
    parser.add_argument("--domain-field", type=str, default="meta.redpajama_set_name")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--max-samples-per-domain", type=int, default=256)
    parser.add_argument("--max-sequences-per-domain", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_layer_indices(spec: str) -> List[int]:
    result: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            result.extend(list(range(start, end + step, step)))
        else:
            result.append(int(part))
    if not result:
        raise ValueError("No layer indices parsed from --layer-indices.")
    return result


def parse_widths(spec: str) -> List[int]:
    widths = sorted({int(part.strip()) for part in spec.split(",") if part.strip()})
    if not widths:
        raise ValueError("No widths parsed from --shared-widths.")
    return widths


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_entropy(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) > 0.0]
    if not vals:
        return 0.0
    total = sum(vals)
    if total <= 0:
        return 0.0
    probs = [v / total for v in vals]
    entropy = -sum(p * math.log(p) for p in probs if p > 0.0)
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 0.0
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def count_for_mass(values: torch.Tensor, mass_fraction: float) -> int:
    if values.numel() == 0:
        return 0
    sorted_vals = torch.sort(values, descending=True).values
    total = float(sorted_vals.sum().item())
    if total <= 0:
        return 0
    cutoff = total * float(mass_fraction)
    running = 0.0
    for idx, value in enumerate(sorted_vals.tolist(), start=1):
        running += float(value)
        if running >= cutoff:
            return idx
    return int(sorted_vals.numel())


@dataclass
class ActivationStats:
    token_count: int = 0
    element_count: int = 0
    shared_element_count: int = 0
    signed_sum: float = 0.0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    shared_abs_sum: float = 0.0
    token_abs_mean_gt_001_count: int = 0
    token_ge_10pct_shared_count: int = 0
    token_ge_25pct_shared_count: int = 0
    token_ge_50pct_shared_count: int = 0
    token_ge_100pct_shared_count: int = 0
    channel_abs_sum: torch.Tensor | None = None
    domain_abs_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    domain_shared_abs_sum: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    domain_token_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    domain_element_count: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def update(self, discarded_values: torch.Tensor, shared_abs_mean: torch.Tensor, domain: str) -> None:
        discarded_abs = discarded_values.abs()
        discarded_abs_mean = discarded_abs.mean(dim=1)
        denom = shared_abs_mean.clamp_min(1e-8)

        self.token_count += int(discarded_values.shape[0])
        self.element_count += int(discarded_values.numel())
        self.shared_element_count += int(shared_abs_mean.numel())
        self.signed_sum += float(discarded_values.sum().item())
        self.abs_sum += float(discarded_abs.sum().item())
        self.sq_sum += float(discarded_values.square().sum().item())
        self.shared_abs_sum += float(shared_abs_mean.sum().item())
        self.token_abs_mean_gt_001_count += int((discarded_abs_mean > 1e-2).sum().item())
        self.token_ge_10pct_shared_count += int((discarded_abs_mean >= denom * 0.10).sum().item())
        self.token_ge_25pct_shared_count += int((discarded_abs_mean >= denom * 0.25).sum().item())
        self.token_ge_50pct_shared_count += int((discarded_abs_mean >= denom * 0.50).sum().item())
        self.token_ge_100pct_shared_count += int((discarded_abs_mean >= denom).sum().item())
        if self.channel_abs_sum is None:
            self.channel_abs_sum = discarded_abs.sum(dim=0).detach().cpu()
        else:
            self.channel_abs_sum += discarded_abs.sum(dim=0).detach().cpu()

        self.domain_abs_sum[domain] += float(discarded_abs.sum().item())
        self.domain_shared_abs_sum[domain] += float(shared_abs_mean.sum().item())
        self.domain_token_count[domain] += int(discarded_values.shape[0])
        self.domain_element_count[domain] += int(discarded_values.numel())

    def to_summary(self) -> Dict[str, object]:
        token_count = max(self.token_count, 1)
        element_count = max(self.element_count, 1)
        shared_element_count = max(self.shared_element_count, 1)
        discarded_abs_mean = self.abs_sum / element_count
        shared_abs_mean = self.shared_abs_sum / shared_element_count
        channel_abs_sum = self.channel_abs_sum if self.channel_abs_sum is not None else torch.zeros(0)
        total_channel_abs = float(channel_abs_sum.sum().item()) if channel_abs_sum.numel() else 0.0
        top10_fraction = 0.0
        top25_fraction = 0.0
        if total_channel_abs > 0 and channel_abs_sum.numel() > 0:
            sorted_vals = torch.sort(channel_abs_sum, descending=True).values
            top10_fraction = float(sorted_vals[: min(10, sorted_vals.numel())].sum().item() / total_channel_abs)
            top25_fraction = float(sorted_vals[: min(25, sorted_vals.numel())].sum().item() / total_channel_abs)
        per_domain = {}
        total_domain_abs = sum(self.domain_abs_sum.values())
        for domain in sorted(self.domain_token_count.keys()):
            domain_tokens = max(int(self.domain_token_count[domain]), 1)
            domain_elements = max(int(self.domain_element_count[domain]), 1)
            domain_abs = float(self.domain_abs_sum.get(domain, 0.0))
            domain_shared_abs = float(self.domain_shared_abs_sum.get(domain, 0.0))
            per_domain[domain] = {
                "token_count": int(self.domain_token_count[domain]),
                "discarded_abs_mean": domain_abs / domain_elements,
                "shared_abs_mean": domain_shared_abs / domain_tokens,
                "discarded_to_shared_abs_mean_ratio": (
                    (domain_abs / domain_elements) / max(domain_shared_abs / domain_tokens, 1e-8)
                ),
                "abs_mass_fraction": (domain_abs / total_domain_abs) if total_domain_abs > 0 else 0.0,
            }
        return {
            "token_count": int(self.token_count),
            "element_count": int(self.element_count),
            "discarded_abs_mean": discarded_abs_mean,
            "discarded_rms": math.sqrt(self.sq_sum / element_count),
            "discarded_signed_mean": self.signed_sum / element_count,
            "shared_abs_mean": shared_abs_mean,
            "discarded_to_shared_abs_mean_ratio": discarded_abs_mean / max(shared_abs_mean, 1e-8),
            "token_abs_mean_gt_001_ratio": self.token_abs_mean_gt_001_count / token_count,
            "token_ge_10pct_shared_ratio": self.token_ge_10pct_shared_count / token_count,
            "token_ge_25pct_shared_ratio": self.token_ge_25pct_shared_count / token_count,
            "token_ge_50pct_shared_ratio": self.token_ge_50pct_shared_count / token_count,
            "token_ge_shared_ratio": self.token_ge_100pct_shared_count / token_count,
            "top10_channel_abs_mass_fraction": top10_fraction,
            "top25_channel_abs_mass_fraction": top25_fraction,
            "channels_for_50pct_abs_mass": count_for_mass(channel_abs_sum, 0.50),
            "channels_for_90pct_abs_mass": count_for_mass(channel_abs_sum, 0.90),
            "normalized_channel_abs_mass_entropy": normalize_entropy(channel_abs_sum.tolist()),
            "normalized_domain_abs_mass_entropy": normalize_entropy(self.domain_abs_sum.values()),
            "top_domain_abs_mass_fraction": (
                max((float(v) for v in self.domain_abs_sum.values()), default=0.0) / total_domain_abs
                if total_domain_abs > 0
                else 0.0
            ),
            "per_domain": per_domain,
        }


class DiscardedActivationCollector:
    def __init__(
        self,
        model: HGRNBitForCausalLM,
        layer_indices: List[int],
        shared_widths: List[int],
        selection_mode: str,
    ) -> None:
        self.model = model
        self.layer_indices = layer_indices
        self.shared_widths = shared_widths
        self.selection_mode = selection_mode
        self.current_domain = "unknown"
        self.stats: Dict[int, Dict[int, ActivationStats]] = {
            width: {layer_idx: ActivationStats() for layer_idx in layer_indices}
            for width in shared_widths
        }
        self.shared_indices: Dict[int, Dict[int, torch.Tensor]] = {}
        self.discarded_indices: Dict[int, Dict[int, torch.Tensor]] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self._prepare_indices()

    def _prepare_indices(self) -> None:
        for width in self.shared_widths:
            self.shared_indices[width] = {}
            self.discarded_indices[width] = {}
            for layer_idx in self.layer_indices:
                mlp = self.model.model.layers[layer_idx].mlp
                shared = select_dense_channel_indices(mlp, width, selection_mode=self.selection_mode).detach().long().cpu()
                intermediate_size = int(mlp.intermediate_size)
                all_indices = torch.arange(intermediate_size, dtype=torch.long)
                keep_mask = torch.ones(intermediate_size, dtype=torch.bool)
                keep_mask[shared] = False
                discarded = all_indices[keep_mask]
                self.shared_indices[width][layer_idx] = shared
                self.discarded_indices[width][layer_idx] = discarded

    def register(self) -> None:
        for layer_idx in self.layer_indices:
            module = self.model.model.layers[layer_idx].mlp.gate_proj

            def _hook(_module, _inputs, output, layer_idx: int = layer_idx) -> None:
                gate, value = output.float().chunk(2, dim=-1)
                activation = swiglu(gate, value).float()
                flat = activation.reshape(-1, activation.shape[-1])
                for width in self.shared_widths:
                    shared_idx = self.shared_indices[width][layer_idx].to(flat.device)
                    discarded_idx = self.discarded_indices[width][layer_idx].to(flat.device)
                    shared_abs_mean = flat.index_select(1, shared_idx).abs().mean(dim=1)
                    discarded_values = flat.index_select(1, discarded_idx)
                    self.stats[width][layer_idx].update(discarded_values, shared_abs_mean, self.current_domain)

            self.handles.append(module.register_forward_hook(_hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def layer_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {}
        for width in self.shared_widths:
            width_payload: Dict[str, object] = {"layers": {}, "aggregate": {}}
            aggregate = ActivationStats()
            for layer_idx in self.layer_indices:
                stats = self.stats[width][layer_idx]
                width_payload["layers"][f"layer_{layer_idx}"] = {
                    "shared_width": int(width),
                    "discarded_channel_count": int(self.discarded_indices[width][layer_idx].numel()),
                    "shared_channel_count": int(self.shared_indices[width][layer_idx].numel()),
                    "summary": stats.to_summary(),
                }
                aggregate.token_count += stats.token_count
                aggregate.element_count += stats.element_count
                aggregate.shared_element_count += stats.shared_element_count
                aggregate.signed_sum += stats.signed_sum
                aggregate.abs_sum += stats.abs_sum
                aggregate.sq_sum += stats.sq_sum
                aggregate.shared_abs_sum += stats.shared_abs_sum
                aggregate.token_abs_mean_gt_001_count += stats.token_abs_mean_gt_001_count
                aggregate.token_ge_10pct_shared_count += stats.token_ge_10pct_shared_count
                aggregate.token_ge_25pct_shared_count += stats.token_ge_25pct_shared_count
                aggregate.token_ge_50pct_shared_count += stats.token_ge_50pct_shared_count
                aggregate.token_ge_100pct_shared_count += stats.token_ge_100pct_shared_count
                if stats.channel_abs_sum is not None:
                    if aggregate.channel_abs_sum is None:
                        aggregate.channel_abs_sum = stats.channel_abs_sum.clone()
                    else:
                        aggregate.channel_abs_sum = torch.cat([aggregate.channel_abs_sum, stats.channel_abs_sum], dim=0)
                for domain, value in stats.domain_abs_sum.items():
                    aggregate.domain_abs_sum[domain] += value
                for domain, value in stats.domain_shared_abs_sum.items():
                    aggregate.domain_shared_abs_sum[domain] += value
                for domain, value in stats.domain_token_count.items():
                    aggregate.domain_token_count[domain] += value
                for domain, value in stats.domain_element_count.items():
                    aggregate.domain_element_count[domain] += value
            width_payload["aggregate"] = {
                "shared_width": int(width),
                "discarded_channel_count_total": int(sum(self.discarded_indices[width][layer].numel() for layer in self.layer_indices)),
                "summary": aggregate.to_summary(),
            }
            payload[f"shared_width_{width}"] = width_payload
        return payload


def build_domain_loaders(args: argparse.Namespace) -> tuple[Dict[str, DataLoader], Dict[str, object]]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    domain_texts, domain_row_counts, preview = read_validation_by_domain(
        val_data_source=args.val_data_source,
        text_field=args.text_field,
        domain_field=args.domain_field,
        max_samples_per_domain=args.max_samples_per_domain,
        seed=args.seed,
        save_samples=False,
    )

    domain_loaders: Dict[str, DataLoader] = {}
    manifest_domains: Dict[str, object] = {}
    for domain in sorted(domain_texts.keys()):
        sequences = build_domain_sequences(
            domain_texts=domain_texts[domain],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_sequences=args.max_sequences_per_domain,
        )
        if not sequences:
            continue
        domain_loaders[domain] = DataLoader(
            sequences,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_domain_batch,
        )
        manifest_domains[domain] = {
            "sampled_text_count": len(domain_texts[domain]),
            "raw_row_count": int(domain_row_counts.get(domain, 0)),
            "sequence_count": int(len(sequences)),
        }
    manifest = {
        "val_data_source": args.val_data_source,
        "domain_field": args.domain_field,
        "text_field": args.text_field,
        "max_samples_per_domain": args.max_samples_per_domain,
        "max_sequences_per_domain": args.max_sequences_per_domain,
        "domains": manifest_domains,
        "sample_preview": preview,
    }
    return domain_loaders, manifest


def run_analysis(args: argparse.Namespace) -> Dict[str, object]:
    device = ensure_cuda_device(args.device)
    model = HGRNBitForCausalLM.from_pretrained(args.checkpoint_path, torch_dtype=torch.bfloat16).to(device)
    model.config.use_moe = False
    model.config.moe_layer_indices = []
    model.eval()

    collector = DiscardedActivationCollector(
        model=model,
        layer_indices=args.layer_indices,
        shared_widths=args.shared_widths,
        selection_mode=args.selection_mode,
    )
    collector.register()
    precision_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[args.precision]

    domain_loaders, domain_manifest = build_domain_loaders(args)
    with torch.no_grad():
        for domain, loader in domain_loaders.items():
            collector.current_domain = domain
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                with precision_context(precision_dtype):
                    model(input_ids=input_ids)

    collector.close()
    payload = {
        "checkpoint_path": args.checkpoint_path,
        "selection_mode": args.selection_mode,
        "layer_indices": args.layer_indices,
        "shared_widths": args.shared_widths,
        "domain_manifest": domain_manifest,
        "results": collector.layer_payload(),
    }
    return payload


def build_digest(payload: Dict[str, object]) -> Dict[str, object]:
    rows = []
    for width_key, width_payload in payload["results"].items():
        width = int(width_key.split("_")[-1])
        summary = width_payload["aggregate"]["summary"]
        rows.append(
            {
                "shared_width": width,
                "discarded_channel_count_total": width_payload["aggregate"]["discarded_channel_count_total"],
                "discarded_abs_mean": summary["discarded_abs_mean"],
                "discarded_to_shared_abs_mean_ratio": summary["discarded_to_shared_abs_mean_ratio"],
                "token_ge_10pct_shared_ratio": summary["token_ge_10pct_shared_ratio"],
                "token_ge_25pct_shared_ratio": summary["token_ge_25pct_shared_ratio"],
                "token_ge_50pct_shared_ratio": summary["token_ge_50pct_shared_ratio"],
                "token_ge_shared_ratio": summary["token_ge_shared_ratio"],
                "normalized_domain_abs_mass_entropy": summary["normalized_domain_abs_mass_entropy"],
                "top_domain_abs_mass_fraction": summary["top_domain_abs_mass_fraction"],
                "normalized_channel_abs_mass_entropy": summary["normalized_channel_abs_mass_entropy"],
                "top10_channel_abs_mass_fraction": summary["top10_channel_abs_mass_fraction"],
                "channels_for_90pct_abs_mass": summary["channels_for_90pct_abs_mass"],
            }
        )
    return {
        "checkpoint_path": payload["checkpoint_path"],
        "layer_indices": payload["layer_indices"],
        "shared_widths": payload["shared_widths"],
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    args.layer_indices = parse_layer_indices(args.layer_indices)
    args.shared_widths = parse_widths(args.shared_widths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        args.output_dir / "run_spec.json",
        {
            "checkpoint_path": args.checkpoint_path,
            "val_data_source": args.val_data_source,
            "tokenizer_path": args.tokenizer_path,
            "layer_indices": args.layer_indices,
            "shared_widths": args.shared_widths,
            "selection_mode": args.selection_mode,
            "domain_field": args.domain_field,
            "text_field": args.text_field,
            "max_samples_per_domain": args.max_samples_per_domain,
            "max_sequences_per_domain": args.max_sequences_per_domain,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "precision": args.precision,
            "device": args.device,
            "seed": args.seed,
        },
    )

    payload = run_analysis(args)
    write_json(args.output_dir / "discarded_activation_coverage.json", payload)
    write_json(args.output_dir / "discarded_activation_digest.json", build_digest(payload))


if __name__ == "__main__":
    main()
