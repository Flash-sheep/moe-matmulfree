#!/usr/bin/env python3
"""Per-domain MoE diagnostic evaluation.

Loads a MoE checkpoint, reads SlimPajama-6B validation parquet grouped by
`meta.redpajama_set_name`, and evaluates PPL, pair usage, expert usage,
and router entropy per domain.

Outputs: JSON report, Markdown summary, CSV tables, divergence matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from scripts.train_moe_lm import ensure_cuda_device, get_precision_dtype, precision_context


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-domain MoE diagnostic evaluation")
    p.add_argument("--config-path", type=Path, required=True)
    p.add_argument("--pretrained-path", type=str, required=True,
                   help="Base dense checkpoint path, e.g. checkpoints/MMfreeLM-370M")
    p.add_argument("--checkpoint-path", type=Path, default=None,
                   help="MoE checkpoint to evaluate. If not set, auto-looks in --output-dir.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Experiment output dir (for reading checkpoint and writing diagnostics)")
    p.add_argument("--val-data-source", type=str, default="datasets/SlimPajama-6B/data")
    p.add_argument("--tokenizer-path", type=str, default="checkpoints/MMfreeLM-370M")
    p.add_argument("--domain-field", type=str, default="meta.redpajama_set_name")
    p.add_argument("--text-field", type=str, default="text")
    p.add_argument("--max-samples-per-domain", type=int, default=256)
    p.add_argument("--max-sequences-per-domain", type=int, default=256)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--precision", type=str, default="bf16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--save-domain-samples", action="store_true", default=False)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
# I/O utilities
# ──────────────────────────────────────────────────────────────

def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Checkpoint resolution
# ──────────────────────────────────────────────────────────────

def resolve_checkpoint(output_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    candidates = [
        output_dir / "checkpoint_best_proxy",
        output_dir / "checkpoint_best",
    ]
    for c in candidates:
        if (c / "config.json").exists() or c.is_dir():
            return c
    # try checkpoint_last
    for sub in sorted(output_dir.glob("checkpoint*"), reverse=True):
        if sub.is_dir():
            return sub
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


# ──────────────────────────────────────────────────────────────
# Model loading (reuse project patterns)
# ──────────────────────────────────────────────────────────────

def load_moe_checkpoint_directly(checkpoint_dir: Path, device_str: str) -> tuple:
    device = ensure_cuda_device(device_str)
    model = HGRNBitForCausalLM.from_pretrained(
        str(checkpoint_dir),
        torch_dtype=torch.bfloat16,
    ).to(device)
    routing_mode = getattr(model.config, "moe_routing_mode", "standard")
    num_experts = getattr(model.config, "moe_num_experts", 6)
    return model, device, routing_mode, num_experts


def load_checkpoint_weights(model: HGRNBitForCausalLM, checkpoint_dir: Path) -> None:
    """Load MoE checkpoint weights (safetensors or pytorch bin)."""
    ckpt_files = sorted(checkpoint_dir.glob("*.safetensors")) + sorted(checkpoint_dir.glob("pytorch_model*.bin"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model weights found in {checkpoint_dir}")

    state_dict = {}
    for fpath in ckpt_files:
        if fpath.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict.update(load_file(str(fpath)))
        else:
            state_dict.update(torch.load(fpath, map_location="cpu"))
    model.load_state_dict(state_dict, strict=False)


# ──────────────────────────────────────────────────────────────
# Data reading (domain-aware, document-boundary preserved)
# ──────────────────────────────────────────────────────────────

def read_validation_by_domain(
    val_data_source: str,
    text_field: str,
    domain_field: str,
    max_samples_per_domain: int,
    seed: int,
    save_samples: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, Any]]:
    """Read validation parquet, group texts by domain.

    Returns:
        domain_texts: {domain: [text1, text2, ...]}
        domain_row_counts: {domain: raw_row_count}
        sample_preview: preview dict if save_samples else {}
    """
    import glob as gl
    import pyarrow.parquet as pq

    val_files = sorted(gl.glob(f"{val_data_source}/validation-*.parquet"))
    if not val_files:
        raise FileNotFoundError(f"No validation parquet found in {val_data_source}")

    rng = __import__("random")
    rng.seed(seed)

    all_texts: Dict[str, List[str]] = defaultdict(list)
    domain_counts: Dict[str, int] = defaultdict(int)

    for vf in val_files:
        pf = pq.ParquetFile(vf)
        for batch in pf.iter_batches(columns=[text_field, domain_field], batch_size=4096):
            texts = batch.column(0).to_pylist()
            metas = batch.column(1).to_pylist()
            for txt, meta in zip(texts, metas):
                if txt is None:
                    continue
                domain = None
                if meta is not None:
                    # meta is a dict with redpajama_set_name
                    domain = meta.get("redpajama_set_name") if isinstance(meta, dict) else None
                if domain is None:
                    domain = "unknown"
                domain_counts[domain] += 1
                if domain_counts[domain] <= max_samples_per_domain:
                    all_texts[domain].append(str(txt))

    # Sample uniformly if more texts than max_samples_per_domain
    sampled: Dict[str, List[str]] = {}
    row_counts: Dict[str, int] = {}
    for domain, texts in all_texts.items():
        row_counts[domain] = domain_counts[domain]
        if len(texts) > max_samples_per_domain:
            sampled[domain] = rng.sample(texts, max_samples_per_domain)
        else:
            sampled[domain] = texts

    preview = {}
    if save_samples:
        for domain, texts in sampled.items():
            preview[domain] = [t[:120].replace("\n", "\\n") for t in texts[:3]]

    return sampled, row_counts, preview


# ──────────────────────────────────────────────────────────────
# Tokenization & sequence construction (per-domain, no cross-domain mixing)
# ──────────────────────────────────────────────────────────────

def build_domain_sequences(
    domain_texts: List[str],
    tokenizer,
    max_length: int,
    max_sequences: int,
) -> List[torch.Tensor]:
    """Tokenize texts within one domain, build max_length+1 sequences.

    Loss shift: input_ids = seq[:-1], labels = seq[1:] (same convention as StreamingTextDataset).
    Only tokens from the same domain are mixed (with EOS separator between texts).
    """
    sequences: List[torch.Tensor] = []
    buffer: List[int] = []

    for text in domain_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if buffer:
            # Insert EOS between texts
            buffer.append(tokenizer.eos_token_id)
        buffer.extend(tokens)
        while len(buffer) >= max_length + 1:
            chunk = buffer[:max_length + 1]
            buffer = buffer[max_length:]
            sequences.append(torch.tensor(chunk, dtype=torch.long))
            if len(sequences) >= max_sequences:
                return sequences[:max_sequences]
    return sequences


def collate_domain_batch(batch: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pad batch of sequences to same length.

    Loss shift convention: The model internally computes next-token prediction
    (shifts labels internally). We pass input_ids as labels (same convention as
    StreamingTextDataset and the training loop).
    """
    max_len = max(s.shape[0] for s in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, :s.shape[0]] = s
    # input_ids = seq[:-1], labels = seq[:-1] (model does internal shift)
    return {"input_ids": padded[:, :-1], "labels": padded[:, :-1]}


# ──────────────────────────────────────────────────────────────
# Router metrics aggregation
# ──────────────────────────────────────────────────────────────

def safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(sum(values) / len(values))


def safe_tensor_mean(accumulator: Optional[torch.Tensor]) -> Optional[List[float]]:
    if accumulator is None:
        return None
    return accumulator.tolist()


def aggregate_router_metrics(
    all_metrics: List[Dict[str, torch.Tensor]],
    num_experts: int,
) -> Dict[str, Any]:
    """Aggregate per-batch router metrics across all batches for one domain.

    all_metrics: list of per-layer-metrics from outputs.router_metrics per batch.
    """
    if not all_metrics:
        return {}

    # Collect scalar metrics
    router_entropies: List[float] = []
    pair_entropies: List[float] = []
    pair_entropies_norm: List[float] = []

    # Collect vector metrics (sum over batches)
    tokens_expert_sum = torch.zeros(num_experts)
    pair_fraction_sum: Optional[torch.Tensor] = None
    free_expert_sum: Optional[torch.Tensor] = None
    num_batches = 0

    for layer_metrics_list in all_metrics:
        for layer_metrics in layer_metrics_list:
            if not layer_metrics:
                continue
            num_batches += 1

            if "router_entropy" in layer_metrics:
                re = layer_metrics["router_entropy"]
                router_entropies.append(float(re.mean().item() if re.ndim > 0 else re.item()))

            if "tokens_per_expert" in layer_metrics:
                tpe = layer_metrics["tokens_per_expert"]
                tokens_expert_sum += tpe.cpu().float()

            if "pair_fraction" in layer_metrics:
                pf = layer_metrics["pair_fraction"].cpu().float()
                if pair_fraction_sum is None:
                    pair_fraction_sum = torch.zeros_like(pf)
                pair_fraction_sum += pf

            if "pair_entropy" in layer_metrics:
                pe = layer_metrics["pair_entropy"]
                pair_entropies.append(float(pe.mean().item() if pe.ndim > 0 else pe.item()))

            if "pair_entropy_normalized" in layer_metrics:
                pe = layer_metrics["pair_entropy_normalized"]
                pair_entropies_norm.append(float(pe.mean().item() if pe.ndim > 0 else pe.item()))

            if "free_expert_fraction" in layer_metrics:
                fef = layer_metrics["free_expert_fraction"].cpu().float()
                if free_expert_sum is None:
                    free_expert_sum = torch.zeros_like(fef)
                free_expert_sum += fef

    result: Dict[str, Any] = {}
    if num_batches > 0:
        if num_batches > 0:
            result["tokens_per_expert"] = (tokens_expert_sum / num_batches).tolist()
        if pair_fraction_sum is not None:
            result["pair_fraction"] = (pair_fraction_sum / num_batches).tolist()
        if free_expert_sum is not None:
            result["free_expert_fraction"] = (free_expert_sum / num_batches).tolist()
    result["router_entropy"] = safe_mean(router_entropies)
    result["pair_entropy"] = safe_mean(pair_entropies)
    result["pair_entropy_normalized"] = safe_mean(pair_entropies_norm)

    # Compute expert-level entropy
    if result.get("tokens_per_expert"):
        tpe = torch.tensor(result["tokens_per_expert"]).clamp_min(1e-9)
        tpe_norm = tpe / tpe.sum()
        expert_entropy = float(-(tpe_norm * tpe_norm.log()).sum().item())
        result["expert_entropy"] = expert_entropy
        result["expert_entropy_normalized"] = expert_entropy / math.log(max(num_experts, 2))

    return result


# ──────────────────────────────────────────────────────────────
# Domain evaluation loop
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_domain(
    model: HGRNBitForCausalLM,
    domain_texts: List[str],
    max_length: int,
    max_sequences: int,
    batch_size: int,
    device: torch.device,
    precision_dtype,
    num_experts: int,
    tokenizer,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate PPL and router metrics for one domain."""
    model.eval()

    seqs = build_domain_sequences(domain_texts, tokenizer, max_length, max_sequences)
    if not seqs:
        return {"num_sequences": 0, "num_tokens": 0, "error": "no_sequences"}, {}

    # Create batches
    batches = [seqs[i:i + batch_size] for i in range(0, len(seqs), batch_size)]

    total_loss = 0.0
    total_lm_loss = 0.0
    total_batches = 0
    total_tokens = 0
    all_router_metrics: List = []

    for batch_seqs in batches:
        batch = collate_domain_batch(batch_seqs)
        input_ids = batch["input_ids"].to(device)
        total_tokens += input_ids.numel()

        with precision_context(precision_dtype):
            outputs = model(
                input_ids=input_ids,
                labels=batch["labels"].to(device),
                output_router_logits=True,
                return_dict=True,
            )

        total_loss += float(outputs.loss.detach().cpu())
        total_lm_loss += float(outputs.lm_loss.detach().cpu()) if outputs.lm_loss is not None else float(outputs.loss.detach().cpu())
        total_batches += 1

        if outputs.router_metrics:
            all_router_metrics.append(outputs.router_metrics)

    if total_batches == 0:
        return {"num_sequences": len(seqs), "num_tokens": 0, "error": "zero_batches"}, {}

    avg_loss = total_loss / total_batches
    eval_result = {
        "num_sequences": len(seqs),
        "num_batches": total_batches,
        "num_tokens": total_tokens,
        "lm_loss": total_lm_loss / total_batches,
        "ppl": float(math.exp(min(avg_loss, 20.0))),
        "val_loss": avg_loss,
    }

    router_result = aggregate_router_metrics(all_router_metrics, num_experts)
    return eval_result, router_result


# ──────────────────────────────────────────────────────────────
# Divergence computation
# ──────────────────────────────────────────────────────────────

def compute_jsd_matrix(domain_data: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Compute pairwise Jensen-Shannon divergence of distributions across domains.

    key: metric key holding a list of floats (e.g. 'pair_fraction', 'tokens_per_expert').
    """
    domains = sorted(domain_data.keys())
    dists: Dict[str, List[float]] = {}
    for d in domains:
        val = domain_data[d].get(key)
        if val and isinstance(val, list) and sum(val) > 0:
            arr = torch.tensor(val).clamp_min(1e-9)
            dists[d] = (arr / arr.sum()).tolist()
        else:
            dists[d] = None

    active = {d: v for d, v in dists.items() if v is not None}
    if len(active) < 2:
        return {"error": "insufficient_domains", "active_domains": list(active.keys())}

    matrix: Dict[str, Dict[str, float]] = {}
    domain_list = sorted(active.keys())
    n = len(domain_list)

    for i in range(n):
        di = domain_list[i]
        matrix[di] = {}
        for j in range(n):
            dj = domain_list[j]
            if i == j:
                matrix[di][dj] = 0.0
            elif j < i:
                matrix[di][dj] = matrix[dj][di]
            else:
                p = torch.tensor(active[di])
                q = torch.tensor(active[dj])
                m = 0.5 * (p + q)
                kl_pm = (p * (p / m).log()).sum().item()
                kl_qm = (q * (q / m).log()).sum().item()
                matrix[di][dj] = float(0.5 * kl_pm + 0.5 * kl_qm)

    return {"domains": domain_list, "matrix": matrix}


def compute_pair_usage_variation(domain_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-pair standard deviation across domains."""
    domains = sorted(domain_data.keys())
    pair_fractions: Dict[str, List[float]] = {}
    for d in domains:
        pf = domain_data[d].get("pair_fraction")
        if pf and isinstance(pf, list):
            for idx, val in enumerate(pf):
                pair_fractions.setdefault(str(idx), []).append(float(val))

    variation = {}
    for pair_id, values in pair_fractions.items():
        if len(values) > 1:
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            variation[pair_id] = {"mean": mean_val, "std": std_val}
    return variation


def compute_expert_usage_variation(domain_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Compute per-expert standard deviation across domains."""
    domains = sorted(domain_data.keys())
    expert_fractions: Dict[str, List[float]] = {}
    for d in domains:
        eu = domain_data[d].get("tokens_per_expert")
        if eu and isinstance(eu, list):
            for idx, val in enumerate(eu):
                expert_fractions.setdefault(str(idx), []).append(float(val))

    variation = {}
    for expert_id, values in expert_fractions.items():
        if len(values) > 1:
            mean_val = sum(values) / len(values)
            std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
            variation[expert_id] = {"mean": mean_val, "std": std_val}
    return variation


# ──────────────────────────────────────────────────────────────
# Text statistics
# ──────────────────────────────────────────────────────────────

def compute_text_stats(texts: List[str]) -> Dict[str, Any]:
    if not texts:
        return {}
    lengths = sorted([len(t) for t in texts])
    n = len(lengths)
    return {
        "raw_num_rows": n,
        "mean_chars": sum(lengths) / n,
        "median_chars": lengths[n // 2],
        "p95_chars": lengths[int(0.95 * (n - 1))] if n > 1 else lengths[0],
        "min_chars": lengths[0],
        "max_chars": lengths[-1],
    }


# ──────────────────────────────────────────────────────────────
# Markdown report generation
# ──────────────────────────────────────────────────────────────

def generate_markdown_report(
    experiment_name: str,
    checkpoint_path: str,
    results: Dict[str, Dict[str, Any]],
    domain_row_counts: Dict[str, int],
    out_path: Path,
) -> None:
    domains = sorted(results.keys())
    lines = [
        f"# Domain Diagnostic Report: {experiment_name}",
        "",
        f"- **Checkpoint**: `{checkpoint_path}`",
        f"- **Val data**: `datasets/SlimPajama-6B/data`",
        f"- **Domain field**: `meta.redpajama_set_name`",
        "",
        "## Per-Domain PPL and Router Metrics",
        "",
        "| Domain | Rows | Seqs | Tokens | LM Loss | PPL | Router Entropy | Pair Entropy | Expert Entropy |",
        "|--------|------|------|--------|---------|-----|----------------|--------------|----------------|",
    ]

    for d in domains:
        r = results[d]
        drc = domain_row_counts.get(d, "?")
        lines.append(
            f"| {d} | {drc} | {r.get('num_sequences','?')} | {r.get('num_tokens','?')} | "
            f"{r.get('lm_loss','?'):.4f}" if isinstance(r.get('lm_loss'), (int, float)) else f"| {d} | {drc} | {r.get('num_sequences','?')} | {r.get('num_tokens','?')} | ? |"
        )

    lines.append("")

    # Pair usage table
    all_pairs = set()
    for d in domains:
        pf = results[d].get("pair_fraction", [])
        if isinstance(pf, list):
            for i in range(len(pf)):
                all_pairs.add(i)

    if all_pairs:
        lines.append("## Per-Domain Pair Usage")
        lines.append("")
        header = "| Domain | " + " | ".join(f"Pair {i}" for i in sorted(all_pairs)) + " |"
        sep = "|--------|" + "|".join("---" for _ in sorted(all_pairs)) + "|"
        lines.append(header)
        lines.append(sep)
        for d in domains:
            pf = results[d].get("pair_fraction", [])
            cells = []
            for i in sorted(all_pairs):
                val = pf[i] if i < len(pf) else 0.0
                cells.append(f"{val:.4f}")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        lines.append("")

    # Expert usage table
    all_experts = set()
    for d in domains:
        eu = results[d].get("tokens_per_expert", [])
        if isinstance(eu, list):
            for i in range(len(eu)):
                all_experts.add(i)

    if all_experts:
        lines.append("## Per-Domain Expert Usage")
        lines.append("")
        header = "| Domain | " + " | ".join(f"E{i}" for i in sorted(all_experts)) + " |"
        sep = "|--------|" + "|".join("---" for _ in sorted(all_experts)) + "|"
        lines.append(header)
        lines.append(sep)
        for d in domains:
            eu = results[d].get("tokens_per_expert", [])
            cells = []
            for i in sorted(all_experts):
                val = eu[i] if i < len(eu) else 0.0
                cells.append(f"{val:.4f}")
            lines.append(f"| {d} | " + " | ".join(cells) + " |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# CSV exports
# ──────────────────────────────────────────────────────────────

def write_csv_tables(results: Dict[str, Dict[str, Any]], out_dir: Path) -> None:
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pair usage CSV
    rows = []
    for domain, r in results.items():
        pf = r.get("pair_fraction", [])
        if isinstance(pf, list):
            for i, v in enumerate(pf):
                rows.append({"domain": domain, "pair_id": i, "pair_fraction": float(v)})
    if rows:
        with open(out_dir / "domain_pair_usage.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["domain", "pair_id", "pair_fraction"])
            w.writeheader()
            w.writerows(rows)

    # Expert usage CSV
    rows = []
    for domain, r in results.items():
        eu = r.get("tokens_per_expert", [])
        if isinstance(eu, list):
            for i, v in enumerate(eu):
                rows.append({"domain": domain, "expert_id": i, "expert_fraction": float(v)})
    if rows:
        with open(out_dir / "domain_expert_usage.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["domain", "expert_id", "expert_fraction"])
            w.writeheader()
            w.writerows(rows)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Resolve paths ──
    config = load_json(args.config_path) if args.config_path.exists() else {}
    experiment_name = config.get("experiment_name", args.output_dir.name)

    checkpoint_path = resolve_checkpoint(args.output_dir, args.checkpoint_path)
    out_dir = args.out_dir or (args.output_dir / "domain_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval_moe_by_domain] Experiment: {experiment_name}")
    print(f"[eval_moe_by_domain] Checkpoint: {checkpoint_path}")
    print(f"[eval_moe_by_domain] Output dir: {out_dir}")

    # ── Load model directly from MoE checkpoint ──
    print(f"[eval_moe_by_domain] Loading MoE checkpoint from {checkpoint_path} ...")
    model, device, routing_mode, num_experts = load_moe_checkpoint_directly(checkpoint_path, args.device)
    moe_layer_indices = list(getattr(model.config, "moe_layer_indices", []) or [])
    print(f"[eval_moe_by_domain] Routing mode: {routing_mode}, experts: {num_experts}, MoE layers: {moe_layer_indices}")
    model.eval()
    precision_dtype, _ = get_precision_dtype(
        type("Cfg", (), {"precision": args.precision})(),
    )

    # ── Load tokenizer ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Read and group validation data by domain ──
    print("[eval_moe_by_domain] Reading validation data by domain ...")
    domain_texts, domain_row_counts, preview = read_validation_by_domain(
        args.val_data_source,
        args.text_field,
        args.domain_field,
        args.max_samples_per_domain,
        args.seed,
        args.save_domain_samples,
    )

    print(f"[eval_moe_by_domain] Found {len(domain_texts)} domains:")
    for d in sorted(domain_texts.keys()):
        print(f"  {d}: {len(domain_texts[d])} sampled / {domain_row_counts[d]} total rows")

    # ── Evaluate per domain ──
    results: Dict[str, Dict[str, Any]] = {}
    for domain in sorted(domain_texts.keys()):
        texts = domain_texts[domain]
        print(f"[eval_moe_by_domain] Evaluating domain: {domain} ({len(texts)} texts) ...")
        eval_result, router_result = evaluate_domain(
            model=model,
            domain_texts=texts,
            max_length=args.max_length,
            max_sequences=args.max_sequences_per_domain,
            batch_size=args.batch_size,
            device=device,
            precision_dtype=precision_dtype,
            num_experts=num_experts,
            tokenizer=tokenizer,
        )
        stats = compute_text_stats(texts)

        combined = {}
        combined.update(stats)
        combined.update(eval_result)
        combined.update(router_result)
        if domain_row_counts.get(domain, 0) < 10:
            combined["low_sample_warning"] = True
        results[domain] = combined

        ppl = combined.get("ppl", "?")
        print(f"  -> PPL={ppl}, seqs={eval_result.get('num_sequences', 0)}, tokens={eval_result.get('num_tokens', 0)}")

    # ── Compute divergence ──
    pair_divergence = compute_jsd_matrix(results, "pair_fraction")
    expert_divergence = compute_jsd_matrix(results, "tokens_per_expert")
    pair_variation = compute_pair_usage_variation(results)
    expert_variation = compute_expert_usage_variation(results)

    # Mean pair JSD
    pair_matrix = pair_divergence.get("matrix", {})
    pair_jsd_values = []
    for di in pair_matrix:
        for dj, val in pair_matrix[di].items():
            if di < dj:
                pair_jsd_values.append(val)
    mean_pair_jsd = sum(pair_jsd_values) / len(pair_jsd_values) if pair_jsd_values else None

    # ── Write outputs ──
    full_report = {
        "experiment_name": experiment_name,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(args.config_path),
        "val_data_source": args.val_data_source,
        "domain_field": args.domain_field,
        "max_samples_per_domain": args.max_samples_per_domain,
        "max_sequences_per_domain": args.max_sequences_per_domain,
        "num_experts": num_experts,
        "moe_layer_indices": moe_layer_indices,
        "domains": results,
        "pair_jsd_matrix": pair_divergence,
        "expert_jsd_matrix": expert_divergence,
        "pair_usage_variation_across_domains": pair_variation,
        "expert_usage_variation_across_domains": expert_variation,
        "mean_pair_jsd_across_domains": mean_pair_jsd,
    }

    write_json(out_dir / "domain_eval_results.json", full_report)
    if args.save_domain_samples:
        write_json(out_dir / "domain_sample_preview.json", preview)

    generate_markdown_report(experiment_name, str(checkpoint_path), results, domain_row_counts, out_dir / "domain_eval_summary.md")
    write_csv_tables(results, out_dir)
    write_json(out_dir / "domain_routing_divergence.json", {
        "pair_jsd_matrix": pair_divergence,
        "expert_jsd_matrix": expert_divergence,
        "mean_pair_jsd": mean_pair_jsd,
        "pair_usage_variation": pair_variation,
        "expert_usage_variation": expert_variation,
    })

    print(f"\n[eval_moe_by_domain] Done. Outputs in {out_dir}/")
    print(f"  - domain_eval_results.json")
    print(f"  - domain_eval_summary.md")
    print(f"  - domain_pair_usage.csv")
    print(f"  - domain_expert_usage.csv")
    print(f"  - domain_routing_divergence.json")
    if mean_pair_jsd is not None:
        print(f"\n  Mean pair JSD across domains: {mean_pair_jsd:.6f}")


if __name__ == "__main__":
    main()
