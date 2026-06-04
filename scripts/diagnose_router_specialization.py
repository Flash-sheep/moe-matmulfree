#!/usr/bin/env python3
"""Comprehensive MoE router specialization diagnostics.

Runs 5 analyses in one pass on a MoE checkpoint:
  1. Hidden-state domain probe (linear classifier)
  2. Router logits by domain
  3. Loss-conditioned routing
  4. Hidden-state cluster → routing correspondence
  5. Pair+free residual contribution (pair+free checkpoints only)

All diagnostics reuse a single model load + forward pass to minimise cost.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mmfreelm.models.hgrn_bit.modeling_hgrn_bit import HGRNBitForCausalLM
from scripts.train_moe_lm import ensure_cuda_device, get_precision_dtype, precision_context


# ────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MoE router specialization diagnostics")
    p.add_argument("--config-path", type=Path)
    p.add_argument("--pretrained-path", type=str, default="checkpoints/MMfreeLM-370M")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint-path", type=Path)
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
    p.add_argument("--layers", type=str, default="12,16,20,23",
                   help="Comma-separated MoE layer indices to probe")
    p.add_argument("--max-hidden-tokens", type=int, default=4096,
                   help="Max hidden-state tokens to cache per layer per domain")
    p.add_argument("--probe-epochs", type=int, default=10)
    p.add_argument("--skip-cluster", action="store_true")
    p.add_argument("--out-dir", type=Path)
    return p.parse_args()


# ────────────────────────────────────────────────────────
# I/O utilities
# ────────────────────────────────────────────────────────

def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, fieldnames: List[str], rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ────────────────────────────────────────────────────────
# Checkpoint resolution
# ────────────────────────────────────────────────────────

def resolve_checkpoint(output_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    for c in [output_dir / "checkpoint_best_proxy", output_dir / "checkpoint_best"]:
        if (c / "config.json").exists() or c.is_dir():
            return c
    for sub in sorted(output_dir.glob("checkpoint*"), reverse=True):
        if sub.is_dir():
            return sub
    raise FileNotFoundError(f"No checkpoint found in {output_dir}")


# ────────────────────────────────────────────────────────
# Data reading
# ────────────────────────────────────────────────────────

def read_validation_by_domain(val_data_source, text_field, domain_field,
                              max_samples, seed) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    import glob as gl
    import pyarrow.parquet as pq
    import random as rng_mod

    rng_mod.seed(seed)
    val_files = sorted(gl.glob(f"{val_data_source}/validation-*.parquet"))
    if not val_files:
        raise FileNotFoundError(f"No validation parquet in {val_data_source}")

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
                domain = meta.get("redpajama_set_name") if isinstance(meta, dict) and meta else None
                if domain is None:
                    domain = "unknown"
                domain_counts[domain] += 1
                if domain_counts[domain] <= max_samples:
                    all_texts[domain].append(str(txt))

    sampled = {}
    rows = {}
    for domain, texts in all_texts.items():
        rows[domain] = domain_counts[domain]
        if len(texts) > max_samples:
            sampled[domain] = rng_mod.sample(texts, max_samples)
        else:
            sampled[domain] = texts
    return sampled, rows


def build_domain_sequences(domain_texts, tokenizer, max_length, max_sequences):
    sequences = []
    buffer = []
    for text in domain_texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if buffer:
            buffer.append(tokenizer.eos_token_id)
        buffer.extend(tokens)
        while len(buffer) >= max_length + 1:
            chunk = buffer[:max_length + 1]
            buffer = buffer[max_length:]
            sequences.append(torch.tensor(chunk, dtype=torch.long))
            if len(sequences) >= max_sequences:
                return sequences
    return sequences


def collate_pad(batch):
    max_len = max(s.shape[0] for s in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, s in enumerate(batch):
        padded[i, :s.shape[0]] = s
    return padded[:, :-1]


# ────────────────────────────────────────────────────────
# Hook infrastructure for hidden-state collection
# ────────────────────────────────────────────────────────

class HiddenCollector:
    """Collect router-input hidden states from specified MoE layers.

    Hooks into SparseMoEBlock's forward_pre_hook to capture the hidden state
    that goes into the router (before expert computation).
    """

    def __init__(self, model, layer_indices, max_tokens_per_layer=4096):
        self.model = model
        self.layer_indices = list(layer_indices)
        self.max_tokens = max_tokens_per_layer
        self.hidden = {li: [] for li in self.layer_indices}
        self.hooks = []
        self._collected = {li: 0 for li in self.layer_indices}

    def _make_hook(self, li):
        moe_block = self.model.model.layers[li].mlp

        def pre_hook(module, args):
            """args[0] is the input hidden state to the MoE block (router input)."""
            if self._collected[li] >= self.max_tokens:
                return
            hs = args[0].detach().cpu()
            remaining = self.max_tokens - self._collected[li]
            flat = hs.reshape(-1, hs.shape[-1])
            n = min(flat.shape[0], remaining)
            self.hidden[li].append(flat[:n])
            self._collected[li] += n

        return moe_block.register_forward_pre_hook(pre_hook)

    def register(self):
        for li in self.layer_indices:
            self.hooks.append(self._make_hook(li))
        return self

    def remove(self):
        for h in self.hooks:
            h.remove()

    def get_tensors(self):
        result = {}
        for li in self.layer_indices:
            if self.hidden[li]:
                result[li] = torch.cat(self.hidden[li], dim=0)
            else:
                result[li] = torch.zeros(0, self.model.config.hidden_size)
        return result

    def reset(self, domain):
        for li in self.layer_indices:
            self.hidden[li] = []
            self._collected[li] = 0


# ────────────────────────────────────────────────────────
# Router logits collector (hook on SparseMoEBlock)
# ────────────────────────────────────────────────────────

class RouterLogitCollector:
    """Collect router logits per forward pass."""

    def __init__(self, model, moe_layer_indices):
        self.model = model
        self.layer_indices = list(moe_layer_indices)
        self.logits = {li: [] for li in self.layer_indices}
        self.tokens_per_expert = {li: [] for li in self.layer_indices}
        self.pair_fraction = {li: [] for li in self.layer_indices}
        self.selected_pair = {li: [] for li in self.layer_indices}
        self.router_entropy = {li: [] for li in self.layer_indices}
        self.hooks = []

    def _make_hook(self, li):
        def hook(module, input, output):
            if hasattr(module, 'router') and hasattr(module.router, 'gate'):
                # Try to capture from the MoE block's output metrics
                pass
        return hook

    def collect_from_metrics(self, all_router_metrics, batch_size):
        """Extract per-layer logits from router_metrics."""
        for lidx, layer_metrics in enumerate(all_router_metrics):
            if lidx >= len(self.layer_indices):
                continue
            li = self.layer_indices[lidx]
            if not layer_metrics:
                continue
            # tokens_per_expert
            if "tokens_per_expert" in layer_metrics:
                self.tokens_per_expert[li].append(layer_metrics["tokens_per_expert"].cpu())
            if "pair_fraction" in layer_metrics:
                self.pair_fraction[li].append(layer_metrics["pair_fraction"].cpu())
            if "router_entropy" in layer_metrics:
                re = layer_metrics["router_entropy"]
                self.router_entropy[li].append(float(re.mean().item() if re.ndim > 0 else re.item()))

    def get_stats(self):
        result = {}
        for li in self.layer_indices:
            layer_stats = {}
            if self.tokens_per_expert[li]:
                stacked = torch.stack(self.tokens_per_expert[li])
                layer_stats["tokens_per_expert_mean"] = stacked.mean(dim=0).tolist()
                layer_stats["tokens_per_expert_std"] = stacked.std(dim=0).tolist()
            if self.pair_fraction[li]:
                stacked = torch.stack(self.pair_fraction[li])
                layer_stats["pair_fraction_mean"] = stacked.mean(dim=0).tolist()
                layer_stats["pair_fraction_std"] = stacked.std(dim=0).tolist()
            if self.router_entropy[li]:
                layer_stats["router_entropy_mean"] = sum(self.router_entropy[li]) / len(self.router_entropy[li])
            result[li] = layer_stats
        return result


# ──────────────────────────────────────────────────────────────
# Diagnostic 1: Domain probe
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_hidden_by_domain(model, domain_texts_dict, tokenizer, max_length,
                              max_sequences, batch_size, device, precision_dtype,
                              probe_layers, max_hidden_tokens, collector_cls):
    """Forward pass per domain, collecting hidden states and router metrics."""
    all_hidden = {d: {} for d in domain_texts_dict}
    all_logits = {d: {} for d in domain_texts_dict}
    all_ppl = {}
    all_losses = {}  # per-sequence losses

    for domain, texts in domain_texts_dict.items():
        seqs = build_domain_sequences(texts, tokenizer, max_length, max_sequences)
        if not seqs:
            continue

        collector = HiddenCollector(model, probe_layers, max_hidden_tokens)
        collector.register()
        logit_collector = RouterLogitCollector(model, probe_layers)

        total_loss = 0.0
        total_batches = 0
        seq_losses = []

        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i + batch_size]
            input_ids = collate_pad(batch_seqs).to(device)

            with precision_context(precision_dtype):
                outputs = model(input_ids=input_ids, labels=input_ids,
                                output_router_logits=True, return_dict=True)

            loss_val = float(outputs.loss.detach().cpu())
            total_loss += loss_val
            total_batches += 1
            seq_losses.append(loss_val)

            if outputs.router_metrics:
                logit_collector.collect_from_metrics(outputs.router_metrics, input_ids.shape[0])

        collector.remove()

        all_hidden[domain] = collector.get_tensors()
        all_logits[domain] = logit_collector.get_stats()
        all_ppl[domain] = float(math.exp(min(total_loss / max(total_batches, 1), 20.0)))
        all_losses[domain] = seq_losses

    return all_hidden, all_logits, all_ppl, all_losses


def run_domain_probe(all_hidden, domain_map, probe_layers, probe_epochs, device):
    """Train linear probes on hidden states to predict domain."""
    domain_names = sorted(all_hidden.keys())
    domain_to_idx = {d: i for i, d in enumerate(domain_names)}
    majority_baseline = 1.0 / max(len(domain_names), 1)

    results = {}
    for li in probe_layers:
        X_list, y_list = [], []
        for domain in domain_names:
            h = all_hidden[domain].get(li)
            if h is None or h.shape[0] == 0:
                continue
            n_samples = h.shape[0]
            X_list.append(h)
            y_list.append(torch.full((n_samples,), domain_to_idx[domain], dtype=torch.long))

        if not X_list or len(set(domain_names)) < 2:
            results[li] = {"error": "insufficient_data", "majority_baseline": majority_baseline}
            continue

        X = torch.cat(X_list, dim=0).float()
        y = torch.cat(y_list, dim=0).long()

        n = X.shape[0]
        n_train = int(n * 0.8)
        perm = torch.randperm(n)
        X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
        X_test, y_test = X[perm[n_train:]], y[perm[n_train:]]

        input_dim = X.shape[1]
        num_classes = len(domain_names)

        probe = nn.Linear(input_dim, num_classes).to(device)
        opt = torch.optim.Adam(probe.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        X_train_gpu = X_train.to(device)
        y_train_gpu = y_train.to(device)
        X_test_gpu = X_test.to(device)
        y_test_gpu = y_test.to(device)

        for epoch in range(probe_epochs):
            probe.train()
            opt.zero_grad()
            logits = probe(X_train_gpu)
            loss = loss_fn(logits, y_train_gpu)
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(X_test_gpu)
            test_preds = test_logits.argmax(dim=1)
            acc = (test_preds == y_test_gpu).float().mean().item()

            # Confusion matrix
            conf = torch.zeros(num_classes, num_classes)
            for i in range(num_classes):
                mask = (y_test_gpu == i)
                if mask.any():
                    for j in range(num_classes):
                        conf[i, j] = (test_preds[mask] == j).float().mean().item()

        results[li] = {
            "accuracy": float(acc),
            "majority_baseline": float(majority_baseline),
            "n_train": n_train,
            "n_test": n - n_train,
            "n_classes": num_classes,
            "classes": domain_names,
            "confusion_matrix": conf.tolist(),
        }
    return results


# ──────────────────────────────────────────────────────────────
# Diagnostic 2: Router logits by domain
# ──────────────────────────────────────────────────────────────

def run_router_logits_analysis(all_logits, all_ppl, domain_names):
    """Analyze router metrics across domains."""
    rows = []
    for domain in domain_names:
        logits = all_logits.get(domain, {})
        for li, stats in logits.items():
            row = {"domain": domain, "layer": li}
            row.update({f"tpe_{i}": stats.get("tokens_per_expert_mean", [0]*6)[i]
                        if i < len(stats.get("tokens_per_expert_mean", [])) else 0.0
                        for i in range(6)})
            row.update({f"pf_{i}": stats.get("pair_fraction_mean", [0]*3)[i]
                        if i < len(stats.get("pair_fraction_mean", [])) else 0.0
                        for i in range(3)})
            row["router_entropy"] = stats.get("router_entropy_mean", 0)
            rows.append(row)

    # Pair JSD across domains
    pair_jsd = compute_pair_jsd_across_domains(all_logits, domain_names)
    return rows, pair_jsd


def compute_pair_jsd_across_domains(all_logits, domain_names):
    """Compute mean pair JSD across domains for each layer."""
    results = {}
    for li in range(30):
        domain_pairs = {}
        for d in domain_names:
            stats = all_logits.get(d, {}).get(li)
            if stats and stats.get("pair_fraction_mean"):
                domain_pairs[d] = stats["pair_fraction_mean"]
        if len(domain_pairs) < 2:
            continue
        jsd_vals = []
        domains = list(domain_pairs.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                p = torch.tensor(domain_pairs[domains[i]]).clamp_min(1e-9)
                q = torch.tensor(domain_pairs[domains[j]]).clamp_min(1e-9)
                p = p / p.sum()
                q = q / q.sum()
                m = 0.5 * (p + q)
                jsd = 0.5 * ((p * (p / m).log()).sum().item() + (q * (q / m).log()).sum().item())
                jsd_vals.append(jsd)
        results[li] = {"mean_jsd": sum(jsd_vals) / len(jsd_vals), "pair_count": len(jsd_vals)}
    return results


# ──────────────────────────────────────────────────────────────
# Diagnostic 3: Loss-conditioned routing
# ──────────────────────────────────────────────────────────────

def run_loss_conditioned_routing(all_hidden, all_logits, all_ppl, all_losses, domain_names):
    """No additional forward needed — use existing data."""
    # We compute per-sequence losses during forward pass, bucket them
    buckets = {"low_25": [], "mid_50": [], "high_25": []}
    domain_loss_buckets = defaultdict(lambda: {"low_25": 0, "mid_50": 0, "high_25": 0})

    all_seq_losses = []
    for domain in domain_names:
        losses = all_losses.get(domain, [])
        for l in losses:
            all_seq_losses.append((domain, l))

    if not all_seq_losses:
        return {"error": "no_data"}

    sorted_losses = sorted(all_seq_losses, key=lambda x: x[1])
    n = len(sorted_losses)
    low_thresh = sorted_losses[n // 4][1]
    high_thresh = sorted_losses[3 * n // 4][1]

    for domain, loss in sorted_losses:
        if loss <= low_thresh:
            buckets["low_25"].append(domain)
            domain_loss_buckets[domain]["low_25"] += 1
        elif loss >= high_thresh:
            buckets["high_25"].append(domain)
            domain_loss_buckets[domain]["high_25"] += 1
        else:
            buckets["mid_50"].append(domain)
            domain_loss_buckets[domain]["mid_50"] += 1

    return {
        "low_threshold": float(low_thresh),
        "high_threshold": float(high_thresh),
        "total_sequences": n,
        "domain_loss_buckets": {d: dict(v) for d, v in domain_loss_buckets.items()},
        "bucket_sizes": {k: len(v) for k, v in buckets.items()},
    }


# ──────────────────────────────────────────────────────────────
# Diagnostic 4: Hidden cluster → routing
# ──────────────────────────────────────────────────────────────

def run_hidden_cluster_analysis(all_hidden, all_logits, domain_names, probe_layers, skip=False):
    if skip:
        return {"skipped": True, "reason": "--skip-cluster flag"}

    results = {}
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"error": "sklearn_not_available"}

    for li in probe_layers:
        X_list, labels_list = [], []
        for domain in domain_names:
            h = all_hidden[domain].get(li)
            if h is None or h.shape[0] < 100:
                continue
            X_list.append(h.float().numpy())
            labels_list.extend([domain] * h.shape[0])

        if not X_list:
            results[li] = {"error": "insufficient_data"}
            continue

        X = np.concatenate(X_list, axis=0)
        n = X.shape[0]
        k = min(6, n // 100)

        if k < 2:
            results[li] = {"error": f"too_few_samples_n={n}"}
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
        cluster_ids = kmeans.fit_predict(X_scaled)

        cluster_domain = defaultdict(lambda: defaultdict(int))
        cluster_pair = defaultdict(lambda: defaultdict(int))
        for i, (cid, d) in enumerate(zip(cluster_ids, labels_list)):
            cluster_domain[cid][d] += 1
        for cid in range(k):
            total = sum(cluster_domain[cid].values())
            for d in cluster_domain[cid]:
                cluster_domain[cid][d] = cluster_domain[cid][d] / max(total, 1)

        results[li] = {
            "k": k,
            "n_samples": n,
            "cluster_sizes": [int((cluster_ids == c).sum()) for c in range(k)],
            "cluster_domain_composition": {str(cid): dict(dists) for cid, dists in cluster_domain.items()},
        }
    return results


# ──────────────────────────────────────────────────────────────
# Diagnostic 5: Pair+free contribution
# ──────────────────────────────────────────────────────────────

def run_pairfree_analysis(all_logits, all_ppl, domain_names, all_losses):
    """Extract free expert stats from logit collector."""
    free_stats = {}
    for domain in domain_names:
        logits = all_logits.get(domain, {})
        domain_free = {}
        for li, stats in logits.items():
            fef = stats.get("free_expert_fraction")
            if fef:
                domain_free[li] = fef
        if domain_free:
            free_stats[domain] = domain_free

    if not free_stats:
        return {"free_expert_available": False}

    return {
        "free_expert_available": True,
        "per_domain_free_expert_usage": free_stats,
    }


# ──────────────────────────────────────────────────────────────
# Main coordinator
# ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    probe_layers = [int(x) for x in args.layers.split(",")]

    checkpoint_path = resolve_checkpoint(args.output_dir, args.checkpoint_path)
    out_dir = args.out_dir or (args.output_dir / "domain_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] Checkpoint: {checkpoint_path}")
    print(f"[diag] Probe layers: {probe_layers}")

    # Load model
    device = ensure_cuda_device(args.device)
    model = HGRNBitForCausalLM.from_pretrained(str(checkpoint_path), torch_dtype=torch.bfloat16).to(device)
    model.eval()

    routing_mode = getattr(model.config, "moe_routing_mode", "standard")
    num_experts = getattr(model.config, "moe_num_experts", 6)
    is_pairfree = (routing_mode == "complement_pair_plus_free")
    print(f"[diag] Routing: {routing_mode}, experts: {num_experts}, pairfree: {is_pairfree}")

    precision_dtype, _ = get_precision_dtype(type("Cfg", (), {"precision": args.precision})())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Read data
    domain_texts, domain_rows = read_validation_by_domain(
        args.val_data_source, args.text_field, args.domain_field,
        args.max_samples_per_domain, args.seed)
    domain_names = sorted(domain_texts.keys())
    print(f"[diag] Domains: {domain_names}")

    # ── Collect hidden states + forward metrics ──
    print("[diag] Collecting hidden states and router metrics ...")
    all_hidden, all_logits, all_ppl, all_losses = collect_hidden_by_domain(
        model, domain_texts, tokenizer, args.max_length,
        args.max_sequences_per_domain, args.batch_size,
        device, precision_dtype, probe_layers, args.max_hidden_tokens,
        HiddenCollector)

    # ── Diagnostic 1: Hidden probe ──
    print("[diag] Diagnostic 1: Domain probe ...")
    probe_results = run_domain_probe(all_hidden, domain_names, probe_layers, args.probe_epochs, device)
    write_json(out_dir / "hidden_domain_probe_results.json", probe_results)

    # ── Diagnostic 2: Router logits ──
    print("[diag] Diagnostic 2: Router logits by domain ...")
    logits_rows, pair_jsd = run_router_logits_analysis(all_logits, all_ppl, domain_names)
    if logits_rows:
        fieldnames = ["domain", "layer", "router_entropy"] + [f"tpe_{i}" for i in range(6)] + [f"pf_{i}" for i in range(3)]
        write_csv_rows(out_dir / "router_logits_by_layer_domain.csv", fieldnames, logits_rows)
    write_json(out_dir / "router_logits_by_domain.json", {
        "domains": domain_names,
        "pair_jsd": pair_jsd,
        "per_domain_ppl": all_ppl,
    })

    # ── Diagnostic 3: Loss-conditioned ──
    print("[diag] Diagnostic 3: Loss-conditioned routing ...")
    loss_results = run_loss_conditioned_routing(all_hidden, all_logits, all_ppl, all_losses, domain_names)
    write_json(out_dir / "loss_conditioned_routing.json", loss_results)

    # ── Diagnostic 4: Hidden cluster ──
    print("[diag] Diagnostic 4: Hidden cluster analysis ...")
    cluster_results = run_hidden_cluster_analysis(all_hidden, all_logits, domain_names, probe_layers, args.skip_cluster)
    write_json(out_dir / "hidden_cluster_routing.json", cluster_results)

    # ── Diagnostic 5: Pair+free ──
    if is_pairfree:
        print("[diag] Diagnostic 5: Pair+free contribution ...")
        pf_results = run_pairfree_analysis(all_logits, all_ppl, domain_names, all_losses)
        write_json(out_dir / "pairfree_contribution.json", pf_results)
    else:
        print("[diag] Skipping pair+free (not a pair+free checkpoint)")

    # ── Aggregate summary ──
    summary_rows = []
    for li in probe_layers:
        pr = probe_results.get(li, {})
        summary_rows.append({
            "checkpoint": str(checkpoint_path),
            "layer": li,
            "probe_accuracy": pr.get("accuracy", "N/A"),
            "majority_baseline": pr.get("majority_baseline", "N/A"),
            "n_classes": pr.get("n_classes", 0),
        })

    summary = {
        "checkpoint": str(checkpoint_path),
        "routing_mode": routing_mode,
        "num_experts": num_experts,
        "per_domain_ppl": all_ppl,
        "probe_summary": summary_rows,
        "pair_jsd_by_layer": pair_jsd,
        "loss_bucket_analysis": loss_results,
    }
    write_json(out_dir / "diagnostics_summary.json", summary)

    # Markdown summary
    md_lines = [
        f"# Router Specialization Diagnostics: {str(checkpoint_path)}",
        f"",
        f"- Routing: {routing_mode}, experts: {num_experts}",
        f"- Domains: {', '.join(domain_names)}",
        f"",
        "## Per-Domain PPL",
        "",
        "| Domain | PPL |",
        "|--------|-----|",
    ]
    for d in domain_names:
        md_lines.append(f"| {d} | {all_ppl.get(d, 'N/A')} |")
    md_lines.extend([
        "",
        "## Hidden Domain Probe",
        "",
        "| Layer | Accuracy | Majority Baseline | Conclusion |",
        "|-------|----------|-------------------|------------|",
    ])
    for li in probe_layers:
        pr = probe_results.get(li, {})
        acc = pr.get("accuracy", 0)
        mb = pr.get("majority_baseline", 0.167)
        conclusion = "domain signal present" if acc > mb * 1.5 else "little domain signal"
        md_lines.append(f"| {li} | {acc:.4f} | {mb:.4f} | {conclusion} |")
    md_lines.extend([
        "",
        "## Router Pair JSD Across Domains",
        "",
        "| Layer | Mean Pair JSD | Pairs | Conclusion |",
        "|-------|--------------|-------|------------|",
    ])
    for li, jsd_info in sorted(pair_jsd.items()):
        mjsd = jsd_info.get("mean_jsd", 0)
        conclusion = "no domain routing" if mjsd < 0.001 else "some domain routing" if mjsd < 0.01 else "clear domain routing"
        md_lines.append(f"| {li} | {mjsd:.6f} | {jsd_info.get('pair_count', 0)} | {conclusion} |")

    write_json(out_dir / "diagnostics_summary.md", md_lines) if False else None
    (out_dir / "diagnostics_summary.md").write_text("\n".join(md_lines))

    print(f"\n[diag] All outputs in {out_dir}/")
    print(f"  hidden_domain_probe_results.json")
    print(f"  router_logits_by_domain.json")
    print(f"  loss_conditioned_routing.json")
    print(f"  hidden_cluster_routing.json")
    if is_pairfree:
        print(f"  pairfree_contribution.json")
    print(f"  diagnostics_summary.json")
    print(f"  diagnostics_summary.md")


if __name__ == "__main__":
    main()
