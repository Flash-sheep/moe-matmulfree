#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from transformers import AutoTokenizer


def percentile(sorted_values: List[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = int(round((len(sorted_values) - 1) * q))
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def summarize_distribution(values: List[int]) -> Dict[str, int]:
    if not values:
        return {"count": 0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(ordered, 0.50),
        "p90": percentile(ordered, 0.90),
        "p99": percentile(ordered, 0.99),
        "max": ordered[-1],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a mixed raw-text parquet dataset built for StreamingTextDataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=100)
    return parser


def run_loader_smoke(dataset_dir: Path, tokenizer_path: str, seq_len: int) -> Dict[str, object]:
    try:
        from torch.utils.data import DataLoader

        from mmfreelm.upcycling import StreamingTextDataset
        from scripts.run_sparse_upcycling import collate_streaming_batch

        ds = StreamingTextDataset(
            data_source=str(dataset_dir),
            tokenizer_path=tokenizer_path,
            max_length=seq_len,
            split="train",
            text_field="text",
            max_samples=2,
        )
        loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_streaming_batch)
        batch = next(iter(loader))
        return {
            "success": True,
            "error": None,
            "batch_shapes": {key: list(value.shape) for key, value in batch.items()},
            "python": sys.executable,
        }
    except Exception:
        pass

    host_python = "/home/fyf/anaconda3/bin/python3"
    inline = f"""
import json
from torch.utils.data import DataLoader
from mmfreelm.upcycling import StreamingTextDataset
from scripts.run_sparse_upcycling import collate_streaming_batch
ds = StreamingTextDataset(
    data_source={str(dataset_dir)!r},
    tokenizer_path={tokenizer_path!r},
    max_length={int(seq_len)},
    split='train',
    text_field='text',
    max_samples=2,
)
loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_streaming_batch)
batch = next(iter(loader))
print(json.dumps({{
    'success': True,
    'error': None,
    'batch_shapes': {{k: list(v.shape) for k, v in batch.items()}},
    'python': {host_python!r},
}}))
"""
    try:
        completed = subprocess.run(
            [host_python, "-c", inline],
            cwd=str(Path(__file__).resolve().parents[1]),
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "batch_shapes": None,
            "python": host_python,
        }


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = Path(args.dataset_dir)
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing dataset_manifest.json in {dataset_dir}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    tokenizer_path = manifest["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import pyarrow.parquet as pq

    train_files = sorted(dataset_dir.glob("train-*.parquet"))
    sample_count = 0
    empty_samples = 0
    short_samples = 0
    metadata_missing = 0
    source_counter = Counter()
    domain_counter = Counter()
    token_counts: List[int] = []
    sample_previews: List[Dict[str, object]] = []

    required_metadata = {"source_dataset", "domain", "original_source", "document_token_count", "sample_id", "mixture_name"}
    for file_path in train_files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=128):
            for row in batch.to_pylist():
                if sample_count >= args.max_samples:
                    break
                text = str(row.get("text", ""))
                if not text.strip():
                    empty_samples += 1
                source = str(row.get("source_dataset", "unknown"))
                domain = str(row.get("domain", "unknown"))
                source_counter[source] += 1
                domain_counter[domain] += 1
                token_count = int(row.get("document_token_count", 0))
                if token_count <= 0:
                    token_count = len(tokenizer.encode(text, add_special_tokens=False))
                token_counts.append(token_count)
                if token_count < 32:
                    short_samples += 1
                if not required_metadata.issubset(set(row.keys())):
                    metadata_missing += 1
                decoded = tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)[:64], skip_special_tokens=True)
                sample_previews.append(
                    {
                        "source_dataset": source,
                        "domain": domain,
                        "document_token_count": token_count,
                        "sample_id": row.get("sample_id"),
                        "text_preview": text[:200],
                        "decoded_preview": decoded[:200],
                    }
                )
                sample_count += 1
            if sample_count >= args.max_samples:
                break
        if sample_count >= args.max_samples:
            break

    report = {
        "dataset_dir": str(dataset_dir),
        "dataset_name": manifest.get("dataset_name"),
        "tokenizer": tokenizer_path,
        "compatible_with_training_loader": bool(manifest.get("compatible_with_training_loader")),
        "output_format": manifest.get("output_format"),
        "seq_len": manifest.get("seq_len"),
        "total_token_target": manifest.get("target_tokens"),
        "total_train_tokens_manifest": manifest.get("actual_tokens"),
        "train_files_found": len(train_files),
        "validation_dir": str(dataset_dir / "validation_domain_balanced"),
        "inspected_samples": sample_count,
        "source_distribution_samples": dict(source_counter),
        "domain_distribution_samples": dict(domain_counter),
        "document_token_count_distribution": summarize_distribution(token_counts),
        "empty_samples": empty_samples,
        "short_samples_lt32_tokens": short_samples,
        "metadata_missing_samples": metadata_missing,
        "metadata_strategy": manifest.get("metadata_strategy"),
        "sample_previews": sample_previews[: min(len(sample_previews), 10)],
    }

    loader_smoke = run_loader_smoke(dataset_dir, tokenizer_path, int(manifest.get("seq_len", 1024)))
    report["loader_smoke"] = loader_smoke
    report["loader_smoke_passed"] = bool(loader_smoke.get("success"))
    report["loader_batch_shapes"] = loader_smoke.get("batch_shapes")
    report["empty_sample_count"] = empty_samples
    report["metadata_completeness"] = {
        "required_metadata_fields_present_for_sampled_rows": metadata_missing == 0,
        "missing_sample_count": metadata_missing,
    }
    report["success"] = bool(
        manifest.get("compatible_with_training_loader")
        and empty_samples == 0
        and metadata_missing == 0
        and loader_smoke.get("success")
    )

    report_path = dataset_dir / "dataset_inspection_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"dataset_dir": str(dataset_dir), "report": str(report_path), "inspected_samples": sample_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
