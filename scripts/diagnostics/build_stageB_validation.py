#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


DOMAIN_MAP = {
    "RedPajamaCommonCrawl": "commoncrawl",
    "RedPajamaC4": "c4",
    "RedPajamaGithub": "code",
    "RedPajamaWikipedia": "wiki",
    "RedPajamaBook": "book",
    "RedPajamaArXiv": "academic",
    "RedPajamaStackExchange": "qa_forum",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed Stage B validation tensor set.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=4096)
    parser.add_argument("--sequence-length", type=int, default=2048)
    return parser.parse_args()


def iter_records(source_files):
    try:
        import zstandard
    except ImportError as exc:
        raise SystemExit("zstandard is required to read local validation .jsonl.zst files") from exc
    for path in source_files:
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                text_stream = __import__("io").TextIOWrapper(reader, encoding="utf-8")
                for line_number, line in enumerate(text_stream):
                    if not line.strip():
                        continue
                    yield path, line_number, json.loads(line)


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output_dir}")
    source_files = sorted(args.source_dir.glob("*.jsonl.zst"))
    if not source_files:
        raise SystemExit(f"No .jsonl.zst validation files found under {args.source_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    array_path = args.output_dir / "validation_sequences.npy"
    sequences = np.lib.format.open_memmap(
        array_path,
        mode="w+",
        dtype=np.int32,
        shape=(args.num_sequences, args.sequence_length + 1),
    )
    token_buffer = []
    metadata_buffer = []
    hashes = []
    domain_documents = Counter()
    documents_read = 0
    sequence_index = 0
    for path, line_number, record in iter_records(source_files):
        text = record.get("text")
        if not text:
            continue
        tokens = tokenizer.encode(str(text), add_special_tokens=False)
        raw_domain = (record.get("meta") or {}).get("redpajama_set_name")
        domain = DOMAIN_MAP.get(raw_domain, "unknown")
        domain_documents[domain] += 1
        documents_read += 1
        token_buffer.extend(tokens)
        metadata_buffer.append(
            {
                "source_file": path.name,
                "line_number": line_number,
                "domain": domain,
                "num_tokens": len(tokens),
            }
        )
        while len(token_buffer) >= args.sequence_length + 1:
            chunk = token_buffer[: args.sequence_length + 1]
            del token_buffer[: args.sequence_length]
            sequences[sequence_index] = np.asarray(chunk, dtype=np.int32)
            hashes.append(
                hashlib.sha256(np.asarray(chunk, dtype=np.int32).tobytes()).hexdigest()
            )
            remaining = args.sequence_length
            while remaining > 0 and metadata_buffer:
                if metadata_buffer[0]["num_tokens"] <= remaining:
                    remaining -= metadata_buffer[0]["num_tokens"]
                    metadata_buffer.pop(0)
                else:
                    metadata_buffer[0]["num_tokens"] -= remaining
                    remaining = 0
            sequence_index += 1
            if sequence_index >= args.num_sequences:
                break
        if sequence_index >= args.num_sequences:
            break
    sequences.flush()
    if sequence_index != args.num_sequences:
        raise SystemExit(
            f"Validation source exhausted at {sequence_index}/{args.num_sequences} sequences"
        )
    hashes_path = args.output_dir / "validation_sample_hashes.txt"
    hashes_path.write_text("\n".join(hashes) + "\n", encoding="utf-8")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_name": "slimpajama_62b_venketh_stageB_pilot_validation",
        "source_dataset": "venketh/SlimPajama-62B",
        "source_split": "validation",
        "source_path": str(args.source_dir),
        "source_files": [str(path) for path in source_files],
        "train_overlap_policy": "official source train/validation split separation",
        "train_overlap_detected": False,
        "tokenizer": args.tokenizer_path,
        "sequence_length": args.sequence_length,
        "stored_width": args.sequence_length + 1,
        "num_sequences": args.num_sequences,
        "effective_tokens": args.num_sequences * args.sequence_length,
        "documents_read": documents_read,
        "domain_documents_read": dict(sorted(domain_documents.items())),
        "array_path": str(array_path),
        "array_dtype": "int32",
        "array_shape": [args.num_sequences, args.sequence_length + 1],
        "hash_algorithm": "sha256_int32_bytes",
        "hashes_path": str(hashes_path),
        "deterministic": True,
        "packing": "online contiguous stream with stride=sequence_length",
        "notes": [
            "Validation is built only from the upstream validation split, not relaxed-30B train shards.",
            "The extra token in each stored row is used to derive shifted labels exactly.",
        ],
    }
    (args.output_dir / "validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
