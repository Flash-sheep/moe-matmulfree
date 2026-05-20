#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert parquet text rows into JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-field", type=str, default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = pq.read_table(args.input, columns=[args.text_field])
    texts = table.column(args.text_field).to_pylist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(texts)
    kept = 0
    chars = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for text in texts:
            text = (text or "").strip()
            if not text:
                continue
            handle.write(json.dumps({args.text_field: text}, ensure_ascii=False) + "\n")
            kept += 1
            chars += len(text)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "rows": total,
                "kept": kept,
                "chars": chars,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
