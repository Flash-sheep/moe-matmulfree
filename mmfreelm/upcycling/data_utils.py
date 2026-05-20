# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class StreamingTextDataset(IterableDataset):
    def __init__(
        self,
        data_source: str,
        tokenizer_path: str,
        max_length: int = 2048,
        split: str = "train",
        text_field: str = "text",
        max_samples: Optional[int] = None,
    ):
        self.data_source = data_source
        self.max_length = max_length
        self.split = split
        self.text_field = text_field
        self.max_samples = max_samples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        buffer = []
        sample_count = 0
        for text in self._text_iterator():
            if self.max_samples and sample_count >= self.max_samples:
                break
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.max_length + 1:
                chunk = buffer[: self.max_length + 1]
                buffer = buffer[self.max_length :]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }
                sample_count += 1
                if self.max_samples and sample_count >= self.max_samples:
                    return

    def _text_iterator(self):
        if os.path.isdir(self.data_source):
            for fpath in self._iter_directory_files(Path(self.data_source)):
                suffix = fpath.suffix.lower()
                if suffix == ".jsonl":
                    with fpath.open(encoding="utf-8") as handle:
                        for line in handle:
                            obj = json.loads(line)
                            yield obj.get(self.text_field, "")
                elif suffix in {".txt", ".md"}:
                    with fpath.open(encoding="utf-8") as handle:
                        yield handle.read()
                elif suffix == ".parquet":
                    yield from self._iter_parquet_file(fpath)
        elif self.data_source.endswith(".jsonl"):
            with open(self.data_source, encoding="utf-8") as handle:
                for line in handle:
                    obj = json.loads(line)
                    yield obj.get(self.text_field, "")
        elif self.data_source.endswith(".parquet"):
            yield from self._iter_parquet_file(Path(self.data_source))
        else:
            from datasets import load_dataset

            dataset = load_dataset(self.data_source, split=self.split, streaming=True)
            for item in dataset:
                yield item.get(self.text_field, "")

    def _iter_directory_files(self, root: Path):
        split_prefixes = {
            "train": ("train",),
            "validation": ("validation", "val"),
            "test": ("test",),
        }
        prefixes = split_prefixes.get(self.split, (self.split,))
        for fpath in sorted(root.rglob("*")):
            if not fpath.is_file():
                continue
            name = fpath.name.lower()
            if fpath.suffix.lower() == ".parquet":
                if not any(name.startswith(prefix) for prefix in prefixes):
                    continue
            yield fpath

    def _iter_parquet_file(self, path: Path):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        if self.text_field not in parquet_file.schema_arrow.names:
            raise KeyError(f"Missing text field `{self.text_field}` in {path}")
        for batch in parquet_file.iter_batches(columns=[self.text_field], batch_size=256):
            for text in batch.column(0).to_pylist():
                if text is not None:
                    yield str(text)
