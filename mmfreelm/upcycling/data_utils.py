# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer


class StreamingTextDataset(IterableDataset):
    SUPPORTED_FILE_SUFFIXES = {".jsonl", ".json", ".txt", ".md", ".parquet"}

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
        self.split = self._normalize_split(split)
        self._selected_directory_files: List[Path] = []
        self._manifest = self._build_manifest()

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
            for fpath in self._selected_directory_files:
                yield from self._iter_file_texts(fpath)
        elif self.data_source.endswith(".jsonl"):
            with open(self.data_source, encoding="utf-8") as handle:
                for line in handle:
                    obj = json.loads(line)
                    yield obj.get(self.text_field, "")
        elif self.data_source.endswith(".json"):
            yield from self._iter_json_file(Path(self.data_source))
        elif self.data_source.endswith(".parquet"):
            yield from self._iter_parquet_file(Path(self.data_source))
        elif self.data_source.endswith((".txt", ".md")):
            yield Path(self.data_source).read_text(encoding="utf-8")
        else:
            from datasets import load_dataset

            dataset = load_dataset(self.data_source, split=self.split, streaming=True)
            for item in dataset:
                yield item.get(self.text_field, "")

    def get_manifest(self) -> Dict[str, object]:
        return dict(self._manifest)

    def _normalize_split(self, split: str) -> str:
        normalized = (split or "train").lower()
        if normalized == "val":
            return "validation"
        return normalized

    def _split_prefixes(self) -> tuple[str, ...]:
        split_prefixes = {
            "train": ("train",),
            "validation": ("validation", "val"),
            "test": ("test",),
        }
        return split_prefixes.get(self.split, (self.split,))

    def _build_manifest(self) -> Dict[str, object]:
        root = Path(self.data_source)
        if root.is_dir():
            return self._build_directory_manifest(root)

        suffix = root.suffix.lower()
        if suffix in self.SUPPORTED_FILE_SUFFIXES:
            total_size_bytes = root.stat().st_size if root.exists() else None
            return {
                "data_source": self.data_source,
                "split": self.split,
                "source_type": "single_file",
                "split_filter_applied": False,
                "reason": "single_file_source",
                "file_count": 1 if root.exists() else 0,
                "first_20_files": [root.name] if root.exists() else [],
                "file_extensions": [suffix] if suffix else [],
                "total_size_bytes": total_size_bytes,
            }

        return {
            "data_source": self.data_source,
            "split": self.split,
            "source_type": "datasets_streaming",
            "split_filter_applied": True,
            "reason": "datasets_library_split_argument",
            "file_count": None,
            "first_20_files": [],
            "file_extensions": [],
            "total_size_bytes": None,
        }

    def _build_directory_manifest(self, root: Path) -> Dict[str, object]:
        all_files = [
            fpath
            for fpath in sorted(root.rglob("*"))
            if fpath.is_file() and fpath.suffix.lower() in self.SUPPORTED_FILE_SUFFIXES
        ]
        prefixes = self._split_prefixes()
        filtered_files = [fpath for fpath in all_files if self._matches_split_prefix(fpath, prefixes)]
        if filtered_files:
            selected_files = filtered_files
            split_filter_applied = True
            reason = "matched_split_prefixed_files"
        else:
            selected_files = all_files
            split_filter_applied = False
            reason = "no split-prefixed files found"
        self._selected_directory_files = selected_files
        first_20_files = [str(path.relative_to(root)) for path in selected_files[:20]]
        extensions = sorted({path.suffix.lower() for path in selected_files})
        total_size_bytes = sum(path.stat().st_size for path in selected_files)
        return {
            "data_source": self.data_source,
            "split": self.split,
            "source_type": "directory",
            "split_filter_applied": split_filter_applied,
            "reason": reason,
            "file_count": len(selected_files),
            "first_20_files": first_20_files,
            "file_extensions": extensions,
            "total_size_bytes": total_size_bytes,
        }

    def _matches_split_prefix(self, path: Path, prefixes: tuple[str, ...]) -> bool:
        name = path.name.lower()
        stem = path.stem.lower()
        relative_parts = [part.lower() for part in path.parts]
        for prefix in prefixes:
            if name.startswith(prefix) or stem.startswith(prefix):
                return True
            if prefix in relative_parts:
                return True
            if any(part.startswith(prefix) for part in relative_parts):
                return True
        return False

    def _iter_file_texts(self, path: Path):
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    obj = json.loads(line)
                    yield obj.get(self.text_field, "")
            return
        if suffix == ".json":
            yield from self._iter_json_file(path)
            return
        if suffix in {".txt", ".md"}:
            with path.open(encoding="utf-8") as handle:
                yield handle.read()
            return
        if suffix == ".parquet":
            yield from self._iter_parquet_file(path)
            return
        raise ValueError(f"Unsupported dataset file suffix: {path}")

    def _iter_json_file(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    yield item.get(self.text_field, "")
        elif isinstance(payload, dict):
            if self.text_field in payload:
                value = payload.get(self.text_field)
                if isinstance(value, list):
                    for item in value:
                        if item is not None:
                            yield str(item)
                elif value is not None:
                    yield str(value)
        else:
            raise TypeError(f"Unsupported JSON structure in {path}")

    def _iter_parquet_file(self, path: Path):
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        if self.text_field not in parquet_file.schema_arrow.names:
            raise KeyError(f"Missing text field `{self.text_field}` in {path}")
        for batch in parquet_file.iter_batches(columns=[self.text_field], batch_size=256):
            for text in batch.column(0).to_pylist():
                if text is not None:
                    yield str(text)
