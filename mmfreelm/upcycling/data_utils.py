# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import random
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info
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
        shard_rank: int = 0,
        shard_world_size: int = 1,
    ):
        self.data_source = data_source
        self.max_length = max_length
        self.split = split
        self.text_field = text_field
        self.max_samples = max_samples
        self.shard_rank = int(shard_rank)
        self.shard_world_size = max(int(shard_world_size), 1)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.split = self._normalize_split(split)
        self._selected_directory_files: List[Path] = []
        self._manifest = self._build_manifest()

    def __iter__(self):
        buffer = []
        sample_count = 0
        worker_info = get_worker_info()
        num_worker_shards = worker_info.num_workers if worker_info is not None else 1
        worker_shard_rank = worker_info.id if worker_info is not None else 0
        combined_world_size = self.shard_world_size * num_worker_shards
        combined_rank = self.shard_rank * num_worker_shards + worker_shard_rank
        for text_index, text in enumerate(self._text_iterator()):
            if combined_world_size > 1 and text_index % combined_world_size != combined_rank:
                continue
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
                "shard_rank": self.shard_rank,
                "shard_world_size": self.shard_world_size,
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


class StatefulPackedParquetDataset(IterableDataset):
    """Deterministic parquet stream with serializable online-packing state.

    Shuffling is deliberately limited to a seeded shard permutation. Row order
    inside each parquet shard is stable, so resume can seek directly to one row
    group and restore the exact packing remainder without replaying the corpus.
    """

    STATE_VERSION = 1

    def __init__(
        self,
        data_source: str,
        tokenizer_path: str,
        max_length: int = 2048,
        split: str = "train",
        text_field: str = "text",
        max_samples: Optional[int] = None,
        shard_rank: int = 0,
        shard_world_size: int = 1,
        seed: int = 0,
        epoch: int = 0,
        shuffle_shards: bool = True,
        emit_state: bool = False,
    ):
        super().__init__()
        self.data_source = str(data_source)
        self.max_length = int(max_length)
        self.split = "validation" if split == "val" else str(split)
        self.text_field = str(text_field)
        self.max_samples = max_samples
        self.shard_rank = int(shard_rank)
        self.shard_world_size = max(int(shard_world_size), 1)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.shuffle_shards = bool(shuffle_shards)
        self.emit_state = bool(emit_state)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._files = self._select_parquet_files()
        self._shard_order = self._make_shard_order()
        self._shard_order_hash = sha256(
            "\n".join(str(path) for path in self._shard_order).encode("utf-8")
        ).hexdigest()
        self._resume_states: Dict[str, Dict[str, object]] = {}
        self._runtime_states: Dict[str, Dict[str, object]] = {}

    def _select_parquet_files(self) -> List[Path]:
        root = Path(self.data_source)
        if root.is_file():
            if root.suffix.lower() != ".parquet":
                raise ValueError("StatefulPackedParquetDataset only supports local parquet files")
            return [root.resolve()]
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {root}")
        prefixes = ("validation", "val") if self.split == "validation" else (self.split,)
        files = [
            path.resolve()
            for path in sorted(root.rglob("*.parquet"))
            if any(path.name.lower().startswith(prefix) for prefix in prefixes)
        ]
        if not files:
            raise FileNotFoundError(f"No {self.split} parquet shards found under {root}")
        return files

    def _make_shard_order(self) -> List[Path]:
        files = list(self._files)
        if self.shuffle_shards:
            random.Random(self.seed + self.epoch).shuffle(files)
        return files

    def _new_worker_state(self, worker_id: int, num_workers: int) -> Dict[str, object]:
        return {
            "version": self.STATE_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "shuffle_shards": self.shuffle_shards,
            "shard_order_hash": self._shard_order_hash,
            "rank": self.shard_rank,
            "world_size": self.shard_world_size,
            "worker_id": worker_id,
            "num_workers": num_workers,
            "max_length": self.max_length,
            "assigned_shard_position": 0,
            "row_group_index": 0,
            "row_offset_in_group": 0,
            "current_shard": None,
            "documents_consumed": 0,
            "sequences_consumed": 0,
            "tokens_emitted": 0,
            "token_buffer": [],
            "metadata_buffer": [],
        }

    def _validate_worker_state(
        self,
        state: Dict[str, object],
        worker_id: int,
        num_workers: int,
    ) -> None:
        expected = {
            "version": self.STATE_VERSION,
            "seed": self.seed,
            "epoch": self.epoch,
            "shuffle_shards": self.shuffle_shards,
            "shard_order_hash": self._shard_order_hash,
            "rank": self.shard_rank,
            "world_size": self.shard_world_size,
            "worker_id": worker_id,
            "num_workers": num_workers,
            "max_length": self.max_length,
        }
        mismatches = {
            key: {"checkpoint": state.get(key), "runtime": value}
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Dataset resume state is incompatible: {mismatches}")

    def _worker_state(self, worker_id: int, num_workers: int) -> Dict[str, object]:
        key = str(worker_id)
        if key in self._resume_states:
            state = deepcopy(self._resume_states[key])
            self._validate_worker_state(state, worker_id, num_workers)
        else:
            state = self._new_worker_state(worker_id, num_workers)
        self._runtime_states[key] = state
        return state

    def state_dict(self) -> Dict[str, object]:
        states = self._runtime_states or self._resume_states
        return {
            "version": self.STATE_VERSION,
            "dataset_type": type(self).__name__,
            "data_source": self.data_source,
            "split": self.split,
            "workers": deepcopy(states),
        }

    def load_state_dict(self, payload: Dict[str, object]) -> None:
        if int(payload.get("version", -1)) != self.STATE_VERSION:
            raise ValueError(f"Unsupported dataset state version: {payload.get('version')}")
        if payload.get("dataset_type") != type(self).__name__:
            raise ValueError(f"Unexpected dataset state type: {payload.get('dataset_type')}")
        if str(payload.get("data_source")) != self.data_source:
            raise ValueError("Dataset state points to a different data source")
        if str(payload.get("split")) != self.split:
            raise ValueError("Dataset state points to a different split")
        workers = payload.get("workers")
        if not isinstance(workers, dict) or not workers:
            raise ValueError("Dataset state does not contain worker states")
        self._resume_states = deepcopy(workers)
        self._runtime_states = {}

    def get_manifest(self) -> Dict[str, object]:
        return {
            "data_source": self.data_source,
            "split": self.split,
            "source_type": "stateful_local_parquet",
            "file_count": len(self._files),
            "shard_order_hash": self._shard_order_hash,
            "seed": self.seed,
            "epoch": self.epoch,
            "shuffle_strategy": "seeded_shard_permutation",
            "row_order": "parquet_physical_order",
            "exact_resume": True,
            "packing_state_saved": True,
        }

    @staticmethod
    def _consume_metadata_segments(
        segments: List[Dict[str, object]],
        consumed_tokens: int,
    ) -> List[Dict[str, object]]:
        remaining = int(consumed_tokens)
        result = deepcopy(segments)
        while remaining > 0 and result:
            head_tokens = int(result[0]["num_tokens"])
            if head_tokens <= remaining:
                remaining -= head_tokens
                result.pop(0)
            else:
                result[0]["num_tokens"] = head_tokens - remaining
                remaining = 0
        return result

    def _assigned_shards(self, worker_id: int, num_workers: int) -> List[Path]:
        combined_world_size = self.shard_world_size * num_workers
        combined_rank = self.shard_rank * num_workers + worker_id
        return [
            path
            for index, path in enumerate(self._shard_order)
            if index % combined_world_size == combined_rank
        ]

    def _iter_rows(
        self,
        path: Path,
        row_group_index: int,
        row_offset_in_group: int,
    ) -> Iterator[tuple[int, int, Dict[str, object]]]:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        if self.text_field not in available:
            raise KeyError(f"Missing text field `{self.text_field}` in {path}")
        optional = ["sample_id", "domain", "source_dataset"]
        columns = [self.text_field] + [name for name in optional if name in available]
        for group_index in range(row_group_index, parquet_file.metadata.num_row_groups):
            table = parquet_file.read_row_group(group_index, columns=columns)
            start = row_offset_in_group if group_index == row_group_index else 0
            names = table.column_names
            for row_index in range(start, table.num_rows):
                row = {name: table[name][row_index].as_py() for name in names}
                yield group_index, row_index, row
            row_offset_in_group = 0

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        state = self._worker_state(worker_id, num_workers)
        assigned_shards = self._assigned_shards(worker_id, num_workers)
        while int(state["assigned_shard_position"]) < len(assigned_shards):
            shard_position = int(state["assigned_shard_position"])
            shard_path = assigned_shards[shard_position]
            state["current_shard"] = str(shard_path)
            start_group = int(state["row_group_index"])
            start_offset = int(state["row_offset_in_group"])
            parquet_file_row_groups = None
            for group_index, row_index, row in self._iter_rows(
                shard_path,
                start_group,
                start_offset,
            ):
                if parquet_file_row_groups is None:
                    import pyarrow.parquet as pq

                    parquet_file_row_groups = pq.ParquetFile(shard_path).metadata.num_row_groups
                text = row.get(self.text_field)
                state["row_group_index"] = group_index
                state["row_offset_in_group"] = row_index + 1
                if text is None:
                    continue
                tokens = self.tokenizer.encode(str(text), add_special_tokens=False)
                state["documents_consumed"] = int(state["documents_consumed"]) + 1
                token_buffer = state["token_buffer"]
                token_buffer.extend(tokens)
                state["metadata_buffer"].append(
                    {
                        "sample_id": row.get("sample_id"),
                        "domain": row.get("domain"),
                        "source_dataset": row.get("source_dataset"),
                        "num_tokens": len(tokens),
                    }
                )
                while len(token_buffer) >= self.max_length + 1:
                    chunk = token_buffer[: self.max_length + 1]
                    del token_buffer[: self.max_length]
                    state["metadata_buffer"] = self._consume_metadata_segments(
                        state["metadata_buffer"],
                        self.max_length,
                    )
                    state["sequences_consumed"] = int(state["sequences_consumed"]) + 1
                    state["tokens_emitted"] = int(state["tokens_emitted"]) + self.max_length
                    self._runtime_states[str(worker_id)] = state
                    input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                    sample = {
                        "input_ids": input_ids,
                        # HGRNBitForCausalLM performs the causal shift internally.
                        "labels": input_ids.clone(),
                        "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                    }
                    if self.emit_state:
                        sample["__worker_id__"] = worker_id
                        sample["__state_json__"] = json.dumps(state, ensure_ascii=True)
                    yield sample
                    if self.max_samples and int(state["sequences_consumed"]) >= self.max_samples:
                        return
            state["assigned_shard_position"] = shard_position + 1
            state["row_group_index"] = 0
            state["row_offset_in_group"] = 0
            state["current_shard"] = None
