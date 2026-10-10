#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import yaml
from transformers import AutoTokenizer


SUPPORTED_FILE_SUFFIXES = {".parquet", ".jsonl", ".json", ".zst"}


@dataclass
class SourceRuntime:
    name: str
    target_ratio: float
    effective_ratio: float
    kind: str
    config: Dict[str, object]
    iterator: Iterator[Dict[str, object]]
    available: bool
    notes: List[str]
    stats: Dict[str, int]


@dataclass
class BalancePlan:
    strategy: str
    requested_target_tokens: int
    actual_target_tokens: int
    domain_target_tokens: Dict[str, int]
    reference_domain_tokens: Dict[str, int]
    reference_total_tokens: int
    min_available_domain: Optional[str]
    min_available_tokens: Optional[int]
    per_domain_budget: Optional[int]
    auto_downgraded: bool
    allow_oversample: bool
    plan_notes: List[str]
    details: Dict[str, object]


def parse_bool(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return payload


def dump_yaml(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def resolve_special_path(value: Optional[str], default: Optional[Path] = None) -> Optional[str]:
    if value is None:
        return None
    if value == "auto_from_existing_project_config":
        if default is None:
            return None
        return str(default)
    if value == "auto_detect_or_fill_me":
        if default is None:
            return None
        return str(default)
    return value


def normalize_domain_name(name: Optional[str]) -> str:
    if not name:
        return "unknown"
    return str(name)


def get_nested_field(record: Dict[str, object], field_path: Optional[str], fallback: Optional[str] = None):
    if not field_path:
        return fallback
    current: object = record
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return fallback
        current = current[part]
    return current


def coarse_domain(source_name: str, raw_domain: Optional[str], mapping: Dict[str, object]) -> str:
    source_map = mapping.get(source_name, {})
    if not isinstance(source_map, dict):
        return normalize_domain_name(raw_domain)
    if raw_domain is None:
        return str(source_map.get("default", "unknown"))
    return str(source_map.get(str(raw_domain), source_map.get("default", normalize_domain_name(raw_domain))))


def normalize_ratio_map(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(value)) for value in raw.values())
    if total <= 0:
        raise ValueError("ratio map total must be positive")
    return {key: max(0.0, float(value)) / total for key, value in raw.items()}


def round_token_targets(target_tokens: int, ratios: Dict[str, float]) -> Dict[str, int]:
    raw_targets = {key: float(target_tokens) * float(ratio) for key, ratio in ratios.items()}
    floor_targets = {key: int(math.floor(value)) for key, value in raw_targets.items()}
    assigned = sum(floor_targets.values())
    remainder = max(0, int(target_tokens) - assigned)
    order = sorted(raw_targets, key=lambda key: (raw_targets[key] - floor_targets[key], key), reverse=True)
    for key in order[:remainder]:
        floor_targets[key] += 1
    return floor_targets


def capped_ratio_projection(
    ratios: Dict[str, float],
    caps: Dict[str, float],
    mins: Dict[str, float],
) -> Dict[str, float]:
    normalized = normalize_ratio_map(ratios)
    result = dict(normalized)
    fixed: Dict[str, float] = {}

    for key, min_value in mins.items():
        if key not in result:
            continue
        min_value = max(0.0, float(min_value))
        if result[key] < min_value:
            fixed[key] = min_value

    for key, cap_value in caps.items():
        if key not in result:
            continue
        cap_value = max(0.0, float(cap_value))
        if result[key] > cap_value:
            fixed[key] = cap_value

    fixed_total = sum(fixed.values())
    if fixed_total >= 1.0:
        return normalize_ratio_map(fixed)

    free_keys = [key for key in result if key not in fixed]
    free_total = sum(result[key] for key in free_keys)
    remaining = 1.0 - fixed_total
    projected = dict(fixed)
    if free_total <= 0:
        for key in free_keys:
            projected[key] = remaining / max(1, len(free_keys))
        return projected
    for key in free_keys:
        projected[key] = result[key] / free_total * remaining
    return projected


def load_reference_domain_tokens(path: Path) -> tuple[Dict[str, int], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    domains = payload.get("domains", {})
    if not isinstance(domains, dict):
        raise TypeError(f"domains must be a mapping in {path}")
    normalized = {str(key): int(value) for key, value in domains.items()}
    total = int(payload.get("actual_tokens") or sum(normalized.values()))
    return normalized, total


def infer_balance_domains(cfg: Dict[str, object], domain_mapping: Dict[str, object]) -> List[str]:
    source_names = [str(source.get("name")) for source in cfg.get("sources", []) if isinstance(source, dict)]
    inferred: List[str] = []
    for source_name in source_names:
        source_map = domain_mapping.get(source_name, {})
        if isinstance(source_map, dict):
            for key, value in source_map.items():
                if key == "default":
                    continue
                normalized = str(value)
                if normalized not in inferred:
                    inferred.append(normalized)
    return inferred


def build_balance_plan(cfg: Dict[str, object], domain_mapping: Dict[str, object]) -> Optional[BalancePlan]:
    balance_cfg = cfg.get("balance", {})
    if not isinstance(balance_cfg, dict):
        return None
    strategy = str(balance_cfg.get("strategy", "none"))
    if strategy in {"none", "", "null"}:
        return None

    target_tokens = int(cfg["target_tokens"])
    domains_cfg = balance_cfg.get("domains")
    domains = [str(item) for item in domains_cfg] if isinstance(domains_cfg, list) and domains_cfg else infer_balance_domains(cfg, domain_mapping)
    if not domains:
        raise ValueError(f"balance strategy `{strategy}` requires a non-empty domain list")

    reference_manifest_value = balance_cfg.get("reference_manifest")
    if not reference_manifest_value:
        raise ValueError(f"balance strategy `{strategy}` requires balance.reference_manifest")
    reference_manifest = Path(str(reference_manifest_value))
    if not reference_manifest.exists():
        raise FileNotFoundError(f"reference manifest not found: {reference_manifest}")
    raw_reference_tokens, reference_total_tokens = load_reference_domain_tokens(reference_manifest)
    reference_domain_tokens = {domain: int(raw_reference_tokens.get(domain, 0)) for domain in domains}

    plan_notes: List[str] = [f"balance strategy enabled: {strategy}"]
    allow_oversample = bool(balance_cfg.get("allow_oversample", False))
    if allow_oversample:
        plan_notes.append("allow_oversample=true")
    else:
        plan_notes.append("allow_oversample=false")

    details: Dict[str, object] = {
        "reference_manifest": str(reference_manifest),
        "domains": domains,
        "reference_total_tokens": int(reference_total_tokens),
        "reference_domain_tokens": reference_domain_tokens,
    }

    if strategy == "relaxed_tempered":
        reference_ratios = normalize_ratio_map({domain: max(1, reference_domain_tokens.get(domain, 0)) for domain in domains})
        temperature = float(balance_cfg.get("domain_temperature", 0.65))
        tempered = {domain: math.pow(ratio, temperature) for domain, ratio in reference_ratios.items()}
        domain_caps = {str(key): float(value) for key, value in dict(balance_cfg.get("domain_caps", {})).items()}
        domain_min_ratios = {str(key): float(value) for key, value in dict(balance_cfg.get("domain_min_ratios", {})).items()}
        projected_ratios = capped_ratio_projection(normalize_ratio_map(tempered), domain_caps, domain_min_ratios)
        domain_target_tokens = round_token_targets(target_tokens, projected_ratios)
        details.update(
            {
                "domain_temperature": temperature,
                "domain_caps": domain_caps,
                "domain_min_ratios": domain_min_ratios,
                "reference_ratios": reference_ratios,
                "projected_ratios": projected_ratios,
            }
        )
        return BalancePlan(
            strategy=strategy,
            requested_target_tokens=target_tokens,
            actual_target_tokens=int(sum(domain_target_tokens.values())),
            domain_target_tokens=domain_target_tokens,
            reference_domain_tokens=reference_domain_tokens,
            reference_total_tokens=int(reference_total_tokens),
            min_available_domain=None,
            min_available_tokens=None,
            per_domain_budget=None,
            auto_downgraded=False,
            allow_oversample=allow_oversample,
            plan_notes=plan_notes,
            details=details,
        )

    if strategy == "strict_equal_domain":
        if allow_oversample:
            raise ValueError("strict_equal_domain does not allow oversampling")
        estimated_total_tokens = int(
            balance_cfg.get("source_total_tokens_estimate")
            or cfg.get("source_provenance", {}).get("source_total_tokens_estimate")
            or reference_total_tokens
        )
        reference_ratios = normalize_ratio_map({domain: max(1, reference_domain_tokens.get(domain, 0)) for domain in domains})
        estimated_available = {domain: int(reference_ratios[domain] * estimated_total_tokens) for domain in domains}
        min_available_domain = min(estimated_available, key=lambda key: estimated_available[key])
        min_available_tokens = int(estimated_available[min_available_domain])
        per_domain_budget_requested = target_tokens // len(domains)
        auto_downgraded = False
        actual_target_tokens = int(target_tokens)
        safety_factor = float(balance_cfg.get("downgrade_safety_factor", 0.95))
        if per_domain_budget_requested > min_available_tokens:
            if bool(balance_cfg.get("auto_downgrade_if_insufficient", True)):
                safe_total = int(math.floor(min_available_tokens * len(domains) * safety_factor))
                safe_total = max(len(domains), safe_total - (safe_total % len(domains)))
                actual_target_tokens = safe_total
                auto_downgraded = True
                plan_notes.append(
                    f"strict balance auto-downgraded from {target_tokens} to {actual_target_tokens} because requested per-domain budget {per_domain_budget_requested} exceeded min available {min_available_tokens} in {min_available_domain}"
                )
            else:
                raise ValueError(
                    f"strict_equal_domain target {target_tokens} is unsafe: per-domain budget {per_domain_budget_requested} exceeds min available {min_available_tokens} in {min_available_domain}"
                )
        per_domain_budget = actual_target_tokens // len(domains)
        actual_target_tokens = per_domain_budget * len(domains)
        domain_target_tokens = {domain: per_domain_budget for domain in domains}
        details.update(
            {
                "source_total_tokens_estimate": estimated_total_tokens,
                "reference_ratios": reference_ratios,
                "estimated_available_domain_tokens": estimated_available,
                "downgrade_safety_factor": safety_factor,
            }
        )
        return BalancePlan(
            strategy=strategy,
            requested_target_tokens=target_tokens,
            actual_target_tokens=actual_target_tokens,
            domain_target_tokens=domain_target_tokens,
            reference_domain_tokens=estimated_available,
            reference_total_tokens=estimated_total_tokens,
            min_available_domain=min_available_domain,
            min_available_tokens=min_available_tokens,
            per_domain_budget=per_domain_budget,
            auto_downgraded=auto_downgraded,
            allow_oversample=False,
            plan_notes=plan_notes,
            details=details,
        )

    raise ValueError(f"unsupported balance strategy: {strategy}")


def discover_split_files(root: Path, split: str) -> List[Path]:
    prefixes = {
        "train": ("train",),
        "validation": ("validation", "val"),
        "test": ("test",),
    }.get(split, (split,))
    all_files = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and (path.suffix.lower() in SUPPORTED_FILE_SUFFIXES or path.name.lower().endswith(".jsonl.zst"))
    ]
    selected = [
        path
        for path in all_files
        if any(
            path.name.lower().startswith(prefix)
            or path.stem.lower().startswith(prefix)
            or prefix in [part.lower() for part in path.parts]
            for prefix in prefixes
        )
    ]
    return selected or all_files


def iter_local_records(source_cfg: Dict[str, object], split: str, stats: Optional[Dict[str, int]] = None) -> Iterator[Dict[str, object]]:
    import pyarrow.parquet as pq

    if stats is None:
        stats = {}
    stats.setdefault("bad_json_lines", 0)
    stats.setdefault("rows_yielded", 0)

    path_value = source_cfg.get("path")
    if not path_value:
        return
    path = Path(str(path_value))
    text_field = str(source_cfg.get("text_field", "text"))
    metadata_fields = source_cfg.get("metadata_fields", {})
    if not isinstance(metadata_fields, dict):
        metadata_fields = {}
    domain_field = metadata_fields.get("domain")

    if path.is_dir():
        files = discover_split_files(path, split)
    else:
        files = [path]

    for file_path in files:
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            parquet_file = pq.ParquetFile(file_path)
            column_roots = {text_field.split(".")[0]}
            if isinstance(domain_field, str):
                column_roots.add(domain_field.split(".")[0])
            columns = [column for column in column_roots if column in parquet_file.schema_arrow.names]
            if text_field.split(".")[0] not in columns:
                raise KeyError(f"Missing text field root `{text_field}` in {file_path}")
            row_index = 0
            for batch in parquet_file.iter_batches(columns=columns, batch_size=256):
                rows = batch.to_pylist()
                for row in rows:
                    text_value = get_nested_field(row, text_field, fallback=None)
                    if text_value is None:
                        row_index += 1
                        continue
                    raw_domain = get_nested_field(row, str(domain_field), fallback="unknown") if domain_field else "unknown"
                    yield {
                        "text": str(text_value),
                        "raw_domain": normalize_domain_name(raw_domain),
                        "original_source": str(file_path.name),
                        "sample_id": f"{source_cfg['name']}:{file_path.name}:{row_index}",
                        "shard_id": file_path.name,
                    }
                    stats["rows_yielded"] += 1
                    row_index += 1
            continue

        if file_path.name.lower().endswith(".jsonl.zst"):
            import zstandard as zstd

            row_index = 0
            with file_path.open("rb") as compressed:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(compressed) as reader:
                    import io

                    text_handle = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
                    for line in text_handle:
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError:
                            stats["bad_json_lines"] += 1
                            row_index += 1
                            continue
                        text_value = get_nested_field(row, text_field, fallback=None)
                        if text_value is None:
                            row_index += 1
                            continue
                        raw_domain = get_nested_field(row, str(domain_field), fallback="unknown") if domain_field else "unknown"
                        yield {
                            "text": str(text_value),
                            "raw_domain": normalize_domain_name(raw_domain),
                            "original_source": str(file_path.name),
                            "sample_id": f"{source_cfg['name']}:{file_path.name}:{row_index}",
                            "shard_id": file_path.name,
                        }
                        stats["rows_yielded"] += 1
                        row_index += 1
            continue

        raise NotImplementedError(f"Unsupported local source format for this builder: {file_path}")


def iter_hf_records(source_cfg: Dict[str, object], split: str, streaming: bool, download: bool, stats: Optional[Dict[str, int]] = None) -> Iterator[Dict[str, object]]:
    from datasets import load_dataset

    if stats is None:
        stats = {}
    stats.setdefault("rows_yielded", 0)

    dataset_name = source_cfg.get("hf_dataset")
    if not dataset_name:
        return
    text_field = str(source_cfg.get("text_field", "text"))
    metadata_fields = source_cfg.get("metadata_fields", {})
    if not isinstance(metadata_fields, dict):
        metadata_fields = {}
    domain_field = metadata_fields.get("domain")

    ds = load_dataset(
        str(dataset_name),
        split=str(source_cfg.get("split", split)),
        streaming=(streaming and not download),
    )
    for idx, row in enumerate(ds):
        if not isinstance(row, dict):
            continue
        text_value = get_nested_field(row, text_field, fallback=None)
        if text_value is None:
            continue
        raw_domain = get_nested_field(row, str(domain_field), fallback=source_cfg.get("default_domain", "unknown")) if domain_field else source_cfg.get("default_domain", "unknown")
        yield {
            "text": str(text_value),
            "raw_domain": normalize_domain_name(raw_domain),
            "original_source": str(dataset_name),
            "sample_id": f"{source_cfg['name']}:{idx}",
            "shard_id": str(source_cfg.get("split", split)),
        }
        stats["rows_yielded"] += 1


class ShardedParquetWriter:
    def __init__(self, output_dir: Path, split_prefix: str, shard_size_tokens: int):
        self.output_dir = output_dir
        self.split_prefix = split_prefix
        self.shard_size_tokens = max(1, shard_size_tokens)
        self.rows: List[Dict[str, object]] = []
        self.current_tokens = 0
        self.shard_index = 0
        self.shards: List[Dict[str, object]] = []

    def add_row(self, row: Dict[str, object]) -> None:
        token_count = int(row.get("document_token_count", 0))
        self.rows.append(row)
        self.current_tokens += token_count
        if self.current_tokens >= self.shard_size_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        shard_name = f"{self.split_prefix}-{self.shard_index:05d}.parquet"
        shard_path = self.output_dir / shard_name
        table = pa.Table.from_pylist(self.rows)
        pq.write_table(table, shard_path)
        self.shards.append(
            {
                "name": shard_name,
                "path": str(shard_path),
                "rows": len(self.rows),
                "tokens": self.current_tokens,
            }
        )
        self.rows = []
        self.current_tokens = 0
        self.shard_index += 1

    def close(self) -> List[Dict[str, object]]:
        self.flush()
        return list(self.shards)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a mixed pretraining dataset in a format compatible with StreamingTextDataset.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--output-dir", required=True, help="Output dataset directory.")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--streaming", type=parse_bool, default=False)
    parser.add_argument("--download", type=parse_bool, default=False)
    parser.add_argument("--dry-run", type=parse_bool, default=False)
    parser.add_argument("--max-docs-per-source", type=int, default=None)
    parser.add_argument("--max-tokens-per-source", type=int, default=None)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument("--timeout", type=int, default=15)
    return parser


def resolve_config(cfg: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    root = repo_root()
    resolved = json.loads(json.dumps(cfg))
    resolved["target_tokens"] = args.target_tokens or int(cfg.get("target_tokens", 0))
    resolved["seq_len"] = args.seq_len or int(cfg.get("seq_len", 1024))
    resolved["seed"] = args.seed or int(cfg.get("seed", 202606))

    tokenizer_cfg = resolved.setdefault("tokenizer", {})
    if not isinstance(tokenizer_cfg, dict):
        raise TypeError("tokenizer config must be a mapping")
    tokenizer_cfg["path"] = args.tokenizer or resolve_special_path(
        tokenizer_cfg.get("path"),
        default=root / "checkpoints" / "MMfreeLM-370M",
    )
    tokenizer_cfg.setdefault("type", "auto")

    output_cfg = resolved.setdefault("output", {})
    if not isinstance(output_cfg, dict):
        raise TypeError("output config must be a mapping")
    output_cfg.setdefault("format", "raw_text_parquet_with_metadata")
    output_cfg.setdefault("shard_size_tokens", 100_000_000)
    output_cfg.setdefault("write_manifest", True)
    output_cfg.setdefault("write_domain_summary", True)
    output_cfg.setdefault("write_source_summary", True)
    output_cfg.setdefault("validation_tokens_per_domain", 131_072)

    safety_cfg = resolved.setdefault("safety", {})
    if not isinstance(safety_cfg, dict):
        raise TypeError("safety config must be a mapping")
    safety_cfg["overwrite"] = args.overwrite or bool(safety_cfg.get("overwrite", False))
    safety_cfg.setdefault("require_explicit_download_for_remote", True)

    sources = resolved.get("sources", [])
    if not isinstance(sources, list) or not sources:
        raise ValueError("config sources must be a non-empty list")
    for source in sources:
        if not isinstance(source, dict):
            raise TypeError("source entry must be a mapping")
        source["name"] = str(source["name"])
        if source["name"] == "slimpajama":
            source["path"] = resolve_special_path(
                source.get("path"),
                default=root / "datasets" / "SlimPajama-6B" / "data",
            )
        source.setdefault("split", "train")
        source.setdefault("text_field", "text")
        source.setdefault("metadata_fields", {})
        source.setdefault("default_domain", "unknown")

    return resolved


def build_source_runtimes(
    cfg: Dict[str, object],
    domain_mapping: Dict[str, object],
    streaming: bool,
    download: bool,
) -> tuple[List[SourceRuntime], List[str]]:
    notes: List[str] = []
    runtimes: List[SourceRuntime] = []
    sources = cfg["sources"]
    available_ratio_total = 0.0

    for source_cfg in sources:
        assert isinstance(source_cfg, dict)
        source_name = str(source_cfg["name"])
        target_ratio = float(source_cfg.get("ratio", 0.0))
        source_notes: List[str] = []
        iterator: Optional[Iterator[Dict[str, object]]] = None
        available = False
        kind = "unavailable"
        stats: Dict[str, int] = {"bad_json_lines": 0, "rows_yielded": 0}

        path_value = source_cfg.get("path")
        if path_value and Path(str(path_value)).exists():
            iterator = iter_local_records(source_cfg, split=str(source_cfg.get("split", "train")), stats=stats)
            available = True
            kind = "local"
        elif source_cfg.get("hf_dataset"):
            if not (streaming or download):
                source_notes.append("remote source skipped: pass --streaming true or --download true to enable HuggingFace access")
            else:
                try:
                    iterator = iter_hf_records(source_cfg, split=str(source_cfg.get("split", "train")), streaming=streaming, download=download, stats=stats)
                    available = True
                    kind = "huggingface_streaming" if streaming and not download else "huggingface_download"
                except Exception as exc:  # noqa: BLE001
                    source_notes.append(f"remote source unavailable: {type(exc).__name__}: {exc}")
        else:
            source_notes.append("source unavailable: no local path and no HuggingFace dataset configured")

        if available:
            available_ratio_total += target_ratio
        runtimes.append(
            SourceRuntime(
                name=source_name,
                target_ratio=target_ratio,
                effective_ratio=0.0,
                kind=kind,
                config=source_cfg,
                iterator=iterator if iterator is not None else iter(()),
                available=available,
                notes=source_notes,
                stats=stats,
            )
        )

    if available_ratio_total <= 0:
        notes.append("no sources available for build")
        return runtimes, notes

    for runtime in runtimes:
        if runtime.available:
            runtime.effective_ratio = runtime.target_ratio / available_ratio_total
        else:
            runtime.effective_ratio = 0.0
            notes.extend(f"{runtime.name}: {note}" for note in runtime.notes)
    for runtime in runtimes:
        bad_json_lines = int(runtime.stats.get("bad_json_lines", 0))
        if bad_json_lines > 0:
            notes.append(f"{runtime.name}: skipped {bad_json_lines} malformed json lines during source iteration")
    return runtimes, notes


def maybe_divert_to_validation(
    row: Dict[str, object],
    validation_tokens_per_domain: int,
    validation_domain_tokens: Dict[str, int],
) -> bool:
    if validation_tokens_per_domain <= 0:
        return False
    domain = str(row["domain"])
    return validation_domain_tokens[domain] < validation_tokens_per_domain


def write_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def flatten_source_provenance(provenance: object) -> Dict[str, object]:
    if not isinstance(provenance, dict):
        return {}
    flattened: Dict[str, object] = {}
    for key in ("source_dataset", "source_note", "not_full_slimpajama_627b"):
        if key in provenance:
            flattened[key] = provenance[key]
    return flattened


def main() -> int:
    args = build_argument_parser().parse_args()
    root = repo_root()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory exists and is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_dir = output_dir / "validation_domain_balanced"
    validation_dir.mkdir(parents=True, exist_ok=True)

    cfg = resolve_config(load_yaml(Path(args.config)), args)
    domain_mapping_path = root / "configs" / "datasets" / "domain_mapping.yaml"
    domain_mapping = load_yaml(domain_mapping_path)
    balance_plan = build_balance_plan(cfg, domain_mapping)
    if balance_plan:
        notes_balance = list(balance_plan.plan_notes)
    else:
        notes_balance = []

    tokenizer_path = str(cfg["tokenizer"]["path"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    random.seed(int(cfg["seed"]))

    runtimes, notes = build_source_runtimes(cfg, domain_mapping, streaming=args.streaming, download=args.download)
    notes.extend(notes_balance)
    active_runtimes = [runtime for runtime in runtimes if runtime.available]
    if not active_runtimes:
        source_provenance = cfg.get("source_provenance", {})
        notes.append("build aborted: no active sources available")
        manifest = {
            "dataset_name": output_dir.name,
            "config_dataset_name": cfg.get("dataset_name"),
            "created_at": now_iso(),
            "target_tokens": int(cfg["target_tokens"]),
            "actual_tokens": 0,
            "seq_len": int(cfg["seq_len"]),
            "tokenizer": tokenizer_path,
            "sources": [],
            "domains": {},
            "output_format": cfg["output"]["format"],
            "num_shards": 0,
            "shards": [],
            "compatible_with_training_loader": True,
            "source_provenance": source_provenance,
            "metadata_batch_note": cfg.get("metadata_batch_note"),
            "balance": vars(balance_plan) if balance_plan else None,
            "notes": notes,
        }
        manifest.update(flatten_source_provenance(source_provenance))
        write_json(output_dir / "dataset_manifest.json", manifest)
        dump_yaml(output_dir / "dataset_build_config_resolved.yaml", cfg)
        return 0

    if args.dry_run:
        source_provenance = cfg.get("source_provenance", {})
        manifest = {
            "dataset_name": output_dir.name,
            "config_dataset_name": cfg.get("dataset_name"),
            "created_at": now_iso(),
            "target_tokens": int(cfg["target_tokens"]),
            "actual_tokens": 0,
            "seq_len": int(cfg["seq_len"]),
            "tokenizer": tokenizer_path,
            "sources": [
                {
                    "name": runtime.name,
                    "target_ratio": runtime.target_ratio,
                    "actual_ratio": 0.0,
                    "tokens": 0,
                    "available": runtime.available,
                    "kind": runtime.kind,
                    "notes": runtime.notes,
                }
                for runtime in runtimes
            ],
            "domains": {},
            "output_format": cfg["output"]["format"],
            "num_shards": 0,
            "shards": [],
            "compatible_with_training_loader": True,
            "source_provenance": source_provenance,
            "metadata_batch_note": cfg.get("metadata_batch_note"),
            "balance": vars(balance_plan) if balance_plan else None,
            "notes": notes + ["dry-run only; no data written"],
        }
        manifest.update(flatten_source_provenance(source_provenance))
        write_json(output_dir / "dataset_manifest.json", manifest)
        dump_yaml(output_dir / "dataset_build_config_resolved.yaml", cfg)
        return 0

    target_tokens = int(balance_plan.actual_target_tokens if balance_plan else cfg["target_tokens"])
    output_cfg = cfg["output"]
    shard_size_tokens = int(output_cfg["shard_size_tokens"])
    validation_tokens_per_domain = int(output_cfg.get("validation_tokens_per_domain", 131_072))
    mixture_name = str(cfg.get("dataset_name", output_dir.name))

    source_train_tokens = Counter()
    source_train_docs = Counter()
    domain_train_tokens = Counter()
    domain_train_docs = Counter()
    source_validation_tokens = Counter()
    source_validation_docs = Counter()
    domain_validation_tokens = Counter()
    domain_validation_docs = Counter()
    validation_domain_tokens = defaultdict(int)
    domain_target_budget = dict(balance_plan.domain_target_tokens) if balance_plan else {}

    per_source_budget = {runtime.name: int(target_tokens * runtime.effective_ratio) for runtime in active_runtimes}
    per_source_consumed = Counter()
    exhausted = set()

    train_writer = ShardedParquetWriter(output_dir, "train", shard_size_tokens=shard_size_tokens)
    validation_writer = ShardedParquetWriter(validation_dir, "validation", shard_size_tokens=max(validation_tokens_per_domain * 2, 1))

    train_tokens_total = 0
    train_docs_total = 0
    domain_skipped_after_budget = Counter()

    def train_complete() -> bool:
        if balance_plan:
            return all(domain_train_tokens[domain] >= budget for domain, budget in domain_target_budget.items())
        return train_tokens_total >= target_tokens

    while not train_complete() and len(exhausted) < len(active_runtimes):
        candidates = [runtime for runtime in active_runtimes if runtime.name not in exhausted]
        if not candidates:
            break
        candidates.sort(
            key=lambda runtime: (
                per_source_consumed[runtime.name] / max(per_source_budget[runtime.name], 1),
                runtime.name,
            )
        )
        runtime = candidates[0]
        try:
            raw_row = next(runtime.iterator)
        except StopIteration:
            exhausted.add(runtime.name)
            notes.append(f"source exhausted before target reached: {runtime.name}")
            continue
        except Exception as exc:  # noqa: BLE001
            exhausted.add(runtime.name)
            notes.append(f"source failed during iteration and was disabled: {runtime.name}: {type(exc).__name__}: {exc}")
            continue

        text = str(raw_row["text"]).strip()
        if not text:
            continue
        token_count = len(tokenizer.encode(text, add_special_tokens=False))
        if token_count <= 0:
            continue
        if args.max_docs_per_source and source_train_docs[runtime.name] >= args.max_docs_per_source:
            exhausted.add(runtime.name)
            notes.append(f"source capped by --max-docs-per-source: {runtime.name}")
            continue
        if args.max_tokens_per_source and per_source_consumed[runtime.name] >= args.max_tokens_per_source:
            exhausted.add(runtime.name)
            notes.append(f"source capped by --max-tokens-per-source: {runtime.name}")
            continue

        raw_domain = normalize_domain_name(raw_row.get("raw_domain"))
        mapped_domain = coarse_domain(runtime.name, raw_domain, domain_mapping)
        if balance_plan:
            budget = domain_target_budget.get(mapped_domain, 0)
            if budget <= 0:
                domain_skipped_after_budget[mapped_domain] += 1
                continue
            if domain_train_tokens[mapped_domain] >= budget:
                domain_skipped_after_budget[mapped_domain] += 1
                continue
        row = {
            "text": text,
            "source_dataset": runtime.name,
            "domain": mapped_domain,
            "original_source": str(raw_row.get("original_source", runtime.name)),
            "raw_domain": raw_domain,
            "token_count": token_count,
            "document_token_count": token_count,
            "sample_id": str(raw_row.get("sample_id", "")),
            "shard_id": str(raw_row.get("shard_id", "")),
            "mixture_name": mixture_name,
            "source_kind": runtime.kind,
        }

        if maybe_divert_to_validation(row, validation_tokens_per_domain, validation_domain_tokens):
            validation_writer.add_row(row)
            validation_domain_tokens[mapped_domain] += token_count
            source_validation_tokens[runtime.name] += token_count
            source_validation_docs[runtime.name] += 1
            domain_validation_tokens[mapped_domain] += token_count
            domain_validation_docs[mapped_domain] += 1
            per_source_consumed[runtime.name] += token_count
            continue

        train_writer.add_row(row)
        per_source_consumed[runtime.name] += token_count
        source_train_tokens[runtime.name] += token_count
        source_train_docs[runtime.name] += 1
        domain_train_tokens[mapped_domain] += token_count
        domain_train_docs[mapped_domain] += 1
        train_tokens_total += token_count
        train_docs_total += 1

    train_shards = train_writer.close()
    validation_shards = validation_writer.close()
    if balance_plan and domain_skipped_after_budget:
        notes.append(
            "rows skipped after domain quota reached: "
            + ", ".join(f"{domain}={count}" for domain, count in sorted(domain_skipped_after_budget.items()))
        )

    actual_source_total = sum(source_train_tokens.values()) or 1
    source_summary = {
        runtime.name: {
            "target_ratio": runtime.target_ratio,
            "effective_ratio_used_for_build": runtime.effective_ratio,
            "actual_ratio": source_train_tokens[runtime.name] / actual_source_total,
            "train_tokens": int(source_train_tokens[runtime.name]),
            "train_docs": int(source_train_docs[runtime.name]),
            "validation_tokens": int(source_validation_tokens[runtime.name]),
            "validation_docs": int(source_validation_docs[runtime.name]),
            "available": runtime.available,
            "kind": runtime.kind,
            "notes": runtime.notes,
        }
        for runtime in runtimes
    }
    domain_summary = {
        "train": {
            domain: {"tokens": int(domain_train_tokens[domain]), "docs": int(domain_train_docs[domain])}
            for domain in sorted(domain_train_tokens)
        },
        "validation_domain_balanced": {
            domain: {"tokens": int(domain_validation_tokens[domain]), "docs": int(domain_validation_docs[domain])}
            for domain in sorted(domain_validation_tokens)
        },
    }

    source_provenance = cfg.get("source_provenance", {})
    manifest = {
        "dataset_name": output_dir.name,
        "config_dataset_name": mixture_name,
        "created_at": now_iso(),
        "target_tokens": target_tokens,
        "requested_target_tokens": int(balance_plan.requested_target_tokens if balance_plan else target_tokens),
        "actual_tokens": int(train_tokens_total),
        "train_documents": int(train_docs_total),
        "validation_actual_tokens": int(sum(source_validation_tokens.values())),
        "seq_len": int(cfg["seq_len"]),
        "tokenizer": tokenizer_path,
        "sources": [
            {
                "name": runtime.name,
                "target_ratio": runtime.target_ratio,
                "effective_ratio_used_for_build": runtime.effective_ratio,
                "actual_ratio": source_summary[runtime.name]["actual_ratio"],
                "tokens": source_summary[runtime.name]["train_tokens"],
                "validation_tokens": source_summary[runtime.name]["validation_tokens"],
                "available": runtime.available,
                "kind": runtime.kind,
            }
            for runtime in runtimes
        ],
        "domains": {domain: int(stats["tokens"]) for domain, stats in domain_summary["train"].items()},
        "output_format": output_cfg["format"],
        "num_shards": len(train_shards),
        "shards": train_shards,
        "validation_domain_balanced_dir": str(validation_dir),
        "validation_num_shards": len(validation_shards),
        "validation_shards": validation_shards,
        "compatible_with_training_loader": True,
        "text_field": "text",
        "metadata_preserved": True,
        "metadata_strategy": "document-level metadata preserved in parquet rows; packed-sequence metadata is not materialized because the current training loader packs online from raw text.",
        "metadata_batch_note": cfg.get("metadata_batch_note"),
        "source_provenance": source_provenance,
        "balance": vars(balance_plan) if balance_plan else None,
        "notes": notes,
    }
    manifest.update(flatten_source_provenance(source_provenance))

    write_json(output_dir / "dataset_manifest.json", manifest)
    write_json(output_dir / "dataset_source_summary.json", source_summary)
    write_json(output_dir / "dataset_domain_summary.json", domain_summary)
    dump_yaml(output_dir / "dataset_build_config_resolved.yaml", cfg)

    validation_manifest = {
        "dataset_name": f"{output_dir.name}_validation_domain_balanced",
        "config_dataset_name": f"{mixture_name}_validation_domain_balanced",
        "created_at": now_iso(),
        "parent_dataset": mixture_name,
        "seq_len": int(cfg["seq_len"]),
        "tokenizer": tokenizer_path,
        "actual_tokens": int(sum(source_validation_tokens.values())),
        "domains": domain_summary["validation_domain_balanced"],
        "num_shards": len(validation_shards),
        "shards": validation_shards,
        "text_field": "text",
        "compatible_with_training_loader": True,
        "notes": notes,
    }
    write_json(validation_dir / "dataset_manifest.json", validation_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
