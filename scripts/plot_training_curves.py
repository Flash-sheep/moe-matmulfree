#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.jsonl or loss_curve.json files.")
    parser.add_argument("--log", action="append", required=True, help="Path to a train_log.jsonl or loss_curve.json file.")
    parser.add_argument("--label", action="append", required=True, help="Label for the corresponding log file.")
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        action="append",
        required=True,
        help="Per-log output directory for individual plots.",
    )
    parser.add_argument("--eval1024", action="append", help="Optional eval_results_1024.json path per log.")
    parser.add_argument("--training-report", action="append", help="Optional training_report.json path per log.")
    parser.add_argument("--comparison-output", type=Path, help="Optional output path for a multi-run comparison figure.")
    parser.add_argument("--window", type=int, default=100)
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or len(values) <= 1:
        return list(values)
    result: List[float] = []
    running_sum = 0.0
    history: List[float] = []
    for value in values:
        history.append(value)
        running_sum += value
        if len(history) > window:
            running_sum -= history.pop(0)
        result.append(running_sum / len(history))
    return result


def _safe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _series_from_entries(entries: Sequence[Dict[str, object]], key: str) -> List[Optional[float]]:
    return [_safe_float(entry.get(key)) for entry in entries]


def _pair_series(xs: Sequence[int], ys: Sequence[Optional[float]]) -> Tuple[List[int], List[float]]:
    filtered_xs: List[int] = []
    filtered_ys: List[float] = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        filtered_xs.append(int(x))
        filtered_ys.append(float(y))
    return filtered_xs, filtered_ys


def _warn(message: str) -> None:
    print(f"warning: {message}")


def _proxy_eval_label(entry: Dict[str, object]) -> str:
    sequences = entry.get("actual_num_sequences", entry.get("proxy_val_actual_num_sequences"))
    if sequences is not None:
        return f"proxy val ({sequences} seq)"
    return "proxy val (legacy)"


def extract_curves_from_train_log(records: List[Dict]) -> Dict[str, object]:
    train_entries: List[Dict[str, object]] = []
    proxy_eval_entries: List[Dict[str, object]] = []
    formal_eval_entries: List[Dict[str, object]] = []
    for record in records:
        if "train_loss" in record or "normalized_total_loss" in record:
            entry = {"step": int(record["step"])}
            for key, value in record.items():
                if key == "step":
                    continue
                entry[key] = value
            train_entries.append(entry)
        if "proxy_val_loss" in record or record.get("val_eval_name") == "proxy_val":
            proxy_eval_entries.append(
                {
                    "step": int(record["step"]),
                    "proxy_val_loss": record.get("proxy_val_loss", record.get("val_loss")),
                    "proxy_val_lm_loss": record.get("proxy_val_lm_loss", record.get("val_lm_loss")),
                    "proxy_val_ppl": record.get("proxy_val_ppl", record.get("val_ppl")),
                    "actual_num_sequences": record.get("proxy_val_actual_num_sequences", record.get("val_actual_num_sequences")),
                    "actual_num_batches": record.get("proxy_val_actual_num_batches", record.get("val_actual_num_batches")),
                    "actual_num_tokens": record.get("proxy_val_actual_num_tokens", record.get("val_actual_num_tokens")),
                    "proxy_val_scope": record.get("proxy_val_scope", record.get("val_eval_scope")),
                    "label": _proxy_eval_label(record),
                }
            )
    return {
        "train_entries": train_entries,
        "proxy_eval_entries": proxy_eval_entries,
        "formal_eval_entries": formal_eval_entries,
    }


def extract_curves_from_loss_curve(payload: Dict) -> Dict[str, object]:
    train_entries = payload.get("train", [])
    proxy_eval_entries = payload.get("eval", [])
    formal_eval_entries = payload.get("formal_eval_1024", [])
    return {
        "train_entries": train_entries,
        "proxy_eval_entries": proxy_eval_entries,
        "formal_eval_entries": formal_eval_entries,
    }


def extract_curves(path: Path) -> Dict[str, object]:
    if path.suffix == ".jsonl":
        return extract_curves_from_train_log(load_jsonl(path))
    payload = load_json(path)
    if "train" in payload:
        return extract_curves_from_loss_curve(payload)
    raise ValueError(f"Unsupported curve source: {path}")


def load_eval1024_summary(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None or not path.exists():
        return None
    payload = load_json(path)
    return {
        "loss": _safe_float(payload.get("val_loss")),
        "lm_loss": _safe_float(payload.get("val_lm_loss", payload.get("val_loss"))),
        "ppl": _safe_float(payload.get("val_ppl")),
        "actual_num_sequences": payload.get("actual_num_sequences", 1024),
        "label": f"formal eval ({payload.get('actual_num_sequences', 1024)} seq)",
    }


def load_training_report_summary(path: Optional[Path]) -> Optional[Dict[str, object]]:
    if path is None or not path.exists():
        return None
    payload = load_json(path)
    formal_eval = payload.get("formal_eval_1024", {})
    if not isinstance(formal_eval, dict):
        return None
    actual_num_sequences = formal_eval.get("actual_num_sequences")
    return {
        "formal_eval_1024_actual_num_sequences": actual_num_sequences,
        "formal_eval_1024_checkpoint_source": formal_eval.get("checkpoint_source"),
        "proxy_eval_scope": payload.get("proxy_eval", {}).get("eval_scope") if isinstance(payload.get("proxy_eval"), dict) else None,
    }


def _plot_line(ax, xs: Sequence[int], ys: Sequence[Optional[float]], label: str, *, style: str = "-", smooth_window: Optional[int] = None) -> bool:
    filtered_xs, filtered_ys = _pair_series(xs, ys)
    if not filtered_xs:
        return False
    if smooth_window is not None:
        filtered_ys = moving_average(filtered_ys, smooth_window)
    ax.plot(filtered_xs, filtered_ys, style, label=label)
    return True


def _plot_proxy_series(ax, proxy_eval_entries: Sequence[Dict[str, object]], key: str) -> bool:
    if not proxy_eval_entries:
        return False
    xs = [int(entry["step"]) for entry in proxy_eval_entries]
    ys = [_safe_float(entry.get(key)) for entry in proxy_eval_entries]
    labels = [str(entry.get("label", "proxy val")) for entry in proxy_eval_entries]
    filtered_xs, filtered_ys = _pair_series(xs, ys)
    if not filtered_xs:
        return False
    ax.plot(filtered_xs, filtered_ys, "--", linewidth=1.0, label=labels[-1] if labels else "proxy val")
    return True


def _plot_formal_hline(ax, formal_eval_summary: Optional[Dict[str, object]], key: str, xmin: int, xmax: int) -> bool:
    if formal_eval_summary is None:
        return False
    y = _safe_float(formal_eval_summary.get(key))
    if y is None:
        return False
    ax.hlines(y, xmin=xmin, xmax=xmax, colors="tab:red", linestyles=":", label=str(formal_eval_summary["label"]))
    return True


def _has_any_series(entries: Sequence[Dict[str, object]], keys: Sequence[str]) -> bool:
    for key in keys:
        if any(_safe_float(entry.get(key)) is not None for entry in entries):
            return True
    return False


def plot_single_run(
    label: str,
    curves: Dict[str, object],
    formal_eval_summary: Optional[Dict[str, object]],
    training_report_summary: Optional[Dict[str, object]],
    output_dir: Path,
    window: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_entries: List[Dict[str, object]] = curves["train_entries"]  # type: ignore[assignment]
    proxy_eval_entries: List[Dict[str, object]] = curves.get("proxy_eval_entries", [])  # type: ignore[assignment]
    if not train_entries:
        raise ValueError(f"No train steps found for plotting: {label}")

    train_steps = [int(entry["step"]) for entry in train_entries]
    xmin, xmax = train_steps[0], train_steps[-1]

    normalized_total = _series_from_entries(train_entries, "normalized_total_loss")
    if all(value is None for value in normalized_total):
        normalized_total = _series_from_entries(train_entries, "train_loss")
    normalized_lm = _series_from_entries(train_entries, "normalized_lm_loss")
    if all(value is None for value in normalized_lm):
        normalized_lm = _series_from_entries(train_entries, "lm_loss")
    normalized_router_aux = _series_from_entries(train_entries, "normalized_router_aux_loss")
    if all(value is None for value in normalized_router_aux):
        normalized_router_aux = _series_from_entries(train_entries, "router_aux_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    plotted |= _plot_line(ax, train_steps, normalized_total, f"{label} normalized_total_loss_ma", smooth_window=window)
    plotted |= _plot_line(ax, train_steps, normalized_router_aux, f"{label} normalized_router_aux_loss_ma", style="-.", smooth_window=window)
    _plot_proxy_series(ax, proxy_eval_entries, "proxy_val_loss")
    _plot_formal_hline(ax, formal_eval_summary, "loss", xmin, xmax)
    if not plotted:
        _warn(f"{label}: no total loss fields found for total_loss_curve.png")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "total_loss_curve.png", dpi=160)
    fig.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    plotted |= _plot_line(ax, train_steps, normalized_lm, f"{label} normalized_lm_loss_ma", smooth_window=window)
    _plot_proxy_series(ax, proxy_eval_entries, "proxy_val_lm_loss")
    _plot_formal_hline(ax, formal_eval_summary, "lm_loss", xmin, xmax)
    if not plotted:
        _warn(f"{label}: no LM loss fields found for lm_loss_curve.png")
    ax.set_xlabel("step")
    ax.set_ylabel("lm_loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lm_loss_curve.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    proxy_steps = [int(entry["step"]) for entry in proxy_eval_entries]
    proxy_lm = [_safe_float(entry.get("proxy_val_lm_loss")) for entry in proxy_eval_entries]
    proxy_ppl = [_safe_float(entry.get("proxy_val_ppl")) for entry in proxy_eval_entries]
    proxy_label = proxy_eval_entries[-1]["label"] if proxy_eval_entries else "proxy val"
    plotted_lm = _plot_line(ax1, proxy_steps, proxy_lm, f"{proxy_label} lm_loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("proxy lm_loss")
    plotted_ppl = False
    if any(value is not None for value in proxy_ppl):
        ax2 = ax1.twinx()
        px, py = _pair_series(proxy_steps, proxy_ppl)
        if px:
            ax2.plot(px, py, ":", color="tab:orange", label="proxy val ppl")
            ax2.set_ylabel("proxy ppl")
            plotted_ppl = True
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2)
    if not plotted_lm and not plotted_ppl:
        _warn(f"{label}: no proxy validation fields found for proxy_val_curve.png")
    elif not plotted_ppl:
        ax1.legend()
    title_bits = []
    if proxy_eval_entries:
        title_bits.append(str(proxy_eval_entries[-1].get("label")))
    if training_report_summary and training_report_summary.get("proxy_eval_scope"):
        title_bits.append(str(training_report_summary["proxy_eval_scope"]))
    if title_bits:
        ax1.set_title(" | ".join(title_bits))
    fig.tight_layout()
    fig.savefig(output_dir / "proxy_val_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    lr_fields = [
        ("lr_moe", "lr_moe"),
        ("lr_shared_expert", "lr_shared_expert"),
        ("lr_backbone", "lr_backbone"),
        ("lr_norm_or_bias", "lr_norm_or_bias"),
        ("lr_embed_lm_head", "lr_embed_lm_head"),
    ]
    plotted = False
    for field, series_label in lr_fields:
        plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, field), series_label)
    if not plotted:
        plotted = _plot_line(ax, train_steps, _series_from_entries(train_entries, "lr"), "legacy lr (old log lacks group lr fields)")
        if plotted:
            ax.set_title("Old log lacks group LR fields; showing legacy lr only")
        else:
            _warn(f"{label}: no LR fields found for lr_curve.png")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lr_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "router_entropy"), "router_entropy")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "expert_entropy"), "expert_entropy")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "pair_entropy"), "pair_entropy")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "free_expert_entropy"), "free_expert_entropy")
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("entropy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "router_curve.png", dpi=160)
    else:
        _warn(f"{label}: router/expert/pair entropy fields missing; skipping router_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "expert_load_imbalance"), "expert_load_imbalance")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "pair_load_imbalance"), "pair_load_imbalance")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "dead_expert_count"), "dead_expert_count")
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("usage / imbalance")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "expert_usage_curve.png", dpi=160)
    else:
        _warn(f"{label}: expert usage imbalance fields missing; skipping expert_usage_curve.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "sparse_to_shared_norm_ratio"), "sparse_to_shared_norm_ratio")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "residual_scale"), "residual_scale")
    plotted |= _plot_line(ax, train_steps, _series_from_entries(train_entries, "active_width_ratio"), "active_width_ratio")
    if plotted:
        ax.set_xlabel("step")
        ax.set_ylabel("residual metric")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "residual_curve.png", dpi=160)
    else:
        _warn(f"{label}: residual/shared metrics missing; skipping residual_curve.png")
    plt.close(fig)


def plot_comparison(
    labels: List[str],
    curve_list: List[Dict[str, object]],
    formal_eval_summaries: List[Optional[Dict[str, object]]],
    output_path: Path,
    window: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, curves, formal_eval_summary in zip(labels, curve_list, formal_eval_summaries):
        train_entries: List[Dict[str, object]] = curves["train_entries"]  # type: ignore[assignment]
        if not train_entries:
            continue
        steps = [int(entry["step"]) for entry in train_entries]
        lm_series = _series_from_entries(train_entries, "normalized_lm_loss")
        if all(value is None for value in lm_series):
            lm_series = _series_from_entries(train_entries, "lm_loss")
        total_series = _series_from_entries(train_entries, "normalized_total_loss")
        if all(value is None for value in total_series):
            total_series = _series_from_entries(train_entries, "train_loss")
        lx, ly = _pair_series(steps, lm_series)
        tx, ty = _pair_series(steps, total_series)
        if lx:
            ax.plot(lx, moving_average(ly, window), label=f"{label} normalized_lm_loss_ma")
        if tx:
            ax.plot(tx, moving_average(ty, window), "--", label=f"{label} normalized_total_loss_ma")
        _plot_formal_hline(ax, formal_eval_summary, "lm_loss", steps[0], steps[-1])
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (len(args.log) == len(args.label) == len(args.output_dir)):
        raise SystemExit("--log, --label, and --output-dir/--out-dir must have the same number of entries.")

    eval1024_paths = args.eval1024 or []
    while len(eval1024_paths) < len(args.log):
        eval1024_paths.append(None)
    training_report_paths = args.training_report or []
    while len(training_report_paths) < len(args.log):
        training_report_paths.append(None)

    if len(eval1024_paths) > len(args.log):
        raise SystemExit("--eval1024 cannot be provided more times than --log.")
    if len(training_report_paths) > len(args.log):
        raise SystemExit("--training-report cannot be provided more times than --log.")

    all_curves: List[Dict[str, object]] = []
    formal_eval_summaries: List[Optional[Dict[str, object]]] = []
    for log_path_str, label, output_dir, eval1024_str, training_report_str in zip(
        args.log,
        args.label,
        args.output_dir,
        eval1024_paths,
        training_report_paths,
    ):
        log_path = Path(log_path_str)
        curves = extract_curves(log_path)
        training_report_summary = load_training_report_summary(Path(training_report_str) if training_report_str else None)
        formal_eval_summary = load_eval1024_summary(Path(eval1024_str) if eval1024_str is not None else None)
        if formal_eval_summary is None and curves.get("formal_eval_entries"):
            formal_entries: List[Dict[str, object]] = curves["formal_eval_entries"]  # type: ignore[assignment]
            if formal_entries:
                formal_entry = formal_entries[-1]
                formal_eval_summary = {
                    "loss": _safe_float(formal_entry.get("val_loss")),
                    "lm_loss": _safe_float(formal_entry.get("formal_eval_1024_lm_loss", formal_entry.get("val_lm_loss"))),
                    "ppl": _safe_float(formal_entry.get("formal_eval_1024_ppl", formal_entry.get("val_ppl"))),
                    "actual_num_sequences": formal_entry.get("actual_num_sequences", 1024),
                    "label": f"formal eval ({formal_entry.get('actual_num_sequences', 1024)} seq)",
                }
        plot_single_run(label, curves, formal_eval_summary, training_report_summary, output_dir, args.window)
        all_curves.append(curves)
        formal_eval_summaries.append(formal_eval_summary)

    if args.comparison_output is not None and len(all_curves) >= 2:
        plot_comparison(args.label, all_curves, formal_eval_summaries, args.comparison_output, args.window)


if __name__ == "__main__":
    main()
