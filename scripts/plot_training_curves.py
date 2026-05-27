#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.jsonl files.")
    parser.add_argument("--log", action="append", required=True, help="Path to a train_log.jsonl file.")
    parser.add_argument("--label", action="append", required=True, help="Label for the corresponding log file.")
    parser.add_argument("--output-dir", type=Path, action="append", required=True, help="Per-log output directory for individual plots.")
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


def extract_curves(records: List[Dict]) -> Dict[str, List[float]]:
    train_steps: List[int] = []
    train_total_loss: List[float] = []
    train_lm_loss: List[float] = []
    train_lr: List[float] = []
    val_steps: List[int] = []
    val_lm_loss: List[float] = []
    for record in records:
        if "train_loss" in record:
            train_steps.append(int(record["step"]))
            train_total_loss.append(float(record.get("train_loss", 0.0)))
            train_lm_loss.append(float(record.get("lm_loss", record.get("train_loss", 0.0))))
            train_lr.append(float(record.get("lr", 0.0)))
        if "val_loss" in record:
            val_steps.append(int(record["step"]))
            val_lm_loss.append(float(record.get("val_loss", 0.0)))
    return {
        "train_steps": train_steps,
        "train_total_loss": train_total_loss,
        "train_lm_loss": train_lm_loss,
        "train_lr": train_lr,
        "val_steps": val_steps,
        "val_lm_loss": val_lm_loss,
    }


def plot_single_run(label: str, curves: Dict[str, List[float]], output_dir: Path, window: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_steps = curves["train_steps"]
    val_steps = curves["val_steps"]
    total_ma = moving_average(curves["train_total_loss"], window)
    lm_ma = moving_average(curves["train_lm_loss"], window)
    lr_values = curves["train_lr"]

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, total_ma, label=f"{label} total_loss_ma")
    if val_steps:
        plt.plot(val_steps, curves["val_lm_loss"], label=f"{label} val_loss", linestyle="--")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, lm_ma, label=f"{label} lm_loss_ma")
    if val_steps:
        plt.plot(val_steps, curves["val_lm_loss"], label=f"{label} val_loss", linestyle="--")
    plt.xlabel("step")
    plt.ylabel("lm_loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lm_loss_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(train_steps, lr_values, label=f"{label} lr")
    plt.xlabel("step")
    plt.ylabel("lr")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "lr_curve.png", dpi=160)
    plt.close()


def plot_comparison(labels: List[str], curve_list: List[Dict[str, List[float]]], output_path: Path, window: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 6))
    for label, curves in zip(labels, curve_list):
        plt.plot(curves["train_steps"], moving_average(curves["train_lm_loss"], window), label=f"{label} lm_loss_ma")
        plt.plot(curves["train_steps"], moving_average(curves["train_total_loss"], window), label=f"{label} total_loss_ma", linestyle="--")
        if curves["val_steps"]:
            plt.plot(curves["val_steps"], curves["val_lm_loss"], label=f"{label} val_loss", linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    if not (len(args.log) == len(args.label) == len(args.output_dir)):
        raise SystemExit("--log, --label, and --output-dir must have the same number of entries.")

    all_curves: List[Dict[str, List[float]]] = []
    for log_path_str, label, output_dir in zip(args.log, args.label, args.output_dir):
        log_path = Path(log_path_str)
        records = load_jsonl(log_path)
        curves = extract_curves(records)
        plot_single_run(label, curves, output_dir, args.window)
        all_curves.append(curves)

    if args.comparison_output is not None and len(all_curves) >= 2:
        plot_comparison(args.label, all_curves, args.comparison_output, args.window)


if __name__ == "__main__":
    main()
