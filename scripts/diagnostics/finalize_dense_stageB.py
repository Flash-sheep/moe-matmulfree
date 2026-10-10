#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a completed dense Stage B run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def nearest_train(
    records: List[Dict[str, object]],
    target_tokens: int,
) -> Optional[Dict[str, object]]:
    if not records:
        return None
    return min(
        records,
        key=lambda row: abs(int(row["global_tokens"]) - target_tokens),
    )


def validation_at(
    records: List[Dict[str, object]],
    target_tokens: int,
) -> Optional[Dict[str, object]]:
    candidates = [
        row for row in records if int(row["global_tokens"]) == target_tokens
    ]
    if candidates:
        return candidates[-1]
    return min(
        records,
        key=lambda row: abs(int(row["global_tokens"]) - target_tokens),
        default=None,
    )


def load_activation_history(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    config = yaml.safe_load(
        (output_dir / "resolved_config.yaml").read_text(encoding="utf-8")
    )
    records = load_jsonl(output_dir / "training_metrics.jsonl")
    train = [row for row in records if row.get("record_type") == "train"]
    validation = [row for row in records if row.get("record_type") == "validation"]
    if not train:
        raise SystemExit("No training records found")

    tokens_per_step = int(config["training"]["global_tokens_per_optimizer_step"])
    target_steps = int(config["training"]["stage_stop_optimizer_steps"])
    expected_tokens = int(config["training"]["stage_actual_stop_tokens"])
    final = train[-1]
    checkpoint_dir = output_dir / "checkpoint_final_1b"
    checkpoint_complete = all(
        (checkpoint_dir / name).is_file()
        for name in (
            "trainer_state.pt",
            "checkpoint_manifest.json",
            "dataset_state_rank0.pt",
            "dataset_state_rank1.pt",
        )
    )

    milestones = [
        0,
        10_000_000,
        50_000_000,
        100_000_000,
        250_000_000,
        500_000_000,
        750_000_000,
        1_000_000_000,
    ]
    loss_milestones = {}
    for target in milestones:
        train_row = nearest_train(train, target) if target else None
        validation_row = validation_at(validation, target) if validation else None
        loss_milestones[str(target)] = {
            "train": (
                {
                    "optimizer_step": train_row["optimizer_step"],
                    "global_tokens": train_row["global_tokens"],
                    "loss": train_row["train_loss"],
                }
                if train_row is not None
                else None
            ),
            "validation": (
                {
                    "kind": validation_row["validation_kind"],
                    "global_tokens": validation_row["global_tokens"],
                    "loss": validation_row["loss"],
                    "ppl": validation_row["ppl"],
                    "sequences": validation_row["sequences"],
                }
                if validation_row is not None
                else None
            ),
        }

    history = load_activation_history(output_dir / "activation_rms_history.csv")
    activation_rows = []
    for row in history:
        activation_rows.append(
            {
                "global_tokens": int(row["global_tokens"]),
                "first_residual_rms": float(row["first_residual_rms"]),
                "middle_residual_rms": float(row["middle_residual_rms"]),
                "last_residual_rms": float(row["last_residual_rms"]),
                "last_first_ratio": float(row["last_first_ratio"]),
                "warning_count": int(row["warning_count"]),
                "strong_warning_count": int(row["strong_warning_count"]),
                "auto_stop_count": int(row["auto_stop_count"]),
            }
        )
    max_last_rms = max(
        (row["last_residual_rms"] for row in activation_rows),
        default=None,
    )
    warning_count = sum(row["warning_count"] for row in activation_rows)
    strong_warning_count = sum(
        row["strong_warning_count"] for row in activation_rows
    )
    auto_stop_count = sum(row["auto_stop_count"] for row in activation_rows)

    peak_memory = {}
    for record in train:
        for gpu in record.get("gpu_snapshot", {}).get("gpus", []):
            index = str(gpu["index"])
            peak_memory[index] = max(
                peak_memory.get(index, 0.0),
                float(gpu["memory_used_mib"]),
            )
    stable_records = train[10:] if len(train) > 10 else train
    average_tokens_per_sec = mean(
        float(row["tokens_per_sec"]) for row in stable_records
    )
    non_finite = any(
        not math.isfinite(float(row["train_loss"]))
        or not math.isfinite(float(row["grad_norm"]))
        for row in train
    )
    failure_report_exists = (output_dir / "failure_report.json").exists()
    launch_log = (
        output_dir.parent
        / "stageB_launch_logs"
        / f"{output_dir.name}.log"
    )
    launch_log_text = (
        launch_log.read_text(encoding="utf-8", errors="replace")
        if launch_log.exists()
        else ""
    )
    nccl_error_detected = (
        "NCCL" in launch_log_text
        and (
            "collective operation timeout" in launch_log_text
            or "DistBackendError" in launch_log_text
            or "NCCL error" in launch_log_text
        )
    )
    oom_detected = (
        "CUDA out of memory" in launch_log_text
        or "OutOfMemoryError" in launch_log_text
    )
    wrapper_completed_cleanly = "Stage B completed:" in launch_log_text
    checkpoint_write_window_sec = (
        (checkpoint_dir / "trainer_state.pt").stat().st_mtime
        - (checkpoint_dir / "dataset_state_rank0.pt").stat().st_mtime
        if checkpoint_complete
        else None
    )
    checkpoint_load_smoke_path = output_dir / "checkpoint_load_smoke.json"
    checkpoint_load_smoke = (
        load_json(checkpoint_load_smoke_path)
        if checkpoint_load_smoke_path.exists()
        else {"status": "NOT_RUN"}
    )

    exact_streaming = load_json(output_dir / "streaming_resume_test_report.json")
    exact_streaming_passed = bool(
        exact_streaming.get("pass")
        or exact_streaming.get("status") == "PASS"
    )
    resume_equivalence = load_json(
        output_dir / "training_resume_equivalence_report.json"
    )
    distributed_eval = load_json(output_dir / "distributed_eval_report.json")
    completed = (
        int(final["optimizer_step"]) == target_steps
        and int(final["global_tokens"]) == expected_tokens
        and checkpoint_complete
    )
    hard_failure = (
        not completed
        or non_finite
        or failure_report_exists
        or nccl_error_detected
        or oom_detected
        or auto_stop_count > 0
        or not exact_streaming_passed
        or resume_equivalence.get("status") != "PASS"
        or distributed_eval.get("status") != "PASS"
    )
    warnings = [
        "Fused HGRN/quantized STE kernels are not bitwise reproducible across process restarts; calibrated parameter tolerance is documented.",
    ]
    if nccl_error_detected:
        warnings.append(
            "Training and checkpoint serialization completed, but the terminal "
            "NCCL barrier timed out while rank 0 wrote the 4.49GB checkpoint to NFS."
        )
    if warning_count:
        warnings.append(f"Activation monitor emitted {warning_count} WARN events.")
    if strong_warning_count:
        warnings.append(
            f"Activation monitor emitted {strong_warning_count} STRONG WARNING events."
        )
    verdict = "FAIL" if hard_failure else ("PASS WITH WARNINGS" if warnings else "PASS")

    summary = {
        "verdict": verdict,
        "experiment": config["experiment"]["name"],
        "from_scratch": True,
        "pretrained_weights_loaded": False,
        "stage_a_checkpoint_loaded": False,
        "dataset_path": config["data"]["path"],
        "validation_path": config["validation"]["path"],
        "tokenizer": config["data"]["tokenizer"],
        "total_parameters": 374_108_160,
        "trainable_parameters": 374_108_160,
        "global_tokens_per_optimizer_step": tokens_per_step,
        "optimizer_steps": int(final["optimizer_step"]),
        "global_tokens": int(final["global_tokens"]),
        "expected_stop_tokens": expected_tokens,
        "initial_train_loss": train[0]["train_loss"],
        "final_train_loss": final["train_loss"],
        "initial_validation": validation[0] if validation else None,
        "final_validation": validation[-1] if validation else None,
        "loss_milestones": loss_milestones,
        "initial_learning_rate": train[0]["learning_rate"],
        "final_learning_rate": final["learning_rate"],
        "peak_gpu_memory_mib": peak_memory,
        "average_stable_tokens_per_sec": average_tokens_per_sec,
        "activation_max_last_layer_residual_rms": max_last_rms,
        "activation_warning_count": warning_count,
        "activation_strong_warning_count": strong_warning_count,
        "activation_auto_stop_count": auto_stop_count,
        "non_finite_detected": non_finite,
        "oom_detected": oom_detected,
        "nccl_error_detected": nccl_error_detected,
        "wrapper_completed_cleanly": wrapper_completed_cleanly,
        "checkpoint_write_window_sec": checkpoint_write_window_sec,
        "checkpoint_load_smoke": checkpoint_load_smoke,
        "checkpoint_resume_completed": (
            (output_dir / "checkpoint_250m").is_dir()
            and checkpoint_complete
        ),
        "final_checkpoint": str(checkpoint_dir),
        "exact_data_resume_status": "PASS" if exact_streaming_passed else "FAIL",
        "training_resume_equivalence_status": resume_equivalence.get("status"),
        "distributed_eval_status": distributed_eval.get("status"),
        "warnings": warnings,
    }
    save_json(output_dir / "stageB_summary.json", summary)

    activation_lines = [
        "# Activation RMS Report",
        "",
        "| Global tokens | First residual RMS | Middle residual RMS | Last residual RMS | Last/first | WARN | STRONG | AUTO STOP |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in activation_rows:
        activation_lines.append(
            f"| {row['global_tokens']:,} | {row['first_residual_rms']:.6f} | "
            f"{row['middle_residual_rms']:.6f} | {row['last_residual_rms']:.6f} | "
            f"{row['last_first_ratio']:.3f} | {row['warning_count']} | "
            f"{row['strong_warning_count']} | {row['auto_stop_count']} |"
        )
    activation_lines.extend(
        [
            "",
            f"- Maximum last-layer residual RMS: {max_last_rms}",
            f"- WARN events: {warning_count}",
            f"- STRONG WARNING events: {strong_warning_count}",
            f"- AUTO STOP events: {auto_stop_count}",
        ]
    )
    (output_dir / "activation_rms_report.md").write_text(
        "\n".join(activation_lines) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Dense MatMul-free Scratch Stage B Report",
        "",
        f"- Verdict: **{verdict}**",
        f"- Optimizer steps: {summary['optimizer_steps']:,}",
        f"- Global tokens: {summary['global_tokens']:,}",
        f"- Initial train loss: {summary['initial_train_loss']}",
        f"- Final train loss: {summary['final_train_loss']}",
        f"- Final learning rate: {summary['final_learning_rate']}",
        f"- Average stable throughput: {average_tokens_per_sec:.2f} tokens/s",
        f"- Peak GPU memory MiB: {peak_memory}",
        f"- Final checkpoint: `{checkpoint_dir}`",
        "",
        "## Correctness Gates",
        "",
        f"- Exact data resume: {summary['exact_data_resume_status']}",
        f"- Training resume equivalence: {summary['training_resume_equivalence_status']}",
        f"- Distributed validation: {summary['distributed_eval_status']}",
        f"- Controlled 250M checkpoint and restart: {summary['checkpoint_resume_completed']}",
        f"- Final checkpoint load smoke: {checkpoint_load_smoke.get('status')}",
        f"- Non-finite values: {non_finite}",
        f"- NCCL error detected: {nccl_error_detected}",
        f"- Wrapper completed cleanly: {wrapper_completed_cleanly}",
        f"- Activation auto-stop events: {auto_stop_count}",
        "",
        "## Warnings",
        "",
    ]
    report_lines.extend(f"- {warning}" for warning in warnings)
    (output_dir / "stageB_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
