
import json
from pathlib import Path

configs = [
    ("complement6e_relaxed_top2_lambda000_alpha005_aux0001_5000step", 0.0),
    ("complement6e_relaxed_top2_lambda020_alpha005_aux0001_5000step", 0.2),
]

cfg_dir = Path("experiments/sparse_upcycling/configs")

for name, lam in configs:
    path = cfg_dir / f"{name}.json"
    cfg = json.loads(path.read_text())

    cfg["experiment_name"] = name
    cfg["description"] = f"6E relaxed complement top2 with coverage penalty lambda={lam}, 5K sanity/screening run."

    # Common top-level fields if present / supported
    cfg["monitor_interval"] = cfg.get("monitor_interval", 100)

    moe = cfg.setdefault("moe", {})
    moe["routing_mode"] = "relaxed_complement_coverage"
    moe["num_experts"] = 6
    moe["num_experts_per_tok"] = 2
    moe["init_method"] = "complement_pair_6e"
    moe["pair_weights"] = "uniform"
    moe["moe_output_scale"] = 2.0
    moe["expert_intermediate_factor"] = 0.5
    moe["noise_alpha"] = 0.05
    moe["noise_mode"] = "relative_std"
    moe["router_aux_loss_coef"] = 0.0001
    moe["use_quantized_experts"] = True
    moe["normalize_topk_prob"] = True
    moe["coverage_penalty_lambda"] = lam
    moe["coverage_penalty_lambda_start"] = lam
    moe["coverage_penalty_lambda_end"] = lam
    moe["coverage_penalty_anneal_steps"] = 0
    moe["allow_arbitrary_expert_pairs"] = True

    freeze = cfg.setdefault("freeze", {})
    freeze["freeze_mode"] = "moe_only"
    freeze["freeze_embeddings"] = True
    freeze["freeze_lm_head"] = True
    freeze["freeze_token_mixer"] = True
    freeze["freeze_non_moe_mlp"] = True
    freeze["freeze_rmsnorm"] = True
    freeze["strict_trainable_check"] = True

    training = cfg.setdefault("training", {})
    training["max_steps"] = 5000
    training["batch_size"] = 2
    training["gradient_accumulation_steps"] = 16
    training["max_length"] = 2048
    training["learning_rate"] = 1e-3
    training["min_lr"] = 1e-4
    training["warmup_steps"] = 500
    training["weight_decay"] = 0.01
    training["grad_clip"] = 1.0
    training["precision"] = "bf16"
    training["log_interval"] = 25
    training["eval_interval"] = 1000
    training["save_interval"] = 1000
    training["max_eval_batches"] = 64
    training["max_val_samples"] = 1024

    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {path} lambda={lam}")
