#!/usr/bin/env bash
set -euo pipefail

cd /home/storage/yjl/moe-matmulfree

PY="/opt/venv/bin/python"
CFG_DIR="experiments/sparse_upcycling/configs"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_SCRIPT_DIR="outputs/run_scripts_${RUN_TS}"

mkdir -p "${RUN_SCRIPT_DIR}"

echo "========== Environment check =========="
pwd
"${PY}" --version
nvidia-smi
tmux ls 2>/dev/null || echo "[INFO] No tmux server/session found."
ps aux | grep run_sparse_upcycling | grep -v grep || true
ps aux | grep docker | grep -v grep || true
echo

echo "========== Generate configs =========="

"${PY}" - <<'PY'
import copy
import json
from pathlib import Path

cfg_dir = Path("experiments/sparse_upcycling/configs")
cfg_dir.mkdir(parents=True, exist_ok=True)

def find_first(candidates, glob_patterns):
    for c in candidates:
        p = Path(c)
        if p.exists():
            return p
    for pat in glob_patterns:
        hits = sorted(cfg_dir.glob(pat))
        if hits:
            return hits[0]
    return None

relaxed_base = find_first(
    [
        "experiments/sparse_upcycling/configs/complement6e_relaxed_top2_lambda000_alpha005_aux0001_5000step.json",
        "experiments/sparse_upcycling/configs/complement6e_relaxed_coverage_top2_alpha005_aux0001_5000step.json",
    ],
    [
        "*relaxed*coverage*5000step.json",
        "*relaxed*top2*5000step.json",
    ],
)

pairfree_base = find_first(
    [
        "experiments/sparse_upcycling/configs/complement6e_pair_plus_free_top3_alpha005_aux0001_5000step.json",
        "experiments/sparse_upcycling/configs/complement6e_pair_plus_free_top3_scale050_alpha005_aux0001_5000step.json",
    ],
    [
        "*pair*free*top3*5000step.json",
        "*pair_plus_free*top3*.json",
    ],
)

strict_base = find_first(
    [
        "experiments/sparse_upcycling/configs/complement6e_half_top2_alpha005_aux0001_5000step.json",
        "experiments/sparse_upcycling/configs/complement6e_aux0001_5000step.json",
    ],
    [
        "*complement6e*aux0001*5000step.json",
        "*complement*6e*top2*5000step.json",
    ],
)

if relaxed_base is None:
    raise SystemExit("[ERROR] Cannot find relaxed base config. Please create or provide an existing relaxed config first.")
if pairfree_base is None:
    raise SystemExit("[ERROR] Cannot find pair+free base config. Please create or provide an existing pair+free config first.")

print(f"[INFO] relaxed_base = {relaxed_base}")
print(f"[INFO] pairfree_base = {pairfree_base}")
print(f"[INFO] strict_base = {strict_base if strict_base else relaxed_base}")

relaxed_base_cfg = json.loads(relaxed_base.read_text())
pairfree_base_cfg = json.loads(pairfree_base.read_text())
strict_base_cfg = json.loads((strict_base if strict_base else relaxed_base).read_text())

def ensure_dict(cfg, key):
    if not isinstance(cfg.get(key), dict):
        cfg[key] = {}
    return cfg[key]

def set_common_training(cfg, max_steps):
    training = ensure_dict(cfg, "training")
    training.update({
        "max_steps": max_steps,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "max_length": 2048,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "precision": "bf16",
        "log_interval": 25,
        "eval_interval": 1000,
        "save_interval": 1000,
        "max_eval_batches": 64,
        "max_val_samples": 1024,
    })
    # Also write common top-level keys for code paths that read cfg.get(...)
    cfg.update({
        "max_steps": max_steps,
        "batch_size": 2,
        "gradient_accumulation_steps": 16,
        "max_length": 2048,
        "learning_rate": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "precision": "bf16",
        "log_interval": 25,
        "eval_interval": 1000,
        "save_interval": 1000,
        "max_eval_batches": 64,
        "max_val_samples": 1024,
    })

def set_common_moe(cfg):
    moe = ensure_dict(cfg, "moe")
    common = {
        "moe_layer_indices": list(range(12, 24)),
        "num_experts": 6,
        "top_k": 2,
        "num_experts_per_tok": 2,
        "expert_intermediate_factor": 0.5,
        "init_method": "complement_pair_6e",
        "pair_weights": "uniform",
        "moe_output_scale": 2.0,
        "noise_alpha": 0.05,
        "noise_mode": "relative_std",
        "router_aux_loss_coef": 0.0001,
        "use_quantized_experts": True,
        "normalize_topk_prob": True,
        "router_jitter_noise": 0.0,
        "router_bias": False,
    }
    cfg.update(common)
    moe.update(common)

def set_moe_only_freeze(cfg):
    freeze = ensure_dict(cfg, "freeze")
    freeze.update({
        "freeze_mode": "moe_only",
        "freeze_embeddings": True,
        "freeze_lm_head": True,
        "freeze_token_mixer": True,
        "freeze_non_moe_mlp": True,
        "freeze_rmsnorm": True,
        "strict_trainable_check": True,
    })
    cfg.update({
        "freeze_mode": "moe_only",
        "freeze_embeddings": True,
        "freeze_lm_head": True,
        "freeze_token_mixer": True,
        "freeze_non_moe_mlp": True,
        "freeze_rmsnorm": True,
        "strict_trainable_check": True,
    })

def set_relaxed(cfg, lam=None, lam_start=None, lam_end=None, anneal_steps=0):
    set_common_moe(cfg)
    moe = ensure_dict(cfg, "moe")
    fields = {
        "routing_mode": "relaxed_complement_coverage",
        "allow_arbitrary_expert_pairs": True,
        "top_k": 2,
        "num_experts_per_tok": 2,
    }
    if lam is not None:
        fields.update({
            "coverage_penalty_lambda": lam,
            "coverage_penalty_lambda_start": lam,
            "coverage_penalty_lambda_end": lam,
            "coverage_penalty_anneal_steps": 0,
        })
    else:
        fields.update({
            "coverage_penalty_lambda": lam_start,
            "coverage_penalty_lambda_start": lam_start,
            "coverage_penalty_lambda_end": lam_end,
            "coverage_penalty_anneal_steps": anneal_steps,
        })
    cfg.update(fields)
    moe.update(fields)
    set_moe_only_freeze(cfg)

def set_strict(cfg):
    set_common_moe(cfg)
    moe = ensure_dict(cfg, "moe")
    fields = {
        "routing_mode": "strict_complement_pair",
        "top_k": 2,
        "num_experts_per_tok": 2,
        "pair_weights": "uniform",
        "moe_output_scale": 2.0,
        "router_aux_loss_coef": 0.0001,
    }
    cfg.update(fields)
    moe.update(fields)
    set_moe_only_freeze(cfg)

def set_pairfree(cfg, scale, max_steps):
    set_common_moe(cfg)
    moe = ensure_dict(cfg, "moe")
    fields = {
        "routing_mode": "complement_pair_plus_free",
        "top_k": 3,
        "num_experts_per_tok": 3,
        "base_pair_top_k": 2,
        "free_expert_top_k": 1,
        "pair_weights": "uniform",
        "moe_output_scale_base_pair": 2.0,
        "free_expert_scale": scale,
        "forbid_free_expert_overlap": True,
        "disallow_free_expert_overlap": True,
        "router_aux_loss_coef": 0.0001,
    }
    cfg.update(fields)
    moe.update(fields)
    set_moe_only_freeze(cfg)
    set_common_training(cfg, max_steps)

experiments = []

def write_cfg(name, cfg):
    path = cfg_dir / f"{name}.json"
    cfg["experiment_name"] = name
    cfg["description"] = cfg.get("description", name)
    path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {path}")
    return path

# GPU0 relaxed lambda 0.0
name = "complement6e_relaxed_top2_lambda000_alpha005_aux0001_5000step"
cfg = copy.deepcopy(relaxed_base_cfg)
set_relaxed(cfg, lam=0.0)
set_common_training(cfg, 5000)
write_cfg(name, cfg)

# GPU1 relaxed lambda 0.05
name = "complement6e_relaxed_top2_lambda005_alpha005_aux0001_5000step"
cfg = copy.deepcopy(relaxed_base_cfg)
set_relaxed(cfg, lam=0.05)
set_common_training(cfg, 5000)
write_cfg(name, cfg)

# GPU2 relaxed lambda 0.2
name = "complement6e_relaxed_top2_lambda020_alpha005_aux0001_5000step"
cfg = copy.deepcopy(relaxed_base_cfg)
set_relaxed(cfg, lam=0.2)
set_common_training(cfg, 5000)
write_cfg(name, cfg)

# GPU3 relaxed anneal 1.0 -> 0.0
name = "complement6e_relaxed_top2_lambda1to0_alpha005_aux0001_5000step"
cfg = copy.deepcopy(relaxed_base_cfg)
set_relaxed(cfg, lam=None, lam_start=1.0, lam_end=0.0, anneal_steps=2000)
set_common_training(cfg, 5000)
write_cfg(name, cfg)

# GPU4 strict control rerun
name = "complement6e_strict_rerun_alpha005_aux0001_5000step"
cfg = copy.deepcopy(strict_base_cfg)
set_strict(cfg)
set_common_training(cfg, 5000)
write_cfg(name, cfg)

# GPU5 pair+free scale 0.5 20K
name = "complement6e_pair_plus_free_top3_scale050_alpha005_aux0001_20000step"
cfg = copy.deepcopy(pairfree_base_cfg)
set_pairfree(cfg, 0.5, 20000)
write_cfg(name, cfg)

# GPU6 pair+free scale 0.25 5K
name = "complement6e_pair_plus_free_top3_scale025_alpha005_aux0001_5000step"
cfg = copy.deepcopy(pairfree_base_cfg)
set_pairfree(cfg, 0.25, 5000)
write_cfg(name, cfg)

# GPU7 pair+free scale 1.0 5K
name = "complement6e_pair_plus_free_top3_scale100_alpha005_aux0001_5000step"
cfg = copy.deepcopy(pairfree_base_cfg)
set_pairfree(cfg, 1.0, 5000)
write_cfg(name, cfg)
PY

echo
echo "========== Experiment plan =========="

EXPS=(
  "0 complement6e_relaxed_top2_lambda000_alpha005_aux0001_5000step moe_relaxed_lam000_5k"
  "1 complement6e_relaxed_top2_lambda005_alpha005_aux0001_5000step moe_relaxed_lam005_5k"
  "2 complement6e_relaxed_top2_lambda020_alpha005_aux0001_5000step moe_relaxed_lam020_5k"
  "3 complement6e_relaxed_top2_lambda1to0_alpha005_aux0001_5000step moe_relaxed_lam1to0_5k"
  "4 complement6e_strict_rerun_alpha005_aux0001_5000step moe_strict_rerun_5k"
  "5 complement6e_pair_plus_free_top3_scale050_alpha005_aux0001_20000step moe_pairfree_s050_20k"
  "6 complement6e_pair_plus_free_top3_scale025_alpha005_aux0001_5000step moe_pairfree_s025_5k"
  "7 complement6e_pair_plus_free_top3_scale100_alpha005_aux0001_5000step moe_pairfree_s100_5k"
)

for item in "${EXPS[@]}"; do
  read -r GPU NAME SESSION <<<"${item}"
  echo "GPU${GPU}: ${NAME} -> tmux ${SESSION}"
done

echo
echo "========== Safety checks =========="

for item in "${EXPS[@]}"; do
  read -r GPU NAME SESSION <<<"${item}"
  CFG="${CFG_DIR}/${NAME}.json"
  OUT="outputs/${NAME}"

  if [ ! -f "${CFG}" ]; then
    echo "[ERROR] Missing config: ${CFG}"
    exit 1
  fi

  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "[ERROR] tmux session already exists: ${SESSION}"
    exit 1
  fi

  if [ -d "${OUT}" ] && [ "$(find "${OUT}" -mindepth 1 | head -n 1 | wc -l)" -ne 0 ]; then
    echo "[ERROR] Output dir already exists and is non-empty: ${OUT}"
    echo "        Refusing to overwrite. Rename/remove it manually if this is only a failed test."
    exit 1
  fi
done

echo "[OK] Safety checks passed."

DO_PREFLIGHT="${DO_PREFLIGHT:-1}"

if [ "${DO_PREFLIGHT}" = "1" ]; then
  echo
  echo "========== Run preflight-only for all experiments =========="
  for item in "${EXPS[@]}"; do
    read -r GPU NAME SESSION <<<"${item}"
    CFG="${CFG_DIR}/${NAME}.json"
    PREFLIGHT_OUT="outputs/${NAME}_preflight_${RUN_TS}"
    PREFLIGHT_LOG="outputs/${NAME}_preflight_${RUN_TS}.log"

    echo
    echo "----- Preflight GPU${GPU}: ${NAME} -----"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/run_sparse_upcycling.py \
      --pretrained-path checkpoints/MMfreeLM-370M \
      --config-path "${CFG}" \
      --data-source datasets/SlimPajama-6B/data \
      --val-data-source datasets/SlimPajama-6B/data \
      --tokenizer-path checkpoints/MMfreeLM-370M \
      --output-dir "${PREFLIGHT_OUT}" \
      --preflight-only \
      2>&1 | tee "${PREFLIGHT_LOG}"
  done
else
  echo "[INFO] DO_PREFLIGHT=0, skipping preflight."
fi

echo
echo "========== Create per-experiment launch scripts =========="

for item in "${EXPS[@]}"; do
  read -r GPU NAME SESSION <<<"${item}"
  CFG="${CFG_DIR}/${NAME}.json"
  OUT="outputs/${NAME}"
  LOG="outputs/${NAME}.host.log"
  LAUNCH_SCRIPT="${RUN_SCRIPT_DIR}/${SESSION}.sh"

  cat > "${LAUNCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd /home/storage/yjl/moe-matmulfree

mkdir -p "${OUT}"

echo "========== Training: ${NAME} =========="
echo "GPU: ${GPU}"
echo "Config: ${CFG}"
echo "Output: ${OUT}"
echo "Start: \$(date)"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/run_sparse_upcycling.py \\
  --pretrained-path checkpoints/MMfreeLM-370M \\
  --config-path "${CFG}" \\
  --data-source datasets/SlimPajama-6B/data \\
  --val-data-source datasets/SlimPajama-6B/data \\
  --tokenizer-path checkpoints/MMfreeLM-370M \\
  --output-dir "${OUT}" \\
  2>&1 | tee "${LOG}"
TRAIN_STATUS=\${PIPESTATUS[0]}
set -e

echo "Training status: \${TRAIN_STATUS}"
if [ "\${TRAIN_STATUS}" -ne 0 ]; then
  echo "[ERROR] Training failed for ${NAME}"
  exit "\${TRAIN_STATUS}"
fi

echo "========== Postprocess: ${NAME} =========="
echo "Postprocess start: \$(date)"

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/postprocess_sparse_upcycling.py \\
  --config-path "${CFG}" \\
  --pretrained-path checkpoints/MMfreeLM-370M \\
  --output-dir "${OUT}" \\
  --val-data-source datasets/SlimPajama-6B/data \\
  --tokenizer-path checkpoints/MMfreeLM-370M \\
  2>&1 | tee -a "${LOG}"
POST_STATUS=\${PIPESTATUS[0]}
set -e

echo "Postprocess status: \${POST_STATUS}"
if [ "\${POST_STATUS}" -ne 0 ]; then
  echo "[ERROR] Postprocess failed for ${NAME}"
  exit "\${POST_STATUS}"
fi

echo "========== Done: ${NAME} =========="
echo "End: \$(date)"
echo "Main metric: ${OUT}/eval_results_1024.json"
EOF

  chmod +x "${LAUNCH_SCRIPT}"
  echo "[OK] ${LAUNCH_SCRIPT}"
done

echo
echo "========== Launch tmux sessions =========="

for item in "${EXPS[@]}"; do
  read -r GPU NAME SESSION <<<"${item}"
  LAUNCH_SCRIPT="${RUN_SCRIPT_DIR}/${SESSION}.sh"

  tmux new -d -s "${SESSION}" "bash ${LAUNCH_SCRIPT}"
  echo "[OK] launched ${SESSION} on GPU${GPU}"
done

echo
echo "========== Launched sessions =========="
tmux ls

echo
echo "========== Monitor commands =========="
echo "tmux ls"
echo "nvidia-smi"
echo

for item in "${EXPS[@]}"; do
  read -r GPU NAME SESSION <<<"${item}"
  echo "tail -f outputs/${NAME}.host.log"
done

echo
echo "========== Attach commands =========="
for item in "${EXPS[@]}"; do
  read -r GPU NAME SESSION <<<"${item}"
  echo "tmux attach -t ${SESSION}"
done

echo
echo "========== Notes =========="
echo "Detach from tmux with: Ctrl-b then d"
echo "Formal metric after completion: outputs/<experiment>/eval_results_1024.json"
echo "Do not use PPL@64 as the main decision criterion."