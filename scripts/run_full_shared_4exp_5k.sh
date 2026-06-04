#!/usr/bin/env bash
set -euo pipefail

WORKDIR="/home/storage/yjl/moe-matmulfree"
PY="/opt/venv/bin/python"
CFG_DIR="experiments/sparse_upcycling/configs"
PRETRAINED_PATH="checkpoints/MMfreeLM-370M"
DATA_SOURCE="datasets/SlimPajama-6B/data"
VAL_DATA_SOURCE="datasets/SlimPajama-6B/data"
TOKENIZER_PATH="checkpoints/MMfreeLM-370M"
DO_PREFLIGHT="${DO_PREFLIGHT:-1}"
LAUNCH_TRAIN="${LAUNCH_TRAIN:-1}"
GPU_MAP="${GPU_MAP:-0,1,2,3}"
ACTIVE_GRACE_MINUTES="${ACTIVE_GRACE_MINUTES:-30}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
RUN_SCRIPT_DIR="outputs/run_scripts_full_shared_${RUN_TS}"
GENERATED_CFG_DIR="outputs/generated_configs_full_shared_${RUN_TS}"

EXPERIMENTS=(
  "full_shared_4x64_top1_alpha005_moe_only_5k|${CFG_DIR}/full_shared_4x64_top1_alpha005_moe_only_5k.json|fs_4x64_moeonly_5k|moe_only|4|64|1|0.05|0.0005|64"
  "full_shared_8x32_top2_alpha005_moe_only_5k|${CFG_DIR}/full_shared_8x32_top2_alpha005_moe_only_5k.json|fs_8x32_moeonly_5k|moe_only|8|32|2|0.05|0.0002|64"
  "full_shared_4x64_top1_alpha005_local_backbone_ft_5k|${CFG_DIR}/full_shared_4x64_top1_alpha005_local_backbone_ft_5k.json|fs_4x64_local_5k|local_backbone_ft|4|64|1|0.05|0.0005|64"
  "full_shared_8x32_top2_alpha005_local_backbone_ft_5k|${CFG_DIR}/full_shared_8x32_top2_alpha005_local_backbone_ft_5k.json|fs_8x32_local_5k|local_backbone_ft|8|32|2|0.05|0.0002|64"
)

GPU_MAP_COMPACT="${GPU_MAP// /}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_MAP_COMPACT}"
declare -A EXP_STATUS
declare -A EXP_REASON
declare -A EXP_CFG_PATH
declare -a LAUNCHABLE_EXPERIMENTS=()

die() {
  echo "ERROR: $*" >&2
  exit 1
}

info() {
  echo "[INFO] $*"
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

dir_nonempty() {
  local path="$1"
  [[ -d "${path}" ]] && find "${path}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .
}

is_recent_path() {
  local path="$1"
  [[ -e "${path}" ]] || return 1
  find "${path}" -mmin "-${ACTIVE_GRACE_MINUTES}" -print -quit 2>/dev/null | grep -q .
}

train_log_has_step_records() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  grep -q '"step"' "${path}"
}

run_py() {
  "$@"
}

run_py_gpu() {
  local gpu="$1"
  shift
  CUDA_VISIBLE_DEVICES="${gpu}" "$@"
}

log_env() {
  cd "${WORKDIR}"
  echo "WORKDIR=${WORKDIR}"
  pwd
  [[ -x "${PY}" ]] || die "Python not found or not executable: ${PY}"
  run_py "${PY}" --version
  nvidia-smi
  tmux ls 2>/dev/null || true
  ps aux | grep run_sparse_upcycling | grep -v grep || true
}

write_config_payload() {
  local target_cfg_path="$1"
  local experiment_name="$2"
  local freeze_mode="$3"
  local num_experts="$4"
  local sparse_expert_width="$5"
  local sparse_top_k="$6"
  local residual_scale_init="$7"
  local router_aux_loss_coef="$8"
  local backbone_lr="0.00003"
  local norm_lr="0.00003"

  if [[ "${freeze_mode}" == "moe_only" ]]; then
    backbone_lr="0.0"
    norm_lr="0.0"
  fi

  mkdir -p "$(dirname "${target_cfg_path}")"
  cat > "${target_cfg_path}" <<EOF
{
  "experiment_name": "${experiment_name}",
  "description": "Full-shared FFN shared-residual experiment with exact shared width 2816, ${num_experts}x${sparse_expert_width} sparse residual experts, top${sparse_top_k}, alpha=${residual_scale_init}, freeze_mode=${freeze_mode}.",
  "seed": 42,
  "monitor_interval": 100,
  "moe_lr": 0.0003,
  "shared_expert_lr": 0.0001,
  "backbone_lr": ${backbone_lr},
  "norm_lr": ${norm_lr},
  "moe": {
    "moe_arch": "shared_residual",
    "layer_indices": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "num_experts": ${num_experts},
    "num_experts_per_tok": ${sparse_top_k},
    "enable_sparse_residual": true,
    "nominal_shared_width": 2816,
    "auto_resolve_shared_width": false,
    "min_shared_width": 2816,
    "shared_width_step": 16,
    "strict_total_param_fair": false,
    "skip_param_budget_resolver": true,
    "shared_init": "dense_prefix",
    "sparse_init": "random_ternary_matched",
    "sparse_expert_width": ${sparse_expert_width},
    "sparse_top_k": ${sparse_top_k},
    "residual_scale_init": ${residual_scale_init},
    "residual_scale_learnable": true,
    "residual_scale_max": 0.5,
    "use_quantized_experts": true,
    "router_aux_loss_coef": ${router_aux_loss_coef},
    "router_jitter_noise": 0.0,
    "router_bias": false,
    "normalize_topk_prob": true
  },
  "freeze": {
    "freeze_mode": "${freeze_mode}",
    "local_backbone_layer_indices": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "freeze_embeddings": true,
    "freeze_lm_head": true,
    "freeze_token_mixer": false,
    "freeze_non_moe_mlp": false,
    "freeze_rmsnorm": false,
    "norm_scope": "moe_layers_all_norm",
    "strict_trainable_check": true,
    "trainable_extra_patterns": []
  },
  "training": {
    "precision": "bf16",
    "device": "cuda",
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "max_steps": 5000,
    "max_length": 2048,
    "learning_rate": 0.0003,
    "min_lr": 0.00001,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "text_field": "text",
    "log_interval": 25,
    "eval_interval": 1000,
    "save_interval": 1000,
    "max_eval_batches": 64,
    "max_val_samples": 1024
  }
}
EOF
  info "Wrote config: ${target_cfg_path}"
}

config_matches_spec() {
  local cfg_path="$1"
  local experiment_name="$2"
  local freeze_mode="$3"
  local num_experts="$4"
  local sparse_expert_width="$5"
  local sparse_top_k="$6"
  local residual_scale_init="$7"
  local router_aux_loss_coef="$8"
  [[ -f "${cfg_path}" ]] || return 1
  "${PYTHON_CHECKER:-python3}" - "${cfg_path}" "${experiment_name}" "${freeze_mode}" "${num_experts}" "${sparse_expert_width}" "${sparse_top_k}" "${residual_scale_init}" "${router_aux_loss_coef}" <<'PY'
import json
import math
import sys

cfg_path, expected_name, expected_freeze_mode, expected_num_experts, expected_width, expected_topk, expected_alpha, expected_aux = sys.argv[1:]
cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
moe = cfg.get("moe", {})
freeze = cfg.get("freeze", {})
assert cfg.get("experiment_name") == expected_name
assert freeze.get("freeze_mode") == expected_freeze_mode
assert moe.get("moe_arch") == "shared_residual"
assert bool(moe.get("enable_sparse_residual", False)) is True
assert int(moe.get("nominal_shared_width", -1)) == 2816
assert bool(moe.get("auto_resolve_shared_width", True)) is False
assert bool(moe.get("strict_total_param_fair", True)) is False
assert bool(moe.get("skip_param_budget_resolver", False)) is True
assert moe.get("shared_init") == "dense_prefix"
assert moe.get("sparse_init") == "random_ternary_matched"
assert int(moe.get("num_experts", -1)) == int(expected_num_experts)
assert int(moe.get("sparse_expert_width", -1)) == int(expected_width)
assert int(moe.get("sparse_top_k", -1)) == int(expected_topk)
assert math.isclose(float(moe.get("residual_scale_init", -1.0)), float(expected_alpha), rel_tol=0.0, abs_tol=1e-9)
assert math.isclose(float(moe.get("router_aux_loss_coef", -1.0)), float(expected_aux), rel_tol=0.0, abs_tol=1e-9)
PY
}

prepare_configs() {
  cd "${WORKDIR}"
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name cfg_path _ freeze_mode num_experts sparse_expert_width sparse_top_k residual_scale_init router_aux_loss_coef _ <<< "${spec}"
    if config_matches_spec "${cfg_path}" "${experiment_name}" "${freeze_mode}" "${num_experts}" "${sparse_expert_width}" "${sparse_top_k}" "${residual_scale_init}" "${router_aux_loss_coef}" >/dev/null 2>&1; then
      EXP_CFG_PATH["${experiment_name}"]="${cfg_path}"
      info "Using existing config: ${cfg_path}"
      continue
    fi
    local generated_cfg_path="${GENERATED_CFG_DIR}/${experiment_name}.json"
    write_config_payload \
      "${generated_cfg_path}" \
      "${experiment_name}" \
      "${freeze_mode}" \
      "${num_experts}" \
      "${sparse_expert_width}" \
      "${sparse_top_k}" \
      "${residual_scale_init}" \
      "${router_aux_loss_coef}"
    EXP_CFG_PATH["${experiment_name}"]="${generated_cfg_path}"
  done
}

verify_feature_support() {
  cd "${WORKDIR}"
  run_py "${PY}" - <<'PY'
from pathlib import Path
from mmfreelm.upcycling.trainable_scope import should_train_parameter

root = Path("/home/storage/yjl/moe-matmulfree")
upcycle_text = (root / "mmfreelm/upcycling/sparse_upcycling.py").read_text(encoding="utf-8")
run_text = (root / "scripts/run_sparse_upcycling.py").read_text(encoding="utf-8")

missing = []
if 'skip_param_budget_resolver: bool = False' not in upcycle_text:
    missing.append("skip_param_budget_resolver is not wired into upcycle_dense_to_moe")
if 'shared_width_resolution_mode": "exact_requested_width"' not in upcycle_text:
    missing.append("exact full-shared parameter budget payload path is missing")
if 'skip_param_budget_resolver=moe_cfg.get("skip_param_budget_resolver", False)' not in run_text:
    missing.append("run_sparse_upcycling.py does not pass skip_param_budget_resolver through")
if 'random_ternary_matched' not in upcycle_text:
    missing.append("random_ternary_matched sparse init support not found")

moe_layers = list(range(12, 24))
shared_ok = should_train_parameter(
    name="model.layers.12.mlp.shared_expert.gate_proj.weight",
    freeze_mode="moe_only",
    moe_layer_indices=moe_layers,
    local_backbone_layer_indices=moe_layers,
    norm_scope="none",
    freeze_embeddings=True,
    freeze_lm_head=True,
)
backbone_blocked = not should_train_parameter(
    name="model.layers.12.attn.i_proj.weight",
    freeze_mode="moe_only",
    moe_layer_indices=moe_layers,
    local_backbone_layer_indices=moe_layers,
    norm_scope="none",
    freeze_embeddings=True,
    freeze_lm_head=True,
)
if not shared_ok:
    missing.append("freeze_mode=moe_only does not currently leave shared_expert trainable")
if not backbone_blocked:
    missing.append("freeze_mode=moe_only still trains non-MoE backbone parameters")

if missing:
    raise SystemExit(
        "Required full-shared capability checks failed:\n- " + "\n- ".join(missing) +
        "\nRefusing to proceed to preflight or launch."
    )
print("Full-shared capability checks passed.")
PY
}

safety_checks() {
  cd "${WORKDIR}"
  [[ ${#GPU_LIST[@]} -eq 4 ]] || die "GPU_MAP must contain exactly 4 comma-separated GPU ids. Got: ${GPU_MAP}"
  LAUNCHABLE_EXPERIMENTS=()

  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name _ session_name freeze_mode num_experts sparse_expert_width sparse_top_k residual_scale_init router_aux_loss_coef _ <<< "${spec}"
    local cfg_path="${EXP_CFG_PATH[${experiment_name}]:-}"
    local output_dir="outputs/${experiment_name}"
    local host_log="outputs/${experiment_name}.host.log"
    local train_log="${output_dir}/train_log.jsonl"
    local eval_1024="${output_dir}/eval_results_1024.json"
    local training_report="${output_dir}/training_report.json"

    [[ -n "${cfg_path}" && -f "${cfg_path}" ]] || die "Missing resolved config for ${experiment_name}: ${cfg_path}"
    config_matches_spec "${cfg_path}" "${experiment_name}" "${freeze_mode}" "${num_experts}" "${sparse_expert_width}" "${sparse_top_k}" "${residual_scale_init}" "${router_aux_loss_coef}" >/dev/null \
      || die "Resolved config does not match expected full-shared semantics: ${cfg_path}"

    if tmux has-session -t "${session_name}" 2>/dev/null; then
      EXP_STATUS["${experiment_name}"]="skip"
      EXP_REASON["${experiment_name}"]="tmux session already exists: ${session_name}"
      continue
    fi
    if [[ -f "${eval_1024}" || -f "${training_report}" ]]; then
      EXP_STATUS["${experiment_name}"]="skip"
      EXP_REASON["${experiment_name}"]="formal outputs already exist"
      continue
    fi
    if is_recent_path "${train_log}" && train_log_has_step_records "${train_log}"; then
      EXP_STATUS["${experiment_name}"]="skip"
      EXP_REASON["${experiment_name}"]="existing train_log.jsonl looks active or recently updated"
      continue
    fi
    if dir_nonempty "${output_dir}" || [[ -e "${host_log}" ]]; then
      EXP_STATUS["${experiment_name}"]="skip"
      EXP_REASON["${experiment_name}"]="stale partial output/log exists; refusing to overwrite"
      continue
    fi
    EXP_STATUS["${experiment_name}"]="launch"
    EXP_REASON["${experiment_name}"]="ready"
    LAUNCHABLE_EXPERIMENTS+=("${experiment_name}")
  done
}

run_preflight() {
  cd "${WORKDIR}"
  mkdir -p "${RUN_SCRIPT_DIR}"

  for idx in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name _ _ _ _ _ _ _ _ _ <<< "${EXPERIMENTS[$idx]}"
    [[ "${EXP_STATUS[${experiment_name}]:-skip}" == "launch" ]] || continue
    local cfg_path="${EXP_CFG_PATH[${experiment_name}]:-}"
    local gpu="${GPU_LIST[$idx]}"
    local preflight_dir="outputs/${experiment_name}_preflight_${RUN_TS}"
    local preflight_log="outputs/${experiment_name}_preflight_${RUN_TS}.log"
    mkdir -p "${preflight_dir}"
    info "Running preflight for ${experiment_name} on GPU ${gpu}"
    run_py_gpu "${gpu}" \
      "${PY}" scripts/preflight_sparse_upcycling.py \
      --pretrained-path "${PRETRAINED_PATH}" \
      --config-path "${cfg_path}" \
      --data-source "${DATA_SOURCE}" \
      --tokenizer-path "${TOKENIZER_PATH}" \
      --output-path "${preflight_dir}/init_verification.json" \
      --run-output-dir "outputs/${experiment_name}" \
      --device cuda > "${preflight_log}" 2>&1

    [[ -f "${preflight_dir}/init_verification.json" ]] || die "Missing preflight init_verification.json for ${experiment_name}"
    [[ -f "${preflight_dir}/trainable_param_summary.json" ]] || die "Missing preflight trainable_param_summary.json for ${experiment_name}"
    [[ -f "${preflight_dir}/optimizer_param_groups.json" ]] || die "Missing preflight optimizer_param_groups.json for ${experiment_name}"
    [[ -f "${preflight_dir}/parameter_budget_verification.json" ]] || die "Missing preflight parameter_budget_verification.json for ${experiment_name}"
  done
}

create_launch_scripts() {
  cd "${WORKDIR}"
  mkdir -p "${RUN_SCRIPT_DIR}"
  for idx in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name _ session_name _ _ _ _ _ _ _ <<< "${EXPERIMENTS[$idx]}"
    [[ "${EXP_STATUS[${experiment_name}]:-skip}" == "launch" ]] || continue
    local cfg_path="${EXP_CFG_PATH[${experiment_name}]:-}"
    local gpu="${GPU_LIST[$idx]}"
    local launch_script="${RUN_SCRIPT_DIR}/${experiment_name}.sh"
    cat > "${launch_script}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${WORKDIR}"
mkdir -p "outputs/${experiment_name}"
export CUDA_VISIBLE_DEVICES="${gpu}"
${PY} scripts/run_sparse_upcycling.py \\
  --pretrained-path ${PRETRAINED_PATH} \\
  --config-path ${cfg_path} \\
  --data-source ${DATA_SOURCE} \\
  --val-data-source ${VAL_DATA_SOURCE} \\
  --tokenizer-path ${TOKENIZER_PATH} \\
  --output-dir outputs/${experiment_name} \\
  --device cuda
${PY} scripts/postprocess_sparse_upcycling.py \\
  --config-path ${cfg_path} \\
  --pretrained-path ${PRETRAINED_PATH} \\
  --output-dir outputs/${experiment_name} \\
  --val-data-source ${VAL_DATA_SOURCE} \\
  --tokenizer-path ${TOKENIZER_PATH}
${PY} scripts/plot_training_curves.py \\
  --log outputs/${experiment_name}/train_log.jsonl \\
  --label ${experiment_name} \\
  --out-dir outputs/${experiment_name} \\
  --eval1024 outputs/${experiment_name}/eval_results_1024.json \\
  --training-report outputs/${experiment_name}/training_report.json
EOF
    chmod +x "${launch_script}"
  done
}

launch_tmux_sessions() {
  cd "${WORKDIR}"
  for idx in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name _ session_name _ _ _ _ _ _ _ <<< "${EXPERIMENTS[$idx]}"
    [[ "${EXP_STATUS[${experiment_name}]:-skip}" == "launch" ]] || continue
    local launch_script="${RUN_SCRIPT_DIR}/${experiment_name}.sh"
    local host_log="outputs/${experiment_name}.host.log"
    tmux new -d -s "${session_name}" "bash '${launch_script}' 2>&1 | tee '${host_log}'"
  done
}

print_monitoring_commands() {
  cd "${WORKDIR}"
  echo
  echo "Monitoring:"
  echo "  tmux ls"
  echo "  nvidia-smi"
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name _ session_name _ _ _ _ _ _ _ <<< "${spec}"
    if [[ "${EXP_STATUS[${experiment_name}]:-skip}" == "launch" ]]; then
      echo "  tail -f outputs/${experiment_name}.host.log"
      echo "  tmux attach -t ${session_name}"
    else
      echo "  SKIPPED ${experiment_name}: ${EXP_REASON[${experiment_name}]}"
    fi
  done
}

main() {
  require_cmd tmux
  require_cmd nvidia-smi

  cd "${WORKDIR}"
  log_env
  prepare_configs
  safety_checks

  if [[ "${DO_PREFLIGHT}" == "1" || "${LAUNCH_TRAIN}" == "1" ]]; then
    verify_feature_support
  fi

  if [[ "${DO_PREFLIGHT}" == "1" ]]; then
    run_preflight
    if [[ ${#LAUNCHABLE_EXPERIMENTS[@]} -gt 0 ]]; then
      "${PYTHON_CHECKER:-python3}" - "${RUN_TS}" "${LAUNCHABLE_EXPERIMENTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

run_ts = sys.argv[1]
selected = set(sys.argv[2:])
root = Path("/home/storage/yjl/moe-matmulfree")
specs = [
    {
        "experiment_name": "full_shared_4x64_top1_alpha005_moe_only_5k",
        "freeze_mode": "moe_only",
        "num_sparse_experts": 4,
        "sparse_expert_width": 64,
        "residual_active_width": 64,
    },
    {
        "experiment_name": "full_shared_8x32_top2_alpha005_moe_only_5k",
        "freeze_mode": "moe_only",
        "num_sparse_experts": 8,
        "sparse_expert_width": 32,
        "residual_active_width": 64,
    },
    {
        "experiment_name": "full_shared_4x64_top1_alpha005_local_backbone_ft_5k",
        "freeze_mode": "local_backbone_ft",
        "num_sparse_experts": 4,
        "sparse_expert_width": 64,
        "residual_active_width": 64,
    },
    {
        "experiment_name": "full_shared_8x32_top2_alpha005_local_backbone_ft_5k",
        "freeze_mode": "local_backbone_ft",
        "num_sparse_experts": 8,
        "sparse_expert_width": 32,
        "residual_active_width": 64,
    },
]

for spec in specs:
    if spec["experiment_name"] not in selected:
        continue
    preflight_dir = root / "outputs" / f"{spec['experiment_name']}_preflight_{run_ts}"
    budget_path = preflight_dir / "parameter_budget_verification.json"
    init_path = preflight_dir / "init_verification.json"
    trainable_summary_path = preflight_dir / "trainable_param_summary.json"
    optimizer_path = preflight_dir / "optimizer_param_groups.json"

    for required_path in (budget_path, init_path, trainable_summary_path, optimizer_path):
        if not required_path.exists():
            raise SystemExit(f"Missing preflight artifact: {required_path}")

    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    init_payload = json.loads(init_path.read_text(encoding="utf-8"))
    trainable_summary = json.loads(trainable_summary_path.read_text(encoding="utf-8"))
    optimizer_summary = json.loads(optimizer_path.read_text(encoding="utf-8"))

    if budget.get("shared_width_resolution_mode") != "exact_requested_width":
        raise SystemExit(f"{spec['experiment_name']}: expected shared_width_resolution_mode=exact_requested_width")
    if int(budget.get("resolved_shared_width", -1)) != 2816:
        raise SystemExit(f"{spec['experiment_name']}: resolved_shared_width must stay exactly 2816")
    if bool(budget.get("enforce_baseline_fair", True)):
        raise SystemExit(f"{spec['experiment_name']}: enforce_baseline_fair should be false")
    if bool(budget.get("enforce_active_width_below_dense", True)):
        raise SystemExit(f"{spec['experiment_name']}: enforce_active_width_below_dense should be false")
    if budget.get("shared_init") != "dense_prefix":
        raise SystemExit(f"{spec['experiment_name']}: shared_init mismatch")
    if budget.get("sparse_init") != "random_ternary_matched":
        raise SystemExit(f"{spec['experiment_name']}: sparse_init mismatch")
    if int(budget.get("num_sparse_experts", -1)) != spec["num_sparse_experts"]:
        raise SystemExit(f"{spec['experiment_name']}: num_sparse_experts mismatch")
    if int(budget.get("sparse_expert_width", -1)) != spec["sparse_expert_width"]:
        raise SystemExit(f"{spec['experiment_name']}: sparse_expert_width mismatch")
    expected_active_width = int(budget["resolved_shared_width"]) + spec["residual_active_width"]
    if int(budget.get("active_width", -1)) != expected_active_width:
        raise SystemExit(f"{spec['experiment_name']}: active_width mismatch")
    if float(budget.get("active_width_ratio_vs_dense", 0.0)) <= 1.0:
        raise SystemExit(f"{spec['experiment_name']}: active_width_ratio_vs_dense should be > 1.0 for full_shared")
    if bool(budget.get("width_budget_passed", True)):
        raise SystemExit(f"{spec['experiment_name']}: width_budget_passed should be false for full_shared")

    if init_payload.get("freeze_mode") != spec["freeze_mode"]:
        raise SystemExit(f"{spec['experiment_name']}: freeze_mode mismatch")
    if spec["freeze_mode"] == "moe_only":
        if int(trainable_summary.get("trainable_backbone_parameter_count", 0)) != 0:
            raise SystemExit(f"{spec['experiment_name']}: moe_only should not train backbone params")
    else:
        if int(trainable_summary.get("local_backbone_parameter_count", 0)) <= 0:
            raise SystemExit(f"{spec['experiment_name']}: local_backbone_ft expected local_backbone_parameter_count > 0")

    if int(trainable_summary.get("trainable_embedding_parameter_count", 0)) != 0:
        raise SystemExit(f"{spec['experiment_name']}: embeddings should be frozen")
    if int(trainable_summary.get("trainable_lm_head_parameter_count", 0)) != 0:
        raise SystemExit(f"{spec['experiment_name']}: lm_head should be frozen")
    if not any(group.get("group_name") == "shared_expert" and int(group.get("param_count", 0)) > 0 for group in optimizer_summary):
        raise SystemExit(f"{spec['experiment_name']}: optimizer groups missing non-empty shared_expert group")
    if not any(group.get("group_name") == "moe" and int(group.get("param_count", 0)) > 0 for group in optimizer_summary):
        raise SystemExit(f"{spec['experiment_name']}: optimizer groups missing non-empty moe group")
PY
    else
      info "No launchable experiments require preflight hard checks."
    fi
  else
    info "Skipping preflight because DO_PREFLIGHT=0"
  fi

  create_launch_scripts

  if [[ "${LAUNCH_TRAIN}" == "1" && ${#LAUNCHABLE_EXPERIMENTS[@]} -gt 0 ]]; then
    launch_tmux_sessions
  elif [[ "${LAUNCH_TRAIN}" == "1" ]]; then
    info "Skipping training launch because no experiments are launchable."
  else
    info "Skipping training launch because LAUNCH_TRAIN=0"
  fi

  print_monitoring_commands
}

main "$@"
