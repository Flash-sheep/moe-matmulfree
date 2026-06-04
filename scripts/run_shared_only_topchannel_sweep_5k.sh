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
RUN_SCRIPT_DIR="outputs/run_scripts_shared_only_topchannel_sweep_${RUN_TS}"

EXPERIMENTS=(
  "shared_only_topchannel_2400_local_backbone_ft_5000step|${CFG_DIR}/shared_only_topchannel_2400_local_backbone_ft_5000step.json|sr_shared_only_2400_5k|2400"
  "shared_only_topchannel_2432_local_backbone_ft_5000step|${CFG_DIR}/shared_only_topchannel_2432_local_backbone_ft_5000step.json|sr_shared_only_2432_5k|2432"
  "shared_only_topchannel_2496_local_backbone_ft_5000step|${CFG_DIR}/shared_only_topchannel_2496_local_backbone_ft_5000step.json|sr_shared_only_2496_5k|2496"
  "shared_only_topchannel_2560_local_backbone_ft_5000step|${CFG_DIR}/shared_only_topchannel_2560_local_backbone_ft_5000step.json|sr_shared_only_2560_5k|2560"
)

GPU_MAP_COMPACT="${GPU_MAP// /}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_MAP_COMPACT}"
declare -A EXP_STATUS
declare -A EXP_REASON
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

write_config_if_missing() {
  local cfg_path="$1"
  local experiment_name="$2"
  local nominal_shared_width="$3"

  if [[ -f "${cfg_path}" ]]; then
    info "Using existing config: ${cfg_path}"
    return 0
  fi

  mkdir -p "$(dirname "${cfg_path}")"
  cat > "${cfg_path}" <<EOF
{
  "experiment_name": "${experiment_name}",
  "description": "Shared-only top-channel sweep with nominal shared width ${nominal_shared_width} and local backbone finetuning on layers 12-23.",
  "seed": 42,
  "monitor_interval": 100,
  "moe_lr": 0.0003,
  "shared_expert_lr": 0.0001,
  "backbone_lr": 0.00003,
  "norm_lr": 0.00003,
  "moe": {
    "moe_arch": "shared_residual",
    "layer_indices": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    "num_experts": 4,
    "num_experts_per_tok": 1,
    "enable_sparse_residual": false,
    "nominal_shared_width": ${nominal_shared_width},
    "auto_resolve_shared_width": true,
    "min_shared_width": 2048,
    "shared_width_step": 16,
    "strict_total_param_fair": true,
    "shared_init": "dense_top_channel",
    "sparse_init": "random_ternary_matched",
    "sparse_expert_width": 128,
    "sparse_top_k": 1,
    "residual_scale_init": 0.10,
    "residual_scale_learnable": true,
    "residual_scale_max": 0.5,
    "use_quantized_experts": true,
    "router_aux_loss_coef": 0.0005,
    "router_jitter_noise": 0.0,
    "router_bias": false,
    "normalize_topk_prob": true
  },
  "freeze": {
    "freeze_mode": "local_backbone_ft",
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
  info "Generated config: ${cfg_path}"
}

generate_configs_if_missing() {
  cd "${WORKDIR}"
  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name cfg_path _ nominal_shared_width <<< "${spec}"
    write_config_if_missing "${cfg_path}" "${experiment_name}" "${nominal_shared_width}"
  done
}

verify_feature_support() {
  cd "${WORKDIR}"
  run_py "${PY}" - <<'PY'
from pathlib import Path

root = Path("/home/storage/yjl/moe-matmulfree")
moe_text = (root / "mmfreelm/modules/moe.py").read_text(encoding="utf-8")
sparse_upcycling_text = (root / "mmfreelm/upcycling/sparse_upcycling.py").read_text(encoding="utf-8")
run_text = (root / "scripts/run_sparse_upcycling.py").read_text(encoding="utf-8")
modeling_text = (root / "mmfreelm/models/hgrn_bit/modeling_hgrn_bit.py").read_text(encoding="utf-8")

missing = []
if 'selection_mode == "dense_top_channel"' not in moe_text:
    missing.append("shared_init=dense_top_channel is not implemented")
if (
    "moe_arch == \"shared_residual\"" not in sparse_upcycling_text
    and "moe_arch\", \"standard\") == \"shared_residual\"" not in run_text
    and "moe_arch\", \"standard\") == \"shared_residual\"" not in modeling_text
):
    missing.append("shared_residual architecture support not found")
if "strict_total_param_fair_passed" not in run_text:
    missing.append("shared_residual parameter fairness checks not found in run script")

if missing:
    raise SystemExit(
        "Required shared-only sweep capability checks failed:\n- " + "\n- ".join(missing) +
        "\nRefusing to proceed to preflight or launch."
    )
print("Shared-only top-channel capability checks passed.")
PY
}

safety_checks() {
  cd "${WORKDIR}"
  [[ ${#GPU_LIST[@]} -eq 4 ]] || die "GPU_MAP must contain exactly 4 comma-separated GPU ids. Got: ${GPU_MAP}"
  LAUNCHABLE_EXPERIMENTS=()

  for spec in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name cfg_path session_name _ <<< "${spec}"
    local output_dir="outputs/${experiment_name}"
    local host_log="outputs/${experiment_name}.host.log"
    local train_log="${output_dir}/train_log.jsonl"
    local eval_1024="${output_dir}/eval_results_1024.json"
    local training_report="${output_dir}/training_report.json"

    [[ -f "${cfg_path}" ]] || die "Missing config: ${cfg_path}"

    "${PYTHON_CHECKER:-python3}" - "${cfg_path}" "${experiment_name}" <<'PY'
import json
import sys
cfg_path, expected_name = sys.argv[1], sys.argv[2]
cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
actual = cfg.get("experiment_name")
if actual != expected_name:
    raise SystemExit(f"config experiment_name mismatch: {cfg_path} expected {expected_name} got {actual}")
PY

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

  if [[ "${LAUNCH_TRAIN}" == "1" && ${#LAUNCHABLE_EXPERIMENTS[@]} -eq 0 ]]; then
    info "No launchable experiments after safety checks. All experiments were skipped."
  fi
}

run_preflight() {
  cd "${WORKDIR}"
  mkdir -p "${RUN_SCRIPT_DIR}"

  for idx in "${!EXPERIMENTS[@]}"; do
    IFS='|' read -r experiment_name cfg_path _ _ <<< "${EXPERIMENTS[$idx]}"
    [[ "${EXP_STATUS[${experiment_name}]:-skip}" == "launch" ]] || continue
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
    IFS='|' read -r experiment_name cfg_path session_name _ <<< "${EXPERIMENTS[$idx]}"
    [[ "${EXP_STATUS[${experiment_name}]:-skip}" == "launch" ]] || continue
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
    IFS='|' read -r experiment_name _ session_name _ <<< "${EXPERIMENTS[$idx]}"
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
    IFS='|' read -r experiment_name _ session_name _ <<< "${spec}"
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
  generate_configs_if_missing
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
    ("shared_only_topchannel_2400_local_backbone_ft_5000step", 2400),
    ("shared_only_topchannel_2432_local_backbone_ft_5000step", 2432),
    ("shared_only_topchannel_2496_local_backbone_ft_5000step", 2496),
    ("shared_only_topchannel_2560_local_backbone_ft_5000step", 2560),
]

for experiment_name, nominal_shared_width in specs:
    if experiment_name not in selected:
        continue
    preflight_dir = root / "outputs" / f"{experiment_name}_preflight_{run_ts}"
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

    if not budget.get("strict_total_param_fair_passed", False):
        raise SystemExit(f"{experiment_name}: strict_total_param_fair_passed is false")
    if int(budget["new_total_params"]) > int(budget["baseline_total_params"]):
        raise SystemExit(f"{experiment_name}: new_total_params exceeds baseline_total_params")
    if float(budget["active_width_ratio_vs_dense"]) >= 1.0:
        raise SystemExit(f"{experiment_name}: active_width_ratio_vs_dense must be < 1.0")
    if budget.get("resolved_shared_width") is None:
        raise SystemExit(f"{experiment_name}: missing resolved_shared_width")
    if not budget.get("width_budget_passed", False):
        raise SystemExit(f"{experiment_name}: width_budget_passed is false")
    if budget.get("shared_init") != "dense_top_channel":
        raise SystemExit(f"{experiment_name}: shared_init mismatch")
    if budget.get("router_params", 0) != 0:
        raise SystemExit(f"{experiment_name}: shared-only config must have router_params == 0")
    if int(budget.get("num_sparse_experts", -1)) != 0:
        raise SystemExit(f"{experiment_name}: shared-only config must have num_sparse_experts == 0")
    if int(budget.get("active_width", -1)) != int(budget.get("resolved_shared_width", -2)):
        raise SystemExit(f"{experiment_name}: active_width must equal resolved_shared_width for shared-only")
    if int(budget.get("resolved_shared_width", 0)) > nominal_shared_width:
        raise SystemExit(f"{experiment_name}: resolved_shared_width exceeds nominal_shared_width")

    if init_payload.get("freeze_mode") != "local_backbone_ft":
        raise SystemExit(f"{experiment_name}: expected freeze_mode local_backbone_ft")
    if int(trainable_summary.get("local_backbone_parameter_count", 0)) <= 0:
        raise SystemExit(f"{experiment_name}: expected local_backbone_parameter_count > 0")
    if int(trainable_summary.get("trainable_embedding_parameter_count", 0)) != 0:
        raise SystemExit(f"{experiment_name}: embeddings should be frozen")
    if int(trainable_summary.get("trainable_lm_head_parameter_count", 0)) != 0:
        raise SystemExit(f"{experiment_name}: lm_head should be frozen")
    if not any(group.get("group_name") == "shared_expert" and int(group.get("param_count", 0)) > 0 for group in optimizer_summary):
        raise SystemExit(f"{experiment_name}: optimizer groups missing non-empty shared_expert group")
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
