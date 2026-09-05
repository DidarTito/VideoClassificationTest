#!/usr/bin/env bash
# =============================================================================
# run_project_steps.sh -- single master run file for the frozen17 project.
#
# Usage:
#   bash run_project_steps.sh                # runs every step whose switch is true
#   bash run_project_steps.sh 10             # run step 10 directly
#   bash run_project_steps.sh 7 8 9          # run several steps in order
#   bash run_project_steps.sh tomorrow       # first-lab priority sequence
#   bash run_project_steps.sh rtx3050        # audit + fill missing + final17 table
#   bash run_project_steps.sh rtx3050-full   # full clean rerun of all 17
#   bash run_project_steps.sh --help
#   bash run_project_steps.sh tomorrow --dry-run   # print commands, run nothing
#
# Environment:
#   PYTHON         python executable (default: python)
#   K400_VAL_ROOT  Kinetics-400 validation video root (required for benchmarks)
#   SSV2_ROOT      SSV2 video root override (default: datasets/ssv2/videos)
# =============================================================================
set -Eeuo pipefail

# ----------------------------- SWITCHES --------------------------------------
RUN_01_ENV_CHECK=false
RUN_02_CHECKPOINT_AUDIT=false
RUN_03_FETCH_MISSING_CHECKPOINTS=false
RUN_04_VALIDATE_K400=false
RUN_05_VALIDATE_SSV2=false
RUN_06_FROZEN17_PREFLIGHT=false
RUN_07_RTX3050_FILL_MISSING=false
RUN_08_RTX3050_BUILD_FINAL17=false
RUN_09_RTX5070_PREFLIGHT=false
RUN_10_RTX5070_K400_ALL17=false
RUN_11_FINETUNE_SMOKE_UNIFORMER_S=false
RUN_12_FINETUNE_SMOKE_VIDEOMAE_B=false
RUN_13_FINETUNE_PREFLIGHT_ALL17=false
RUN_14_FINETUNE_ALL17=false
RUN_15_EVAL_FINETUNED_ALL=false
RUN_16_RTX5070_SSV2_DEVICE_BENCHMARK=false
RUN_17_BUILD_FINAL_TABLES=false
RUN_18_GENERATE_REPORT=false
# ------------------------------------------------------------------------------

PYTHON="${PYTHON:-python}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT/logs/project_steps"
DRY_RUN=false
mkdir -p "$LOG_DIR"
cd "$ROOT"

# Full-training order (Phase 15). UniFormer-S and VideoMAE-B lead because they
# are the two smoke-tested priority models; families are grouped so debugging
# one adapter carries to its siblings.
TRAIN_ORDER=(
  uniformer-s videomae-b
  videoswin-t videoswin-s videoswin-b
  video-focalnet-t video-focalnet-s video-focalnet-b
  uniformer-b
  mvit-v1-b-16x4 mvit-v1-b-32x3
  timesformer-b zeroi2v-b16-8f
  dualformer-t dualformer-s
  omnivore-b-in21k dualformer-b-in21k
)

banner() {
  echo
  echo "======================================================================"
  echo "  $1"
  echo "======================================================================"
}

run_logged() {
  # run_logged <log-name> <command...>
  local name="$1"; shift
  local log="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${name}.log"
  echo "+ $*"
  echo "  (log: $log)"
  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] skipped"
    return 0
  fi
  if "$@" 2>&1 | tee "$log"; then
    echo "STEP OK: $name"
  else
    local code=$?
    echo "STEP FAILED: $name (exit $code). Full log: $log" >&2
    return "$code"
  fi
}

require_k400_root() {
  if [ "$DRY_RUN" = true ]; then return 0; fi
  if [ -z "${K400_VAL_ROOT:-}" ] || [ ! -d "${K400_VAL_ROOT:-}" ]; then
    echo "K400_VAL_ROOT is not set or not a directory; export it first." >&2
    exit 1
  fi
}

# ------------------------------- STEPS ---------------------------------------
step_01_environment() {
  banner "STEP 01: environment check"
  run_logged 01_env "$PYTHON" scripts/preflight_rtx5070.py --env-only
}

step_02_checkpoint_audit() {
  banner "STEP 02: frozen17 checkpoint audit"
  run_logged 02_audit "$PYTHON" scripts/audit_frozen17.py
}

step_03_fetch_missing_checkpoints() {
  banner "STEP 03: fetch missing K400 checkpoints"
  run_logged 03_fetch "$PYTHON" scripts/fetch_missing_k400_checkpoints.py
}

step_04_validate_k400() {
  banner "STEP 04: validate K400 subset"
  require_k400_root
  run_logged 04_k400 "$PYTHON" - <<'PYEOF'
import csv, os, sys
from pathlib import Path
manifest = Path("manifests/k400_1000_seed0.csv")
if not manifest.is_file():
    sys.exit(f"missing {manifest}")
rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
if len(rows) != 1000:
    sys.exit(f"{manifest} must list exactly 1000 clips, found {len(rows)}")
root = Path(os.environ["K400_VAL_ROOT"])
missing = [r for r in rows if not (root / r["RelativePath"]).is_file()]
print(f"manifest clips: {len(rows)}; missing on disk: {len(missing)}")
if missing:
    for r in missing[:10]:
        print("  missing:", r["RelativePath"])
    sys.exit(1)
print("K400 fixed subset OK")
PYEOF
}

step_05_validate_ssv2() {
  banner "STEP 05: validate SSV2 dataset"
  run_logged 05_ssv2 "$PYTHON" scripts/validate_ssv2.py ${SSV2_ROOT:+--root "$SSV2_ROOT"}
}

step_06_frozen17_preflight() {
  banner "STEP 06: frozen17 K400 load preflight (17 models)"
  run_logged 06_preflight "$PYTHON" scripts/preflight_rtx5070.py
}

step_prepare_model_backends() {
  banner "MODEL BACKENDS: prepare pinned ZeroI2V/MMAction2 environment"
  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] bash scripts/setup_rtx5070_linux.sh mmaction2-modern"
  elif [ "$(uname -s)" != Linux ]; then
    echo "SKIP backend installation: Linux RTX5070 host required"
  elif [ ! -x .venv-mmaction2-modern/bin/python ]; then
    if ! run_logged model_backends bash scripts/setup_rtx5070_linux.sh mmaction2-modern; then
      echo "ZeroI2V environment remains BLOCKED_RUNTIME; continuing with READY_EXACT models." >&2
      return 0
    fi
  else
    echo "Pinned mmaction2-modern environment already exists."
  fi
  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] .venv-mmaction2-modern/bin/python scripts/preflight_zeroi2v.py"
  elif [ "$(uname -s)" = Linux ]; then
    if ! run_logged zeroi2v_preflight .venv-mmaction2-modern/bin/python scripts/preflight_zeroi2v.py; then
      echo "ZeroI2V exact forward remains BLOCKED_RUNTIME; continuing with READY_EXACT models." >&2
    fi
  fi
}

step_07_rtx3050_fill_missing() {
  banner "STEP 07: RTX3050 -- audit old results and run only missing/invalid"
  require_k400_root
  run_logged 07_audit "$PYTHON" scripts/audit_rtx3050.py
  local models
  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] would read reports/rtx3050_frozen17_status.csv and benchmark every non-REUSE model"
    return 0
  fi
  models=$("$PYTHON" - <<'PYEOF'
import csv
rows = list(csv.DictReader(open("reports/rtx3050_frozen17_status.csv", encoding="utf-8")))
print(" ".join(r["canonical_key"] for r in rows if r["action"] != "REUSE"))
PYEOF
)
  if [ -z "$models" ]; then
    echo "Nothing to run: all 17 RTX3050 results are verified reusable."
    return 0
  fi
  echo "Models needing a run: $models"
  # shellcheck disable=SC2086
  run_logged 07_run "$PYTHON" run_benchmark.py \
    --videos-dir "${K400_VAL_ROOT:-}" \
    --manifest manifests/k400_1000_seed0.csv \
    --num-clips 1000 --batch-size 1 --seed 0 \
    --models $models
}

step_08_rtx3050_build_final17() {
  banner "STEP 08: RTX3050 -- build exact 17-row final table"
  run_logged 08_final17 "$PYTHON" scripts/build_rtx3050_final17.py
}

step_09_rtx5070_preflight() {
  banner "STEP 09: RTX5070 preflight (env + 17 model loads)"
  run_logged 09_preflight "$PYTHON" scripts/preflight_rtx5070.py
}

step_10_rtx5070_k400_all17() {
  banner "STEP 10: RTX5070 K400 benchmark -- every READY_EXACT model"
  require_k400_root
  local ready_models
  if [ "$DRY_RUN" = true ]; then
    echo "  [dry-run] $PYTHON run_benchmark.py --videos-dir \"${K400_VAL_ROOT:-}\" --manifest manifests/k400_1000_seed0.csv --num-clips 1000 --batch-size 1 --seed 0 --models READY_EXACT --output-dir results/k400_rtx5070_frozen17"
    return 0
  fi
  ready_models=$("$PYTHON" - <<'PYEOF'
from training.registry import FROZEN17_KEYS, model_spec
print(" ".join(k for k in FROZEN17_KEYS if model_spec(k)["checkpoint_status"] == "READY_EXACT"))
PYEOF
)
  echo "READY_EXACT models: $ready_models"
  # shellcheck disable=SC2086
  run_logged 10_bench "$PYTHON" run_benchmark.py \
    --videos-dir "${K400_VAL_ROOT:-}" \
    --manifest manifests/k400_1000_seed0.csv \
    --num-clips 1000 --batch-size 1 --seed 0 \
    --models $ready_models \
    --output-dir results/k400_rtx5070_frozen17
}

step_11_finetune_smoke_uniformer_s() {
  banner "STEP 11: SSV2 fine-tune smoke test -- UniFormer-S"
  run_logged 11_smoke_uniformer "$PYTHON" scripts/finetune_ssv2.py --model uniformer-s --smoke --allow-partial-data
}

step_12_finetune_smoke_videomae_b() {
  banner "STEP 12: SSV2 fine-tune smoke test -- VideoMAE-B"
  run_logged 12_smoke_videomae "$PYTHON" scripts/finetune_ssv2.py --model videomae-b --smoke --allow-partial-data
}

step_13_finetune_preflight_all17() {
  banner "STEP 13: fine-tune preflight -- all 17"
  run_logged 13_ft_preflight "$PYTHON" scripts/preflight_finetune.py
}

step_14_finetune_all17() {
  banner "STEP 14: full SSV2 fine-tuning -- all 17 (long)"
  local key
  for key in "${TRAIN_ORDER[@]}"; do
    banner "STEP 14: fine-tune $key"
    run_logged "14_train_${key}" "$PYTHON" scripts/finetune_ssv2.py --model "$key"
  done
}

step_15_eval_finetuned_all() {
  banner "STEP 15: evaluate all fine-tuned best checkpoints on SSV2 val"
  local key best found=false
  for key in "${TRAIN_ORDER[@]}"; do
    best=$(ls -1t runs/ssv2/"$key"/*/best.pth 2>/dev/null | head -n 1 || true)
    if [ -z "$best" ]; then
      echo "SKIP $key: no runs/ssv2/$key/*/best.pth yet"
      continue
    fi
    found=true
    run_logged "15_eval_${key}" "$PYTHON" scripts/eval_ssv2.py --model "$key" --checkpoint "$best"
  done
  if [ "$found" = false ] && [ "$DRY_RUN" = false ]; then
    echo "No fine-tuned checkpoints found under runs/ssv2/." >&2
    return 1
  fi
}

step_16_rtx5070_ssv2_device_benchmark() {
  banner "STEP 16: RTX5070 SSV2 deployment benchmark (batch size 1)"
  local key best found=false
  for key in "${TRAIN_ORDER[@]}"; do
    best=$(ls -1t runs/ssv2/"$key"/*/best.pth 2>/dev/null | head -n 1 || true)
    if [ -z "$best" ]; then
      echo "SKIP $key: no fine-tuned checkpoint"
      continue
    fi
    found=true
    run_logged "16_bench_${key}" "$PYTHON" scripts/benchmark_finetuned_ssv2.py \
      --model "$key" --checkpoint "$best" \
      --output results/ssv2_finetuned_rtx5070
  done
  if [ "$found" = false ] && [ "$DRY_RUN" = false ]; then
    echo "No fine-tuned checkpoints found under runs/ssv2/." >&2
    return 1
  fi
}

step_17_build_final_tables() {
  banner "STEP 17: build final tables"
  run_logged 17_tables "$PYTHON" scripts/build_rtx3050_final17.py
}

step_18_generate_report() {
  banner "STEP 18: generate progress report"
  run_logged 18_report "$PYTHON" scripts/generate_progress_report.py
}

# ------------------------------ DISPATCH -------------------------------------
usage() {
  sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  echo
  echo "Steps:"
  echo "  01 environment check              10 RTX5070 K400 benchmark (READY_EXACT)"
  echo "  02 checkpoint audit               11 fine-tune smoke: UniFormer-S"
  echo "  03 fetch missing checkpoints      12 fine-tune smoke: VideoMAE-B"
  echo "  04 validate K400 subset           13 fine-tune preflight (17)"
  echo "  05 validate SSV2 dataset          14 full fine-tuning (17, long)"
  echo "  06 frozen17 load preflight        15 eval fine-tuned checkpoints"
  echo "  07 RTX3050 fill missing           16 RTX5070 SSV2 device benchmark"
  echo "  08 RTX3050 build final17          17 build final tables"
  echo "  09 RTX5070 preflight              18 generate progress report"
}

run_step() {
  case "$1" in
    1|01) step_01_environment ;;
    2|02) step_02_checkpoint_audit ;;
    3|03) step_03_fetch_missing_checkpoints ;;
    4|04) step_04_validate_k400 ;;
    5|05) step_05_validate_ssv2 ;;
    6|06) step_06_frozen17_preflight ;;
    7|07) step_07_rtx3050_fill_missing ;;
    8|08) step_08_rtx3050_build_final17 ;;
    9|09) step_09_rtx5070_preflight ;;
    10) step_10_rtx5070_k400_all17 ;;
    11) step_11_finetune_smoke_uniformer_s ;;
    12) step_12_finetune_smoke_videomae_b ;;
    13) step_13_finetune_preflight_all17 ;;
    14) step_14_finetune_all17 ;;
    15) step_15_eval_finetuned_all ;;
    16) step_16_rtx5070_ssv2_device_benchmark ;;
    17) step_17_build_final_tables ;;
    18) step_18_generate_report ;;
    *) echo "Unknown step: $1" >&2; usage; exit 2 ;;
  esac
}

main() {
  local args=()
  local arg
  for arg in "$@"; do
    case "$arg" in
      --help|-h) usage; exit 0 ;;
      --dry-run) DRY_RUN=true ;;
      *) args+=("$arg") ;;
    esac
  done

  if [ "${#args[@]}" -gt 0 ]; then
    case "${args[0]}" in
      tomorrow)
        banner "TOMORROW MODE: first-lab priorities"
        step_01_environment
        step_02_checkpoint_audit
        step_04_validate_k400
        step_05_validate_ssv2
        step_prepare_model_backends
        step_06_frozen17_preflight
        step_10_rtx5070_k400_all17
        step_11_finetune_smoke_uniformer_s
        step_12_finetune_smoke_videomae_b
        step_18_generate_report
        run_logged final_status "$PYTHON" scripts/frozen17_status.py
        ;;
      rtx3050)
        banner "RTX3050 MODE: audit, fill missing, build final17"
        step_02_checkpoint_audit
        step_07_rtx3050_fill_missing
        step_08_rtx3050_build_final17
        step_18_generate_report
        ;;
      rtx3050-full)
        banner "RTX3050 FULL MODE: clean rerun of all 17"
        require_k400_root
        run_logged rtx3050_full "$PYTHON" run_benchmark.py \
          --videos-dir "${K400_VAL_ROOT:-}" \
          --manifest manifests/k400_1000_seed0.csv \
          --num-clips 1000 --batch-size 1 --seed 0 \
          --models frozen17
        step_08_rtx3050_build_final17
        ;;
      *)
        for arg in "${args[@]}"; do
          run_step "$arg"
        done
        ;;
    esac
  else
    [ "$RUN_01_ENV_CHECK" = true ] && step_01_environment
    [ "$RUN_02_CHECKPOINT_AUDIT" = true ] && step_02_checkpoint_audit
    [ "$RUN_03_FETCH_MISSING_CHECKPOINTS" = true ] && step_03_fetch_missing_checkpoints
    [ "$RUN_04_VALIDATE_K400" = true ] && step_04_validate_k400
    [ "$RUN_05_VALIDATE_SSV2" = true ] && step_05_validate_ssv2
    [ "$RUN_06_FROZEN17_PREFLIGHT" = true ] && step_06_frozen17_preflight
    [ "$RUN_07_RTX3050_FILL_MISSING" = true ] && step_07_rtx3050_fill_missing
    [ "$RUN_08_RTX3050_BUILD_FINAL17" = true ] && step_08_rtx3050_build_final17
    [ "$RUN_09_RTX5070_PREFLIGHT" = true ] && step_09_rtx5070_preflight
    [ "$RUN_10_RTX5070_K400_ALL17" = true ] && step_10_rtx5070_k400_all17
    [ "$RUN_11_FINETUNE_SMOKE_UNIFORMER_S" = true ] && step_11_finetune_smoke_uniformer_s
    [ "$RUN_12_FINETUNE_SMOKE_VIDEOMAE_B" = true ] && step_12_finetune_smoke_videomae_b
    [ "$RUN_13_FINETUNE_PREFLIGHT_ALL17" = true ] && step_13_finetune_preflight_all17
    [ "$RUN_14_FINETUNE_ALL17" = true ] && step_14_finetune_all17
    [ "$RUN_15_EVAL_FINETUNED_ALL" = true ] && step_15_eval_finetuned_all
    [ "$RUN_16_RTX5070_SSV2_DEVICE_BENCHMARK" = true ] && step_16_rtx5070_ssv2_device_benchmark
    [ "$RUN_17_BUILD_FINAL_TABLES" = true ] && step_17_build_final_tables
    [ "$RUN_18_GENERATE_REPORT" = true ] && step_18_generate_report
    echo
    echo "Done. Enable steps by editing the RUN_XX switches at the top, or run"
    echo "  bash run_project_steps.sh --help"
  fi
}

main "$@"
