#!/usr/bin/env bash
set -Eeuo pipefail

: "${K400_VAL_ROOT:?Set K400_VAL_ROOT to the Kinetics-400 validation root}"
PYTHON="${PYTHON:-python}"

# This list is intentionally explicit and comes from RTX3050_EXACT_STATUS.md.
# Blocked rows are never smuggled into this rerun.
"$PYTHON" run_benchmark.py \
  --videos-dir "$K400_VAL_ROOT" \
  --manifest manifests/k400_1000_seed0.csv \
  --num-clips 1000 --batch-size 1 --seed 0 \
  --models uniformer-s mvit-v1-b-16x4 videomae-b
