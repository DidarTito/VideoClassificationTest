#!/usr/bin/env bash
set -u
FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1
mkdir -p runs/ssv2
SUMMARY="runs/ssv2/all17_summary.csv"
echo "model,status,exit_code" > "$SUMMARY"
overall=0
MODELS=(uniformer-s dualformer-t video-focalnet-t videoswin-t mvit-v1-b-16x4 video-focalnet-s mvit-v1-b-32x3 dualformer-s videoswin-s zeroi2v-b16-8f video-focalnet-b uniformer-b videomae-b dualformer-b-in21k omnivore-b-in21k videoswin-b timesformer-b)
for model in "${MODELS[@]}"; do
  if python -c "from training.registry import model_spec; raise SystemExit(0 if model_spec('$model')['checkpoint_status'] in {'BLOCKED_EXACT_CHECKPOINT','MISSING_EXACT_CHECKPOINT'} else 1)"; then
    echo "$model,blocked_checkpoint,0" >> "$SUMMARY"; continue
  fi
  if [[ $FORCE -eq 0 ]] && find "runs/ssv2/$model" -name complete.json -print -quit 2>/dev/null | grep -q .; then
    echo "$model,already_completed,0" >> "$SUMMARY"; continue
  fi
  extra=(); [[ $FORCE -eq 1 ]] && extra=(--force)
  latest=$(find "runs/ssv2/$model" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)
  if [[ $FORCE -eq 0 && -n "$latest" && -f "$latest/last.pth" && ! -f "$latest/complete.json" ]]; then
    extra+=(--run-id "$(basename "$latest")")
  fi
  python scripts/finetune_ssv2.py --model "$model" --device cuda "${extra[@]}"
  code=$?
  if [[ $code -eq 0 ]]; then status=completed; else status=failed; overall=1; fi
  echo "$model,$status,$code" >> "$SUMMARY"
done
echo "Summary: $SUMMARY"
exit "$overall"
