#!/usr/bin/env bash

cd /home/esai01/ESAI_Didar/VideoClassificationTest || exit 1

PY=/home/esai01/miniconda3/bin/python
ZERO_ROOT=/media/esai01/10A2D9F6A2D9DFF0/ssv2_full_ft
ZERO=$ZERO_ROOT/zeroi2v-b16-8f
OUT=/media/esai01/10A2D9F6A2D9DFF0/ssv2_meeting9d

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "################################################################"
echo "# MEETING DEADLINE FULL FT"
echo "################################################################"
date

echo
echo "===== ZEROI2V EXISTING CHECKPOINT ====="

EPOCH=$(
"$PY" - <<PY
import torch
x=torch.load(
 "$ZERO/checkpoints/latest.pt",
 map_location="cpu",
 weights_only=False
)
print(int(x.get("epoch",-1)))
PY
)

echo "existing ZeroI2V epoch = $EPOCH"

if [ "$EPOCH" -lt 30 ]; then
    echo
    echo "===== CONTINUE ZEROI2V: e${EPOCH} -> e30 ====="
    echo "NOTE: continuation of existing pilot, NOT clean-from-start e30."

    "$PY" -u scripts/finetune_ssv2_full.py \
        --models zeroi2v-b16-8f \
        --epochs 30 \
        --resume \
        --batch-size 16 \
        --effective-batch 16 \
        --num-workers 8 \
        --output-dir "$ZERO_ROOT"

    RC=$?

    if [ "$RC" -ne 0 ]; then
        echo "ZeroI2V stopped rc=$RC"
        exit "$RC"
    fi
else
    echo "ZeroI2V already at >=30 -> skip"
fi

echo
echo "################################################################"
echo "# ZEROI2V DONE -> DEADLINE PRIORITY NEW MODELS"
echo "# FocalNet-T -> MViT16x4 -> FocalNet-S -> VideoSwin-T -> ..."
echo "################################################################"

"$PY" -u scripts/finetune_meeting9d.py --run

RC=$?

echo
echo "MASTER FINISHED rc=$RC"
date
exit "$RC"
