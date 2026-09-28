# SSV2 remaining 11 final results

Valid SSV2 accuracy requires either an exact 174-class released checkpoint or K400 initialization, a replaced 174-class head, and complete-split fine-tuning. K400-only scores on SSV2 labels are not used here.

## Audit

- Genuine exact SSV2 checkpoints for these 11 variants: **0/11**
- Require K400 -> SSV2 fine-tuning: **11/11**
- Official train videos present: **86680/168913** (missing 82233)
- Official val videos present: **12767/24777** (missing 12010)
- Train/val ID overlap: 0
- Final eval manifest SHA256: `c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02`
- This machine GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB). Fine-tuning was requested on RTX 5090 when data is complete.

Missing official `.webm` IDs:
- `results/ssv2_remaining11_final/missing_train_ids.txt`
- `results/ssv2_remaining11_final/missing_val_ids.txt`

Place the missing files in `datasets/ssv2/videos/<id>.webm` from the official Something-Something V2 release, then rerun:

```powershell
.\venv\Scripts\python.exe -u scripts\run_ssv2_remaining11_final.py --run
```

## Per-model status

| Model | SSV2Method | Status | MeasuredTop1 | Notes |
| --- | --- | --- | --- | --- |
| dualformer-t | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| video-focalnet-t | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| videoswin-t | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| mvit-v1-b-16x4 | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| video-focalnet-s | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| mvit-v1-b-32x3 | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| dualformer-s | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| videoswin-s | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| zeroi2v-b16-8f | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| dualformer-b-in21k | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |
| omnivore-b-in21k | K400_PRETRAINED_THEN_SSV2_FINETUNED | BLOCKED_INCOMPLETE_SSV2_TRAIN | N/A | No exact released 174-class SSV2 checkpoint for this frozen variant. Blocked: missing 82233 train and 12010 val official .webm files under datasets/ssv2/videos/. |

## Why training did not start

The project fine-tune runner refuses `--allow-partial-data` except as a non-final smoke. Training on 86,680/168,913 train videos would leak no eval clips, but it is not a valid complete-split SSV2 result.

