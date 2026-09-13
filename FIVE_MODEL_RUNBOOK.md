# Reproduce the five-model 1,000-clip benchmarks

Run from the repository root with the project environment installed (see
`README.md` and `LINUX_RTX5070_SETUP.md`). The requested models, in run order,
are `uniformer-s`, `video-focalnet-t`, `uniformer-b`, `videomae-b`, and
`dualformer-t`. Both datasets use one full-video view decoded to 32 frames at
224×224, batch size 1, FP32, two untimed warm-ups, and the same per-clip CUDA
timing and NVML power loop. The progress bar updates after each measured chunk
has stopped power sampling.

## K400: fixed 1,000 clips

Place K400 class folders under `datasets/kinetics400/` and the exact source
checkpoints under the paths in `configs/frozen17.yaml`. Do not substitute a
checkpoint with a different frame recipe. The manifest
`manifests/k400_1000_seed0.csv` is tracked and has SHA256
`b8ec0d92f50b4ae5dff46754f62c2ef1b2b8467332f88bb0cc29c8e6172ff749`.

```bash
python run_benchmark.py --device cuda --dataset k400 --num-clips 1000 \
  --manifest manifests/k400_1000_seed0.csv --seed 0 --frames 32 --size 224 \
  --models uniformer-s video-focalnet-t uniformer-b videomae-b dualformer-t \
  --power nvml --output-dir results/five_model_k400 --preflight

python run_benchmark.py --device cuda --dataset k400 --num-clips 1000 \
  --manifest manifests/k400_1000_seed0.csv --seed 0 --frames 32 --size 224 \
  --models uniformer-s video-focalnet-t uniformer-b videomae-b dualformer-t \
  --power nvml --output-dir results/five_model_k400 --resume
```

The preflight must report all five ready before treating the matrix as
complete. It now reports a nonzero exit when any explicitly requested frozen
pair is blocked. The CSV is under
`results/five_model_k400/<detected-device>/k400/full_1000_seed0/`.

## SSV2: fixed 1,000 validation clips and final checkpoints

Keep the selected validation videos under `datasets/ssv2/videos/`. The tracked
`manifests/ssv2_1000_seed0.csv` has SHA256
`c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02`.
The runner verifies every selected ID against the official validation
annotation and resolves all 1,000 integer class labels. It accepts only a
completed, non-smoke, non-partial `best.pth` from
`runs/ssv2/<model>/<run-id>/`, together with its training sidecar files.
The source K400 checkpoints are also needed to construct the exact graphs.

Create a JSON file such as `final_ssv2_checkpoints.json` with exactly these
five entries (replace each run ID with the selected final run):

```json
{
  "uniformer-s": "runs/ssv2/uniformer-s/<run-id>/best.pth",
  "video-focalnet-t": "runs/ssv2/video-focalnet-t/<run-id>/best.pth",
  "uniformer-b": "runs/ssv2/uniformer-b/<run-id>/best.pth",
  "videomae-b": "runs/ssv2/videomae-b/<run-id>/best.pth",
  "dualformer-t": "runs/ssv2/dualformer-t/<run-id>/best.pth"
}
```

```bash
python scripts/benchmark_finetuned_ssv2.py \
  --checkpoint-map final_ssv2_checkpoints.json --preflight

python scripts/benchmark_finetuned_ssv2.py \
  --checkpoint-map final_ssv2_checkpoints.json --device cuda --power nvml --resume
```

The SSV2 CSV is `results/finetuned/ssv2_five_model_1000_seed0/device_metrics.csv`.
Each row records the checkpoint path and its SHA256; `--resume` rejects a
changed checkpoint, sample, device, or measurement setup. It can retry failed
rows without repeating successful ones. A final run with five completed rows
is required. For one final model, the same runner also accepts
`--model <key> --checkpoint runs/ssv2/<key>/<run-id>/best.pth`.

## Current asset limits

The local checkpoint audit marks all five K400 sources `READY_EXACT`.
The frozen UniFormer-B 32×4 source matches the exact first-party Sense-X
Hugging Face LFS SHA256 recorded in `configs/frozen17.yaml`. The older 16×4
UniFormer-B artifact is a distinct checkpoint and recipe.
There are currently no completed final SSV2 fine-tuning runs under `runs/ssv2/`.
The checked-in `reports/SSV2_DATASET_AUDIT.md` also marks the local full
training pool incomplete; a teammate needs the complete official training and
validation videos before producing legitimate final checkpoints.
The scripts fail before timing when those requirements are unmet; they do not
fall back to the historical released SSV2 classifiers.
