# Frozen-17 K400 to SSV2 fine-tuning

The active experiment is exactly the 17 entries in `configs/frozen17.yaml`.
Legacy released-SSV2 and RTX 3050 results remain historical evidence only.
Every training run starts from a K400 classifier, verifies the 400-way head,
replaces only that head with a 174-way layer, trains on the official SSV2 train
split, and selects the highest validation Top-1 (earlier epoch wins a tie).

## Required order

```bash
python scripts/audit_checkpoints.py --models frozen17
python scripts/fetch_missing_k400_checkpoints.py --models mvit-v1-b-16x4 dualformer-s
python scripts/audit_checkpoints.py --models frozen17
python scripts/validate_ssv2.py
python scripts/preflight_finetune.py --models uniformer-s videomae-b --device cuda --level 4
python scripts/preflight_finetune.py --models frozen17 --device cuda --level 4
```

Do not proceed when dataset validation fails. The current local pool is
incomplete; validation reports the missing official IDs rather than silently
training on a subset.

## Train and evaluate

```bash
python scripts/finetune_ssv2.py --model uniformer-s --device cuda --smoke
python scripts/finetune_ssv2.py --model uniformer-s --device cuda
bash scripts/run_finetune_all17.sh
python scripts/eval_ssv2.py --model uniformer-s --checkpoint runs/ssv2/uniformer-s/<run-id>/best.pth --split val --device cuda
```

The all-17 runner is sequential, records each exit code, continues after
failures, skips checkpoint-blocked/completed models, and resumes a run when
the same run directory is explicitly reused. `--force` permits a new rerun;
it does not change model identity.

Run directories contain resolved configuration, environment and Git
provenance, checkpoint source, metrics, last/best checkpoints, best metadata,
and `failure.json` only on failure. Training microbatch and effective batch are
separate; gradient accumulation preserves the target effective batch. An OOM
must be resolved by lowering only microbatch and raising accumulation—never by
changing frames or resolution.

## Final RTX 5070 deployment measurement

Only a selected fine-tuned `best.pth` is valid:

```bash
python scripts/benchmark_finetuned_ssv2.py \
  --model uniformer-s \
  --checkpoint runs/ssv2/uniformer-s/<run-id>/best.pth \
  --device cuda --power nvml --num-clips 1000 \
  --output results/finetuned/ssv2_rtx5070/<run-id>
```

This wrapper reuses the existing synchronized CUDA/NVML Stage-2 measurement,
enforces batch size 1, uses the fixed seeded official-validation ID pool and
writes only below `results/finetuned/ssv2_rtx5070`. It cannot overwrite legacy
K400 or released-SSV2 results. NS-E, NS-M and NS# use the newly measured SSV2
accuracy percentage, total timed-set seconds, average watts and peak MiB.

## Scientific blockers

- DualFormer-B (IN21K, 82.9) remains blocked. The public K400 Base file is the
  IN1K 81.1 model and is explicitly recorded as incompatible.
- MViTv1 16x4 and DualFormer-S exact files are currently absent locally; the
  fetcher contains only their official model-zoo Google Drive IDs.
- ZeroI2V-B and the local VideoMAE-B export remain
  `READY_NEEDS_CONFIG_VALIDATION` until their pinned backend preflights pass.
- The current checkout does not contain the complete SSV2 train/validation
  video set, so full training is intentionally blocked by dataset validation.
