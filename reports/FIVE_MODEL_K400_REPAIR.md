# Five-model K400 repair

Base commit: `7f24c5d20e823bbcd7cfcde195a4b59e3e0710d9`

The K400 preflight now accepts exactly the checkpoint selected in
`configs/frozen17.yaml` for each requested model. It validates the configured
path and full SHA-256 without searching for substitute checkpoints.

## Root causes

- `uniformer-s`: the inspector and duplicate registry still required the
  16x8 checkpoint. The frozen benchmark target is the official K400 16x4
  checkpoint. Its official configuration uses 16 model frames and sampling
  rate 4.
- `uniformer-b`: the inspector globbed all Base checkpoints and compared each
  one with the 16x4 hash. It could reject the selected 32x4 checkpoint, while a
  local 16x4 file could hide the error. Validation is now limited to the exact
  configured 32x4 path and SHA.
- `videomae-b`: the inspector and duplicate registry required an untracked
  Hugging Face conversion. The repository adapter already constructs the
  vendored official VideoMAE graph and strictly loads the official final K400
  classifier `.pth`, so the hidden conversion requirement was removed.

## Exact checkpoint inventory

| Model | Exact path | Bytes | SHA-256 | Classes | Model frames |
| --- | --- | ---: | --- | ---: | ---: |
| uniformer-s | `checkpoints/uniformer/uniformer_small_k400_16x4.pth` | 85,735,325 | `4829a6abe934a1c8083f377afea621f01d16bc4a93ce6ae74d7c5bea7eff6a7b` | 400 | 16 |
| video-focalnet-t | `checkpoints/focalnet/video-focalnet_tiny_kinetics400.pth` | 198,333,209 | `86fb70adbd28e0fc8c9ca8c7ed36d52482a0c58507ad15820c8849bfb634b9ed` | 400 | 8 |
| uniformer-b | `checkpoints/uniformer/uniformer_base_k400_32x4.pth` | 199,525,573 | `b432450566cdc4f50222e4851b5d304bba4444a699cdbb43fa529ceb18ae1f8e` | 400 | 32 |
| videomae-b | `checkpoints/videomae/videomae_vit_b_k400_1600e_ft.pth` | 173,103,787 | `655e6d74924d76dd194ed72904b1541864ff27b0afe0be2c5e9e9c6778dd334d` | 400 | 16 |
| dualformer-t | `checkpoints/dualformer/dualformer_tiny_patch244_window877.pth` | 101,764,429 | `f27c1c1996e4d27bc5c71f2bf3f690d933d8ea378030fb9e9e93c816872deb46` | 400 | 32 |

The machine-readable audit is in
`reports/five_model_checkpoint_inventory.csv`. It contains exactly these five
rows and reports `READY_EXACT` for all five.

## VideoMAE proof

The selected artifact has a top-level `module` mapping with 162 tensors.
`head.weight` is `[400, 768]` and the patch projection is
`[768, 3, 2, 16, 16]`. It has no decoder tensors and no mask token. The
vendored final-classification graph has 12 transformer blocks and
86,534,800 parameters; strict state-dict loading succeeds. This establishes
that the artifact is the final K400 classifier, not a pretraining-only model.

## Changed files

- `check_models.py`: exact frozen-config checkpoint and SHA validation for the
  three repaired models.
- `models/__init__.py`: duplicate registry entries aligned with the frozen
  paths, model-frame counts, and published values.
- `configs/frozen17.yaml`: records the observed VideoMAE runtime parameter
  count and its source.
- `scripts/audit_five_model_checkpoints.py`: deterministic five-row, CPU-only
  artifact audit.
- `tests/test_five_model_k400_repair.py`: regression coverage for the three
  root causes, including an actual VideoMAE strict load.
- `tests/test_requested_model_metadata.py`: expected metadata aligned with the
  frozen model choices.
- `reports/five_model_checkpoint_inventory.csv`: generated forensic inventory.

`scripts/benchmark_finetuned_ssv2.py` was already modified before this repair
and is not part of it. The SSV2 loader was not reverted or changed.

## Verification

- Relevant suite: 51 tests passed, including 58 subtests.
- Exact requested K400 preflight: 15/15 required asset pairs ready and
  `READY: 5 models`.
- One-clip CUDA smoke: all five models completed and produced `[1, 400]`
  outputs. The result CSV preserves the existing format and records the exact
  checkpoint path; the accompanying inventory records its SHA.
- Progress timing regression: the progress update occurs after power sampling
  stops, and inference timing surrounds only the model call.
- `five_model_checkpoints.tar.gz`: all five members match the inventory path,
  size, and SHA, so no rebuild is required.

The full 1000-clip benchmark was not run.

## Teammate handoff

For a teammate staying on the recorded base commit, after the reviewed repair
commit is available:

```powershell
git switch --detach 7f24c5d20e823bbcd7cfcde195a4b59e3e0710d9
git cherry-pick <repair-commit>
venv\Scripts\python.exe scripts\audit_five_model_checkpoints.py
venv\Scripts\python.exe run_benchmark.py --device cuda --dataset k400 --dataset-root datasets/kinetics400 --manifest manifests/k400_1000_seed0.csv --num-clips 1000 --batch-size 1 --precision fp32 --seed 0 --warmup 2 --power nvml --models uniformer-s video-focalnet-t uniformer-b videomae-b dualformer-t --output-dir results/five_model_k400 --preflight
```

If the repair is merged to `origin/main`, use this update sequence instead:

```powershell
git switch main
git pull --ff-only origin main
venv\Scripts\python.exe scripts\audit_five_model_checkpoints.py
venv\Scripts\python.exe run_benchmark.py --device cuda --dataset k400 --dataset-root datasets/kinetics400 --manifest manifests/k400_1000_seed0.csv --num-clips 1000 --batch-size 1 --precision fp32 --seed 0 --warmup 2 --power nvml --models uniformer-s video-focalnet-t uniformer-b videomae-b dualformer-t --output-dir results/five_model_k400 --preflight
```

Suggested commit message: `Repair exact five-model K400 checkpoint preflight`
