# Project progress -- frozen17 energy-efficiency study

_Auto-generated 2026-09-05 by `scripts/generate_progress_report.py`; all numbers come from artifacts in `reports/` and `results/`._

## Frozen17 checkpoint status

- 17 models total (frozen list; thresholds never re-run)
- 14/17 exact K400 checkpoints ready: uniformer-s, dualformer-t, video-focalnet-t, videoswin-t, mvit-v1-b-16x4, video-focalnet-s, mvit-v1-b-32x3, dualformer-s, videoswin-s, video-focalnet-b, videomae-b, omnivore-b-in21k, videoswin-b, timesformer-b
- 1 need backend config validation: zeroi2v-b16-8f
- 2 blocked/missing: uniformer-b, dualformer-b-in21k

## RTX3050 (Experiment A)

- verified reusable measurements: 11/17 (dualformer-t, video-focalnet-t, videoswin-t, video-focalnet-s, mvit-v1-b-32x3, dualformer-s, videoswin-s, video-focalnet-b, omnivore-b-in21k, videoswin-b, timesformer-b)
- need rerun (config/provenance mismatch): 3 (uniformer-s, mvit-v1-b-16x4, videomae-b)
- never measured: 0
- blocked: 3 (zeroi2v-b16-8f, uniformer-b, dualformer-b-in21k)
- final 17-row table not built yet (needs verified measurements for all 17)

## RTX5070

- tomorrow READY_EXACT benchmark set: 14/17 (uniformer-s, dualformer-t, video-focalnet-t, videoswin-t, mvit-v1-b-16x4, video-focalnet-s, mvit-v1-b-32x3, dualformer-s, videoswin-s, video-focalnet-b, videomae-b, omnivore-b-in21k, videoswin-b, timesformer-b)
- previous/stale load preflight: 15/17 models OK (see `reports/RTX5070_PREFLIGHT.md` for its host and date)
- K400 benchmark not yet run (step 10 of `run_project_steps.sh`)

## SSV2 fine-tuning (Experiment B)

- dataset validation: report exists (`reports/SSV2_DATASET_AUDIT.md`)
- fine-tune preflight: 2/2 models pass (uniformer-s, videomae-b)
- completed trainings: 0 (no `runs/ssv2/<model>/<ts>/best_metadata.json` yet)

## Recent concrete work

- 56 files added/modified in the working tree (uncommitted, see `git status`)
- frozen17 registry + `configs/frozen17.yaml` as single source of truth
- exact MViTv1 16x4 and MViTv1 32x3 integrations
- exact DualFormer-S integration
- checkpoint audit, SSV2 quarantine, safe strict checkpoint loading
- corrected official VideoMAE-B 1600-epoch K400 classifier
- explicit DualFormer-B IN21K and UniFormer-B 32x4 blockers
- RTX3050 consistency audit and 17-row final-table builder
- `run_project_steps.sh` master runner (tomorrow / rtx3050 modes)

## Next 3 actions

1. On the RTX5070 box: `bash run_project_steps.sh tomorrow` (env check, audits, 17-model preflight, K400 benchmark, two fine-tune smoke tests).
2. Resolve remaining blockers (UniFormer-B 32x4 artifact; DualFormer-B IN21K checkpoint; ZeroI2V Linux runtime preflight).
3. Fill RTX3050 gaps: `bash run_project_steps.sh rtx3050` after mounting the K400 subset.
