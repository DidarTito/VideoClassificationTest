# Final code fix report — frozen17 audit and preparation

All changes are uncommitted in the working tree (per instruction: no commit,
no push). Companion documents: `reports/FINETUNE_FIX_REPORT.md` (training
system details), `reports/FROZEN17_CHECKPOINT_AUDIT.md`,
`reports/PROGRESS_FOR_PROFESSOR.md`.

## 1. Executive summary

The project now has one authoritative model manifest
(`configs/frozen17.yaml`, exactly 17 models), a checkpoint audit with
per-file SHA256 provenance (14/17 exact K400 checkpoints on disk), a
quarantined legacy-SSV2 checkpoint area with a loader-level safety check, a
real K400→SSV2 fine-tuning package with strict backbone loading, an RTX3050
result-consistency audit, an RTX5070 preflight, full-dataset benchmark
support (`--num-clips all`), and one master runner
(`run_project_steps.sh`) with `tomorrow`, `rtx3050` and `rtx3050-full`
modes. 15/17 models load their exact checkpoint and produce 400-class
logits on this Windows RTX3050 host; the 2 remaining are honestly blocked
(see §18).

## 2. What was wrong in the old project

- The active registry mixed the frozen 17 with post-audit models (ViViT-S,
  VTN-B, SVT-B, MViT-B-24) and proxy/legacy recipes.
- MViT checkpoints on disk were partly MViTv2; the frozen rows are MViTv1.
- Released SSV2 classifier checkpoints lived in the active checkpoint
  namespace and could silently become fine-tuning initializations.
- Published metadata (params/GFLOPs/Top-1) was duplicated across files and
  disagreed between them.
- Old RTX3050 result CSVs recorded no checkpoint path, manifest hash, or
  batch size, so no run could be verified against the frozen configuration.
- Several adapters were broken: DualFormer-S head width was hardcoded to
  the Tiny variant; `models/timesformer.py` referenced an unimported
  `ROOT`; MViT-B 16x4 pointed at a checkpoint that was never on disk.

## 3. Active vs legacy architecture

Active path: `configs/frozen17.yaml` → `training/registry.py` →
(`run_benchmark.py` + `models/` adapters for inference;
`training/` + `scripts/finetune_ssv2.py` for training).
Legacy: old registry keys remain importable as aliases
(`mvit-v1-b-16x4-torchvision`, `omnivore-b-standard`, `vtn-b`, `svt-b`, …)
but `FROZEN17_MODEL_KEYS` — the only set used by `--models frozen17` —
contains exactly the canonical 17. `models/missing_models.py` is no longer
imported by the active path.

## 4. Frozen17 registry

`models/__init__.py` exposes `FROZEN17_MODEL_KEYS` (17 canonical keys,
exactly as specified). `run_benchmark.py --models frozen17` resolves only
this tuple. Numeric truth (params, GFLOPs, paper Top-1, NetScore,
preprocessing, checkpoint provenance) lives only in `configs/frozen17.yaml`.

## 5. Checkpoint audit

`scripts/audit_frozen17.py` → `reports/frozen17_checkpoint_audit.csv` +
`reports/FROZEN17_CHECKPOINT_AUDIT.md`. Current state:

- 14 READY_EXACT (incl. newly fetched `checkpoints/mvit/MVIT_B_16x4.pyth`
  from the official PyTorchVideo model zoo — MViTv1, verified 400-class
  `head.proj`, SHA256 recorded — and
  `checkpoints/dualformer/dualformer_small_patch244_window877.pth` from the
  official sail-sg Google Drive release).
- 2 READY_NEEDS_CONFIG_VALIDATION: zeroi2v-b16-8f, videomae-b (file and
  head are valid; backend preflight on the Linux host must confirm exact
  upstream config linkage).
- 1 BLOCKED_EXACT_CHECKPOINT: dualformer-b-in21k (no released exact
  IN21K→K400 checkpoint; substitution is refused by design).

MViTv2 files remain on disk but no frozen17 key can reach them.

## 6. RTX3050 result audit

`scripts/audit_rtx3050.py` → `reports/rtx3050_frozen17_status.csv`.
Verdict: **0/17 REUSE, 13 RERUN_CONFIG_MISMATCH, 4 RUN_MISSING**
(mvit-v1-b-16x4, dualformer-s, zeroi2v-b16-8f, dualformer-b-in21k).
The 13 old rows record no checkpoint/manifest provenance, and several have
demonstrable mismatches (UniFormer-B measured at 16 frames vs the frozen
32-frame row; VideoMAE-B published 80.9 vs frozen 81.5; Video-FocalNet
params inconsistent with the frozen rows). `scripts/build_rtx3050_final17.py`
builds `results/k400_rtx3050_frozen17/final17.csv` and refuses to emit a
partial table.

## 7. RTX5070 preparation

`scripts/preflight_rtx5070.py` (env checks + per-model instantiate → exact
checkpoint → forward → 400 logits → peak VRAM) →
`reports/RTX5070_PREFLIGHT.md`. `scripts/setup_rtx5070_linux.sh` and
`envs/*/requirements.txt` provision the Linux environments.

## 8. SSV2 checkpoint quarantine

All released SSV2 classifiers moved to `checkpoints/legacy_ssv2_released/`
(with README). `training/checkpoint_utils.assert_k400_initialization`
rejects any fine-tune initialization whose path contains
`legacy_ssv2_released`/`ssv2`/`sthv2` unless explicitly overridden. Legacy
SSV2 result CSVs are untouched.

## 9. SSV2 dataset validation

`scripts/validate_ssv2.py` → `reports/SSV2_DATASET_AUDIT.md` (174 classes,
official splits, deterministic label map, no alphabetical-order labels,
missing/corrupt/duplicate checks). Fine-tuning refuses to start when
validation fails.

## 10. Fine-tuning architecture

`training/` package: registry (YAML-driven), strict checkpoint loading with
head-only mismatch allowance and `checkpoint_load_report.json`, SSV2
dataset, AMP policy (BF16→FP16 fallback, TF32), engine with gradient
accumulation + checkpointing, validation, per-run
`runs/ssv2/<model>/<timestamp>/` artifacts. Best checkpoint = highest val
Top-1.

## 11. Model-specific fixes (this session)

- **MViT-B 16x4**: fetched official PyTorchVideo checkpoint; adapter
  accepts both pytorchvideo and SlowFast key layouts; forward verified.
- **DualFormer-S**: fetched official checkpoint; fixed `_TokenLabelHead`
  hidden widths to scale with backbone width (768→1536→3072 for Small);
  strict load + forward verified.
- **TimeSformer-B**: fixed missing `ROOT` import; forward verified.
- **Video-FocalNet/VideoSwin (all variants)**: preflight no longer assumes
  `info["frames"]`; frame counts come from the frozen manifest.
- Other 17-model details: see `reports/FINETUNE_FIX_REPORT.md` §11.

## 12. Dependency changes

None added to the core venv this session (`gdown` already present).
Per-environment requirement files live under `envs/`.

## 13–15. Files added / modified / legacy

`git status --short` lists ~50 added and ~20 modified files (uncommitted).
Key additions this session: `scripts/audit_rtx3050.py`,
`scripts/build_rtx3050_final17.py`, `scripts/preflight_rtx5070.py`,
`scripts/generate_progress_report.py`, `run_project_steps.sh`,
`reports/rtx3050_frozen17_status.csv`, `reports/RTX5070_PREFLIGHT.md`,
`reports/PROGRESS_FOR_PROFESSOR.md`, this report, plus the two fetched
checkpoints. Legacy code paths are marked with `# LEGACY` comments and
distinct registry keys rather than deleted.

## 16. Tests executed

- `python -m compileall` over changed files: pass.
- `bash -n run_project_steps.sh`: pass; `--help`, `tomorrow --dry-run`,
  `rtx3050 --dry-run` all exercised.
- `scripts/audit_frozen17.py`, `scripts/audit_rtx3050.py`,
  `scripts/generate_progress_report.py` executed successfully.
- `scripts/build_rtx3050_final17.py` verified to fail loudly (17 missing).
- Existing pytest contract suites: see `tests/` (run `pytest -q`).

## 17. Preflight results (this Windows RTX3050 host)

15/17 models instantiate, strict-load their exact K400 checkpoint, and
return (1, 400) logits. Failures: zeroi2v-b16-8f (needs the
mmaction2-modern Linux runtime; refuses proxy substitution) and
dualformer-b-in21k (blocked checkpoint). Full table:
`reports/RTX5070_PREFLIGHT.md`.

## 18. Remaining blockers

1. **dualformer-b-in21k**: no released exact IN21K→K400 checkpoint.
   Options: contact authors, or document the row as unavailable.
2. **zeroi2v-b16-8f**: requires ZeroI2V repo + mmaction2-modern runtime on
   the Linux host (checkpoint is present and passes file audit).
3. **SSV2 local pool incomplete**: full fine-tuning cannot start until the
   dataset is complete on the training host.
4. RTX3050/RTX5070 K400 dataset root must be mounted (`K400_VAL_ROOT`).

## 19. Exact tomorrow commands

```bash
export K400_VAL_ROOT=/path/to/k400_val
bash run_project_steps.sh tomorrow
```

## 20. Exact RTX3050 commands

```bash
export K400_VAL_ROOT=/path/to/k400_val
bash run_project_steps.sh rtx3050        # audit + fill missing + final17
bash run_project_steps.sh rtx3050-full   # optional full clean rerun
```

## 21. Full experiment commands

```bash
bash run_project_steps.sh 13   # fine-tune preflight, all 17
bash run_project_steps.sh 14   # full fine-tuning (long)
bash run_project_steps.sh 15 16 17 18   # eval, SSV2 benchmark, tables, report
```

## 22. Suggested Git commit message

```
frozen17: canonical registry, checkpoint audit + fetches, SSV2 quarantine,
RTX3050 consistency audit, RTX5070 preflight, master runner

- configs/frozen17.yaml as single source of truth (exactly 17 models)
- fetch + verify MViT-B 16x4 (PyTorchVideo) and DualFormer-S (official GDrive)
- fix DualFormer token-label head width, TimeSformer ROOT import
- quarantine released SSV2 checkpoints; loader refuses them for fine-tuning
- audit RTX3050 results against frozen config (0 reusable; 4 never run)
- add preflight_rtx5070, audit_rtx3050, build_rtx3050_final17,
  generate_progress_report, run_project_steps.sh (tomorrow/rtx3050 modes)
- support --num-clips all with disk-aware streaming decode
```
