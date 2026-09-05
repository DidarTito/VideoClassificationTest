# Frozen-17 K400 to SSV2 fixing report

## 1. Executive summary

The repository was an inference-first Stage-1/Stage-2 benchmark whose active
17-row selection still contained ViViT-S, SVT-B, VTN-B and MViT-B-24. It also
mixed dataset-specific released SSV2 checkpoints with K400 inference adapters,
had no end-to-end transfer-training registry, and could not distinguish a
permissible classifier mismatch from arbitrary `strict=False` backbone loss.

This change adds a separate K400-to-SSV2 training system without rewriting the
legacy inference path. `configs/frozen17.yaml` is the sole active training
manifest and contains exactly the user-frozen keys. It adds safe audit/fetch,
strict loading and 174-head replacement, deterministic SSV2 validation,
preflight levels, resumable one-GPU training/evaluation, Linux environment
separation, and a final-results-only RTX 5070 batch-one benchmark wrapper.

Checkpoint audit result (updated after fetching MViT-B 16x4 from the official
PyTorchVideo model zoo and DualFormer-S from the official Google Drive
release): **14 READY_EXACT, 2 READY_NEEDS_CONFIG_VALIDATION, and 1
BLOCKED_EXACT_CHECKPOINT**. Full training is still blocked because the local
SSV2 pool is incomplete.

## 2. Original project architecture

`screen_models.py` built a literature table and threshold screen.
`stage2_pipeline.py` selected registry keys, decoded/cached a fixed clip set,
then delegated to `run_benchmark.py`. Model modules were inference wrappers
which loaded local/vendored checkpoints and applied family preprocessing.
`run_benchmark.py` synchronized CUDA timings, sampled power through the shared
`power.py` NVML logic, reset/recorded peak allocation, calculated measured
accuracy, and called `metrics.py` for NS-E, NS-M and NS#. This measurement code
is reused by `scripts/benchmark_finetuned_ssv2.py`.

The RTX 3050 K400 results and six released-SSV2 checkpoint results remain
legacy-only. They were not moved, edited or overwritten. The old public model
registry remains for compatibility, but training resolves only the new YAML
manifest and cannot select threshold-derived or banned models.

## 3. Frozen 17 registry

| Model | Canonical key | Backend | Frames | K400 checkpoint | 174 head | Status |
|---|---|---|---:|---|---|---|
| UniFormer-S | uniformer-s | core/UniFormer | 16x4 | uniformer_small_k400_16x4.pth | `head` | READY_EXACT |
| DualFormer-T | dualformer-t | legacy MMAction | 32x2 | dualformer_tiny...pth | `cls_head.fc_cls` | READY_EXACT |
| Video-FocalNet-T | video-focalnet-t | native | 8x8 | tiny_kinetics400.pth | `head` | READY_EXACT |
| VideoSwin-T | videoswin-t | legacy MMAction | 32x2 | tiny...kinetics400_1k.pth | `cls_head.fc_cls` | READY_EXACT |
| MViTv1 B 16x4 | mvit-v1-b-16x4 | SlowFast/PyTorchVideo | 16x4 | MVIT_B_16x4.pyth | `head.proj` | READY_EXACT |
| Video-FocalNet-S | video-focalnet-s | native | 8x8 | small_kinetics400.pth | `head` | READY_EXACT |
| MViTv1 B 32x3 | mvit-v1-b-32x3 | SlowFast/PyTorchVideo | 32x3 | MVIT_B_32x3...pyth | `head.proj` | READY_EXACT |
| DualFormer-S | dualformer-s | legacy MMAction | 32x2 | dualformer_small...pth | `cls_head.fc_cls` | READY_EXACT |
| VideoSwin-S | videoswin-s | legacy MMAction | 32x2 | small...kinetics400_1k.pth | `cls_head.fc_cls` | READY_EXACT |
| ZeroI2V B/16 | zeroi2v-b16-8f | modern MMAction2 | 8x16 | CLIP-B_k400_8x16x1.pth | `cls_head.fc_cls` | READY_NEEDS_CONFIG_VALIDATION |
| Video-FocalNet-B | video-focalnet-b | native | 8x8 | base_kinetics400.pth | `head` | READY_EXACT |
| UniFormer-B | uniformer-b | core/UniFormer | 16x4 | uniformer_base_k400_16x4.pth | `head` | READY_EXACT |
| VideoMAE-B | videomae-b | native/HF export | 16x4 | model.safetensors | `classifier` | READY_NEEDS_CONFIG_VALIDATION |
| DualFormer-B IN21K | dualformer-b-in21k | legacy MMAction | 32x2 | none | `cls_head.fc_cls` | BLOCKED_EXACT_CHECKPOINT |
| Omnivore-B IN21K | omnivore-b-in21k | native | 32x40 | swinB_In21k...torch | `heads.video.1` | READY_EXACT |
| VideoSwin-B | videoswin-b | legacy MMAction | 32x2 | base...kinetics400_1k.pth | `cls_head.fc_cls` | READY_EXACT |
| TimeSformer-B | timesformer-b | native | 8x32 | divST_8x32...pyth | `model.head` | READY_EXACT |

Stage-1 GFLOPs in this table's manifest are per-view values. In particular,
DualFormer 240/636/1072 are not reused as frozen per-view values, and ZeroI2V
422 is not used; the requested 60/159/268 and 140.67 values are recorded.

## 4. Checkpoint audit

The machine-generated full table, SHA256 values, safely observed state roots
and head evidence are in `checkpoint_audit.csv` and `CHECKPOINT_AUDIT.md`.
All twelve exact-ready files opened safely and exposed a 400-way classifier.
MViTv1 32x3 contains `model_state/head.proj`; the local MViTv2 files instead
contain the v2 `head.projection` schema and are never candidates. Omnivore's
file contains strict `trunk` and `heads` maps with `video.1` output 400.
TimeSformer contains `model_state/model.head` output 400.

ZeroI2V's pickle requires an MMEngine `HistoryBuffer` global and was not opened
with unrestricted pickle on the audit host; it therefore remains config-
validation pending. VideoMAE's safetensors export has a 400-way classifier,
but its exact official-recipe equivalence still needs runtime validation.

## 5. Missing checkpoint resolution

Nothing was silently substituted or fabricated. The fetcher records the
official SlowFast Google Drive ID `194g...Y-QL` for MViTv1 B-Conv 16x4 and the
official DualFormer-S ID `1dN...L3TY`. Downloads use `.part`, resume, hash and
atomic rename and never overwrite an existing file.

The present `dualformer_base_patch244_window877.pth` hashes to
`6d7414ca...679eca`; it is the public IN1K K400 Base 81.1 artifact and is
recorded in the manifest as scientifically incompatible. The official repo
does not publish the requested IN21K-to-K400 82.9 checkpoint/config. That row
requires an author-provided artifact or a documented reproduction and remains
blocked.

## 6. Dependency/environment audit

Audit host: Windows, Python 3.10.0 at the user Python installation, Torch
2.5.1+cu121, driver 595.97, RTX 3050 Laptop GPU (capability 8.6), CUDA available
and BF16 reported supported. This is not the target RTX 5070 and no target
performance claim was made. The existing environment also has corrupt
pandas/matplotlib installation files (`ValueError: source code string cannot
contain null bytes`).

Four Linux environments are defined: core-modern, SlowFast, legacy MMAction,
and modern MMAction2. They keep a Blackwell-capable Torch/CUDA base while
isolating mutually incompatible framework APIs. The setup script performs a
real CUDA tensor test. Legacy Torch/CUDA downgrades are prohibited.

## 7. Files changed

- `configs/frozen17.yaml`: authoritative frozen identities, inputs, heads,
  checkpoint/provenance/status, Stage-1 facts and recipe policy.
- `configs/ssv2/*.yaml`: default controlled recipe and one provenance-marked
  model resolution file for every frozen key.
- `training/registry.py`: validates 17 unique keys and 174 classes.
- `training/checkpoint_utils.py`: safe state unwrapping, conservative prefix
  removal, exact backbone validation, head-only mismatch and metadata report.
- `training/amp.py`, `engine.py`, `scheduler.py`, `validation.py`: BF16/FP16,
  accumulation, finite-loss checks, AdamW step, cosine warmup and Top-1/Top-5.
- `training/datasets/ssv2.py`: official JSON/numeric mapping, duplicates,
  deterministic IDs and audit hash.
- `training/adapters/base.py`, `runtime.py`: training boundary around strict
  existing constructors and model-specific input call semantics.
- `scripts/audit_checkpoints.py`: CSV/Markdown forensic audit and SHA256.
- `scripts/fetch_missing_k400_checkpoints.py`: official-only safe fetch.
- `scripts/validate_ssv2.py`: training gate and dataset report.
- `scripts/preflight_finetune.py`: Levels 1-4 checks and 17-row CSV.
- `scripts/finetune_ssv2.py`: run provenance, resume, last/best and failures.
- `scripts/eval_ssv2.py`: selected-best full-validation entry point.
- `scripts/run_finetune_all17.sh`: sequential skip/continue/summary runner.
- `scripts/benchmark_finetuned_ssv2.py`: batch-one final result wrapper.
- `scripts/setup_rtx5070_linux.sh`, `envs/*`: isolated reproducible Linux setup.
- `tests/test_finetune_contract.py`: frozen registry, input metadata, strict
  mismatch, head output, output isolation and batch-one contracts.
- `checkpoints/manifest.csv`: verified UniFormer-S and incompatible public
  DualFormer-B records.
- `README.md`, `FINETUNE_SSV2.md`, `LINUX_RTX5070_SETUP.md`: lab entry points.
- `reports/*`: generated checkpoint/data audits and honest preflight status.
- `models/__init__.py`: canonical compatibility aliases only; the training
  shortlist itself is not sourced from this legacy registry.

All other dirty files visible in `git status` predated this task and were
preserved. They must be reviewed separately before a future commit.

## 8. Model-specific fixes

UniFormer-S/B now distinguish the 16x4 K400 initialization from paper test
views; neither SSV2 file is used to start transfer. Both MViT rows explicitly
say v1 and point to v1 configs; v2 weights cannot validate. DualFormer T/S use
their true per-view metadata; Base IN21K is blocked. FocalNet Base reuses the
official SSV2 family recipe while T/S are labelled family-adapted. VideoSwin
T/S/B preserve 32-frame legacy-MMAction semantics. ZeroI2V is the exact B/16
8-frame name, not the released L SSV2 model. VideoMAE distinguishes K400
classification from masked-pretraining `.pth` files and from direct SSV2
pretraining. Omnivore validates its IN21K joint-training builder and K400 head.
TimeSformer is labelled a controlled K400-to-SSV2 transfer rather than its
published non-K400 SSV2 recipe. The executable recipes currently use an
explicit controlled fallback; upstream family configs were audited for input
semantics but have not been falsely labelled `official_exact` until their full
optimizer/augmentation schedules are ported and verified on the target host.

## 9. SSV2 data validation

The official vendored numeric lists contain **168,913 train** and **24,777
validation** records, exactly 174 classes and zero ID overlap. The mapping
comes from the official `labels.json`, never sorted directories. Annotation
identity is `d87d7000...14c5b` and is suitable for cache versioning.

The local pool is incomplete: **82,233 train IDs and 12,010 validation IDs are
missing**. Corrupt decoding was not requested in this fast audit. Dataset
status is therefore not ready, and training exits before constructing a model.

## 10. Fine-tune implementation

Strict existing constructors first load the full 400-class artifact. The
training layer verifies the declared output layer is exactly `nn.Linear(...,
400)`, replaces only it with 174 outputs, uses truncated-normal weight and zero
bias initialization, and records the policy. The independent safe loader fails
on any missing, unexpected or shape-mismatched backbone tensor and writes its
full report before raising.

Training supports fixed configuration/seeds, AMP policy, accumulation,
clipping, warmup/cosine schedule, last/best checkpoints, resume, per-epoch
validation, finite loss/logit guards, CSV metrics, failure JSON and earlier-
epoch tie behavior. Microbatch is distinct from batch-one deployment.

## 11. Preflight results

No target-hardware training preflight was fabricated. `FINETUNE_PREFLIGHT.csv`
has all 17 rows. A CPU Level-1 probe passed model load and 174-head replacement
for VideoMAE-B; Transformers exposed its q/v-bias schema conversion, so it
correctly remains config-pending. UniFormer-S reached import but failed because
the active system Python lacks `timm`; the documented target environment
includes it. All target-host forward/backward/smoke fields remain `NOT_RUN`.
Levels 1-4 are implemented,
including import/build/head, finite `[B,174]` forward, backward/optimizer with
a changed backbone parameter, smoke validation and strict save/resume. Run
UniFormer-S and VideoMAE-B first after the dataset and target environments are
ready.

## 12. RTX 5070 optimizations

The setup retains modern CUDA; BF16 autocast is preferred and FP16 uses
GradScaler. Transfers are non-blocking, loader pinning/persistence/prefetch are
configured, accumulation and `zero_grad(set_to_none=True)` are used, and
gradient checkpointing capability/provenance is recorded per model. DDP/FSDP
is not used. Automatic `torch.compile`, automatic resolution reduction and
automatic frame reduction are prohibited. Largest stable microbatches, VRAM,
TF32 and compile compatibility remain target-host measurements, not guesses.

## 13. Exact commands to run next

```bash
bash scripts/setup_rtx5070_linux.sh core-modern
source .venv-core-modern/bin/activate
python scripts/validate_ssv2.py --check-decode
python scripts/audit_checkpoints.py --models frozen17
python scripts/fetch_missing_k400_checkpoints.py --models mvit-v1-b-16x4 dualformer-s
python scripts/preflight_finetune.py --models uniformer-s videomae-b --device cuda --level 4
python scripts/preflight_finetune.py --models frozen17 --device cuda --level 4
python scripts/finetune_ssv2.py --model uniformer-s --device cuda --smoke
bash scripts/run_finetune_all17.sh
python scripts/eval_ssv2.py --model uniformer-s --checkpoint runs/ssv2/uniformer-s/<run-id>/best.pth --split val --device cuda
python scripts/benchmark_finetuned_ssv2.py --model uniformer-s --checkpoint runs/ssv2/uniformer-s/<run-id>/best.pth --device cuda --power nvml --num-clips 1000 --output results/finetuned/ssv2_rtx5070/<run-id>
```

Activate the model's environment from the table before each family; the
all-17 orchestration should be split by environment if shell activation cannot
be automated reliably in the lab scheduler.

## 14. Known blockers

1. The local SSV2 dataset is incomplete, so no scientifically valid full run
   can start.
2. Exact MViTv1 16x4 and DualFormer-S files are not local.
3. DualFormer-B IN21K 82.9 is not publicly resolved; IN1K 81.1 is rejected.
4. ZeroI2V and VideoMAE exact config/runtime identities need Linux preflight.
5. The current Windows Python has corrupted pandas/matplotlib files; five
   existing test modules cannot import for that environmental reason.
6. No RTX 5070 was present, so BF16 throughput, stable microbatches, VRAM,
   backward, smoke/resume and deployment measurements remain unmeasured.

## 15. Reproducibility record

- Pre-change Git commit: `6e3718c6eed0b1dc3a2fd5b8300669fb67d542cc`
- Branch: `main`
- Audit host: Python 3.10.0, Torch 2.5.1+cu121, driver 595.97, Windows RTX 3050
- Frozen seed policy: seed 0 first; additional seeds are explicit later.
- New focused tests: 6/6 passed.
- `python -m compileall training scripts tests`: passed.
- Full existing suite: 79 runnable tests passed; five modules failed import
  because installed pandas/matplotlib source contains null bytes.
- Git diff includes substantial pre-existing user work; use `git diff --stat`
  immediately before committing to capture the final reviewed state.

## 16. Suggested Git commit message

Title: `Add strict frozen-17 K400 to SSV2 fine-tuning pipeline`

Body:

- freeze canonical 17-model identities and checkpoint provenance
- audit/fetch exact K400 starts and block scientific substitutions
- validate official SSV2 mapping and train/validation separation
- add strict head-only transfer, preflight, training, evaluation and resume
- isolate legacy backends while retaining a Blackwell-capable CUDA stack
- preserve legacy results and add RTX 5070 batch-one final measurement path

No commit or push was performed.
