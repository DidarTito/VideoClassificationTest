# Efficient Video Transformer Deployment Benchmark

This repository evaluates how to choose an efficient video-classification
transformer for a particular deployment platform. The same methodology is
intended to run on consumer GPUs such as an RTX 3050 or RTX 5090 and on edge
devices such as Jetson AGX Orin, while preserving the exact model variant,
checkpoint, dataset, and input protocol used for every measurement.

## K400 to SSV2 fine-tuning

The frozen 17-model transfer experiment is separate from the legacy inference
results. Start with [FINETUNE_SSV2.md](FINETUNE_SSV2.md), prepare the Linux
RTX 5070 environments with [LINUX_RTX5070_SETUP.md](LINUX_RTX5070_SETUP.md),
and review [reports/CHECKPOINT_AUDIT.md](reports/CHECKPOINT_AUDIT.md) before
launching any training. The canonical list is `configs/frozen17.yaml`; it is
not regenerated from Stage-1 thresholds.

The two starting benchmarks are Kinetics-400 (K400) and
Something-Something-v2 (SSV2). Results from the two datasets are never mixed:
K400 checkpoints have 400-class heads, while SSV2 checkpoints have 174-class
heads and must be evaluated with SSV2 labels.

## Tomorrow's run

Run these commands from the repository root.

### Environment check

```powershell
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Preflight

```powershell
python run_benchmark.py --preflight --device auto --dataset k400
```

### Smoke test

```powershell
python run_benchmark.py --device auto --dataset k400 --models uniformer-s --num-clips 5 --smoke-test
```

### Quick 100 clips

```powershell
python run_benchmark.py --device auto --dataset k400 --num-clips 100 --quick
```

### Full 1000 clips

```powershell
python run_benchmark.py --device auto --dataset k400 --num-clips 1000 --resume
```

## Availability is dataset-specific

Stage 1 contains 36 literature candidates and filters them using published
accuracy and complexity. Seventeen rows satisfy the current Stage 2 screening
target. For K400, 15 of those 17 have locally runnable exact checkpoints.
ViViT-S is excluded because the available ViViT weight is a different model
variant; SVT-B is excluded because the available self-supervised backbone has
no trained 400-class classification head. The benchmark does not substitute a
proxy architecture or a random head.

The repository does **not** claim that the same 15 models are runnable on
SSV2. Six exact 174-class candidates pass the local read-only asset preflight:
UniFormer-S, UniFormer-B, VideoSwin-B, VideoFocalNet-B, TimeSformer-B, and
VideoMAE-B. All six now have completed 1,000-clip RTX 3050 measurements with
strict `(1, 174)` logits and labeled accuracy. A ZeroI2V SSV2 weight is
present, but it is not one of the selected 15 candidates and its legacy
MMAction/MMCV runtime is not currently runnable on this Windows environment.

Run the local preflight instead of relying on this summary, because checkpoint
availability changes as artifacts are added:

```powershell
venv\Scripts\python.exe check_dataset_models.py --dataset k400
venv\Scripts\python.exe check_dataset_models.py --dataset ssv2
```

## Methodology

Stage 1 screens candidates with published K400 top-1 accuracy, GFLOPs, and
parameter count. The current Stage 2/Pareto selection region is top-1 accuracy
at least 78%, complexity at most 300 GFLOPs, and fewer than 300 million
parameters. The plot encodes accuracy on the y-axis, GFLOPs on the x-axis,
parameter count by color, and marks selected models with stars.

Stage 2 uses a common device protocol and records:

- measured top-1 accuracy on the selected labeled split;
- latency per clip and total inference-only time;
- peak inference GPU memory;
- average GPU power sampled through NVML (or a platform fallback);
- energy per clip; and
- NetScore variants NS-E, NS-M, and NS# with the paper's 1/8 efficiency weight.

Published top-1 accuracy and measured common-view top-1 accuracy are separate
columns. Device rankings should use measured accuracy when labels are
available. Full protocol details and the score formulas are in
[RUNNING_STAGE1_STAGE2.md](RUNNING_STAGE1_STAGE2.md).

## Current RTX 3050 results

The July 22, 2026 rerun completed all 15 exact K400 classifier pairs and all 6
available exact SSV2 pairs on 1,000 cached clips per model. K400 inference-only
time was 3,694.5 seconds (61.58 minutes); SSV2 was 1,428.9 seconds (23.82
minutes). The combined report preserves all nine unavailable SSV2 rows as
missing rather than inventing proxy results:

- [K400 device metrics](results/rtx3050/k400_1000_seed0_20260722/device_metrics.csv)
- [SSV2 device metrics](results/rtx3050/ssv2_1000_seed0_20260722/device_metrics.csv)
- [Cross-dataset report](results/rtx3050/cross_dataset_1000_seed0_20260722/stage2_cross_dataset_report.md)
- [Experiment notes and limitations](results/rtx3050/EXPERIMENT_NOTES_20260722.md)
- [Stage 1 screening figure](figures/stage1_screening_k400.png)

On this run, UniFormer-S has the best NS# on both datasets. Video-FocalNet-T
has the highest local K400 measured accuracy and NS-E, while VideoSwin-B has
the highest local SSV2 accuracy but much higher latency, energy, and memory.
These are local single-view results, not replacements for canonical
paper-reported multi-view accuracy.

## Environment

Python 3.10 is the currently exercised environment. Install a PyTorch build
that matches the target device first; do not assume that one CUDA wheel is
correct for RTX 3050, RTX 5090, and Jetson. The current Windows environment
uses PyTorch 2.6.0 and torchvision 0.21.0 with CUDA 12.4. For example:

```powershell
py -3.10 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 `
  --index-url https://download.pytorch.org/whl/cu124
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install -r requirements-device-metrics.txt
```

That CUDA 12.4 wheel is the preserved RTX 3050 baseline, not the recommended
Blackwell wheel. `--preflight` executes a real CUDA kernel before loading any
checkpoint. If it reports a CUDA-architecture compatibility error on the RTX
5070, run this separately and repeat preflight (the benchmark never changes
the environment itself):

```powershell
python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

PyTorch 2.7 introduced Blackwell support and official CUDA 12.8 wheels; the
version pair above is listed for Windows and Linux in the official
[previous-version installer](https://pytorch.org/get-started/previous-versions/).

For Jetson, install NVIDIA's JetPack-compatible PyTorch wheel instead of the
desktop CUDA command above. Optional Google Drive artifact retrieval uses
`requirements-artifacts.txt`.

Initialize third-party model sources according to
[THIRD_PARTY.md](THIRD_PARTY.md) after the parent Git repository is created.

## Quick start

Run unit and contract tests without loading multi-gigabyte checkpoints:

```powershell
venv\Scripts\python.exe -m unittest discover -s tests -v
```

Generate the Stage 1 screening plot:

```powershell
venv\Scripts\python.exe scripts\plot_stage1_screening.py `
  --output stage1_pareto_plot.png
```

Use the commands in **Tomorrow's run** for K400. The legacy wrapper remains
available for reproducing older result layouts, but new runs should use the
central `run_benchmark.py` configuration and automatic device naming.

## Checkpoints and datasets are not Git blobs

The local checkpoint collection is tens of gigabytes, and many individual
files exceed GitHub's 100 MiB regular-Git limit. The datasets are larger still
and have their own redistribution terms. Consequently:

- `checkpoints/`, `datasets/`, virtual environments, clip caches, videos,
  and in-progress benchmark directories are ignored;
- `checkpoints/manifest.csv` records filename, byte size, SHA-256 digest,
  dataset hint, and provenance fields without committing the binaries;
- official upstream download URLs are preferred over rehosting third-party
  weights;
- Google Drive or GitHub Release assets may be used only after checking each
  checkpoint's redistribution license; and
- final, reviewed CSV/Markdown summaries and plots under `results/` belong in
  Git, while datasets, decoded caches, and checkpoint binaries remain local.

Use the artifact helper to refresh or verify the local inventory:

```powershell
venv\Scripts\python.exe scripts\checkpoint_artifacts.py refresh
venv\Scripts\python.exe scripts\checkpoint_artifacts.py verify
```

The helper can download rows carrying a verified `source_url` or
`google_drive_id`. Nineteen of the 21 benchmark model/dataset pairs have an
official retrieval source; MViT-B-24 and VTN-B remain source-unresolved. No
checkpoint-specific redistribution license was verified, so every benchmark
binary remains marked `unverified-do-not-redistribute`.

The current inventory contains 46 artifacts totaling 21.055 GiB; 35 exceed
100 MiB and one exceeds 2 GiB. GitHub [blocks regular Git files above
100 MiB](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).
Putting the complete collection in Git LFS is also a poor default: GitHub Free
currently includes 10 GiB of LFS storage and has a 2 GB per-file limit, both
below this inventory ([LFS limits and
billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs)).
GitHub Releases allow large binary distribution, but every asset must remain
under 2 GiB, so the largest checkpoint would need to be split
([release limits](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)).
The safe publication default is therefore Git for code, small reviewed results,
and the checksum manifest; official upstream storage or a license-reviewed
Google Drive/Release mirror for weights.

## Repository status

The working directory is being prepared for
[DidarTito/VideoClassificationTest](https://github.com/DidarTito/VideoClassificationTest).
At the time of this cleanup, the local `.git` directory is empty; no commit,
branch, remote, or push has been created. Repository initialization and remote
history reconciliation must happen only after active benchmarks finish and
the final artifact inventory is reviewed.
