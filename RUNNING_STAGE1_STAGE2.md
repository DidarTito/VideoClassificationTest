# Portable Stage 1-2 runs

The benchmark registry contains the 17 requested model keys. Model loading is
lazy, checkpoint loading is strict, and every candidate is measured with one
uniformly sampled full-video view at batch size 1. `--device` selects the Torch
device; `--device-name` is only the label recorded in the output files.

Stage 2 records latency, power, energy, peak inference VRAM, measured accuracy,
and the adapter's actual temporal input (`FramesUsed`) for every successful
model. It ranks the candidates with the reference paper's 1/8-weight variants:

- `NS-E(1/8)`: accuracy with total inference time and average power
- `NS-M(1/8)`: accuracy with peak inference VRAM
- `NS#(1/8)`: accuracy with time, power, and peak inference VRAM jointly

Higher is better for all three scores. The score accuracy is measured top-1
when ground-truth labels are available and published top-1 otherwise; the CSV
identifies the choice in `MetricAccuracySource`.

With accuracy `A` expressed in percent, total timed-set inference `t` in
seconds, average power `P` in watts, peak VRAM `r` in MiB, and `w = 1/8`, the
implemented paper definitions are:

- `NS-E = 20 log10(A^2 / ((t P)^w))`
- `NS-M = 20 log10(A^2 / r^w)`
- `NS# = 20 log10(A^2 / ((r t P)^w))`

Install the direct NVML binding used by `--power nvml` with:

```powershell
venv\Scripts\python.exe -m pip install -r requirements-device-metrics.txt
```

## Output files

`stage2_pipeline.py` writes all deliverables under `--output`:

- `netscore_screening.csv`
- `netscore_filtered_models.csv`
- `device_metrics.csv`
- `ranking_ns_e.csv`
- `ranking_ns_m.csv`
- `ranking_ns_hash.csv`
- `netscore_variants.png` when matplotlib is installed and all required
  measurements are finite
- `summary_report.md`

The report and console output also give the inference-only time estimate for
all successfully measured candidates. This is the sum of `TotalTime(s)` across
models (mean timed latency times clip count); it excludes video decoding,
checkpoint/model loading, warm-up, and power-logger startup.

## Kinetics-400

Run the read-only checkpoint/source preflight first:

```powershell
venv\Scripts\python.exe check_models.py
```

The current workspace is expected to report 15/17 ready. ViViT-S has no exact
checkpoint (the local file is ViViT-B), and the local SVT file is a
self-supervised backbone without a trained 400-class head. They are not
replaced with proxy models or random heads.

Run the complete screening, measurement, ranking, plot, and report pipeline:

```powershell
venv\Scripts\python.exe stage2_pipeline.py `
  --videos-dir datasets\kinetics400 `
  --dataset k400 `
  --device cuda `
  --power nvml `
  --device-name RTX3050 `
  --max-clips 1000 `
  --output results_stage2_k400_1000
```

Use `--models <registry-key> ...` to override the screened runnable set. The
same command can target `cpu`, `cuda:1`, `mps`, or another supported Torch
device. Energy-dependent scores are unavailable when power measurement is
disabled or unsupported; `NS-M(1/8)` can still be reported when peak VRAM is a
positive measurement.

## Something-Something-v2

The installed dataset uses this layout:

```text
datasets/ssv2/
  labels/
    labels.json
    train.json
    validation.json
    test.json
  videos/
    <video-id>.webm
```

Six compatible local 174-class checkpoint candidates pass the read-only asset
preflight:

- `uniformer-s`
- `uniformer-b`
- `videoswin-b`
- `timesformer-b`
- `video-focalnet-b`
- `videomae-b`

The previously completed device-results baseline contains the first four
models only. TimeSformer-B and VideoMAE-B are part of the new rerun and should
not be added to a ranking until strict 174-class loading and labeled inference
complete successfully.

The downloaded filenames under `datasets/ssv2/labels/` are not reliable split
identifiers in this installation: `validation.json` contains the 168,913-row
training annotation.  Use the vendored official 24,777-row validation list
below.  Of those validation IDs, 12,767 videos are present locally.

Run the validation benchmark with the video directory and the verified split
annotation explicitly:

```powershell
venv\Scripts\python.exe stage2_pipeline.py `
  --videos-dir datasets\ssv2\videos `
  --dataset ssv2 `
  --annotations third_party\UniFormer\video_classification\data_list\sthv2\somesomev2_rgb_validation_split.txt `
  --split-name validation `
  --models uniformer-s uniformer-b videoswin-b timesformer-b video-focalnet-b videomae-b `
  --device cuda `
  --power nvml `
  --device-name RTX3050 `
  --max-clips 1000 `
  --output results_stage2_ssv2_1000
```

For SSV2 directory mode, `--annotations` is required. Before seeded sampling,
the pipeline loads the annotation IDs and restricts the local videos to that
split. It then resolves every selected video's numeric 174-class ground truth
from the same file. This prevents train/test videos in the shared video folder
from leaking into a validation run. The runner validates `(1, 174)` logits and
records `Dataset`, `Split`, and `AnnotationFile` in every result row.

Decoded clips default to 32 frames at 224x224. Adapters may select or require a
different temporal input, so use the per-row `FramesUsed` value in tables and
slides rather than assuming a common 16- or 32-frame count for every model.

## Caching and smoke tests

Directory runs use a split-aware sharded clip cache under the selected video
directory. The annotation-ID fingerprint is part of the cache name, so caches
from different SSV2 splits are not reused. Pass `--no-cache` to force decoding
without reading or writing that cache.

For a quick wiring check before a long run, lower `--max-clips` and select one
model, while keeping the same dataset, annotation, device, and power options:

```powershell
venv\Scripts\python.exe stage2_pipeline.py `
  --videos-dir datasets\ssv2\videos `
  --dataset ssv2 `
  --annotations third_party\UniFormer\video_classification\data_list\sthv2\somesomev2_rgb_validation_split.txt `
  --split-name validation `
  --models uniformer-s `
  --device cuda `
  --power nvml `
  --max-clips 2 `
  --output results_stage2_ssv2_smoke
```

## Checkpoint caveats

- `checkpoints/videomae/VideoMAE-ViT-*_1600epoch.pth` files are pretraining
  encoder/decoder weights, not action-classification checkpoints. VideoMAE-B
  loads offline from `checkpoints/videomae/base-finetuned-kinetics/` when that
  Transformers export exists, otherwise from an already populated local Hugging
  Face cache. It does not download during a benchmark.
- Published table rows sometimes use a different view protocol from the exact
  local checkpoint. The CSV reports the adapter's actual parameter count,
  frame count, and measured device metrics; common-view measured accuracy is
  the fair device-run comparison.
