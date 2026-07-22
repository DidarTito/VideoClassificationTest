# Stage 2 device metrics and deployment NetScore results

## Executive result

The requested SAM replacement is complete.  Stage 2 now reports the reference
paper's deployment-only `NS-E`, `NS-M`, and `NS#` variants at `w = 1/8`.

- **SSV2:** UniFormer-S leads NS-E and NS#; UniFormer-B leads NS-M by 0.07.
- **K400:** Video-FocalNet-T leads NS-E; UniFormer-S leads NS-M and NS#.
- All new result folders are SAM-free.  The original K400 source is preserved.

With accuracy `A` in percent, total timed-set inference `t` in seconds, average
power `P` in watts, and peak inference VRAM `r` in MiB:

- `NS-E = 20 log10(A^2 / (tP)^(1/8))`
- `NS-M = 20 log10(A^2 / r^(1/8))`
- `NS# = 20 log10(A^2 / (rtP)^(1/8))`

The implementation reproduces the supplied paper's ViT-S example
(`A=89.7`, `t=21.3`, `P=144.2`, `r=1084`) as 69.39, 70.52, and 61.81.

## Something-Something-v2 measurement

Protocol: RTX 3050 Laptop GPU, batch size 1, one uniformly sampled full-video
view, 32 decoded frames at 224x224, model-specific temporal subsampling, and
direct `pynvml/NVML` power polling at 20 ms.  The seeded sample contains 1,000
unique videos and 159/174 classes from the verified official validation split.

| Model | Frames | Published top-1 (%) | Measured top-1 (%) | Latency (ms/clip) | Avg power (W) | Energy (J/clip) | Total time (s) | Total energy (J) | Peak VRAM (MiB) | NS-E | NS-M | NS# |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| UniFormer-S | 16 | 67.7 | 65.1 | 78.9 | 44.32 | 3.496 | 78.878 | 3,495.66 | 214 | **63.68** | 66.72 | **57.86** |
| UniFormer-B | 16 | 70.4 | 67.1 | 164.4 | 54.70 | 8.994 | 164.436 | 8,994.28 | 324 | 63.18 | **66.79** | 56.91 |
| VideoSwin-B | 32 | 69.6 | **69.5** | 604.3 | 58.59 | 35.403 | 604.281 | 35,403.13 | 2,156 | 62.31 | 65.35 | 53.97 |
| Video-FocalNet-B | 8 | 71.1 | 61.1 | 230.0 | 56.31 | 12.948 | 229.964 | 12,948.26 | 1,250 | 61.16 | 63.70 | 53.42 |

Estimated inference-only time for all four candidates is **1,077.559 s
(17.96 min)**.  Their summed estimated inference energy is **60,841.33 J
(16.90 Wh)**.  Decoding, checkpoint loading, warm-up, host preprocessing, and
power-logger startup are intentionally excluded from the time estimate.

### SSV2 intermediate conclusions

- **UniFormer-S is the deployment default:** it has the minimum latency, power,
  energy, and peak VRAM, so it wins both energy-aware NS-E and joint NS#.
- **UniFormer-B is the memory-score specialist:** its 2.0-point measured
  accuracy gain is enough to edge UniFormer-S on softly weighted NS-M, despite
  using 110 MiB more VRAM.  It does not win NS-E or NS# because it takes 2.08x
  as long and 2.57x as much energy per clip.  The exact NS-M margin is only
  0.075, so that ordering should be treated as sampling-sensitive.
- **VideoSwin-B is the measured-accuracy leader (69.5%)**, but its 604.3 ms
  latency, 35.4 J/clip, and 2,156 MiB peak VRAM push it to third on all three
  deployment variants.
- **Video-FocalNet-B is protocol-sensitive:** its measured single-view result
  is 10.0 points below its published multi-view result.  Its last-place scores
  here describe this common-view device protocol, not its canonical paper
  evaluation.  In this measured subset UniFormer-S strictly dominates it on
  accuracy, latency, power, energy, and memory; checkpoint/protocol effects
  should be investigated before making a general architectural claim.

The published SSV2 metadata comes from the official model zoos for
[UniFormer](https://github.com/Sense-X/UniFormer),
[Video Swin](https://github.com/SwinTransformer/Video-Swin-Transformer), and
[Video-FocalNets](https://github.com/TalalWasim/Video-FocalNets).

## Kinetics-400 NetScore retrofit

The existing 15-model, 1,000-clip K400 measurements were not rerun.  They were
recomputed into a separate folder with measured top-1 accuracy, total pass
time, average power, and the legacy peak-allocation values interpreted as MiB.

| Variant | Leader | Score | Main reason |
|---|---|---:|---|
| NS-E (1/8) | Video-FocalNet-T | **70.09** | Highest measured accuracy (95.9%) with low total inference energy |
| NS-M (1/8) | UniFormer-S | **72.75** | 92.2% measured accuracy at the minimum 216 MiB peak VRAM |
| NS# (1/8) | UniFormer-S | **63.65** | Best joint time-power-memory balance |

The K400 aggregate inference-only estimate is **3,785.880 s (63.10 min)**.
Because the legacy CSV does not encode split, clip count, or power-backend
fields, those details must accompany a presentation of this table; its folder
name and prior protocol establish the 1,000-clip common-view run.

## Scope and interpretation

- The 1/8 exponent is applied independently to every selected deployment cost.
  It is intentionally soft, so the squared accuracy numerator remains
  influential; a resource minimum does not automatically win every variant.
- Rankings are meaningful within one dataset, sample, input protocol, device,
  software stack, and precision.  Do not rank K400 and SSV2 rows together.
- Only 12,767 of the official validation split's 24,777 videos are installed
  locally (51.53%).  The 1,000-video result is therefore a reproducible sample
  of the locally available validation pool, not the complete SSV2 benchmark.
- This completed results table contains four exact 174-class SSV2 checkpoints.
  TimeSformer-B and VideoMAE-B have since been added as locally available
  candidates for the new rerun, but have no device rows in this report yet.
  The local ZeroI2V SSV2 checkpoint is excluded because its required
  MMAction/MMCV inference adapter is not available; no proxy or random head was
  substituted.
- NVML reports instantaneous GPU board power.  Energy is the protocol's
  `average power × timed inference`, and peak VRAM is PyTorch maximum allocated
  memory, matching the supplied reference methodology.

## Validation and deliverables

- All four checkpoints load strictly and emit finite `(1, 174)` logits.
- All SSV2 rows have 1,000 clips, measured accuracy, positive NVML sample
  counts, finite measurements, and `PowerBackend = pynvml/NVML`.
- The automated suite passes 40/40 tests, including metric reference values,
  annotation formats, split filtering, cache recovery, exact clip counts,
  chunking, and mocked NVML cleanup.
- Full per-model tables, three separate rankings, reports, and plots are in
  `results_stage2_k400_1000_netscore/` and
  `results_stage2_ssv2_1000_netscore/`.
