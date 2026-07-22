# Stage 2 cross-dataset device report

Availability manifest: `stage2_candidate_availability.csv`.

> **Accuracy provenance:** `PublishedTop1(%)` is the canonical paper/model-zoo value under its paper-specific evaluation protocol. `MeasuredTop1Local(%)` is the local single-view device result on the clip count recorded in the source CSV. They are deliberately separate.

> **Missing-data rule:** an em dash means the value was not supplied or is not applicable. This report never copies K400 values into SSV2 rows, never substitutes a proxy checkpoint, and never ranks missing values.

> **Ranking scope:** every rank is within one dataset and only among rows carrying that score. K400 and SSV2 scores are not ranked against each other.

## Candidate and SSV2 availability matrix

| # | Model | K400 result | SSV2 availability | SSV2 result | SSV2 reason |
| --- | --- | --- | --- | --- | --- |
| 1 | UniFormer-B | measured_complete | official_pair_supported | measured_complete | Official UniFormer-B 174-class SSV2 checkpoint and configuration are released and the exact repository adapter is registered. |
| 2 | UniFormer-S | measured_complete | official_pair_supported | measured_complete | Official UniFormer-S 174-class SSV2 checkpoint and configuration are released and the exact repository adapter is registered. |
| 3 | MViT-B-24, 32x3 | measured_complete | no_exact_variant_pair | not_available | No exact supervised SSV2 checkpoint matching the selected MViT-B-24 32x3 v1 configuration is registered; MViTv2 is a different model and is not substituted. |
| 4 | MViT-B, 32x3 | measured_complete | no_exact_variant_pair | not_available | No exact supervised SSV2 checkpoint matching the selected MViT-B 32x3 v1 configuration is registered; MViTv2 is a different model and is not substituted. |
| 5 | VideoSwin-T | measured_complete | no_exact_variant_pair | not_available | The integrated Video Swin SSV2 release provides the selected Base configuration but not an exact Tiny classifier pair; no proxy is substituted. |
| 6 | VideoSwin-S | measured_complete | no_exact_variant_pair | not_available | The integrated Video Swin SSV2 release provides the selected Base configuration but not an exact Small classifier pair; no proxy is substituted. |
| 7 | VideoSwin-B | measured_complete | official_pair_supported | measured_complete | Official VideoSwin-B 174-class SSV2 checkpoint and configuration are released and the exact repository adapter is registered. |
| 8 | TimeSformer-B | measured_complete | official_pair_supported | measured_complete | Official TimeSformer-B 174-class SSV2 checkpoint and configuration are released and the exact repository adapter is registered. |
| 9 | Video-FocalNet-T | measured_complete | no_exact_variant_pair | not_available | The integrated Video-FocalNets SSV2 release provides Base but not an exact Tiny classifier pair; no proxy is substituted. |
| 10 | Video-FocalNet-S | measured_complete | no_exact_variant_pair | not_available | The integrated Video-FocalNets SSV2 release provides Base but not an exact Small classifier pair; no proxy is substituted. |
| 11 | Video-FocalNet-B | measured_complete | official_pair_supported | measured_complete | Official Video-FocalNet-B 174-class SSV2 checkpoint and configuration are released and the exact repository adapter is registered. |
| 12 | DualFormer-T | measured_complete | no_supervised_ssv2_release | not_available | No exact released and registered 174-class SSV2 classifier is available for DualFormer-T; its K400 head cannot be reused. |
| 13 | Omnivore-B | measured_complete | no_supervised_ssv2_release | not_available | No official 174-class SSV2 head is released and registered for Omnivore-B; its K400 or multitask heads are not substituted. |
| 14 | VideoMAE-B | measured_complete | official_pair_supported | measured_complete | Official VideoMAE-B 174-class SSV2 checkpoint and configuration are released and the exact repository adapter is registered. |
| 15 | VTN-B | measured_complete | no_supervised_ssv2_release | not_available | No exact released and registered 174-class SSV2 classifier is available for VTN-B; its K400 head cannot be reused. |

## NetScore leaders

| Dataset | Metric | Leader | Score | Rows ranked |
| --- | --- | --- | --- | --- |
| Kinetics-400 | NS-E(1/8) | Video-FocalNet-T | 70.06 | 15 |
| Kinetics-400 | NS-M(1/8) | UniFormer-S | 72.75 | 15 |
| Kinetics-400 | NS#(1/8) | UniFormer-S | 63.69 | 15 |
| Something-Something-v2 | NS-E(1/8) | UniFormer-S | 63.49 | 6 |
| Something-Something-v2 | NS-M(1/8) | UniFormer-B | 66.79 | 6 |
| Something-Something-v2 | NS#(1/8) | UniFormer-S | 57.66 | 6 |

## Kinetics-400 results

Coverage: 15 complete, 0 partial, 0 available but not measured, and 0 not available. Supplied clip counts: 1,000.

| Model | ResultStatus | PublishedTop1(%) | MeasuredTop1Local(%) | AccuracyDeltaLocalMinusPublished(pp) | Latency(ms/clip) | AvgPower(W) | Energy(J/clip) | PeakVRAM(MiB) | NS-E(1/8) | RankNS-EWithinDataset | NS-M(1/8) | RankNS-MWithinDataset | NS#(1/8) | RankNS#WithinDataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniFormer-B | measured_complete | 82.00 | 92.00 | 10.00 | 163.60 | 55.12 | 9.02 | 324.00 | 68.66 | 4 | 72.28 | 2 | 62.39 | 3 |
| UniFormer-S | measured_complete | 80.80 | 92.20 | 11.40 | 78.90 | 53.31 | 4.21 | 216.00 | 69.53 | 2 | 72.75 | 1 | 63.69 | 1 |
| MViT-B-24, 32x3 | measured_complete | 80.40 | 90.00 | 9.60 | 282.90 | 57.64 | 16.31 | 1050.00 | 67.64 | 10 | 70.62 | 11 | 60.09 | 11 |
| MViT-B, 32x3 | measured_complete | 80.20 | 90.80 | 10.60 | 230.60 | 56.17 | 12.96 | 855.00 | 68.04 | 9 | 70.99 | 8 | 60.71 | 7 |
| VideoSwin-T | measured_complete | 78.80 | 89.70 | 10.90 | 186.90 | 56.01 | 10.47 | 909.00 | 68.06 | 8 | 70.71 | 10 | 60.67 | 8 |
| VideoSwin-S | measured_complete | 80.60 | 93.30 | 12.70 | 307.00 | 56.04 | 17.20 | 1106.00 | 68.21 | 7 | 71.19 | 7 | 60.60 | 10 |
| VideoSwin-B | measured_complete | 80.60 | 91.70 | 11.10 | 436.20 | 56.28 | 24.55 | 1564.00 | 67.52 | 12 | 70.51 | 12 | 59.53 | 13 |
| TimeSformer-B | measured_complete | 78.00 | 92.70 | 14.70 | 170.00 | 55.53 | 9.44 | 856.00 | 68.75 | 3 | 71.35 | 6 | 61.41 | 5 |
| Video-FocalNet-T | measured_complete | 79.80 | 95.90 | 16.10 | 89.50 | 54.03 | 4.84 | 847.00 | 70.06 | 1 | 71.95 | 3 | 62.74 | 2 |
| Video-FocalNet-S | measured_complete | 81.40 | 80.70 | -0.70 | 155.30 | 55.16 | 8.57 | 1138.00 | 66.44 | 15 | 68.63 | 15 | 58.80 | 14 |
| Video-FocalNet-B | measured_complete | 83.60 | 88.60 | 5.00 | 233.30 | 56.10 | 13.09 | 1561.00 | 67.61 | 11 | 69.91 | 13 | 59.62 | 12 |
| DualFormer-T | measured_complete | 79.50 | 89.90 | 10.40 | 128.70 | 54.93 | 7.07 | 819.00 | 68.53 | 5 | 70.87 | 9 | 61.24 | 6 |
| Omnivore-B | measured_complete | 84.00 | 89.80 | 5.80 | 605.40 | 58.46 | 35.40 | 2442.00 | 66.76 | 14 | 69.66 | 14 | 58.29 | 15 |
| VideoMAE-B | measured_complete | 80.90 | 91.30 | 10.40 | 165.00 | 55.86 | 9.21 | 511.00 | 68.51 | 6 | 71.65 | 5 | 61.74 | 4 |
| VTN-B | measured_complete | 77.72 | 92.00 | 14.28 | 461.10 | 57.62 | 26.57 | 547.00 | 67.49 | 13 | 71.71 | 4 | 60.64 | 9 |

### Data-bounded observations

- Highest local measured accuracy: Video-FocalNet-T (95.90).
- Lowest supplied latency: UniFormer-S (78.90) ms/clip.
- Lowest supplied energy: UniFormer-S (4.21) J/clip.
- Lowest supplied peak VRAM: UniFormer-S (216.00) MiB.
- These observations describe only supplied local rows; absent candidates are not treated as zero and cannot win a ranking.

## Something-Something-v2 results

Coverage: 6 complete, 0 partial, 0 available but not measured, and 9 not available. Supplied clip counts: 1,000.

| Model | ResultStatus | PublishedTop1(%) | MeasuredTop1Local(%) | AccuracyDeltaLocalMinusPublished(pp) | Latency(ms/clip) | AvgPower(W) | Energy(J/clip) | PeakVRAM(MiB) | NS-E(1/8) | RankNS-EWithinDataset | NS-M(1/8) | RankNS-MWithinDataset | NS#(1/8) | RankNS#WithinDataset |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniFormer-B | measured_complete | 70.40 | 67.10 | -3.30 | 163.10 | 52.51 | 8.56 | 323.00 | 63.24 | 3 | 66.79 | 1 | 56.96 | 2 |
| UniFormer-S | measured_complete | 67.70 | 65.10 | -2.60 | 78.80 | 52.95 | 4.17 | 215.00 | 63.49 | 1 | 66.71 | 2 | 57.66 | 1 |
| MViT-B-24, 32x3 | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| MViT-B, 32x3 | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VideoSwin-T | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VideoSwin-S | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VideoSwin-B | measured_complete | 69.60 | 69.50 | -0.10 | 601.10 | 58.98 | 35.45 | 2156.00 | 62.31 | 4 | 65.35 | 4 | 53.97 | 4 |
| TimeSformer-B | measured_complete | 59.10 | 57.60 | -1.50 | 167.10 | 55.36 | 9.25 | 750.00 | 60.50 | 6 | 63.23 | 6 | 53.31 | 5 |
| Video-FocalNet-T | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Video-FocalNet-S | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Video-FocalNet-B | measured_complete | 71.10 | 61.10 | -10.00 | 228.10 | 57.44 | 13.10 | 1461.00 | 61.15 | 5 | 63.53 | 5 | 53.24 | 6 |
| DualFormer-T | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| Omnivore-B | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |
| VideoMAE-B | measured_complete | 70.80 | 68.60 | -2.20 | 190.70 | 56.76 | 10.83 | 809.00 | 63.37 | 2 | 66.18 | 3 | 56.10 | 3 |
| VTN-B | not_available | — | — | — | — | — | — | — | — | — | — | — | — | — |

### Data-bounded observations

- Highest local measured accuracy: VideoSwin-B (69.50).
- Lowest supplied latency: UniFormer-S (78.80) ms/clip.
- Lowest supplied energy: UniFormer-S (4.17) J/clip.
- Lowest supplied peak VRAM: UniFormer-S (215.00) MiB.
- These observations describe only supplied local rows; absent candidates are not treated as zero and cannot win a ranking.

## SSV2 missing or incomplete rows

| Model | DatasetAvailability | ResultStatus | AvailabilityReason | ResultNote |
| --- | --- | --- | --- | --- |
| MViT-B-24, 32x3 | no_exact_variant_pair | not_available | No exact supervised SSV2 checkpoint matching the selected MViT-B-24 32x3 v1 configuration is registered; MViTv2 is a different model and is not substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| MViT-B, 32x3 | no_exact_variant_pair | not_available | No exact supervised SSV2 checkpoint matching the selected MViT-B 32x3 v1 configuration is registered; MViTv2 is a different model and is not substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| VideoSwin-T | no_exact_variant_pair | not_available | The integrated Video Swin SSV2 release provides the selected Base configuration but not an exact Tiny classifier pair; no proxy is substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| VideoSwin-S | no_exact_variant_pair | not_available | The integrated Video Swin SSV2 release provides the selected Base configuration but not an exact Small classifier pair; no proxy is substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| Video-FocalNet-T | no_exact_variant_pair | not_available | The integrated Video-FocalNets SSV2 release provides Base but not an exact Tiny classifier pair; no proxy is substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| Video-FocalNet-S | no_exact_variant_pair | not_available | The integrated Video-FocalNets SSV2 release provides Base but not an exact Small classifier pair; no proxy is substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| DualFormer-T | no_supervised_ssv2_release | not_available | No exact released and registered 174-class SSV2 classifier is available for DualFormer-T; its K400 head cannot be reused. | No exact supervised model/dataset pair; numeric fields remain blank. |
| Omnivore-B | no_supervised_ssv2_release | not_available | No official 174-class SSV2 head is released and registered for Omnivore-B; its K400 or multitask heads are not substituted. | No exact supervised model/dataset pair; numeric fields remain blank. |
| VTN-B | no_supervised_ssv2_release | not_available | No exact released and registered 174-class SSV2 classifier is available for VTN-B; its K400 head cannot be reused. | No exact supervised model/dataset pair; numeric fields remain blank. |
