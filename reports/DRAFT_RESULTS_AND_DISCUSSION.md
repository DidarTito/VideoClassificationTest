# Draft Results and Discussion

## Experimental Setup

We evaluated frozen checkpoints on an RTX 3050 Laptop GPU using deterministic 1,000-video manifests, batch size 1, FP32, seed 0, 224-pixel inputs, two warmups and one repeat. Each common input contained 32 decoded frames, temporally resampled by the model adapter. NVML power sampling used a 20 ms interval. Host conversion and decoding were excluded from latency; adapter normalization, transfer and forward were included. Classification used the original dataset-specific heads. NS-E, NS-M and NS# used measured accuracy in percentage points, total timed inference seconds, watts and MiB.

## K400 Results

- highest measured Top-1: **video-focalnet-t**, 95.9 (MeasuredTop1).
- lowest latency: **video-focalnet-t**, 146.739 (LatencyMs).
- lowest average power: **video-focalnet-t**, 29.373 (PowerW).
- lowest energy/clip: **video-focalnet-t**, 4.31 (EnergyPerClipJ).
- lowest VRAM: **dualformer-t**, 516.805 (PeakVRAMMiB).
- highest NS-E: **video-focalnet-t**, 70.187 (NS-E).
- highest NS-M: **video-focalnet-t**, 72.434 (NS-M).
- highest NS#: **video-focalnet-t**, 63.348 (NS#).

The lowest-latency model, video-focalnet-t, has 1.000x the throughput of the highest-accuracy model, video-focalnet-t; their measured Top-1 difference is 0.000 percentage points. video-focalnet-t uses 1.000x less energy per clip than video-focalnet-t.

## SSV2 Accuracy Results

- highest measured Top-1: **uniformer-b**, 67.1 (MeasuredTop1).
- lowest latency: **uniformer-s**, 78.66 (LatencyMs).
- lowest average power: **uniformer-s**, 40.341 (PowerW).
- lowest energy/clip: **uniformer-s**, 3.173 (EnergyPerClipJ).
- lowest VRAM: **uniformer-s**, 214.231 (PeakVRAMMiB).
- highest NS-E: **uniformer-s**, 63.789 (NS-E).
- highest NS-M: **uniformer-b**, 66.792 (NS-M).
- highest NS#: **uniformer-s**, 57.962 (NS#).

The lowest-latency model, uniformer-s, has 2.066x the throughput of the highest-accuracy model, uniformer-b; their measured Top-1 difference is 2.000 percentage points. uniformer-s uses 2.756x less energy per clip than uniformer-b.

The smallest absolute SSV2 paper-to-measured difference is uniformer-s: -2.6 percentage points (measured minus published).

## Accuracy–Efficiency Trade-offs

The measured deployment profiles above separate maximum accuracy from minimum latency, energy and memory. NS# provides a specified combined score rather than a universal best model. Full measured values and equations are in the master table and validation report.

## Model Family Scaling

| Family | Dataset | Baseline | Compared | LatencyMsBaseline | LatencyMsCompared | LatencyMsMultiplier | EnergyPerClipJMultiplier | PeakVRAMMiBMultiplier | RuntimeParamsMMultiplier | AccuracyChangePP | Limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniFormer | ssv2 | uniformer-s | uniformer-b | 78.66 | 162.51 | 2.066 | 2.756 | 1.513 | 2.335 | 2 | single runs; recipe/pretraining and run conditions may differ |
| DualFormer | k400 | dualformer-t | dualformer-s | 196.294 | 245.618 | 1.251 | 2.184 | 1.58 | 2.24 | 2 | single runs; recipe/pretraining and run conditions may differ |
| DualFormer | ssv2_device_only | dualformer-t | dualformer-s | 129.223 | 247.328 | 1.914 | 2.381 | 1.579 | 2.24 | N/A | single runs; recipe/pretraining and run conditions may differ |
| DualFormer | ssv2_device_only | dualformer-t | dualformer-b-in21k | 129.223 | 351.88 | 2.723 | 3.469 | 2.243 | 3.971 | N/A | IN1K Base artifact; not exact frozen IN21K |
| Video-FocalNet | k400 | video-focalnet-t | video-focalnet-s | 146.739 | 249.961 | 1.703 | 1.789 | 1.538 | 1.791 | -15.2 | single runs; recipe/pretraining and run conditions may differ |
| Video-FocalNet | k400 | video-focalnet-t | video-focalnet-b | 146.739 | 335.172 | 2.284 | 2.718 | 2.33 | 3.176 | -7.3 | single runs; recipe/pretraining and run conditions may differ |
| Video-FocalNet | ssv2_device_only | video-focalnet-t | video-focalnet-s | 87.418 | 149.924 | 1.715 | 1.859 | 1.538 | 1.791 | N/A | single runs; recipe/pretraining and run conditions may differ |
| MViT | ssv2_device_only | mvit-v1-b-16x4 | mvit-v1-b-32x3 | 96.926 | 224.36 | 2.315 | 2.538 | 2.415 | 1 | N/A | single runs; recipe/pretraining and run conditions may differ |
| VideoSwin | k400 | videoswin-t | videoswin-s | 278.245 | 395.849 | 1.423 | 1.665 | 1.107 | 1.769 | 3.6 | single runs; recipe/pretraining and run conditions may differ |
| VideoSwin | k400 | videoswin-t | videoswin-b | 278.245 | 499.655 | 1.796 | 2.498 | 1.502 | 3.127 | 2 | single runs; recipe/pretraining and run conditions may differ |

## Cross-Dataset Observations

| Model | K400Top1 | SSV2Top1 | SSV2MinusK400PP | SSV2/K400Latency | SSV2/K400Energy | SSV2/K400VRAM | Comparability |
| --- | --- | --- | --- | --- | --- | --- | --- |
| uniformer-s | 92.2 | 65.1 | -27.1 | 0.593 | 0.948 | 0.998 | dataset-specific recipes; historical K400 recipe mismatch |
| uniformer-b | 92 | 67.1 | -24.9 | 0.597 | 1.057 | 0.999 | dataset-specific recipes; historical K400 recipe mismatch |

## Checkpoint Availability

The frozen17 shortlist was selected before considering SSV2 checkpoint availability. Six exact variants have genuine local released SSV2 classifiers; three have verified published SSV2 values but no exact local checkpoint; eight lack exact SSV2 accuracy evidence in the audited releases. No proxy size or architecture supplies any SSV2 accuracy row. K400_ON_SSV2_INPUT rows are excluded from SSV2 accuracy rankings and all NS scores. This is a methodological availability limitation; missing values remain N/A.

## Limitations

Largest absolute paper-to-measured discrepancies in the recorded settings: k400/video-focalnet-t +16.100 percentage points; k400/timesformer-b +14.700 percentage points; k400/videoswin-s +12.700 percentage points; k400/uniformer-s +11.400 percentage points. These are protocol-specific discrepancies, not proof of a particular cause.

Results describe fixed local 1,000-clip subsets and one uniformly sampled full-video view, not full official validation. Paper protocols can use multiple views, different temporal sampling, resolution, preprocessing or checkpoint recipes; no specific discrepancy cause is proven. K400 and SSV2 are different tasks, label spaces, checkpoints and training recipes: accuracy differences are not pure transfer degradation. Measurements were taken in separate runs with one repeat and no confidence intervals; thermal conditions and other GPU activity may differ. FPS means clips/s, not decoded frames/s. Power is whole-GPU NVML power; energy is power times timed adapter inference duration. Runtime GFLOPs use fvcore with one multiply-add counted as one FLOP; unsupported operators are omitted and listed in GFLOPsSource. Where an exact K400 graph and frame count have a separate runtime trace, its input-shape-invariant FLOP count is reused with provenance; otherwise historical K400 FLOPs have inconsistent view conventions and are retained as metadata, not compared as uniform runtime profiles. VideoMAE's SSV2 2400-epoch model zoo recipe uses no external pretraining data; it is not asserted to be K400-to-SSV2 transfer.

## Intermediate Conclusions

The current package contains 2/17 fresh genuine SSV2 measurements and 11/17 exact-compatible historical K400 rows. Three additional SSV2 rows are published-only evidence; unavailable accuracy and incompatible frozen recipes are not substituted. Quantified selections are conditional on this hardware and protocol.
