# Final K400 table

Measured Top-1 on fixed local K400 1000-clip subset. Published K400 Top-1 is not zero-shot accuracy. Ranks include only exact frozen recipes; four historical recipe mismatches retain their measurements but have no rank. The IN1K DualFormer Base release is not substituted for IN21K. Source paths, hashes and hardware are in the master CSV.

| Rank by NS# | Model | Published K400 Top-1 | Measured K400 Top-1 | GFLOPs | Latency (ms) | Power (W) | Energy/Clip (J) | Total Energy (J) | Peak VRAM (MiB) | NS-E | NS-M | NS# | Evidence status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N/A | uniformer-s | 80.8 | 92.2 | 167 | 132.75 | 25.217 | 3.348 | 3347.534 | 214.675 | 69.777 | 72.76 | 63.948 | MEASURED_K400_RECIPE_MISMATCH |
| 3 | dualformer-t | 79.5 | 89.9 | 60.088 | 196.294 | 32.057 | 6.293 | 6292.602 | 516.805 | 68.653 | 71.367 | 61.87 | MEASURED_EXACT_K400 |
| 1 | video-focalnet-t | 79.8 | 95.9 | 63.139 | 146.739 | 29.373 | 4.31 | 4310.144 | 543.629 | 70.187 | 72.434 | 63.348 | MEASURED_EXACT_K400 |
| 7 | videoswin-t | 78.8 | 89.7 | 88 | 278.245 | 34.035 | 9.47 | 9470.003 | 909.273 | 68.171 | 70.715 | 60.774 | MEASURED_EXACT_K400 |
| N/A | mvit-v1-b-16x4 | 78.4 | 85.4 | 70.5 | 163.287 | 30.226 | 4.935 | 4935.46 | 353.774 | 68.025 | 70.886 | 61.653 | MEASURED_K400_RECIPE_MISMATCH |
| 10 | video-focalnet-s | 81.4 | 80.7 | 124.54 | 249.961 | 30.853 | 7.712 | 7711.958 | 836.271 | 66.557 | 68.969 | 59.251 | MEASURED_EXACT_K400 |
| 5 | mvit-v1-b-32x3 | 80.2 | 90.8 | 170.367 | 223.821 | 52.681 | 11.791 | 11791.007 | 855.647 | 68.145 | 70.993 | 60.814 | MEASURED_EXACT_K400 |
| 4 | dualformer-s | 80.6 | 91.9 | 158.687 | 245.618 | 55.957 | 13.744 | 13744.001 | 816.675 | 68.187 | 71.252 | 60.907 | MEASURED_EXACT_K400 |
| 6 | videoswin-s | 80.6 | 93.3 | 166 | 395.849 | 39.83 | 15.767 | 15766.69 | 1006.771 | 68.301 | 71.288 | 60.794 | MEASURED_EXACT_K400 |
| N/A | zeroi2v-b16-8f | 83 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NO_EXACT_K400_MEASUREMENT |
| 8 | video-focalnet-b | 83.6 | 88.6 | 149 | 335.172 | 34.947 | 11.713 | 11713.102 | 1266.519 | 67.726 | 70.141 | 59.969 | MEASURED_EXACT_K400 |
| N/A | uniformer-b | 82 | 92 | 387 | 272.249 | 30.373 | 8.269 | 8268.974 | 324.595 | 68.758 | 72.273 | 62.48 | MEASURED_K400_RECIPE_MISMATCH |
| N/A | videomae-b | 80.9 | 91.3 | 180 | 286.711 | 30.185 | 8.654 | 8654.255 | 411.837 | 68.576 | 71.882 | 62.039 | MEASURED_K400_RECIPE_MISMATCH |
| N/A | dualformer-b-in21k | 82.9 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NO_EXACT_K400_MEASUREMENT |
| 11 | omnivore-b-in21k | 84 | 89.8 | 320.946 | 659.483 | 51.993 | 34.289 | 34288.723 | 2142.583 | 66.793 | 69.804 | 58.466 | MEASURED_EXACT_K400 |
| 9 | videoswin-b | 80.6 | 91.7 | 282 | 499.655 | 47.341 | 23.654 | 23654.013 | 1365.554 | 67.56 | 70.657 | 59.722 | MEASURED_EXACT_K400 |
| 2 | timesformer-b | 78 | 92.7 | 196 | 263.712 | 34.009 | 8.969 | 8968.695 | 550.136 | 68.801 | 71.832 | 61.95 | MEASURED_EXACT_K400 |

Results describe fixed local 1,000-clip subsets and one uniformly sampled full-video view, not full official validation. Paper protocols can use multiple views, different temporal sampling, resolution, preprocessing or checkpoint recipes; no specific discrepancy cause is proven. K400 and SSV2 are different tasks, label spaces, checkpoints and training recipes: accuracy differences are not pure transfer degradation. Measurements were taken in separate runs with one repeat and no confidence intervals; thermal conditions and other GPU activity may differ. FPS means clips/s, not decoded frames/s. Power is whole-GPU NVML power; energy is power times timed adapter inference duration. Runtime GFLOPs use fvcore with one multiply-add counted as one FLOP; unsupported operators are omitted and listed in GFLOPsSource. Where an exact K400 graph and frame count have a separate runtime trace, its input-shape-invariant FLOP count is reused with provenance; otherwise historical K400 FLOPs have inconsistent view conventions and are retained as metadata, not compared as uniform runtime profiles. VideoMAE's SSV2 2400-epoch model zoo recipe uses no external pretraining data; it is not asserted to be K400-to-SSV2 transfer.
