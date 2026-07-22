# Stage 1 thresholds: why these three cutoffs?

> **Decision:** use the thresholds as an empirical, deployment-oriented *screen*, then make the deployment decision from measured RTX 3050 latency, peak VRAM, power, energy, and accuracy. They are not universal hardware limits.

## Slide 1 — Rationale for the screening gates

| Gate | Why this value? | Evidence in the 36-model K400 candidate set |
|---|---|---|
| **Top-1 ≥ 78%** | Establishes a non-negotiable K400 quality floor before rewarding efficiency. `78.0%` is the lower edge of the competitive transformer cluster in this survey and keeps the original TimeSformer baseline, whose paper reports 78.0% K400 top-1. | There is an empirical gap from 76.0% to 78.0%; the gate retains 34/36 candidates and removes only the two sub-78% models. It is deliberately permissive, so a low-cost model cannot win solely by sacrificing too much accuracy. |
| **GFLOPs ≤ 300** | Keeps the paper-reported compute of one candidate configuration in the lower-compute cluster while preserving strong models close to the boundary. | Among accuracy- and parameter-qualified candidates, 282 GFLOPs is the largest value below the cutoff and the next value is 369.5 GFLOPs. Thus 300 lies in a natural 87.5-GFLOP gap. Video Swin-B demonstrates that the retained edge is still useful: 80.6% top-1, 282 GFLOPs, and 88.1M parameters in its reference table. |
| **Parameters < 300M** | Limits static weight storage and leaves substantial room in a 4GB GPU for activations, framework state, inputs, and temporary workspaces. At 300M parameters, weights alone occupy about **1.12 GiB in FP32** or **0.56 GiB in FP16**. | The cutoff also separates the surveyed sub-300M group from the next large-model group beginning at 304M. The target machine reports an RTX 3050 Laptop GPU with 4096 MiB VRAM; NVIDIA lists RTX 3050 Laptop variants with 4GB or 6GB, so this study is specifically about the installed 4GB device. |

**Result:** the intersection retains **17 of 36** candidates for device measurement. The canonical published-configuration rows are in [`data/stage1_candidates_k400.csv`](data/stage1_candidates_k400.csv).

## Slide 2 — What the thresholds do *not* guarantee

- **78% is a study-specific quality floor, not a universal definition of “good.”** Published K400 values can use different pretraining, clips, crops, and test views. Final comparison therefore uses the common local evaluation protocol.
- **300 GFLOPs is not a latency limit.** FLOPs omit memory-access cost, parallelism, kernel efficiency, and hardware/software effects; models with similar FLOPs can have different wall-clock latency.
- **300M parameters is not an out-of-memory boundary.** Parameter storage is only part of peak VRAM. A smaller model may still exceed 4GB because of activations or workspaces, while a larger model may fit with lower precision, quantization, or offloading.
- **Therefore Stage 1 rejects obvious mismatches; Stage 2 establishes deployability.** Peak VRAM, latency, power, and energy must be measured on the exact RTX 3050, batch size, precision, frame count, and inference-view protocol.

This qualification follows *The Efficiency Misnomer*: no single proxy such as FLOPs or parameter count is sufficient for practical efficiency, and low FLOPs do not necessarily imply low latency.

## Primary and official sources

- Bertasius, Wang, and Torresani, [*Is Space-Time Attention All You Need for Video Understanding?*](https://proceedings.mlr.press/v139/bertasius21a.html), ICML 2021; the [paper PDF](https://proceedings.mlr.press/v139/bertasius21a/bertasius21a.pdf) reports 78.0% K400 top-1 for TimeSformer.
- Liu et al., [*Video Swin Transformer*](https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Video_Swin_Transformer_CVPR_2022_paper.pdf), CVPR 2022; its K400 table reports the 80.6% / 282-GFLOP / 88.1M-parameter retained-boundary example and explicitly lists its test-view protocol.
- Dehghani et al., [*The Efficiency Misnomer*](https://openreview.net/pdf?id=uQShkCh8C97), ICLR 2022.
- NVIDIA, [GeForce RTX 30-Series Laptop GPU specifications](https://www.nvidia.com/en-us/geforce/laptops/30-series/), listing RTX 3050 Laptop GPU memory configurations.

*Implementation note:* the slide and `screen_models.py` now both use
`GFLOPs ≤ 300`. No surveyed candidate is exactly 300 GFLOPs, so synchronizing
the operator does not change the selected 17-model set.

## Reproducing the Stage 1 figure

Run `python scripts/plot_stage1_screening.py`. It validates all 36 rows,
recomputes the classic NetScore through `metrics.netscore`, checks the exact
17-model selection, and writes 600-dpi PNG plus vector SVG/PDF versions to
`figures/`. The figure uses published Kinetics-400 literature configurations;
it does not substitute the device-run accuracy or checkpoint-specific FLOPs.
