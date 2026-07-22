# RTX 3050 experiment notes - 2026-07-22

## Completed scope

- Device: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB.
- Protocol: batch size 1, one uniformly sampled full-video view, 32 decoded
  frames at 224 px, model-specific temporal resampling, seed 0.
- K400: 15/15 exact runnable classifier pairs, 1,000 identical cached clips
  per model, 3,694.5 s inference-only total.
- SSV2: 6/6 released and integrated exact classifier pairs, 1,000 identical
  clips selected from the official validation ID list, 1,428.9 s
  inference-only total.
- SSV2 missing set: 9/15 K400-runnable candidates have no exact released and
  integrated supervised SSV2 pair. Their numeric cells remain empty.

Raw device tables and rankings are in the adjacent dated K400 and SSV2
directories. The 30-row availability-aware comparison is in
`cross_dataset_1000_seed0_20260722`.

## Validity corrections made during the rerun

The first K400 diagnostic pass exposed that the generated `.clip_cache_*`
directory was being counted as a 401st class by the shared alphabetical label
helper. This shifted predictions for most K400 adapters and caused both MViT
adapters to reject the vocabulary. The helper now excludes hidden artifact
directories, requires exactly 400 visible class folders, and has regression
tests. The invalid diagnostic CSV was overwritten by a clean cached rerun in
which all 15 rows succeeded and all deployment NetScores are finite.

## Accuracy scope

The local K400 collection is a reproducible folder-labeled 1,000-clip subset,
not a claim of canonical K400 validation accuracy. Its unusually high values
must not replace paper/model-zoo accuracy. The SSV2 sample is split-safe and
drawn from official validation IDs, but it is still a single-view 1,000-clip
subset rather than each paper's multi-view protocol. Published and local
measured accuracy therefore remain separate columns.

## Thermal observation

During the long sequential passes, `nvidia-smi` reported software thermal
slowdown at 86-87 C. The tables are honest sustained-load measurements, but
model order and temperature can confound small latency differences. Before a
paper-level comparison, repeat each device experiment with randomized model
order, a fixed pre-row temperature/cooldown rule, and at least three seeds or
replicates. Large deployment gaps remain informative; small gaps should not
be treated as decisive from this single run.

## Intermediate conclusions

- UniFormer-S is the best balanced deployment choice on this RTX 3050: it has
  the best NS# on K400 (63.69) and SSV2 (57.66), and the lowest latency,
  energy, and peak VRAM in both measured sets.
- K400 Video-FocalNet-T has the highest local measured accuracy (95.9%) and
  best NS-E (70.06). Compared with UniFormer-S it uses 3.8 times the peak VRAM
  (847 vs 216 MiB), so UniFormer-S remains stronger when memory matters.
- SSV2 VideoSwin-B has the highest local accuracy (69.5%) but costs 601.1 ms,
  35.452 J/clip, and 2,156 MiB. VideoMAE-B is only 0.9 percentage points lower
  at 68.6%, while using 190.7 ms, 10.826 J, and 809 MiB; it is the stronger
  accuracy-efficiency compromise among the larger models.
- SSV2 TimeSformer-B is reasonably fast (167.1 ms) but its 57.6% local
  accuracy yields NS# 53.31. Video-FocalNet-B reaches 61.1% but has the lowest
  SSV2 NS# (53.24) because of higher latency, energy, and memory.
- GPU power lies in a relatively narrow band (roughly 52.5-59.0 W on SSV2),
  so accuracy, latency, and memory drive most practical separation here.
- The 10-point gap between Video-FocalNet-B's published SSV2 accuracy and this
  local single-view result should trigger a preprocessing/multi-view protocol
  audit; it is not evidence that the published result is wrong.

## Recommended next measurement design

1. Use the same split manifest and decoded-cache fingerprint on every device.
2. Record driver, CUDA/JetPack, PyTorch, precision, clocks, and temperature.
3. Randomize model order across repetitions and enforce a temperature rule.
4. Report median and dispersion across at least three repetitions/seeds.
5. Keep accuracy scope explicit and never rank K400 and SSV2 scores together.
6. Recompute NS-E, NS-M, and NS# independently per device and dataset.
