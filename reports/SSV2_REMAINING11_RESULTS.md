# SSV2 remaining 11 results

Saved 11/11 task rows. Each model has an independently saved JSON record.

Manifest: `manifests/ssv2_1000_seed0.csv`; SHA256: `c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02`.
Batch 1; FP32; seed 0; warmup 2; one repeat; requested NVML interval 20 ms.
Historical measurements are reused only after checksum, manifest, settings, head-shape, and metric consistency checks.
The six excluded models were not rerun. No training or Git push was performed.

The existing SSV2 protocol decodes 32 uniformly spaced full-video frames, resizes the short side to 224, and center crops 224x224. Adapters retain their own 8/16/32-frame selection and normalization. K400 training strides in frozen17 are provenance metadata, not the temporal sampling used by these historical SSV2 measurements. These are single-view device measurements, not paper multi-view evaluations.

Latency and its standard deviation are milliseconds; FPS is clips/second; power is W; energy is J; VRAM is peak allocated CUDA MiB. Energy = mean sampled power times timed inference duration. GFLOPs use the existing fvcore trace (one MAC = one FLOP); unsupported operators are omitted and preserved in each JSON record.

K400 classifiers are scored against official SSV2 integer labels. AccuracyValid is TRUE only for a 174-class SSV2 head.

| Model | Protocol | AccuracyValid | Top1 | Top5 | GFLOPs | Latency | FPS | Power | Energy/Clip | Total Energy | VRAM | NS# | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dualformer-t | PRETRAIN_K400_ON_SSV2 | FALSE | 0.100 | 1.000 | 60.088 | 195.620 | 5.112 | 28.048 | 5.487 | 5486.743 | 515.930 | -56.130 | COMPLETED |
| video-focalnet-t | PRETRAIN_K400_ON_SSV2 | FALSE | 0.300 | 1.300 | 63.139 | 149.293 | 6.698 | 25.599 | 3.822 | 3821.830 | 544.431 | -36.711 | COMPLETED |
| videoswin-t | PRETRAIN_K400_ON_SSV2 | FALSE | 0.100 | 1.200 | 88.062 | 277.754 | 3.600 | 31.806 | 8.834 | 8834.252 | 910.076 | -57.263 | COMPLETED |
| mvit-v1-b-16x4 | PRETRAIN_K400_ON_SSV2 | FALSE | 0.100 | 0.700 | 70.804 | 161.304 | 6.199 | 27.569 | 4.447 | 4446.928 | 354.899 | -55.495 | COMPLETED |
| video-focalnet-s | PRETRAIN_K400_ON_SSV2 | FALSE | 0.100 | 1.500 | 124.540 | 247.737 | 4.037 | 28.670 | 7.103 | 7102.532 | 837.871 | -56.936 | COMPLETED |
| mvit-v1-b-32x3 | PRETRAIN_K400_ON_SSV2 | FALSE | 0.000 | 1.100 | 170.367 | 351.302 | 2.847 | 31.226 | 10.970 | 10969.660 | 855.647 | N/A | COMPLETED |
| dualformer-s | PRETRAIN_K400_ON_SSV2 | FALSE | 0.200 | 1.100 | 158.687 | 349.250 | 2.863 | 34.919 | 12.195 | 12195.325 | 816.675 | -45.454 | COMPLETED |
| videoswin-s | PRETRAIN_K400_ON_SSV2 | FALSE | 0.100 | 1.000 | 166.125 | 395.733 | 2.527 | 37.446 | 14.819 | 14818.640 | 1008.911 | -57.937 | COMPLETED |
| zeroi2v-b16-8f | PRETRAIN_K400_ON_SSV2 | FALSE | 0.200 | 1.300 | 162.966 | 226.162 | 4.422 | 31.786 | 7.189 | 7188.719 | 474.748 | -44.292 | COMPLETED |
| dualformer-b-in21k | PRETRAIN_K400_ON_SSV2 | FALSE | 0.200 | 1.300 | 267.965 | 435.426 | 2.297 | 42.388 | 18.457 | 18456.818 | 1157.904 | -46.283 | COMPLETED |
| omnivore-b-in21k | PRETRAIN_K400_ON_SSV2 | FALSE | 0.300 | 1.200 | 320.946 | 648.793 | 1.541 | 50.565 | 32.806 | 32806.480 | 2341.701 | -40.629 | COMPLETED |

SSV2 checkpoints used: 0/11
K400 pretrained checkpoints used: 11/11
Successful device benchmarks: 11/11
Valid SSV2 accuracy models: 0/11
Failed models: 0/11

## Checkpoint audit

No exact SSV2 checkpoint for these 11 variants was identified in the local inventory and official source audit. The previous audit and source snapshots are retained in `results/ssv2_11_exact_accuracy/`.

- [dual_zoo.txt](https://github.com/sail-sg/dualformer): No exact SSV2 release found in the official model zoo or prior source audit.
- [focal_zoo.txt](https://github.com/TalalWasim/Video-FocalNets): Official SSV2 release is Base, not Tiny or Small.
- [swin_zoo.txt](https://github.com/SwinTransformer/Video-Swin-Transformer): Official SSV2 release is Base, not Tiny or Small.
- [mvit_zoo.txt](https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md): No exact MViTv1 B 16x4/32x3 SSV2 release found; MViTv2 and B-24 are different variants.
- [zero_zoo.txt](https://github.com/MCG-NJU/ZeroI2V): Official SSV2 checkpoint is L/14 with 32 frames, not B/16 with 8 frames.
- [omnivore_zoo.txt](https://github.com/facebookresearch/omnivore/tree/main/omnivore): Released IN21K Omnivore-B video head has 400 classes; local OmniMAE is a different model.

## Model notes

- `dualformer-t`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `video-focalnet-t`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `videoswin-t`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `mvit-v1-b-16x4`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `video-focalnet-s`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `mvit-v1-b-32x3`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `dualformer-s`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `videoswin-s`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `zeroi2v-b16-8f`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.
- `dualformer-b-in21k`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring. Local checkpoint is the IN1K K400 Base release, not an IN21K SSV2 classifier.
- `omnivore-b-in21k`: Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring.

## Resume

```powershell
.\venv\Scripts\python.exe -u scripts\run_ssv2_remaining11.py --run-missing
```

Successful raw results are verified and reused on resume. Failed models remain retryable. Checkpoint mismatches are recorded instead of substituting another variant.
