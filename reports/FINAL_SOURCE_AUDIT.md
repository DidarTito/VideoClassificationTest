# Final source audit

Git HEAD: `ab7330e8586bf7701f8dbd0cf7a43ad4be79b8d7`; origin/main: `ab7330e8586bf7701f8dbd0cf7a43ad4be79b8d7`.

The working tree was already dirty. No commit or push was made. The source inventory records CSV hashes before final reporting.

Controlled primary K400 source: `results/NVIDIA_GeForce_RTX_3050_Laptop_GPU/k400/full_1000_seed0/device_metrics.csv`. The two completed rows in `results/missing_k400_1h/NVIDIA_GeForce_RTX_3050_Laptop_GPU/k400/full_1000_seed0/device_metrics.csv` supply the later DualFormer-S measurement and the canonical MViT-B 32x3 alias. All imported rows explicitly record 1,000 clips, the controlled K400 manifest hash and the same RTX 3050 GPU UUID. The 10,777-clip run, smoke tests, and unrelated architectures are excluded.

The older `results/rtx3050/k400_1000_seed0_20260722/device_metrics.csv` is preserved but not silently mixed with the controlled primary source.

## Historical frozen-recipe mismatches

- `uniformer-s`: historical 16x8 checkpoint and preprocessing differ from frozen 16x4. Retained as measured historical evidence, excluded from exact rankings.
- `uniformer-b`: historical 16-frame K400 checkpoint differs from frozen 32x4. Retained as measured historical evidence, excluded from exact rankings.
- `videomae-b`: historical Hugging Face checkpoint/preprocessing differ from frozen official 1600e checkpoint. Retained as measured historical evidence, excluded from exact rankings.
- `mvit-v1-b-16x4`: historical input had 32 model frames; frozen recipe requires 16. Retained as measured historical evidence, excluded from exact rankings.
- `dualformer-b-in21k`: the local IN1K Base checkpoint and its prior measured CSV are preserved; they do not establish an exact IN21K accuracy row.
- `zeroi2v-b16-8f`: no completed compatible loader measurement; no proxy architecture used.

## Primary SSV2 evidence

- `uniformer-s`: [official model zoo](https://github.com/Sense-X/UniFormer/tree/main/video_classification#something-something-v2).
- `uniformer-b`: [official model zoo](https://github.com/Sense-X/UniFormer/tree/main/video_classification#something-something-v2).
- `videomae-b`: [official model zoo](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#something-something-v2).
- `videoswin-b`: [official model zoo](https://github.com/SwinTransformer/Video-Swin-Transformer#something-something-v2).
- `timesformer-b`: [official model zoo](https://github.com/facebookresearch/TimeSformer#model-zoo).
- `video-focalnet-b`: [official model zoo](https://github.com/TalalWasim/Video-FocalNets#model-zoo).
- `mvit-v1-b-16x4`: Top-1 64.7, Top-5 89.2; [original paper, Table 6](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf).
- `mvit-v1-b-32x3`: Top-1 67.1, Top-5 90.8; [original paper, Table 6](https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf).
- `omnivore-b-in21k`: Top-1 71.4, Top-5 93.5; [original paper, Table 6](https://openaccess.thecvf.com/content/CVPR2022/papers/Girdhar_Omnivore_A_Single_Model_for_Many_Visual_Modalities_CVPR_2022_paper.pdf).
