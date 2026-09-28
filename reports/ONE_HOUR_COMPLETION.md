# One-Hour Benchmark Completion

Generated from the historical RTX 3050 CSVs and the new `missing_k400_1h` CSV; blocked rows contain no invented measurements.

## Environment and Dataset Checks

- Device: NVIDIA GeForce RTX 3050 Laptop GPU; PyTorch 2.6.0+cu124; CUDA 12.4; driver 595.97; 4,095.5 MiB VRAM; compute capability 8.6.
- K400 manifest: SHA256 `b8ec0d92f50b4ae5dff46754f62c2ef1b2b8467332f88bb0cc29c8e6172ff749`; 1,000 rows; 400 canonical class directories (the extra directory is the ignored clip cache).
- SSV2 manifest: SHA256 `c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02`; 1,000 rows; existing validation annotations were used for the historical six rows.

## K400

| Model | ModelKey | Status | MeasuredTop1(%) | PublishedTop1(%) | Latency(ms) | AvgPower(W) | EnergyPerClip(J) | TotalEnergy(J) | PeakVRAM(MiB) | NS-E | NS-M | NS# | Checkpoint | SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MViT-B, 32x3 | mvit-v1-b | COMPLETED | 90.8 | 80.2 | 230.6 | 56.17 | 12.955 | 12954.64 | 855.0 | 68.04 | 70.99 | 60.71 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\mvit\MVIT_B_32x3_f294077834.pyth | ee62dd4c41f4cf0622a690c55bb3fea5c5685443cf0b4cb444327961047e5064 |
| DualFormer-S | dualformer-s | COMPLETED | 91.9 | 80.6 | 245.61842410007375 | 55.956718329963124 | 13.744000974017254 | 13744.000974017254 | 816.6748046875 | 68.187337512282 | 71.252497562108 | 60.90721461894554 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\dualformer\dualformer_small_patch244_window877.pth | 1a377476e1244c86fdf0f2bb639ac88906e85ad328d951fc648567ce0ba17cce |
| ZeroI2V ViT-B/16, 8f | zeroi2v-b16-8f | BLOCKED_LOADER |  |  |  |  |  |  |  |  |  |  | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\ZeroI2V\CLIP-B_k400_8x16x1.pth | f62f2e10aa0ae0cb755046f9d84b5ecffd58d4a5f1ae043ace88c3b27621929f |
| DualFormer-B (IN21K) | dualformer-b-in21k | BLOCKED_PROVENANCE |  |  |  |  |  |  |  |  |  |  | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\dualformer\dualformer_base_patch244_window877.pth | 6d7414ca4ff1d010cddb8ebb8c05594921dafd1165da7811bb68a3c6bb679eca |

New K400 rows completed: 1/4 requested priority rows. `mvit-v1-b` already existed; DualFormer-S is newly completed; ZeroI2V and DualFormer-B-IN21K remain blocked.

## SSV2

| Model | ModelKey | Status | MeasuredTop1(%) | PublishedTop1(%) | Latency(ms) | AvgPower(W) | EnergyPerClip(J) | TotalEnergy(J) | PeakVRAM(MiB) | NS-E | NS-M | NS# | Checkpoint | SHA256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| UniFormer-S | uniformer-s | COMPLETED | 65.1 | 67.7 | 78.8 | 52.95 | 4.173 | 4173.25 | 215.0 | 63.49 | 66.71 | 57.66 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\legacy_ssv2_released\uniformer_small_sthv2_16_prek400.pth | 4f5df1bf569e7e0ead9ae9ea43ecef5a852ef14dcf67575e36aa3ce8edd0ed9f |
| UniFormer-B | uniformer-b | COMPLETED | 67.1 | 70.4 | 163.1 | 52.51 | 8.563 | 8563.34 | 323.0 | 63.24 | 66.79 | 56.96 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\legacy_ssv2_released\uniformer_base_sthv2_16_prek400.pth | b3091fbfd2b47845988831599a0325c426b6ec84aee6a545c152fafe84adeffe |
| VideoMAE-B | videomae-b | COMPLETED | 68.6 | 70.8 | 190.7 | 56.76 | 10.826 | 10825.76 | 809.0 | 63.37 | 66.18 | 56.1 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\legacy_ssv2_released\videomae_vit_b_ssv2_2400e.pth | 25f4060e564ca716ff1f1804f5aba92f9240da723e22b383e306273ca91de377 |
| VideoSwin-B | videoswin-b | COMPLETED | 69.5 | 69.6 | 601.1 | 58.98 | 35.452 | 35452.26 | 2156.0 | 62.31 | 65.35 | 53.97 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\legacy_ssv2_released\swin_base_patch244_window1677_sthv2.pth | f7fe417a0659680ab9db4f5e02c39b3f2825b58f0e244d4fbe3de6658fe135fc |
| TimeSformer-B | timesformer-b | COMPLETED | 57.6 | 59.1 | 167.1 | 55.36 | 9.251 | 9250.83 | 750.0 | 60.5 | 63.23 | 53.31 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\legacy_ssv2_released\TimeSformer_divST_8_224_SSv2.pyth | 7a606efdda9345f1e2d83e10cc2b03196edf157ff7e9a07a19368352614109c4 |
| Video-FocalNet-B | video-focalnet-b | COMPLETED | 61.1 | 71.1 | 228.1 | 57.44 | 13.103 | 13102.67 | 1461.0 | 61.15 | 63.53 | 53.24 | C:\Users\DIDAR\Documents\VideoClassificationTest\checkpoints\legacy_ssv2_released\video-focalnet_base_ssv2.pth | 187e4e9469de75ae36252d07418d9802dde16fc3a7beb9815ec949eff25dd9d6 |
| DualFormer-T | dualformer-t | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| Video-FocalNet-T | video-focalnet-t | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| VideoSwin-T | videoswin-t | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| MViT-B, 16x4 | mvit-v1-b-16x4 | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| Video-FocalNet-S | video-focalnet-s | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| MViT-B, 32x3 | mvit-v1-b-32x3 | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| DualFormer-S | dualformer-s | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| VideoSwin-S | videoswin-s | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| ZeroI2V ViT-B/16, 8f | zeroi2v-b16-8f | BLOCKED_LOADER |  |  |  |  |  |  |  |  |  |  |  |  |
| DualFormer-B (IN21K) | dualformer-b-in21k | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |
| Omnivore-B (IN21K) | omnivore-b-in21k | NO_RELEASED_EXACT_SSV2_CHECKPOINT |  |  |  |  |  |  |  |  |  |  |  |  |

New SSV2 rows completed: 0/11. The six existing measured rows were preserved and not rerun.

## Explicit Blockers

- `NO_RELEASED_EXACT_SSV2_CHECKPOINT`: exact Tiny/Small variants for DualFormer, Video-FocalNet, and VideoSwin are not in the official model tables; MViT v1 SSV2 classifier releases were not found; Omnivore has no exact frozen SSV2 classifier release; DualFormer-B-IN21K is not the published K400 Base checkpoint.
- `BLOCKED_LOADER`: ZeroI2V has a local K400 checkpoint, but its exact adapter requires the official ZeroI2V MMAction2 plugin/runtime, which is not available in this Windows environment. The official project documents SSV2 ViT-L/14 32-frame, not the frozen ViT-B/16 8-frame key.
- Existing SSV2 rows are released external baselines, not controlled fine-tuned final checkpoints.

The merged machine-readable table is `results/master_benchmark_results.csv`.
