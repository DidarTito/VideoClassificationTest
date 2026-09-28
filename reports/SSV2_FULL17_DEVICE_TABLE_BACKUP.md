Models without released exact SSV2 checkpoints were evaluated using their exact K400 checkpoints on the same SSV2 video inputs for device-performance metrics only. Their classification accuracy is not reported because the K400 classifier predicts 400 Kinetics classes rather than the 174 Something-Something V2 classes.

# SSV2 full17 device performance

Fixed input: `manifests/ssv2_1000_seed0.csv`; 1,000 clips; batch 1; FP32; seed 0; warmup 2; repeat 1; NVML at 20 ms. The six genuine SSV2 classification rows are preserved from the historical CSV without rerunning. `N/A` means unavailable or inapplicable; pending/failed rows have no invented device values.

FPS is clips/second (1,000 / latency in milliseconds). 32 frames are decoded; ModelFrames is the temporal input to each model. Energy is average sampled GPU power multiplied by measured inference time. Peak VRAM is peak allocated CUDA memory. New GFLOPs values use an fvcore trace of the loaded inference graph (one multiply-add counted as one FLOP); unsupported operators are listed per row in the CSV and are excluded from the count. Historical GFLOPs remain unchanged source metadata.

The `dualformer-b-in21k` key uses the existing official IN1K K400 Base release specified in frozen17; its checkpoint is not relabeled as IN21K. The checkpoint path and actual SHA256 identify the artifact.

| Model | CheckpointProtocol | Checkpoint | CheckpointSHA256 | PublishedSSV2Top1 | MeasuredSSV2Top1 | LatencyMs | PowerW | EnergyPerClipJ | TotalEnergyJ | PeakVRAMMiB | FPS | GFLOPs | RuntimeParamsM | ModelFrames | AccuracyValid | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uniformer-s | GENUINE_SSV2 | checkpoints/legacy_ssv2_released/uniformer_small_sthv2_16_prek400.pth | 4f5df1bf569e7e0ead9ae9ea43ecef5a852ef14dcf67575e36aa3ce8edd0ed9f | 67.7 | 65.1 | 78.8 | 52.95 | 4.173 | 4173.25 | 215.0 | 12.69 | 125.0 | 21.284462 | 16 | TRUE | PRESERVED_GENUINE_SSV2 |
| dualformer-t | K400_ON_SSV2_INPUT | checkpoints/dualformer/dualformer_tiny_patch244_window877.pth | f27c1c1996e4d27bc5c71f2bf3f690d933d8ea378030fb9e9e93c816872deb46 | N/A | N/A | 129.2233446000173 | 44.961387868535525 | 5.810060918230803 | 5810.0609182308035 | 515.93017578125 | 7.738539836554162 | 60.088415744 | 21.97256 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| video-focalnet-t | K400_ON_SSV2_INPUT | checkpoints/focalnet/video-focalnet_tiny_kinetics400.pth | 86fb70adbd28e0fc8c9ca8c7ed36d52482a0c58507ad15820c8849bfb634b9ed | N/A | N/A | 87.41757250003138 | 49.60398533619456 | 4.336259984417282 | 4336.259984417282 | 543.595703125 | 11.439347620864684 | 63.138975744 | 49.553272 | 8 | FALSE | COMPLETED_DEVICE_ONLY |
| videoswin-t | K400_ON_SSV2_INPUT | checkpoints/videoswin/swin_tiny_patch244_window877_kinetics400_1k.pth | 9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc | N/A | N/A | 183.19355720011663 | 54.87350463758159 | 10.052472510595667 | 10052.472510595668 | 910.07568359375 | 5.45870725632355 | 88.06219776 | 28.15807 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| mvit-v1-b-16x4 | K400_ON_SSV2_INPUT | checkpoints/mvit/MVIT_B_16x4.pyth | f2771e3919cd9339f4301f25476e91be26b2ebfdf11b633df2701e4867283007 | N/A | N/A | 96.9257278998557 | 50.33533009394234 | 4.878788508434874 | 4878.788508434874 | 353.7744140625 | 10.317178128733854 | 70.803884928 | 36.610672 | 16 | FALSE | COMPLETED_DEVICE_ONLY |
| video-focalnet-s | K400_ON_SSV2_INPUT | checkpoints/focalnet/video-focalnet_small_kinetics400.pth | 5031b6372b6a91b7a0be4762bc58c8d47318896a91a3f4fe0a9730d28cef4bd2 | N/A | N/A | 149.92421900018235 | 53.781380567640355 | 8.063131478355062 | 8063.131478355063 | 835.7919921875 | 6.670036413521578 | 124.539949056 | 88.735168 | 8 | FALSE | COMPLETED_DEVICE_ONLY |
| mvit-v1-b-32x3 | K400_ON_SSV2_INPUT | checkpoints/mvit/MVIT_B_32x3_f294077834.pyth | ee62dd4c41f4cf0622a690c55bb3fea5c5685443cf0b4cb444327961047e5064 | N/A | N/A | 224.3602768001001 | 55.18541746494516 | 12.381415537764175 | 12381.415537764175 | 854.5224609375 | 4.457116982838175 | 170.366942592 | 36.61144 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| dualformer-s | K400_ON_SSV2_INPUT | checkpoints/dualformer/dualformer_small_patch244_window877.pth | 1a377476e1244c86fdf0f2bb639ac88906e85ad328d951fc648567ce0ba17cce | N/A | N/A | 247.3283483999403 | 55.93848616155291 | 13.835173394329797 | 13835.173394329797 | 814.4873046875 | 4.043208174353545 | 158.686847232 | 49.215664 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| videoswin-s | K400_ON_SSV2_INPUT | checkpoints/videoswin/swin_small_patch244_window877_kinetics400_1k.pth | 7e05aa13cfa1aa99fe78c1ccac96c263344b315fe5d98a165b77b1f7532ce500 | N/A | N/A | 297.705222800163 | 56.882867746717494 | 16.93432681604874 | 16934.32681604874 | 1007.78564453125 | 3.3590273982907513 | 166.124814336 | 49.816678 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| zeroi2v-b16-8f | K400_ON_SSV2_INPUT | checkpoints/ZeroI2V/CLIP-B_k400_8x16x1.pth | f62f2e10aa0ae0cb755046f9d84b5ecffd58d4a5f1ae043ace88c3b27621929f | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FALSE | SKIPPED_FAILED |
| video-focalnet-b | GENUINE_SSV2 | checkpoints/legacy_ssv2_released/video-focalnet_base_ssv2.pth | 187e4e9469de75ae36252d07418d9802dde16fc3a7beb9815ec949eff25dd9d6 | 71.1 | 61.1 | 228.1 | 57.44 | 13.103 | 13102.67 | 1461.0 | 4.38 | 149.0 | 157.157566 | 8 | TRUE | PRESERVED_GENUINE_SSV2 |
| uniformer-b | GENUINE_SSV2 | checkpoints/legacy_ssv2_released/uniformer_base_sthv2_16_prek400.pth | b3091fbfd2b47845988831599a0325c426b6ec84aee6a545c152fafe84adeffe | 70.4 | 67.1 | 163.1 | 52.51 | 8.563 | 8563.34 | 323.0 | 6.13 | 290.0 | 49.697262 | 16 | TRUE | PRESERVED_GENUINE_SSV2 |
| videomae-b | GENUINE_SSV2 | checkpoints/legacy_ssv2_released/videomae_vit_b_ssv2_2400e.pth | 25f4060e564ca716ff1f1804f5aba92f9240da723e22b383e306273ca91de377 | 70.8 | 68.6 | 190.7 | 56.76 | 10.826 | 10825.76 | 809.0 | 5.24 | 180.0 | 86.361006 | 32 | TRUE | PRESERVED_GENUINE_SSV2 |
| dualformer-b-in21k | K400_ON_SSV2_INPUT | checkpoints/dualformer/dualformer_base_patch244_window877.pth | 6d7414ca4ff1d010cddb8ebb8c05594921dafd1165da7811bb68a3c6bb679eca | N/A | N/A | 351.8802829002525 | 57.27293644660035 | 20.153217079357912 | 20153.217079357913 | 1157.21630859375 | 2.841875628147855 | 267.965246464 | 87.247632 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| omnivore-b-in21k | K400_ON_SSV2_INPUT | checkpoints/omnivore/swinB_In21k_checkpoint.torch | 55707dba9353d1e2be1ae67bf989e468493f487e2f3ad44f951c249e1bd8d2e8 | N/A | N/A | 594.9389392998346 | 58.424393105782244 | 34.758946463590654 | 34758.94646359066 | 2341.82763671875 | 1.6808447622824443 | 320.946176 | 90.114643 | 32 | FALSE | COMPLETED_DEVICE_ONLY |
| videoswin-b | GENUINE_SSV2 | checkpoints/legacy_ssv2_released/swin_base_patch244_window1677_sthv2.pth | f7fe417a0659680ab9db4f5e02c39b3f2825b58f0e244d4fbe3de6658fe135fc | 69.6 | 69.5 | 601.1 | 58.98 | 35.452 | 35452.26 | 2156.0 | 1.66 | 320.6 | 88.834038 | 32 | TRUE | PRESERVED_GENUINE_SSV2 |
| timesformer-b | GENUINE_SSV2 | checkpoints/legacy_ssv2_released/TimeSformer_divST_8_224_SSv2.pyth | 7a606efdda9345f1e2d83e10cc2b03196edf157ff7e9a07a19368352614109c4 | 59.1 | 57.6 | 167.1 | 55.36 | 9.251 | 9250.83 | 750.0 | 5.98 | 196.0 | 121.392558 | 32 | TRUE | PRESERVED_GENUINE_SSV2 |

## Execution status

- `uniformer-s`: PRESERVED_GENUINE_SSV2
- `dualformer-t`: COMPLETED_DEVICE_ONLY
- `video-focalnet-t`: COMPLETED_DEVICE_ONLY
- `videoswin-t`: COMPLETED_DEVICE_ONLY
- `mvit-v1-b-16x4`: COMPLETED_DEVICE_ONLY
- `video-focalnet-s`: COMPLETED_DEVICE_ONLY
- `mvit-v1-b-32x3`: COMPLETED_DEVICE_ONLY
- `dualformer-s`: COMPLETED_DEVICE_ONLY
- `videoswin-s`: COMPLETED_DEVICE_ONLY
- `zeroi2v-b16-8f`: SKIPPED_FAILED — RuntimeError: ZeroI2V ViT-B/16 (8f) is not runnable in this environment (checkpoint present): the model is an mmaction2 plugin from https://github.com/MCG-NJU/ZeroI2V, which is not vendored, and the mmaction2-modern runtime (envs/mmaction2-modern) is required. Run it on the Linux RTX 5070 host after setting up that environment; do not substitute a proxy architecture.
- `video-focalnet-b`: PRESERVED_GENUINE_SSV2
- `uniformer-b`: PRESERVED_GENUINE_SSV2
- `videomae-b`: PRESERVED_GENUINE_SSV2
- `dualformer-b-in21k`: COMPLETED_DEVICE_ONLY
- `omnivore-b-in21k`: COMPLETED_DEVICE_ONLY
- `videoswin-b`: PRESERVED_GENUINE_SSV2
- `timesformer-b`: PRESERVED_GENUINE_SSV2
