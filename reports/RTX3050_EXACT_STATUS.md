# RTX3050 exact frozen17 status

Raw execution is **15/17 completed**. This is not a scientific 15/17 final table.

| Model | Category | Existing result | Evidence |
|---|---|---|---|
| UniFormer-S | **RERUN_CONFIGURATION** | `results\rtx3050\k400_1000_seed0_20260722\device_metrics.csv` | exact_architecture=True, exact_checkpoint=False, exact_frames=True, same_subset=True, same_preprocessing=False |
| DualFormer-T | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| Video-FocalNet-T | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| VideoSwin-T | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| MViT-B, 16x4 | **RERUN_CONFIGURATION** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=False, same_subset=True, same_preprocessing=True |
| Video-FocalNet-S | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| MViT-B, 32x3 | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| DualFormer-S | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| VideoSwin-S | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| ZeroI2V ViT-B/16, 8f | **BLOCKED_RUNTIME** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=False, exact_checkpoint=False, exact_frames=False, same_subset=False, same_preprocessing=False |
| Video-FocalNet-B | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| UniFormer-B | **BLOCKED_EXACT_CHECKPOINT** | `results\rtx3050\k400_1000_seed0_20260722\device_metrics.csv` | exact_architecture=False, exact_checkpoint=False, exact_frames=False, same_subset=False, same_preprocessing=False |
| VideoMAE-B | **RERUN_CONFIGURATION** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=False, exact_frames=True, same_subset=True, same_preprocessing=False |
| DualFormer-B (IN21K) | **BLOCKED_EXACT_CHECKPOINT** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=False, exact_checkpoint=False, exact_frames=False, same_subset=False, same_preprocessing=False |
| Omnivore-B (IN21K) | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| VideoSwin-B | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |
| TimeSformer-B | **VALID_REUSE** | `results\NVIDIA_GeForce_RTX_3050_Laptop_GPU\k400\full_1000_seed0\device_metrics.csv` | exact_architecture=True, exact_checkpoint=True, exact_frames=True, same_subset=True, same_preprocessing=True |

## Counts

- BLOCKED_EXACT_CHECKPOINT: 2
- BLOCKED_RUNTIME: 1
- RERUN_CONFIGURATION: 3
- VALID_REUSE: 11

Video-FocalNet note: the official released checkpoint graphs contain 49.553M / 88.735M / 157.389M parameters. The smaller 28M / 49M / 88M values describe the spatial image backbones and are retained as paper metadata, not used to reject strictly compatible video checkpoints.

UniFormer-B note: the official model zoo links a distinct 32x4/82.9 checkpoint. The local 16x4/82.0 artifact is not silently relabeled.
