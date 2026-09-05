# Linux RTX 5070 setup

The current checkout was inspected on Windows with an RTX 3050 Laptop GPU.
These commands target the separate Linux RTX 5070 machine and must be run
there before CUDA preflight. They deliberately retain a modern Blackwell-
capable PyTorch stack instead of downgrading the machine for old research code.

```bash
git status --short
bash scripts/setup_rtx5070_linux.sh core-modern
source .venv-core-modern/bin/activate
python scripts/audit_checkpoints.py --models frozen17
python scripts/validate_ssv2.py
```

Create the other isolated environments only for their assigned families:

```bash
bash scripts/setup_rtx5070_linux.sh slowfast
bash scripts/setup_rtx5070_linux.sh mmaction-legacy
bash scripts/setup_rtx5070_linux.sh mmaction2-modern
```

| Environment | Families | Reason |
|---|---|---|
| `core-modern` | UniFormer, Video-FocalNet, VideoMAE, Omnivore, TimeSformer | Native/standalone graphs can use modern PyTorch. |
| `slowfast` | MViTv1 | Vendored SlowFast config/model code and its additional utilities. |
| `mmaction-legacy` | DualFormer, VideoSwin | These sources vendor legacy MMCV/MMAction APIs. Keep them isolated. |
| `mmaction2-modern` | ZeroI2V | Its checkpoint/config stack uses modern MMEngine/MMAction2 serialization. |

The setup script prints Python, Torch, CUDA runtime, GPU name/capability and
BF16 support, executes a CUDA matrix multiplication, synchronizes it, and then
runs `nvidia-smi`. PyTorch 2.7.1+cu128 is the conservative recorded baseline;
if the lab standardizes on a newer Blackwell-supported wheel, record the exact
wheel and rerun all preflights. Never install the legacy environment's old
CUDA 10/11 Torch build on the RTX 5070 host.

The code uses one GPU only. BF16 is selected when supported, otherwise FP16
uses a gradient scaler. TF32 may be enabled after documenting it in the run
configuration. `torch.compile` is intentionally not automatic for legacy
backends.
