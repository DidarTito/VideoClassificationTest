#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${1:-core-modern}"
case "$ENV_NAME" in core-modern|slowfast|mmaction-legacy|mmaction2-modern) ;; *) echo "Unknown environment: $ENV_NAME"; exit 2;; esac
python3 -m venv ".venv-$ENV_NAME"
source ".venv-$ENV_NAME/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
if [ "$ENV_NAME" = mmaction2-modern ]; then
  # ZeroI2V vendors the MMAction2 plugin implementation at a pinned commit.
  # MMAction2's compiled dependency must be selected after Torch/CUDA exist.
  python -m pip install openmim
  mim install 'mmcv>=2.0.0rc1,<2.1.0'
  python -m pip install -r "envs/$ENV_NAME/requirements.txt"
  python -m pip install -e third_party/ZeroI2V
  python -m pip install 'git+https://github.com/openai/CLIP.git'
else
  python -m pip install -r "envs/$ENV_NAME/requirements.txt"
fi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA validation failed")
print("gpu:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16:", torch.cuda.is_bf16_supported())
x=torch.randn(1024,1024,device="cuda"); y=x@x; torch.cuda.synchronize()
print("tensor test:", y.shape, bool(torch.isfinite(y).all()))
PY
nvidia-smi
