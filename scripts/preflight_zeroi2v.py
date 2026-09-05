#!/usr/bin/env python3
"""Linux-only exact ZeroI2V K400 load and one-forward preflight."""

import argparse
import hashlib
from pathlib import Path
import platform
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "third_party" / "ZeroI2V"
CONFIG = VENDOR / "configs" / "recognition" / "vit_zero_clip" / "CLIP-B_k400_8x16x1.py"
CHECKPOINT = ROOT / "checkpoints" / "ZeroI2V" / "CLIP-B_k400_8x16x1.pth"
EXPECTED_SHA256 = "f62f2e10aa0ae0cb755046f9d84b5ecffd58d4a5f1ae043ace88c3b27621929f"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if platform.system() != "Linux":
        raise SystemExit("ZeroI2V exact runtime preflight is prepared for Linux only")
    if not CONFIG.is_file() or not CHECKPOINT.is_file():
        raise SystemExit(f"missing config/checkpoint: {CONFIG}, {CHECKPOINT}")
    if sha256(CHECKPOINT) != EXPECTED_SHA256:
        raise SystemExit("ZeroI2V checkpoint SHA256 mismatch; refusing unsafe load")

    sys.path.insert(0, str(VENDOR))
    from mmengine.config import Config
    from mmaction.registry import MODELS
    from mmaction.utils import register_all_modules

    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(CONFIG))
    # The exact K400 checkpoint supplies the whole graph. Avoid downloading a
    # second CLIP initializer during a checkpoint-load preflight.
    cfg.model.backbone.pretrained = None
    model = MODELS.build(cfg.model)

    # weights_only=False is isolated here and only occurs after the official
    # artifact hash is verified. MMEngine metadata requires its registered
    # Python classes during deserialization.
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload.get("model", payload))
    model_keys = set(model.state_dict())
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            stripped = {key[len(prefix):]: value for key, value in state.items()}
            if set(stripped) & model_keys:
                state = stripped
                break
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model = model.eval().to(device)
    clip = torch.rand(1, 3, 8, 224, 224, device=device)
    with torch.inference_mode():
        features = model.backbone(clip)
        logits = model.cls_head(features)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if tuple(logits.shape) != (1, 400) or not torch.isfinite(logits).all():
        raise SystemExit(f"invalid ZeroI2V logits: {tuple(logits.shape)}")
    print(f"ZEROI2V_PREFLIGHT=PASS logits={tuple(logits.shape)} device={device}")


if __name__ == "__main__":
    main()
