"""ZeroI2V ViT-B/16 8-frame adapter (MCG-NJU/ZeroI2V) - currently BLOCKED.

The exact checkpoint exists (checkpoints/ZeroI2V/CLIP-B_k400_8x16x1.pth,
verified SHA256 in configs/frozen17.yaml), but running it requires:

1. the ZeroI2V repository itself (its model is a custom mmaction2 plugin;
   the repo is NOT vendored under third_party/), and
2. an mmaction2 2.x runtime (see envs/mmaction2-modern/requirements.txt),
   which has no compatible build for this Windows/torch combination.

Plan of record: on the Linux RTX 5070 machine, create the mmaction2-modern
environment, clone https://github.com/MCG-NJU/ZeroI2V, and load the
checkpoint through mmaction2's ``init_recognizer`` with the official
``CLIP-B_k400_8x16x1`` config. Until that preflight passes, this adapter
refuses to run rather than substituting a proxy architecture.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ZeroI2VModel:
    MODEL_ZOO = {
        "B": {
            "accuracy": 83.0,
            "params_m": 86.0,
            "gflops": 140.67,
            "frames": 8,
            "checkpoint": ROOT / "checkpoints/ZeroI2V/CLIP-B_k400_8x16x1.pth",
        }
    }

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        info = self.MODEL_ZOO["B"]
        checkpoint = info["checkpoint"]
        detail = (
            "checkpoint present" if checkpoint.is_file() else "checkpoint missing"
        )
        raise RuntimeError(
            "ZeroI2V ViT-B/16 (8f) is not runnable in this environment "
            f"({detail}): the model is an mmaction2 plugin from "
            "https://github.com/MCG-NJU/ZeroI2V, which is not vendored, and "
            "the mmaction2-modern runtime (envs/mmaction2-modern) is required. "
            "Run it on the Linux RTX 5070 host after setting up that "
            "environment; do not substitute a proxy architecture."
        )
