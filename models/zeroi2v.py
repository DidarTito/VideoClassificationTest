from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ZeroI2VModel:
    """ZeroI2V (CLIP-B adapted to video, MCG-NJU/ZeroI2V).

    Checkpoints exist at checkpoints/ZeroI2V/, but the implementation is built
    on mmaction2/mmcv 2.x, whose compiled CUDA ops do not build or load
    against torch 2.6 on Windows (the same blocker previously hit with
    Video-Swin's mmaction dependency). Documented as not runnable here;
    revisit on the Jetson (Linux) where a matching mmcv wheel may exist.
    """
    MODEL_ZOO = {"B": {"accuracy": 83.0, "params_m": 86, "gflops": 422}}

    def __init__(self, variant="B", device="cuda"):
        raise RuntimeError(
            "ZeroI2V is not runnable in this environment: it requires the "
            "mmaction2/mmcv runtime, which has no compatible build for "
            "torch 2.6 on Windows. The model is documented as skipped."
        )
