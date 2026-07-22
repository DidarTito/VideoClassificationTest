"""Availability guard for the exact supervised SVT-B classifier."""
from .common import ModelUnavailableError, ROOT, require_dataset


class SVTModel:
    MODEL_ZOO = {"B": {"accuracy": 78.1, "params_m": 86, "gflops": 180}}

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        self.dataset = require_dataset(dataset, ("k400",), "SVT-B")
        checkpoint = ROOT / "checkpoints" / "svt" / "kinetics400_vitb_ssl.pth"
        raise ModelUnavailableError(
            "SVT-B cannot run as a Kinetics-400 classifier: "
            f"{checkpoint} is the released self-supervised backbone/projection "
            "checkpoint and contains no trained 400-class head. TimeSformer or a "
            "new random head is intentionally not substituted."
        )
