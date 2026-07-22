import torch
from torchvision.models.video import (
    mvit_v1_b, mvit_v2_s,
    MViT_V1_B_Weights, MViT_V2_S_Weights,
)


class MViTModel:
    MODEL_ZOO = {
        "V1-B": {"builder": mvit_v1_b, "weights": MViT_V1_B_Weights.KINETICS400_V1,
                 "frames": 16, "accuracy": 78.5, "params_m": 36.6, "gflops": 70.6},
        "V2-S": {"builder": mvit_v2_s, "weights": MViT_V2_S_Weights.KINETICS400_V1,
                 "frames": 16, "accuracy": 80.8, "params_m": 34.5, "gflops": 64.7},
    }
    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)

    def __init__(self, variant="V2-S", device="cuda"):
        self.variant = variant
        self.info = self.MODEL_ZOO[variant]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = self.info["builder"](weights=self.info["weights"])
        self.model.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, clip):
        # MViT has fixed temporal positional embeddings: exactly `frames` frames required
        idx = torch.linspace(0, clip.shape[1] - 1, self.info["frames"]).long()
        clip = (clip[:, idx] - self.MEAN) / self.STD
        clip = clip.permute(0, 2, 1, 3, 4)
        return self.model(clip.to(self.device))

    @property
    def class_names(self):
        return list(self.info["weights"].meta["categories"])

    @property
    def name(self): return f"MViT-{self.variant}"
    @property
    def accuracy(self): return self.info["accuracy"]
    @property
    def gflops(self): return self.info["gflops"]
    @property
    def parameters(self): return int(self.info["params_m"] * 1e6)
