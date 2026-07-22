"""ViViT adapters without cross-variant checkpoint substitution."""
import torch

from .common import ModelUnavailableError, require_dataset, sample_frames, validate_logits


class ViViTModel:
    MODEL_ZOO = {
        "B": {
            "hf_id": "google/vivit-b-16x2-kinetics400",
            "frames": 32,
            "accuracy": 80.0,
            "params_m": 87.9,
            "gflops": 455.2,
        }
    }

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        self.dataset = require_dataset(dataset, ("k400",), "ViViT")
        variant = variant.upper()
        if variant == "S":
            raise ModelUnavailableError(
                "ViViT-S cannot run: no exact ViViT-S Kinetics-400 architecture/checkpoint "
                "is present, and the official public ViViT model zoo provides B/L releases. "
                "ViViT-B is intentionally not substituted for ViViT-S."
            )
        if variant not in self.MODEL_ZOO:
            raise ValueError(f"Unknown ViViT variant: {variant}")

        from transformers import VivitForVideoClassification

        self.variant = variant
        self.info = self.MODEL_ZOO[variant]
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        self.model = VivitForVideoClassification.from_pretrained(self.info["hf_id"])
        self.model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.info["frames"])
        clip = clip.mul(2.0).sub(1.0).to(self.device)
        logits = self.model(pixel_values=clip).logits
        return validate_logits(logits, batch_size=clip.shape[0])

    @property
    def class_names(self):
        labels = self.model.config.id2label
        return [labels[i] for i in range(len(labels))]

    @property
    def name(self):
        return f"ViViT-{self.variant}"

    @property
    def accuracy(self):
        return self.info["accuracy"]

    @property
    def gflops(self):
        return self.info["gflops"]

    @property
    def parameters(self):
        return sum(p.numel() for p in self.model.parameters())
