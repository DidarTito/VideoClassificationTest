import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ACTIONCLIP_ROOT = ROOT / "third_party" / "ActionCLIP"

# 16 prompt templates from ActionCLIP's utils/Text_Prompt.py
TEXT_AUG = [
    "a photo of action {}", "a picture of action {}", "Human action of {}",
    "{}, an action", "{} this is an action", "{}, a video of action",
    "Playing action of {}", "{}", "Playing a kind of action, {}",
    "Doing a kind of action, {}", "Look, the human is {}",
    "Can you recognize the action of {}?", "Video classification of {}",
    "A video of {}", "The man is {}", "The woman is {}",
]


def _strip_module(state_dict):
    return {k[len("module."):] if k.startswith("module.") else k: v
            for k, v in state_dict.items()}


def _kinetics_classes():
    """Class names from the dataset folder layout; falls back to generic
    labels (text features only affect predictions, not energy/latency)."""
    d = ROOT / "datasets" / "kinetics400"
    if d.is_dir():
        names = sorted(p.name for p in d.iterdir() if p.is_dir())
        if names:
            return names
    return [f"action {i}" for i in range(400)]


class ActionCLIPModel:
    MODEL_ZOO = {
        "B8":  {"checkpoint": ROOT / "checkpoints/actionclip/vit-b-16-8f.pt", "frames": 8,
                "accuracy": 82.1, "params_m": 86, "gflops": 280},
        "B16": {"checkpoint": ROOT / "checkpoints/actionclip/vit-b-16-16f.pt", "frames": 16,
                "accuracy": 83.8, "params_m": 86, "gflops": 560},
    }
    # CLIP normalization
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 1, 3, 1, 1)
    STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 1, 3, 1, 1)

    def __init__(self, variant="B8", device="cuda"):
        self.variant = variant
        self.info = self.MODEL_ZOO[variant]
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if not self.info["checkpoint"].is_file():
            raise FileNotFoundError(f"ActionCLIP checkpoint missing: {self.info['checkpoint']}")
        if str(ACTIONCLIP_ROOT) not in sys.path:
            sys.path.insert(0, str(ACTIONCLIP_ROOT))
        from clip.model import build_model
        from modules.Visual_Prompt import visual_prompt

        ckpt = torch.load(self.info["checkpoint"], map_location="cpu")
        model_state = _strip_module(ckpt["model_state_dict"])
        fusion_state = _strip_module(ckpt["fusion_model_state_dict"])

        t = self.info["frames"]
        self.model = build_model(model_state, T=t).float().to(self.device).eval()
        self.fusion = visual_prompt("Transf", model_state, t).float().to(self.device).eval()
        self.fusion.load_state_dict(fusion_state)

        # Encode class-name prompts once; averaged per class over the 16 templates.
        import clip as actionclip_pkg
        classes = _kinetics_classes()
        self._class_names = classes
        with torch.no_grad():
            feats = []
            for template in TEXT_AUG:
                tokens = actionclip_pkg.tokenize(
                    [template.format(c) for c in classes]).to(self.device)
                f = self.model.encode_text(tokens)
                feats.append(f / f.norm(dim=-1, keepdim=True))
            self.text_features = torch.stack(feats).mean(0)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def __call__(self, clip):
        # clip: (B, T, C, H, W) in [0, 1]
        t = self.info["frames"]
        idx = torch.linspace(0, clip.shape[1] - 1, t).long()
        clip = (clip[:, idx] - self.MEAN) / self.STD
        b, tt, c, h, w = clip.shape
        frames = clip.reshape(b * tt, c, h, w).to(self.device)
        feats = self.model.encode_image(frames).view(b, tt, -1)
        feats = self.fusion(feats)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return 100.0 * feats @ self.text_features.T

    @property
    def class_names(self): return self._class_names

    @property
    def name(self): return f"ActionCLIP-{self.variant}"
    @property
    def accuracy(self): return self.info["accuracy"]
    @property
    def gflops(self): return self.info["gflops"]
    @property
    def parameters(self): return int(self.info["params_m"] * 1e6)
