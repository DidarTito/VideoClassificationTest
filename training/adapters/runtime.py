"""Bridge strict inference constructors to trainable raw PyTorch graphs."""
import torch
from torch import nn

from models import MODEL_REGISTRY
from training.checkpoint_utils import assert_k400_initialization, replace_classifier
from training.registry import ROOT, model_spec

LEGACY_KEYS = {
    "mvit-v1-b-16x4": "mvit-v1-b-16x4",
    "mvit-v1-b-32x3": "mvit-v1-b",
    "zeroi2v-b16-8f": "zeroi2v-b",
    "omnivore-b-in21k": "omnivore-b",
}


class TrainableModel(nn.Module):
    def __init__(self, raw, spec, native_slowfast=False):
        super().__init__(); self.raw = raw; self.spec = spec
        self.native_slowfast = native_slowfast

    def forward(self, clip):
        # Dataset contract is B,T,C,H,W in [0,1].
        inp = self.spec["input"]
        frames = int(inp["num_frames"])
        if clip.shape[1] != frames:
            indices = torch.linspace(0, clip.shape[1] - 1, frames, device=clip.device).round().long()
            clip = clip.index_select(1, indices)
        mean = clip.new_tensor(inp["mean"]).view(1, 1, 3, 1, 1)
        std = clip.new_tensor(inp["std"]).view(1, 1, 3, 1, 1)
        clip = (clip - mean) / std
        family = self.spec["family"]
        if family in {"uniformer"}:
            out = self.raw([clip.permute(0, 2, 1, 3, 4).contiguous()])
        elif family == "mvit-v1":
            x = clip.permute(0, 2, 1, 3, 4).contiguous()
            out = self.raw([x]) if self.native_slowfast else self.raw(x)
        elif family in {"dualformer", "videoswin", "timesformer"}:
            out = self.raw(clip.permute(0, 2, 1, 3, 4).contiguous())
        elif family == "omnivore":
            out = self.raw(clip.permute(0, 2, 1, 3, 4).contiguous(), input_type="video")
        elif family == "videomae":
            out = self.raw(clip.permute(0, 2, 1, 3, 4).contiguous())
        else:
            out = self.raw(clip)
        if hasattr(out, "logits"): out = out.logits
        return out


def build_trainable(key, device):
    spec = model_spec(key)
    status = spec["checkpoint_status"]
    if status not in {"READY_EXACT", "READY_NEEDS_CONFIG_VALIDATION"}:
        raise RuntimeError(f"{key} checkpoint status is {status}; training is blocked")
    checkpoint = ROOT / spec["k400_checkpoint"]
    assert_k400_initialization(checkpoint)
    legacy = LEGACY_KEYS.get(key, key)
    if legacy not in MODEL_REGISTRY:
        raise RuntimeError(f"No exact runtime adapter is registered for {key}")
    wrapper = MODEL_REGISTRY[legacy](device=device, dataset="k400")
    raw = wrapper.model
    replace_classifier(raw, spec["classifier"]["path"], 174)
    model = TrainableModel(raw, spec, getattr(wrapper, "_native_slowfast", False))
    model.to(device).train()
    return model, wrapper
