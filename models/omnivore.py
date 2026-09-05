"""Checkpoint-backed Omnivore video-classification adapter."""
from contextlib import contextmanager
import importlib
import importlib.util
import sys
import types

import torch

from .common import (
    AdapterMetadata,
    ROOT,
    checkpoint_path,
    normalize,
    require_file,
    sample_frames,
    validate_logits,
    require_dataset,
)


_VENDOR_ROOT = ROOT / "third_party" / "omnivore"


@contextmanager
def _vendor_on_path():
    prefixes = ("omnivore", "omnivision")
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    for name in saved:
        sys.modules.pop(name, None)
    path = str(_VENDOR_ROOT)
    sys.path.insert(0, path)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def _load_official_model_module():
    """Load only Omnivore's inference graph, skipping optional Hydra setup.

    Importing ``omnivision.models`` normally executes its package initializer,
    which depends on Hydra solely for config-driven training helpers.  The
    published Omnivore factory and Swin inference graph do not use those
    helpers, so expose the models directory as a temporary namespace package
    and load the official model definition directly.
    """
    models_dir = _VENDOR_ROOT / "omnivision" / "models"
    namespace = types.ModuleType("omnivision.models")
    namespace.__path__ = [str(models_dir)]
    namespace.__package__ = "omnivision.models"
    sys.modules["omnivision.models"] = namespace

    model_path = _VENDOR_ROOT / "omnivore" / "models" / "omnivore_model.py"
    spec = importlib.util.spec_from_file_location(
        "_benchmark_official_omnivore_model", model_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official Omnivore model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OmnivoreModel(AdapterMetadata):
    """Omnivore Swin-B variants with checkpoint identity kept explicit."""

    MODEL_ZOO = {
        "B-IN21K": {
            "checkpoint": checkpoint_path("omnivore", "swinB_In21k_checkpoint.torch"),
            "builder": "omnivore_swinB_imagenet21k",
            "name": "Omnivore-B (IN21K)",
            "accuracy": 84.0,
            "gflops": 282.0,
            "frames": 32,
            "sampling_rate": 40,
        },
        "B-STANDARD": {
            "checkpoint": checkpoint_path("omnivore", "Omnivore_swinB_checkpoint.torch"),
            "builder": "omnivore_swinB",
            "name": "Omnivore-B (standard)",
            "accuracy": 83.3,
            "gflops": 282.0,
            "frames": 32,
            "sampling_rate": 40,
        }
    }

    def __init__(self, variant="B-IN21K", device="cuda", dataset="k400"):
        self.dataset = require_dataset(dataset, ("k400",), "Omnivore-B")
        variant = variant.upper()
        if variant == "B":
            variant = "B-IN21K"
        if variant not in self.MODEL_ZOO:
            raise ValueError(f"Unknown Omnivore-B variant: {variant}")
        self.variant = variant
        self.info = self.MODEL_ZOO[variant]
        checkpoint_path = require_file(self.info["checkpoint"])
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )

        require_file(_VENDOR_ROOT / "omnivore" / "models" / "omnivore_model.py")
        with _vendor_on_path():
            omnivore_model = _load_official_model_module()
            builder = getattr(omnivore_model, self.info["builder"])
            model = builder(pretrained=False)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not {
            "trunk", "heads"
        }.issubset(checkpoint):
            raise RuntimeError(f"Unexpected Omnivore checkpoint structure: {checkpoint_path}")
        model.trunk.load_state_dict(checkpoint["trunk"], strict=True)
        model.heads.load_state_dict(checkpoint["heads"], strict=True)
        video_head = model.heads["video"]
        if getattr(video_head[-1], "out_features", None) != 400:
            raise RuntimeError(f"{checkpoint_path} does not provide a 400-class video head")

        self.model = model.eval().to(self.device)
        self.frames = int(self.info["frames"])
        self.sampling_rate = int(self.info["sampling_rate"])

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.frames).to(self.device)
        clip = normalize(clip, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        clip = clip.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(clip, input_type="video")
        return validate_logits(logits, batch_size=clip.shape[0])

    @property
    def name(self):
        return self.info["name"]
