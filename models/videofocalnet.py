"""Strict adapters for official Video-FocalNet Kinetics-400 checkpoints."""
import importlib.util
import sys

import torch

from .common import (
    AdapterMetadata,
    ROOT,
    load_exact_state_dict,
    normalize,
    require_file,
    sample_frames,
    validate_logits,
    canonical_dataset,
    dataset_num_classes,
)


_SOURCE = ROOT / "third_party" / "Video-FocalNets" / "classification" / "videofocalnet.py"
_MODULE_NAME = "_local_video_focalnet"


def _video_focalnet_class():
    module = sys.modules.get(_MODULE_NAME)
    if module is None:
        source = require_file(_SOURCE, f"Official Video-FocalNet source is missing: {_SOURCE}")
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import Video-FocalNet source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_MODULE_NAME] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(_MODULE_NAME, None)
            raise
    return module.VideoFocalNet


class VideoFocalNetModel(AdapterMetadata):
    MODEL_ZOO = {
        "T": {
            "checkpoint": ROOT / "checkpoints" / "focalnet" / "video-focalnet_tiny_kinetics400.pth",
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "drop_path_rate": 0.2,
            "accuracy": 79.8,
            "gflops": 63.0,
        },
        "S": {
            "checkpoint": ROOT / "checkpoints" / "focalnet" / "video-focalnet_small_kinetics400.pth",
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "drop_path_rate": 0.3,
            "accuracy": 81.4,
            "gflops": 124.0,
        },
        "B": {
            "checkpoint": ROOT / "checkpoints" / "focalnet" / "video-focalnet_base_kinetics400.pth",
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "drop_path_rate": 0.5,
            "accuracy": 83.6,
            "gflops": 149.0,
        },
    }

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        variant = variant.upper()
        if variant not in self.MODEL_ZOO:
            raise ValueError(f"Unknown Video-FocalNet variant: {variant}")
        self.variant = variant
        self.dataset = canonical_dataset(dataset)
        self.info = dict(self.MODEL_ZOO[variant])
        if self.dataset == "ssv2":
            if variant != "B":
                from .common import ModelUnavailableError
                raise ModelUnavailableError(
                    f"Video-FocalNet-{variant} has no SSV2 checkpoint in checkpoints/focalnet"
                )
            self.info.update({
                "checkpoint": ROOT / "checkpoints" / "focalnet" / "video-focalnet_base_ssv2.pth",
                "accuracy": 71.1,
            })
        checkpoint_path = require_file(self.info["checkpoint"])
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )

        model_cls = _video_focalnet_class()
        model = model_cls(
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=dataset_num_classes(self.dataset),
            embed_dim=self.info["embed_dim"],
            depths=self.info["depths"],
            drop_path_rate=self.info["drop_path_rate"],
            focal_levels=[2, 2, 2, 2],
            focal_windows=[3, 3, 3, 3],
            num_frames=8,
            tubelet_size=1,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise RuntimeError(
                f"{checkpoint_path} must contain the official nested 'model' state dict"
            )
        load_exact_state_dict(model, checkpoint["model"], checkpoint_path)
        if getattr(model.head, "out_features", None) != self.num_classes:
            raise RuntimeError(f"{checkpoint_path} does not provide a {self.num_classes}-class head")

        self.model = model.eval().to(self.device)
        self.frames = 8
        self.sampling_rate = 4

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.frames).to(self.device)
        clip = normalize(clip, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        logits = self.model(clip)
        return validate_logits(logits, batch_size=clip.shape[0], classes=self.num_classes)

    @property
    def name(self):
        return f"Video-FocalNet-{self.variant}"
