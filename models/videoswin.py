"""Strict adapters for the local official Video Swin checkpoints."""
from contextlib import contextmanager
import importlib.util
import logging
import sys
import types

import torch
from torch import nn

from .common import (
    AdapterMetadata,
    ModelUnavailableError,
    ROOT,
    canonical_dataset,
    dataset_num_classes,
    normalize,
    require_file,
    sample_frames,
    validate_logits,
)

_VENDOR_ROOT = ROOT / "third_party" / "Video-Swin-Transformer"
_SOURCE = _VENDOR_ROOT / "mmaction" / "models" / "backbones" / "swin_transformer.py"


class _RegistryShim:
    def register_module(self, cls=None, **_kwargs):
        return cls if cls is not None else (lambda value: value)


@contextmanager
def _import_shims():
    saved = {name: module for name, module in list(sys.modules.items())
             if name == "mmcv" or name.startswith("mmcv.") or
             name == "mmaction" or name.startswith("mmaction.")}
    for name in saved:
        sys.modules.pop(name, None)
    mmcv = types.ModuleType("mmcv")
    runner = types.ModuleType("mmcv.runner")
    runner.load_checkpoint = lambda *_args, **_kwargs: None
    mmaction = types.ModuleType("mmaction")
    mmaction.__path__ = [str(_VENDOR_ROOT / "mmaction")]
    utils = types.ModuleType("mmaction.utils")
    utils.get_root_logger = lambda: logging.getLogger("videoswin")
    models = types.ModuleType("mmaction.models")
    models.__path__ = [str(_VENDOR_ROOT / "mmaction" / "models")]
    builder = types.ModuleType("mmaction.models.builder")
    builder.BACKBONES = _RegistryShim()
    backbones = types.ModuleType("mmaction.models.backbones")
    backbones.__path__ = [str(_VENDOR_ROOT / "mmaction" / "models" / "backbones")]
    sys.modules.update({"mmcv": mmcv, "mmcv.runner": runner, "mmaction": mmaction,
                        "mmaction.utils": utils, "mmaction.models": models,
                        "mmaction.models.builder": builder,
                        "mmaction.models.backbones": backbones})
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == "mmcv" or name.startswith("mmcv.") or name == "mmaction" or name.startswith("mmaction."):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


def _backbone_class():
    require_file(_SOURCE, f"Official Video Swin source is missing: {_SOURCE}")
    with _import_shims():
        name = "mmaction.models.backbones.swin_transformer"
        spec = importlib.util.spec_from_file_location(name, _SOURCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module.SwinTransformer3D


class _I3DHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def forward(self, features):
        return self.fc_cls(self.dropout(self.avg_pool(features)).flatten(1))


class _Classifier(nn.Module):
    def __init__(self, backbone_cls, info, num_classes):
        super().__init__()
        self.backbone = backbone_cls(
            pretrained=None, pretrained2d=False, patch_size=(2, 4, 4),
            embed_dim=info["embed_dim"], depths=info["depths"],
            num_heads=info["heads"], window_size=info["window"],
            drop_path_rate=info["drop_path"], patch_norm=True,
        )
        self.cls_head = _I3DHead(info["embed_dim"] * 8, num_classes)

    def forward(self, video):
        return self.cls_head(self.backbone(video))


class VideoSwinModel(AdapterMetadata):
    MODEL_ZOO = {
        "T": {"embed_dim": 96, "depths": [2, 2, 6, 2], "heads": [3, 6, 12, 24],
              "window": (8, 7, 7), "drop_path": 0.1, "accuracy": 78.8,
              "gflops": 88.0, "params_m": 28.2,
              "checkpoint": "swin_tiny_patch244_window877_kinetics400_1k.pth"},
        "S": {"embed_dim": 96, "depths": [2, 2, 18, 2], "heads": [3, 6, 12, 24],
              "window": (8, 7, 7), "drop_path": 0.2, "accuracy": 80.6,
              "gflops": 166.0, "params_m": 49.8,
              "checkpoint": "swin_small_patch244_window877_kinetics400_1k.pth"},
        "B": {"embed_dim": 128, "depths": [2, 2, 18, 2], "heads": [4, 8, 16, 32],
              "window": (8, 7, 7), "drop_path": 0.3, "accuracy": 80.6,
              "gflops": 282.0, "params_m": 88.0,
              "checkpoint": "swin_base_patch244_window877_kinetics400_1k.pth"},
    }

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        self.variant = variant.upper()
        if self.variant not in self.MODEL_ZOO:
            raise ValueError(f"Unknown VideoSwin variant: {variant}")
        self.dataset = canonical_dataset(dataset)
        self.info = dict(self.MODEL_ZOO[self.variant])
        if self.dataset == "ssv2":
            if self.variant != "B":
                raise ModelUnavailableError(
                    f"VideoSwin-{self.variant} has no SSV2 checkpoint in checkpoints/videoswin"
                )
            self.info.update(checkpoint="swin_base_patch244_window1677_sthv2.pth",
                             window=(16, 7, 7), drop_path=0.4,
                             accuracy=69.6, gflops=320.6)
        checkpoint_path = require_file(ROOT / "checkpoints" / "videoswin" / self.info["checkpoint"])
        self.device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
        model = _Classifier(_backbone_class(), self.info, dataset_num_classes(self.dataset))
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = payload.get("state_dict") if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected Video Swin checkpoint structure: {checkpoint_path}")
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)
        self.frames = 32

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.frames).to(self.device)
        clip = normalize(clip, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        logits = self.model(clip.permute(0, 2, 1, 3, 4).contiguous())
        return validate_logits(logits, batch_size=clip.shape[0], classes=self.num_classes)

    @property
    def name(self):
        return f"VideoSwin-{self.variant}"

    @property
    def parameters(self):
        return sum(parameter.numel() for parameter in self.model.parameters())
