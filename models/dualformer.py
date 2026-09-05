"""Strict standalone adapters for the official DualFormer Kinetics models.

Supported variants: T (exact local checkpoint) and S (exact architecture from
the official config; requires the official Small K400 checkpoint on disk).
DualFormer-B (IN21K) is blocked: no public exact IN21K->K400 checkpoint
exists and the IN1K Base release is a forbidden substitute.
"""
from contextlib import contextmanager
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    AdapterMetadata,
    ROOT,
    checkpoint_path,
    load_exact_state_dict,
    normalize,
    require_file,
    sample_frames,
    validate_logits,
    require_dataset,
)


_VENDOR_ROOT = ROOT / "third_party" / "DualFormer"
_SOURCE = _VENDOR_ROOT / "mmaction" / "models" / "backbones" / "dualformer.py"
_CHECKPOINT = checkpoint_path("dualformer", "dualformer_tiny_patch244_window877.pth")
_CHECKPOINT_S = checkpoint_path("dualformer", "dualformer_small_patch244_window877.pth")

# Exact architecture rows from the official vendored configs
# (configs/_base_/models/dualformer/dualformer_{tiny,small,base}.py).
_ARCH = {
    "T": {
        "embed_dims": [64, 128, 256, 512],
        "num_heads": [2, 4, 8, 16],
        "depths": [2, 2, 10, 4],
        "head_in": 512,
    },
    "S": {
        "embed_dims": [96, 192, 384, 768],
        "num_heads": [3, 6, 12, 24],
        "depths": [2, 2, 18, 2],
        "head_in": 768,
    },
}


class _RegistryShim:
    """The official backbone only needs MMAction's decorator at import time."""

    def register_module(self, cls=None, **_kwargs):
        def decorate(value):
            return value

        return decorate(cls) if cls is not None else decorate


@contextmanager
def _dualformer_import_shims():
    """Expose only the legacy import names needed by the official backbone.

    DualFormer bundled an old MMAction2 tree whose broad import path conflicts
    with modern mmcv packages.  Classification inference needs none of the
    runner/registry machinery, so temporary minimal modules let us execute the
    authors' unmodified backbone source without importing legacy training code.
    """
    prefixes = ("mmcv", "mmaction")
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)
    }
    for name in saved:
        sys.modules.pop(name, None)

    mmcv = types.ModuleType("mmcv")
    mmcv.__path__ = []
    runner = types.ModuleType("mmcv.runner")

    def unavailable_checkpoint_loader(*_args, **_kwargs):
        raise RuntimeError("Legacy mmcv checkpoint loading is disabled")

    runner.load_checkpoint = unavailable_checkpoint_loader
    mmcv.runner = runner

    mmaction = types.ModuleType("mmaction")
    mmaction.__path__ = [str(_VENDOR_ROOT / "mmaction")]
    utils = types.ModuleType("mmaction.utils")
    utils.get_root_logger = lambda: logging.getLogger("dualformer")
    models = types.ModuleType("mmaction.models")
    models.__path__ = [str(_VENDOR_ROOT / "mmaction" / "models")]
    builder = types.ModuleType("mmaction.models.builder")
    builder.BACKBONES = _RegistryShim()
    backbones = types.ModuleType("mmaction.models.backbones")
    backbones.__path__ = [str(_VENDOR_ROOT / "mmaction" / "models" / "backbones")]

    mmaction.utils = utils
    mmaction.models = models
    models.builder = builder
    models.backbones = backbones
    injected = {
        "mmcv": mmcv,
        "mmcv.runner": runner,
        "mmaction": mmaction,
        "mmaction.utils": utils,
        "mmaction.models": models,
        "mmaction.models.builder": builder,
        "mmaction.models.backbones": backbones,
    }
    sys.modules.update(injected)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        importlib.invalidate_caches()


def _official_backbone_class():
    source = require_file(_SOURCE, f"Official DualFormer source is missing: {_SOURCE}")
    with _dualformer_import_shims():
        module_name = "mmaction.models.backbones.dualformer"
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load official DualFormer source: {source}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.DualFormer


class _TokenLabelHead(nn.Module):
    """Exact published training head; only ``fc_cls`` is used during testing."""

    def __init__(self, in_channels=512, with_token_linear=True):
        super().__init__()
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_cls = nn.Linear(in_channels, 400)
        # Hidden widths scale with the backbone output width in the released
        # checkpoints: Tiny 512 -> 1024 -> 2048 -> 400, Small 768 -> 1536 ->
        # 3072 -> 400.
        self.token_linear = (
            nn.Sequential(
                nn.Linear(in_channels, 2 * in_channels),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(2 * in_channels, 4 * in_channels),
                nn.Dropout(0.5),
                nn.ReLU(inplace=True),
                nn.Linear(4 * in_channels, 400),
            )
            if with_token_linear
            else None
        )

    def forward(self, features):
        pooled = self.avg_pool(features)
        pooled = self.dropout(pooled).reshape(features.shape[0], -1)
        return self.fc_cls(pooled)


class _DualFormerClassifier(nn.Module):
    def __init__(self, backbone_class, arch, with_token_linear=True):
        super().__init__()
        self.backbone = backbone_class(
            pretrained=None,
            pretrained2d=True,
            video_size=(32, 224, 224),
            patch_size=(2, 4, 4),
            in_chans=3,
            num_classes=1000,
            embed_dims=arch["embed_dims"],
            num_heads=arch["num_heads"],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            depths=arch["depths"],
            local_sizes=[(8, 7, 7)] * 4,
            fine_pysizes=[(8, 7, 7), (8, 7, 7), (8, 7, 7), (16, 7, 7)],
            coarse_pysizes=[(4, 4, 4)] * 4,
            temporal_pooling=[-1, 1, 1, 1],
        )
        self.cls_head = _TokenLabelHead(arch["head_in"], with_token_linear)

    def forward(self, video):
        return self.cls_head(self.backbone(video))


class DualFormerModel(AdapterMetadata):
    MODEL_ZOO = {
        "T": {
            "checkpoint": _CHECKPOINT,
            "accuracy": 79.5,
            "gflops": 240.0,
            "frames": 32,
            "sampling_rate": 2,
            "name": "DualFormer-T",
        },
        "S": {
            "checkpoint": _CHECKPOINT_S,
            "accuracy": 80.6,
            "gflops": 159.0,
            "frames": 32,
            "sampling_rate": 2,
            "name": "DualFormer-S",
        },
    }

    def __init__(self, variant="T", device="cuda", dataset="k400"):
        variant = str(variant).upper()
        if variant in {"B", "B-IN21K"}:
            # Frozen row 14 requires the IN21K-pretrained -> K400 Base
            # configuration (82.9). The only public Base checkpoint
            # (checkpoints/dualformer/dualformer_base_patch244_window877.pth)
            # is the IN1K 81.1 model and is a forbidden substitute.
            raise RuntimeError(
                "DualFormer-B (IN21K) is BLOCKED: no public exact "
                "IN21K-pretrained K400 checkpoint exists. The local "
                "dualformer_base_patch244_window877.pth is the IN1K (81.1) "
                "model and must not be silently substituted."
            )
        if variant not in self.MODEL_ZOO:
            raise ValueError(
                f"Unknown DualFormer variant {variant!r}; supported: T, S "
                "(B-IN21K is blocked pending an exact checkpoint)"
            )
        self.dataset = require_dataset(
            dataset, ("k400",), self.MODEL_ZOO[variant]["name"]
        )
        self.variant = variant
        self.info = self.MODEL_ZOO[variant]
        checkpoint_path = require_file(
            self.info["checkpoint"],
            f"{self.info['name']} cannot run because the official Kinetics-400 "
            f"checkpoint is missing: {self.info['checkpoint']}. A randomly "
            "initialized model is not a valid benchmark target.",
        )
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )

        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(state, dict) or not state:
            raise RuntimeError(f"Unexpected DualFormer checkpoint structure: {checkpoint_path}")
        # The Tiny release was trained with the auxiliary token-label head;
        # build the head to match the checkpoint keys exactly so the load
        # remains strict for every released variant.
        with_token_linear = any(
            key.startswith("cls_head.token_linear") for key in state
        )
        model = _DualFormerClassifier(
            _official_backbone_class(), _ARCH[variant], with_token_linear
        )
        load_exact_state_dict(model, state, checkpoint_path)
        if model.cls_head.fc_cls.out_features != 400:
            raise RuntimeError(f"{checkpoint_path} does not provide a 400-class head")

        # The official test path ignores the auxiliary token-label head. It was
        # strict-loaded above for identity validation, then removed before GPU
        # transfer so benchmark memory/parameters describe the inference graph.
        model.cls_head.token_linear = None
        self.model = model.eval().to(self.device)
        self.frames = int(self.info["frames"])
        self.sampling_rate = int(self.info["sampling_rate"])
        self.checkpoint_compatibility = "strict"

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.frames).to(dtype=torch.float32)
        if clip.shape[-2:] != (224, 224):
            batch, frames, channels, height, width = clip.shape
            clip = F.interpolate(
                clip.reshape(batch * frames, channels, height, width),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, frames, channels, 224, 224)
        clip = normalize(
            clip.to(self.device),
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
        logits = self.model(clip.permute(0, 2, 1, 3, 4).contiguous())
        return validate_logits(logits, batch_size=clip.shape[0])

    @property
    def name(self):
        return str(self.info["name"])


__all__ = ["DualFormerModel"]
