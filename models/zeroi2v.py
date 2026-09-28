"""Strict inference adapter for the vendored ZeroI2V ViT-B/16, 8-frame model.

Runs the authors' original, unreparameterized checkpoint graph directly;
MMAction's optional training/detection stack is not needed for inference.
"""
from contextlib import contextmanager
import importlib.util
import math
import sys
import types

import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    AdapterMetadata, ROOT, checkpoint_path, isolated_vendor_package,
    load_exact_state_dict, normalize, require_dataset, require_file,
    sample_frames, validate_logits,
)

VENDOR = ROOT / "third_party/ZeroI2V"


@contextmanager
def _local_grammar_cache():
    # YAPF (imported by MMEngine) otherwise writes to the user profile.
    import platformdirs
    original = platformdirs.user_cache_dir
    cache = ROOT / ".cache/zeroi2v/yapf"
    cache.mkdir(parents=True, exist_ok=True)
    def cache_dir(*args, **kwargs):
        app = kwargs.get("appname", args[0] if args else None)
        return str(cache) if app == "YAPF" else original(*args, **kwargs)
    platformdirs.user_cache_dir = cache_dir
    driver = sys.modules.get("yapf_third_party._ylib2to3.pgen2.driver")
    if driver is not None:
        driver.user_cache_dir = cache_dir
    try:
        yield
    finally:
        platformdirs.user_cache_dir = original
        driver = sys.modules.get("yapf_third_party._ylib2to3.pgen2.driver")
        if driver is not None:
            driver.user_cache_dir = original


def load_checkpoint_state(path):
    """Load weights safely, allowing only explicit author metadata types."""
    import numpy as np
    with _local_grammar_cache():
        from mmengine.logging.history_buffer import HistoryBuffer
    safe = [HistoryBuffer, (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
            (np.core.multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
            np.dtype, np.ndarray, type(np.dtype("float64")), type(np.dtype("float32")),
            type(np.dtype("int64"))]
    with torch.serialization.safe_globals(safe):
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"No official state_dict in {path}")
    if state.get("cls_head.fc_cls.weight", torch.empty(0)).shape != (400, 768):
        raise RuntimeError("ZeroI2V B/16 requires its exact 400 x 768 K400 classifier")
    if state.get("backbone.conv1.weight", torch.empty(0)).shape != (768, 3, 16, 16):
        raise RuntimeError("Checkpoint is not ZeroI2V ViT-B/16")
    if state.get("backbone.temporal_embedding", torch.empty(0)).shape != (1, 8, 768):
        raise RuntimeError("Checkpoint is not the 8-frame ZeroI2V variant")
    return state


def _legacy_attention(q, k, v, mask=None, dropout_p=0.0):
    """Restore the PyTorch 1.x primitive used by the authors' STDHA source."""
    scores = torch.bmm(q / math.sqrt(q.size(-1)), k.transpose(-2, -1))
    if mask is not None:
        scores = scores + mask
    weights = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        weights = F.dropout(weights, p=dropout_p)
    return torch.bmm(weights, v), weights


def _load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, require_file(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _backbone_class():
    with _local_grammar_cache():
        from mmengine.registry import Registry
        import mmcv.cnn.bricks  # noqa: F401
        import clip  # noqa: F401
    with isolated_vendor_package("mmaction", VENDOR):
        for name, relative in [("mmaction", "mmaction"), ("mmaction.models", "mmaction/models"),
                               ("mmaction.models.backbones", "mmaction/models/backbones"),
                               ("mmaction.models.common", "mmaction/models/common")]:
            module = types.ModuleType(name)
            module.__path__ = [str(VENDOR / relative)]
            sys.modules[name] = module
        registry = types.ModuleType("mmaction.registry")
        registry.MODELS = Registry("zeroi2v_inference", scope="zeroi2v_inference")
        sys.modules[registry.__name__] = registry
        attention = _load_source("mmaction.models.common.stdha", VENDOR / "mmaction/models/common/stdha.py")
        # Keep this compatibility function local to ZeroI2V, not global Torch.
        attention.F = types.SimpleNamespace(**{name: getattr(F, name) for name in dir(F)})
        attention.F._scaled_dot_product_attention = _legacy_attention
        sys.modules["mmaction.models.common"].STDHA = attention.STDHA
        backbone = _load_source("mmaction.models.backbones.vit_zero_clip",
                                VENDOR / "mmaction/models/backbones/vit_zero_clip.py")
        return backbone.ViT_Zero_CLIP


class _I3DHead(nn.Module):
    """Official I3DHead inference operations and checkpoint key names."""
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc_cls = nn.Linear(768, 400)

    def forward(self, x):
        return self.fc_cls(self.dropout(self.avg_pool(x)).reshape(x.shape[0], -1))


class _Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _backbone_class()(
            pretrained=None, input_resolution=224, patch_size=16, num_frames=8,
            width=768, layers=12, heads=12, adapter_scale=0.5, num_tadapter=2,
            stdha_cfg=dict(shift_div=12, divide_head=False))
        self.cls_head = _I3DHead()

    def forward(self, x):
        return self.cls_head(self.backbone(x))


class ZeroI2VModel(AdapterMetadata):
    MODEL_ZOO = {"B": dict(accuracy=83.0, params_m=86.0, gflops=140.67, frames=8,
                           checkpoint=checkpoint_path("ZeroI2V", "CLIP-B_k400_8x16x1.pth"))}

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        if str(variant).upper() != "B":
            raise ValueError("Only the exact ZeroI2V ViT-B/16 8-frame variant is supported")
        self.dataset = require_dataset(dataset, ("k400",), "ZeroI2V ViT-B/16 (8f)")
        self.variant = "B"
        self.info = dict(self.MODEL_ZOO["B"])
        path = require_file(self.info["checkpoint"])
        self.info["checkpoint"] = str(path.resolve())
        self.device = torch.device(device)
        state = load_checkpoint_state(path)
        model = _Classifier()
        load_exact_state_dict(model, state, path)
        self.model = model.eval().requires_grad_(False).to(self.device)
        self.frames = 8
        self.sampling_rate = 16
        self.checkpoint_compatibility = "strict"

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.frames).to(device=self.device, dtype=torch.float32)
        # The exact official config specifies normalization in [0,255].
        clip = normalize(clip, tuple(v / 255 for v in (122.769, 116.74, 104.04)),
                         tuple(v / 255 for v in (68.493, 66.63, 70.321)))
        logits = self.model(clip.permute(0, 2, 1, 3, 4).contiguous())
        return validate_logits(logits, batch_size=clip.shape[0], classes=self.num_classes)

    @property
    def name(self):
        return "ZeroI2V ViT-B/16 (8f, original checkpoint graph)"
