"""Strict SlowFast/PyTorchVideo adapters for the original MViT models.

The public SlowFast ``PTVMViT`` class is a thin wrapper around
``pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers``.
Using that constructor directly for MViT-B avoids importing all of SlowFast
(including optional Detectron2 and distributed-training dependencies) while
producing the exact parameter names used by the official checkpoint.

The deeper B-24 checkpoint has existed in both PyTorchVideo and native
SlowFast formats.  Its architecture is therefore selected from the checkpoint
keys.  Native SlowFast is imported only when that checkpoint is present and
actually uses the native format.
"""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .kinetics_labels import kinetics400_classes
from .common import require_dataset


_ROOT = Path(__file__).resolve().parents[1]
_SLOWFAST_ROOT = _ROOT / "third_party" / "SlowFast"
_B24_CONFIG = _SLOWFAST_ROOT / "configs" / "Kinetics" / "MVIT_B_32x3_CONV.yaml"


class MViTSlowFastModel:
    """Runnable Kinetics-400 MViT adapter for ``run_benchmark.py``.

    Args:
        variant: ``"B-32x3"`` (default) or ``"B-24-32x3"``.
        device: Requested Torch device. CUDA requests fall back to CPU when
            CUDA is unavailable.
        checkpoint: Optional checkpoint override. Both defaults are stored in
            the workspace so a clean clone never depends on a user cache.
    """

    MODEL_ZOO = {
        "B-32x3": {
            "name": "MViT-B, 32x3",
            "accuracy": 80.2,
            "parameters": 36_600_000,
            "gflops": 170.0,
            "checkpoint": _ROOT
            / "checkpoints"
            / "mvit"
            / "MVIT_B_32x3_f294077834.pyth",
        },
        "B-24-32x3": {
            "name": "MViT-B-24, 32x3",
            "accuracy": 80.4,
            "parameters": 52_900_000,
            "gflops": 236.0,
            "checkpoint": _ROOT
            / "checkpoints"
            / "mvit"
            / "MVIT_B_24_32x3.pyth",
        },
    }

    _ALIASES = {
        "B": "B-32x3",
        "B-32X3": "B-32x3",
        "B32X3": "B-32x3",
        "B-24": "B-24-32x3",
        "B24": "B-24-32x3",
        "B-24-32X3": "B-24-32x3",
        "B24-32X3": "B-24-32x3",
    }

    NUM_FRAMES = 32
    CROP_SIZE = 224
    MEAN = (0.45, 0.45, 0.45)
    STD = (0.225, 0.225, 0.225)

    def __init__(
        self,
        variant: str = "B-32x3",
        device: str | torch.device | None = "cuda",
        checkpoint: str | Path | None = None,
        dataset: str = "k400",
    ) -> None:
        self.dataset = require_dataset(dataset, ("k400",), "MViT")
        self.variant = self._canonical_variant(variant)
        self.info = self.MODEL_ZOO[self.variant]
        self.device = self._resolve_device(device)
        self.checkpoint_path = Path(checkpoint or self.info["checkpoint"]).expanduser()

        if not self.checkpoint_path.is_file():
            if self.variant == "B-24-32x3":
                raise FileNotFoundError(
                    "MViT-B-24, 32x3 is unavailable: expected a compatible "
                    f"checkpoint at '{self.checkpoint_path}'. Place the checkpoint "
                    "there or pass checkpoint=... explicitly."
                )
            raise FileNotFoundError(
                f"Official MViT-B, 32x3 checkpoint not found at "
                f"'{self.checkpoint_path}'."
            )

        checkpoint_obj = _torch_load_checkpoint(self.checkpoint_path)
        state = _extract_state_dict(checkpoint_obj, self.checkpoint_path)
        schema = _checkpoint_schema(state)

        if self.variant == "B-32x3":
            if schema != "pytorchvideo":
                raise RuntimeError(
                    "The MViT-B, 32x3 checkpoint is not in the expected official "
                    f"PyTorchVideo format (detected: {schema})."
                )
            model = _build_pytorchvideo_mvit(depth=16)
            self._native_slowfast = False
        elif schema == "pytorchvideo":
            model = _build_pytorchvideo_mvit(depth=24)
            self._native_slowfast = False
        elif schema == "slowfast":
            model = _build_native_slowfast_b24()
            self._native_slowfast = True
        else:
            raise RuntimeError(
                "Could not identify the MViT-B-24 checkpoint format. Expected "
                "PyTorchVideo keys such as 'patch_embed.patch_model.weight' or "
                "native SlowFast keys such as 'patch_embed.proj.weight'."
            )

        _load_state_dict_strict(model, state, self.checkpoint_path)
        # Release optimizer/cfg references before copying the model to its device.
        del checkpoint_obj, state

        self.model = model.eval().to(self.device)
        self._class_names = kinetics400_classes()
        if len(self._class_names) != 400:
            raise RuntimeError(
                f"Expected 400 Kinetics class names, found {len(self._class_names)}."
            )
        self.checkpoint_compatibility = "strict"

    @staticmethod
    def _canonical_variant(variant: str) -> str:
        key = str(variant).strip().upper().replace("_", "-").replace("×", "X")
        key = key.replace(" ", "").replace(",", "")
        canonical = MViTSlowFastModel._ALIASES.get(key)
        if canonical is None:
            choices = ", ".join(MViTSlowFastModel.MODEL_ZOO)
            raise ValueError(f"Unknown MViT variant '{variant}'. Choose one of: {choices}.")
        return canonical

    @staticmethod
    def _resolve_device(device: str | torch.device | None) -> torch.device:
        requested = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if requested.type == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return requested

    @torch.inference_mode()
    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Classify ``clip`` shaped ``(B, T, C, H, W)`` in the range [0, 1]."""
        if not isinstance(clip, torch.Tensor):
            raise TypeError(f"clip must be a torch.Tensor, got {type(clip).__name__}")
        if clip.ndim != 5:
            raise ValueError(f"clip must have shape (B,T,C,H,W), got {tuple(clip.shape)}")
        if clip.shape[1] < 1 or clip.shape[2] != 3:
            raise ValueError(
                "clip must contain at least one frame and exactly three RGB channels; "
                f"got {tuple(clip.shape)}"
            )

        clip = clip.to(dtype=torch.float32)
        indices = torch.linspace(
            0,
            clip.shape[1] - 1,
            self.NUM_FRAMES,
            device=clip.device,
        ).round().to(torch.long)
        clip = clip.index_select(1, indices)

        if clip.shape[-2:] != (self.CROP_SIZE, self.CROP_SIZE):
            batch, frames, channels, height, width = clip.shape
            clip = F.interpolate(
                clip.reshape(batch * frames, channels, height, width),
                size=(self.CROP_SIZE, self.CROP_SIZE),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch, frames, channels, self.CROP_SIZE, self.CROP_SIZE)

        mean = clip.new_tensor(self.MEAN).view(1, 1, 3, 1, 1)
        std = clip.new_tensor(self.STD).view(1, 1, 3, 1, 1)
        clip = ((clip - mean) / std).permute(0, 2, 1, 3, 4).contiguous()
        clip = clip.to(self.device, non_blocking=self.device.type == "cuda")

        output = self.model([clip]) if self._native_slowfast else self.model(clip)
        if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[1] != 400:
            shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output).__name__
            raise RuntimeError(f"MViT must return (B, 400); received {shape}")
        return output

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    @property
    def name(self) -> str:
        return str(self.info["name"])

    @property
    def accuracy(self) -> float:
        return float(self.info["accuracy"])

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @property
    def gflops(self) -> float:
        return float(self.info["gflops"])

    @property
    def num_classes(self) -> int:
        return 400


class MViT_B_32x3(MViTSlowFastModel):
    """Convenience class for the 16-block MViT-B, 32x3 model."""

    def __init__(
        self,
        device: str | torch.device | None = "cuda",
        checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__("B-32x3", device=device, checkpoint=checkpoint)


class MViT_B_24_32x3(MViTSlowFastModel):
    """Convenience class for the deeper 24-block MViT-B, 32x3 model."""

    def __init__(
        self,
        device: str | torch.device | None = "cuda",
        checkpoint: str | Path | None = None,
    ) -> None:
        super().__init__("B-24-32x3", device=device, checkpoint=checkpoint)


def _build_pytorchvideo_mvit(depth: int) -> torch.nn.Module:
    """Mirror vendored SlowFast's PTVMViT constructor without broad imports."""
    try:
        from pytorchvideo.models.vision_transformers import (
            create_multiscale_vision_transformers,
        )
    except ImportError as exc:
        raise ImportError(
            "MViT requires PyTorchVideo, which is a dependency of the vendored "
            "third_party/SlowFast implementation."
        ) from exc

    if depth == 16:
        transitions = (1, 3, 14)
        droppath = 0.2
    elif depth == 24:
        transitions = (2, 5, 21)
        droppath = 0.3
    else:  # Internal programming error, not a user-selectable configuration.
        raise ValueError(f"Unsupported MViT depth: {depth}")

    multipliers = [[index, 2.0] for index in transitions]
    strides = [[index, 1, 2, 2] for index in transitions]
    return create_multiscale_vision_transformers(
        spatial_size=224,
        temporal_size=32,
        cls_embed_on=True,
        sep_pos_embed=True,
        depth=depth,
        norm="layernorm",
        input_channels=3,
        patch_embed_dim=96,
        conv_patch_embed_kernel=(3, 7, 7),
        conv_patch_embed_stride=(2, 4, 4),
        conv_patch_embed_padding=(1, 3, 3),
        enable_patch_embed_norm=False,
        use_2d_patch=False,
        num_heads=1,
        mlp_ratio=4.0,
        qkv_bias=True,
        dropout_rate_block=0.0,
        droppath_rate_block=droppath,
        pooling_mode="conv",
        pool_first=False,
        embed_dim_mul=multipliers,
        atten_head_mul=multipliers,
        pool_q_stride_size=strides,
        pool_kv_stride_size=None,
        pool_kv_stride_adaptive=[1, 8, 8],
        pool_kvq_kernel=[3, 3, 3],
        head_dropout_rate=0.5,
        head_activation=None,
        head_num_classes=400,
    )


def _build_native_slowfast_b24() -> torch.nn.Module:
    """Build native SlowFast MViT-B-24 with an isolated, lazy vendor import."""
    if not _B24_CONFIG.is_file():
        raise FileNotFoundError(
            f"Vendored SlowFast B-24 config not found at '{_B24_CONFIG}'."
        )

    vendor_root = _SLOWFAST_ROOT.resolve()
    existing_slowfast = {
        name: module
        for name, module in sys.modules.items()
        if name == "slowfast" or name.startswith("slowfast.")
    }
    for module in existing_slowfast.values():
        module_file = getattr(module, "__file__", None)
        if module_file and vendor_root not in Path(module_file).resolve().parents:
            raise RuntimeError(
                "Cannot load native MViT-B-24 because a different 'slowfast' "
                f"package is already imported from '{module_file}'."
            )

    inserted_path = str(vendor_root) not in sys.path
    if inserted_path:
        sys.path.insert(0, str(vendor_root))

    # SlowFast imports these optional pieces even though classification does
    # not use ROIAlign or distributed collectives. Add minimal import-time
    # shims and remove them as soon as construction is complete.
    fake_detectron = "detectron2" not in sys.modules
    if fake_detectron:
        detectron_module = types.ModuleType("detectron2")
        layers_module = types.ModuleType("detectron2.layers")

        class _UnavailableROIAlign(torch.nn.Module):
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise ImportError("Detectron2 ROIAlign is unavailable")

        layers_module.ROIAlign = _UnavailableROIAlign
        detectron_module.layers = layers_module
        sys.modules["detectron2"] = detectron_module
        sys.modules["detectron2.layers"] = layers_module

    distributed_module = importlib.import_module("pytorchvideo.layers.distributed")
    distributed_shims = {
        "cat_all_gather": lambda tensor, local=False: tensor,
        "get_local_process_group": lambda: None,
        "get_local_rank": lambda: 0,
        "get_local_size": lambda: 1,
        "init_distributed_training": lambda *_args, **_kwargs: None,
    }
    added_distributed_names: list[str] = []
    for name, value in distributed_shims.items():
        if not hasattr(distributed_module, name):
            setattr(distributed_module, name, value)
            added_distributed_names.append(name)

    try:
        importlib.import_module("slowfast")
        if "slowfast.models" not in sys.modules:
            models_namespace = types.ModuleType("slowfast.models")
            models_namespace.__path__ = [str(vendor_root / "slowfast" / "models")]
            models_namespace.__package__ = "slowfast.models"
            sys.modules["slowfast.models"] = models_namespace

        get_cfg = importlib.import_module("slowfast.config.defaults").get_cfg
        mvit_class = importlib.import_module(
            "slowfast.models.video_model_builder"
        ).MViT
        cfg = get_cfg()
        cfg.merge_from_file(str(_B24_CONFIG))
        cfg.NUM_GPUS = 0
        model = mvit_class(cfg)
    finally:
        for name in added_distributed_names:
            delattr(distributed_module, name)
        if fake_detectron:
            sys.modules.pop("detectron2.layers", None)
            sys.modules.pop("detectron2", None)
        # Do not leave the vendored package occupying the global ``slowfast``
        # name when it was not already there before this adapter ran.
        for name in list(sys.modules):
            if (
                name == "slowfast" or name.startswith("slowfast.")
            ) and name not in existing_slowfast:
                sys.modules.pop(name, None)
        sys.modules.update(existing_slowfast)
        if inserted_path:
            try:
                sys.path.remove(str(vendor_root))
            except ValueError:
                pass

    return model


def _torch_load_checkpoint(path: Path) -> Any:
    """Load tensors safely, using mmap when supported by the installed Torch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # Torch versions predating weights_only/mmap.
        return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        if "mmap" not in str(exc).lower():
            raise
        return torch.load(path, map_location="cpu", weights_only=True)


def _extract_state_dict(checkpoint: Any, path: Path) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Checkpoint '{path}' is not a mapping.")

    for key in ("model_state", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping) and candidate:
            if all(isinstance(name, str) for name in candidate):
                return candidate

    if checkpoint and all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in checkpoint.items()
    ):
        return checkpoint
    raise RuntimeError(
        f"Checkpoint '{path}' contains no model_state/state_dict tensor mapping."
    )


def _checkpoint_schema(state: Mapping[str, torch.Tensor]) -> str:
    keys = tuple(state)
    if any(key.endswith("patch_embed.patch_model.weight") for key in keys):
        return "pytorchvideo"
    if any(key.endswith("patch_embed.proj.weight") for key in keys):
        return "slowfast"
    return "unknown"


def _state_dict_candidates(
    state: Mapping[str, torch.Tensor],
) -> list[tuple[str, Mapping[str, torch.Tensor]]]:
    candidates: list[tuple[str, Mapping[str, torch.Tensor]]] = [("original", state)]
    prefixes = ("module.model.", "module.", "model.", "_orig_mod.")
    for prefix in prefixes:
        if state and all(key.startswith(prefix) for key in state):
            candidates.append(
                (
                    f"stripped '{prefix}'",
                    {key[len(prefix) :]: value for key, value in state.items()},
                )
            )
    return candidates


def _load_state_dict_strict(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    path: Path,
) -> None:
    expected = model.state_dict()
    best_description = "original"
    best_details: tuple[list[str], list[str], list[str]] | None = None
    best_score: int | None = None

    for description, candidate in _state_dict_candidates(state):
        missing = sorted(set(expected) - set(candidate))
        unexpected = sorted(set(candidate) - set(expected))
        mismatched = sorted(
            key
            for key in set(expected).intersection(candidate)
            if tuple(expected[key].shape) != tuple(candidate[key].shape)
        )
        score = len(missing) + len(unexpected) + len(mismatched)
        if best_score is None or score < best_score:
            best_score = score
            best_description = description
            best_details = (missing, unexpected, mismatched)
        if score == 0:
            # Keep strict=True even after the explicit compatibility audit.
            model.load_state_dict(candidate, strict=True)
            return

    assert best_details is not None
    missing, unexpected, mismatched = best_details

    def sample(items: list[str]) -> str:
        return ", ".join(items[:5]) if items else "none"

    raise RuntimeError(
        f"Checkpoint '{path}' is not strictly compatible with the constructed "
        f"model ({best_description}): missing={len(missing)} [{sample(missing)}], "
        f"unexpected={len(unexpected)} [{sample(unexpected)}], "
        f"shape_mismatches={len(mismatched)} [{sample(mismatched)}]."
    )


__all__ = ["MViTSlowFastModel", "MViT_B_32x3", "MViT_B_24_32x3"]
