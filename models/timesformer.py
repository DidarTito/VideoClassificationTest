"""Checkpoint-local TimeSformer adapter (no network downloads)."""
import importlib
import sys
import types
import collections.abc

import torch
import torch.nn.functional as F

from .common import (AdapterMetadata, ROOT, normalize, require_dataset,
                     require_file, sample_frames, validate_logits)


class TimeSformerModel(AdapterMetadata):
    MODEL_ZOO = {
        "B": {"checkpoint": ROOT / "checkpoints" / "timesformer" /
              "TimeSformer_divST_8x32_224_K400.pyth", "frames": 8, "size": 224,
              "accuracy": 78.0, "gflops": 196.0},
        "HR": {"checkpoint": ROOT / "checkpoints" / "timesformer" /
               "TimeSformer_divST_16x16_448_K400.pyth", "frames": 16, "size": 448,
               "accuracy": 79.7, "gflops": 1703.0},
    }
    SSV2_MODEL_ZOO = {
        "B": {
            "checkpoint": ROOT / "checkpoints" / "timesformer" /
                          "TimeSformer_divST_8_224_SSv2.pyth",
            "frames": 8,
            "size": 224,
            "accuracy": 59.1,
            "gflops": 196.0,
        },
    }

    @staticmethod
    def _strict_checkpoint_state(checkpoint):
        """Return the exact model state stored by the official trainer.

        TimeSformer's upstream convenience loader uses ``strict=False`` and
        can silently leave a randomly initialized classifier behind.  The
        benchmark must never do that, especially when switching between the
        400-way K400 and 174-way SSV2 heads.
        """
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("model_state"), dict
        ):
            raise RuntimeError(
                f"Unexpected TimeSformer checkpoint structure: {checkpoint}"
            )
        state = {}
        for key, value in payload["model_state"].items():
            if key.startswith("model."):
                key = key[len("model."):]
            elif key.startswith("module."):
                key = key[len("module."):]
            state[key] = value
        return state

    def __init__(self, variant="B", device="cuda", dataset="k400"):
        self.dataset = require_dataset(dataset, ("k400", "ssv2"), "TimeSformer")
        self.variant = variant.upper()
        model_zoo = self.SSV2_MODEL_ZOO if self.dataset == "ssv2" else self.MODEL_ZOO
        if self.variant not in model_zoo:
            raise ValueError(f"Unknown TimeSformer variant: {variant}")
        self.info = model_zoo[self.variant]
        checkpoint = require_file(self.info["checkpoint"])
        vendor = ROOT / "third_party" / "TimeSformer"
        sys.path.insert(0, str(vendor))
        try:
            # Upstream models/__init__.py eagerly imports obsolete SlowFast
            # helpers.  The TimeSformer graph only needs its own submodules.
            importlib.import_module("timesformer")
            if "torch._six" not in sys.modules:
                torch_six = types.ModuleType("torch._six")
                torch_six.container_abcs = collections.abc
                sys.modules["torch._six"] = torch_six
            namespace = types.ModuleType("timesformer.models")
            namespace.__path__ = [str(vendor / "timesformer" / "models")]
            namespace.__package__ = "timesformer.models"
            sys.modules["timesformer.models"] = namespace
            TimeSformer = importlib.import_module("timesformer.models.vit").TimeSformer
            model = TimeSformer(
                img_size=self.info["size"], patch_size=16,
                num_classes=self.num_classes,
                num_frames=self.info["frames"], attention_type="divided_space_time",
                pretrained_model=str(checkpoint),
            )
            # Re-load strictly after the upstream constructor.  Its loader is
            # intentionally permissive for transfer learning; this benchmark
            # requires an exact supervised classifier instead.
            state = self._strict_checkpoint_state(checkpoint)
            try:
                model.model.load_state_dict(state, strict=True)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Checkpoint is incompatible with {checkpoint}: {exc}"
                ) from exc
        finally:
            try:
                sys.path.remove(str(vendor))
            except ValueError:
                pass
        self.device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.info["frames"])
        size = self.info["size"]
        if clip.shape[-2:] != (size, size):
            b, t, c, h, w = clip.shape
            clip = F.interpolate(clip.reshape(b * t, c, h, w), (size, size),
                                 mode="bilinear", align_corners=False).reshape(b, t, c, size, size)
        clip = normalize(clip, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        logits = self.model(clip.permute(0, 2, 1, 3, 4).contiguous().to(self.device))
        return validate_logits(
            logits, batch_size=clip.shape[0], classes=self.num_classes
        )

    @property
    def name(self):
        return f"TimeSformer-{self.variant}"
