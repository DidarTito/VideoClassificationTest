"""Strict adapters for the official UniFormer Kinetics-400 checkpoints."""
from pathlib import Path

import torch

from .common import (
    AdapterMetadata,
    ModelUnavailableError,
    ROOT,
    checkpoint_path as configured_checkpoint_path,
    isolated_vendor_package,
    load_exact_state_dict,
    normalize,
    require_file,
    sample_frames,
    validate_logits,
    canonical_dataset,
    dataset_class_names,
    dataset_num_classes,
)


_VENDOR_ROOT = ROOT / "third_party" / "UniFormer" / "video_classification"
_CLASS_MAP = _VENDOR_ROOT / "data_list" / "k400" / "kinetics_400_categroies.txt"


def _checkpoint_class_names():
    """Return the non-alphabetical label order used by official checkpoints."""
    class_map = require_file(_CLASS_MAP)
    names = [None] * 400
    for line in class_map.read_text(encoding="utf-8").splitlines():
        name, index = line.rsplit("\t", 1)
        index = int(index)
        if not 0 <= index < len(names) or names[index] is not None:
            raise RuntimeError(f"Invalid UniFormer class mapping entry: {line!r}")
        names[index] = name.strip()
    if any(name is None for name in names):
        raise RuntimeError(f"Incomplete UniFormer Kinetics class mapping: {class_map}")
    return names


class UniFormerModel(AdapterMetadata):
    """UniFormer using the checked-out official SlowFast implementation.

    The benchmark pipeline supplies ``(B,T,C,H,W)`` clips in ``[0,1]``;
    UniFormer consumes a one-element pathway list containing
    ``(B,C,T,H,W)``.
    """

    MODEL_ZOO = {
        "S": {
            "config": _VENDOR_ROOT / "exp" / "uniformer_s16x4_k400" / "config.yaml",
            "checkpoint": configured_checkpoint_path("uniformer", "uniformer_small_k400_16x4.pth"),
            "frames": 16,
            "sampling_rate": 4,
            "accuracy": 80.8,
            "gflops": 41.8,
        },
        "B": {
            "config": _VENDOR_ROOT / "exp" / "uniformer_b32x4_k400" / "config.yaml",
            "checkpoint": configured_checkpoint_path("uniformer", "uniformer_base_k400_32x4.pth"),
            "frames": 32,
            "sampling_rate": 4,
            "accuracy": 82.9,
            "gflops": 259.0,
        },
    }
    def __init__(self, variant="S", device="cuda", dataset="k400"):
        variant = variant.upper()
        if variant not in self.MODEL_ZOO:
            raise ValueError(f"Unknown UniFormer variant: {variant}")
        self.variant = variant
        self.dataset = canonical_dataset(dataset)
        self.info = dict(self.MODEL_ZOO[variant])
        if self.dataset == "ssv2":
            stem = "base" if variant == "B" else "small"
            self.info.update({
                "config": _VENDOR_ROOT / "exp" / f"uniformer_{variant.lower()}16_sthv2_prek400" / "config.yaml",
                # LEGACY released classifier, quarantined per Phase 3.
                "checkpoint": configured_checkpoint_path("legacy_ssv2_released", f"uniformer_{stem}_sthv2_16_prek400.pth"),
                "frames": 16,
                "sampling_rate": 4,
                # Official K400-pretrained 16x3x1 SSV2 model-zoo rows.
                "accuracy": 70.4 if variant == "B" else 67.7,
                "gflops": 290.0 if variant == "B" else 125.0,
            })

        if self.dataset == "k400" and variant == "B" and not Path(self.info["checkpoint"]).is_file():
            wrong = configured_checkpoint_path("legacy_ssv2_released", "uniformer_base_sthv2_16_prek400.pth")
            detail = (
                f" The present {wrong.name} checkpoint has a 174-class "
                "Something-Something-v2 head and is intentionally rejected."
                if wrong.is_file()
                else ""
            )
            raise ModelUnavailableError(
                "UniFormer-B Kinetics-400 cannot run: the exact 400-class "
                f"checkpoint is missing ({self.info['checkpoint']}).{detail}"
            )

        config_path = require_file(self.info["config"])
        checkpoint_path = require_file(self.info["checkpoint"])
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )

        # UniFormer and the local Meta SlowFast checkout both use the top-level
        # package name ``slowfast``. Isolate the import while constructing it.
        with isolated_vendor_package("slowfast", _VENDOR_ROOT):
            from slowfast.config.defaults import get_cfg
            from slowfast.models.uniformer import Uniformer

            cfg = get_cfg()
            cfg.merge_from_file(str(config_path))
            cfg.NUM_GPUS = 0
            cfg.UNIFORMER.PRETRAIN_NAME = None
            model = Uniformer(cfg)

        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise RuntimeError(f"Unexpected checkpoint payload in {checkpoint_path}")
        for key in ("model_state", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
        load_exact_state_dict(model, state, checkpoint_path)
        expected_classes = dataset_num_classes(self.dataset)
        if getattr(model.head, "out_features", None) != expected_classes:
            raise RuntimeError(f"{checkpoint_path} does not provide a {expected_classes}-class head")

        self.model = model.eval().to(self.device)
        self.frames = int(self.info["frames"])
        self.sampling_rate = int(self.info["sampling_rate"])
        self._class_names = (_checkpoint_class_names() if self.dataset == "k400"
                             else dataset_class_names(self.dataset))

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.frames).to(self.device)
        clip = normalize(clip, (0.45, 0.45, 0.45), (0.225, 0.225, 0.225))
        logits = self.model([clip.permute(0, 2, 1, 3, 4).contiguous()])
        return validate_logits(logits, batch_size=clip.shape[0], classes=self.num_classes)

    @property
    def name(self):
        return f"UniFormer-{self.variant}"

    @property
    def class_names(self):
        return list(self._class_names)
