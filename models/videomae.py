"""VideoMAE classifier adapter with offline-only checkpoint loading.

The ``VideoMAE-ViT-*_1600epoch.pth`` workspace files are pretraining
encoder/decoder checkpoints and do not contain a K400 classifier.  They are
never mislabeled as fine-tuned models here.  Put a Transformers export in
``checkpoints/videomae/base-finetuned-kinetics`` or use an already populated
Hugging Face cache.
"""
import importlib.util
from pathlib import Path
import sys

import torch

from .common import (
    AdapterMetadata,
    ModelUnavailableError,
    ROOT,
    dataset_class_names,
    normalize,
    require_dataset,
    require_file,
    sample_frames,
    validate_logits,
)


def _restore_qv_biases(model, source):
    """Map the checkpoint's q_bias/v_bias names to Transformers 5.x."""
    from safetensors.torch import load_file
    source_path = Path(source)
    if source_path.is_dir():
        weights_path = source_path / "model.safetensors"
    else:
        from transformers.utils.hub import cached_file
        weights_path = Path(cached_file(source, "model.safetensors", local_files_only=True))
    state = load_file(str(weights_path), device="cpu")
    restored = 0
    for index, layer in enumerate(model.videomae.encoder.layer):
        attention = layer.attention.attention
        prefix = f"videomae.encoder.layer.{index}.attention.attention"
        q_bias = state.get(f"{prefix}.q_bias")
        v_bias = state.get(f"{prefix}.v_bias")
        if q_bias is None or v_bias is None:
            continue
        attention.query.bias.data.copy_(q_bias)
        attention.value.bias.data.copy_(v_bias)
        attention.key.bias.data.zero_()
        restored += 1
    if restored != len(model.videomae.encoder.layer):
        raise RuntimeError(
            f"Restored VideoMAE q/v biases for {restored} layers; expected "
            f"{len(model.videomae.encoder.layer)} from {weights_path}"
        )


def _load_official_ssv2_model(checkpoint):
    """Strict-load the official 174-way VideoMAE-B fine-tuned graph."""
    source_path = ROOT / "third_party" / "VideoMAE" / "modeling_finetune.py"
    require_file(source_path)
    module_name = "_benchmark_official_videomae_finetune"
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load official VideoMAE source from {source_path}"
            )
        module = importlib.util.module_from_spec(spec)
        # timm's register_model decorator looks the defining module up while
        # the file is executing, so it must already be visible here.
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
    model = module.vit_base_patch16_224(
        pretrained=False,
        num_classes=174,
        all_frames=16,
        tubelet_size=2,
        use_mean_pooling=True,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("module"), dict):
        raise RuntimeError(f"Unexpected VideoMAE checkpoint structure: {checkpoint}")
    try:
        model.load_state_dict(payload["module"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint is incompatible with {checkpoint}: {exc}"
        ) from exc
    if getattr(model.head, "out_features", None) != 174:
        raise RuntimeError(f"{checkpoint} does not provide a 174-class head")
    return model


class VideoMAEModel(AdapterMetadata):
    MODEL_ZOO = {
        "B1600": {"local": ROOT / "checkpoints" / "videomae" / "base-finetuned-kinetics",
                  "hf_id": "MCG-NJU/videomae-base-finetuned-kinetics",
                  "frames": 16, "accuracy": 80.9, "gflops": 180.0},
        "L": {"local": ROOT / "checkpoints" / "videomae" / "large-finetuned-kinetics",
              "hf_id": "MCG-NJU/videomae-large-finetuned-kinetics",
              "frames": 16, "accuracy": 84.7, "gflops": 597.0},
    }
    SSV2_MODEL_ZOO = {
        "B1600": {
            "checkpoint": ROOT / "checkpoints" / "videomae" /
                          "videomae_vit_b_ssv2_2400e.pth",
            "frames": 16,
            "accuracy": 70.8,
            "gflops": 180.0,
        },
    }

    def __init__(self, variant="B1600", device="cuda", dataset="k400"):
        self.dataset = require_dataset(dataset, ("k400", "ssv2"), "VideoMAE")
        self.variant = variant.upper()
        model_zoo = self.SSV2_MODEL_ZOO if self.dataset == "ssv2" else self.MODEL_ZOO
        if self.variant not in model_zoo:
            if self.dataset == "ssv2":
                raise ModelUnavailableError(
                    f"VideoMAE-{self.variant} has no SSV2 classifier checkpoint"
                )
            raise ValueError(f"Unknown VideoMAE variant: {variant}")
        self.info = model_zoo[self.variant]
        if self.dataset == "ssv2":
            checkpoint = require_file(self.info["checkpoint"])
            model = _load_official_ssv2_model(checkpoint)
        else:
            from transformers import VideoMAEForVideoClassification
            source = str(
                self.info["local"]
                if Path(self.info["local"]).is_dir()
                else self.info["hf_id"]
            )
            try:
                model = VideoMAEForVideoClassification.from_pretrained(
                    source, local_files_only=True
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Fine-tuned VideoMAE classifier is not available offline. Expected "
                    f"{self.info['local']} or cached {self.info['hf_id']}. The 1600epoch .pth "
                    "files are pretraining checkpoints without classification heads."
                ) from exc
            _restore_qv_biases(model, source)
            if model.classifier.out_features != 400:
                raise RuntimeError(f"{source} does not provide a 400-class head")
        self.device = torch.device(device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu")
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def __call__(self, clip):
        clip = sample_frames(clip, self.info["frames"]).to(self.device)
        clip = normalize(clip, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        if self.dataset == "ssv2":
            logits = self.model(clip.permute(0, 2, 1, 3, 4).contiguous())
        else:
            logits = self.model(pixel_values=clip).logits
        return validate_logits(
            logits, batch_size=clip.shape[0], classes=self.num_classes
        )

    @property
    def name(self):
        return "VideoMAE-B" if self.variant == "B1600" else "VideoMAE-L"

    @property
    def class_names(self):
        if self.dataset == "ssv2":
            return dataset_class_names(self.dataset)
        labels = self.model.config.id2label
        return [labels[index] for index in range(len(labels))]
