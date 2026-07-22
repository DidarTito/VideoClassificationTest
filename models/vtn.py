"""Strict standalone port of the official ViT-B Video Transformer Network.

The upstream implementation lives in ``bomri/SlowFast``.  Keeping the small
model graph here avoids replacing the project's newer SlowFast checkout and
lets the complete local Kinetics-400 checkpoint run on current timm and
Transformers releases.
"""
import torch
from torch import nn
import torch.nn.functional as F

from .common import (
    AdapterMetadata,
    ROOT,
    load_exact_state_dict,
    normalize,
    require_file,
    sample_frames,
    validate_logits,
    require_dataset,
)


class _VTNNetwork(nn.Module):
    """Parameter graph matching bomri/SlowFast's registered ``VTN`` class."""

    def __init__(self):
        super().__init__()
        import timm
        from transformers import LongformerConfig, LongformerModel

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0,
            drop_path_rate=0.0,
            drop_rate=0.0,
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, 768))

        config = LongformerConfig(
            attention_mode="sliding_chunks",
            intermediate_size=3072,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            attention_window=[18, 18, 18],
            num_hidden_layers=3,
            num_attention_heads=12,
            pad_token_id=-1,
            max_position_embeddings=288,
            hidden_size=768,
            vocab_size=288,
        )
        self.temporal_encoder = LongformerModel(config, add_pooling_layer=False)
        # Official VTN supplies frame features through inputs_embeds and removes
        # this unused module. Its old Transformers version persisted the buffer.
        self.temporal_encoder.embeddings.word_embeddings = None
        embeddings = self.temporal_encoder.embeddings
        embeddings._buffers.pop("position_ids", None)
        embeddings.register_buffer(
            "position_ids", torch.arange(288).expand((1, -1)), persistent=True
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(768, 400),
        )

    def forward(self, video, frame_position_ids, backbone_chunk_size=8):
        batch, channels, frames, height, width = video.shape
        flat = video.permute(0, 2, 1, 3, 4).reshape(
            batch * frames, channels, height, width
        )
        features = []
        for start in range(0, flat.shape[0], backbone_chunk_size):
            features.append(self.backbone(flat[start : start + backbone_chunk_size]))
        features = torch.cat(features, dim=0).reshape(batch, frames, -1)

        cls_tokens = self.cls_token.expand(batch, -1, -1)
        embeddings = torch.cat((cls_tokens, features), dim=1)
        attention_mask = torch.ones(
            (batch, frames + 1), dtype=torch.long, device=video.device
        )
        attention_mask[:, 0] = 2  # global attention for the classification token

        # vtn_helper.pad_to_window_size_local pads to 2 * config window (36).
        window_multiple = 2 * self.temporal_encoder.config.attention_window[0]
        padding = (window_multiple - embeddings.shape[1] % window_multiple) % window_multiple
        embeddings = F.pad(
            embeddings.permute(0, 2, 1), (0, padding), value=-1
        ).permute(0, 2, 1)
        attention_mask = F.pad(attention_mask, (0, padding), value=0)
        position_ids = F.pad(frame_position_ids, (1, padding), value=0).long()

        token_type_ids = torch.zeros_like(attention_mask)
        token_type_ids[:, 0] = 1
        valid_mask = attention_mask.ne(0)
        position_ids = position_ids % 286
        position_ids[:, 0] = 286
        position_ids[~valid_mask] = 287

        encoded = self.temporal_encoder(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=embeddings,
            return_dict=True,
        ).last_hidden_state
        return self.mlp_head(encoded[:, 0])


class VTNModel(AdapterMetadata):
    """ViT-B-VTN with the official complete local Kinetics-400 weights."""

    MODEL_ZOO = {
        "B": {
            "checkpoint": ROOT / "checkpoints" / "vtn" / "VTN_VIT_B_KINETICS.pyth",
            # Official bomri/SlowFast model-zoo result for this exact checkpoint.
            "accuracy": 77.72,
            # Paper full-video protocol: 250 uniformly sampled frames.
            "gflops": 4218.0,
        }
    }
    FULL_VIDEO_FRAMES = 250
    TRAIN_SAMPLING_RATE = 8

    def __init__(self, variant="B", device="cuda", backbone_chunk_size=8, dataset="k400"):
        self.dataset = require_dataset(dataset, ("k400",), "VTN-B")
        variant = variant.upper()
        if variant != "B":
            raise ValueError(f"Only VTN-B is supported, received {variant}")
        self.variant = variant
        self.info = self.MODEL_ZOO[variant]
        self.device = torch.device(
            device if not str(device).startswith("cuda") or torch.cuda.is_available() else "cpu"
        )
        self.backbone_chunk_size = int(backbone_chunk_size)
        if self.backbone_chunk_size < 1:
            raise ValueError("backbone_chunk_size must be positive")

        checkpoint_path = require_file(self.info["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state"), dict
        ):
            raise RuntimeError(f"Unexpected VTN checkpoint structure: {checkpoint_path}")
        model = _VTNNetwork()
        load_exact_state_dict(model, checkpoint["model_state"], checkpoint_path)
        if model.mlp_head[-1].out_features != 400:
            raise RuntimeError(f"{checkpoint_path} does not provide a 400-class head")
        self.model = model.eval().to(self.device)
        self._last_frames = 32

    @torch.inference_mode()
    def __call__(self, clip):
        if clip.shape[1] > self.FULL_VIDEO_FRAMES:
            clip = sample_frames(clip, self.FULL_VIDEO_FRAMES)
        clip = normalize(
            clip.to(self.device),
            (0.45, 0.45, 0.45),
            (0.225, 0.225, 0.225),
        )
        frames = clip.shape[1]
        self._last_frames = int(frames)
        position_ids = torch.arange(frames, device=self.device).mul(
            self.TRAIN_SAMPLING_RATE
        )
        position_ids = position_ids.unsqueeze(0).expand(clip.shape[0], -1)
        video = clip.permute(0, 2, 1, 3, 4).contiguous()
        logits = self.model(video, position_ids, self.backbone_chunk_size)
        return validate_logits(logits, batch_size=clip.shape[0])

    @property
    def name(self):
        return f"VTN-B ({self._last_frames}-frame view)"

    @property
    def gflops(self):
        # Spatial ViT cost dominates and scales almost linearly with frames.
        return self.info["gflops"] * self._last_frames / self.FULL_VIDEO_FRAMES
