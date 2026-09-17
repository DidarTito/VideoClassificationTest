"""Lazy model registry and authoritative Stage 2 model metadata.

Factories import implementation modules only when a model is selected.  The
requested-model metadata is deliberately lightweight as well: importing this
package does not import Torch, third-party model sources, or checkpoint files.

``REQUESTED_MODEL_SPECS`` is the single source of truth for the 17 Stage 2
candidates.  The legacy ``MODEL_REGISTRY`` and availability constants remain
available for existing commands and are derived from those specifications.
"""
from dataclasses import dataclass
from importlib import import_module
from typing import Mapping


class LazyFactory:
    """Import and construct one adapter on demand."""

    def __init__(self, module, class_name, *args, **kwargs):
        self.module = module
        self.class_name = class_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, **overrides):
        cls = getattr(import_module(self.module), self.class_name)
        kwargs = dict(self.kwargs)
        kwargs.update(overrides)
        return cls(*self.args, **kwargs)

    def __repr__(self):
        return f"LazyFactory({self.module}.{self.class_name})"


def _lazy(module, class_name, *args, **kwargs):
    return LazyFactory(module, class_name, *args, **kwargs)


@dataclass(frozen=True)
class PublishedValues:
    """Published values associated with one exact model/checkpoint recipe."""

    top1_percent: float
    params_m: float
    gflops: float


@dataclass(frozen=True)
class PreprocessingSpec:
    """Model-side preprocessing after the common 32-frame video decode.

    ``frames`` is the actual temporal length received by the network, not the
    number of frames initially decoded.  The common decoded input is a batch-1
    ``(B,T,C,H,W)`` float32 tensor in ``[0, 1]``.
    """

    frames: int
    resolution: tuple[int, int]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    model_input_layout: str
    temporal_sampling: str = "uniform resampling from the 32-frame full-video view"
    decoded_frames: int = 32


@dataclass(frozen=True)
class DatasetModelSpec:
    """Availability and scientific identity for one model/dataset pair."""

    dataset: str
    num_classes: int
    available: bool
    checkpoint: str | None
    preprocessing: PreprocessingSpec | None
    published: PublishedValues | None
    availability_reason: str
    incompatible_artifact: str | None = None


@dataclass(frozen=True)
class RequestedModelSpec:
    """Authoritative metadata for one requested Stage 2 candidate."""

    key: str
    display_name: str
    builder: LazyFactory
    screening: PublishedValues
    datasets: Mapping[str, DatasetModelSpec]

    def for_dataset(self, dataset: str) -> DatasetModelSpec:
        """Return the exact dataset pair, accepting the public dataset aliases."""
        key = str(dataset).strip().lower().replace("_", "-")
        aliases = {
            "k400": "k400",
            "kinetics400": "k400",
            "kinetics-400": "k400",
            "ssv2": "ssv2",
            "sth-v2": "ssv2",
            "sthv2": "ssv2",
            "something-something-v2": "ssv2",
        }
        try:
            return self.datasets[aliases[key]]
        except KeyError as exc:
            raise ValueError(f"Unsupported dataset {dataset!r}") from exc


_MEAN_045 = (0.45, 0.45, 0.45)
_STD_0225 = (0.225, 0.225, 0.225)
_MEAN_IMAGENET = (0.485, 0.456, 0.406)
_STD_IMAGENET = (0.229, 0.224, 0.225)
_RESOLUTION = (224, 224)


def _preprocessing(frames, mean, std, layout, temporal_sampling=None):
    kwargs = {}
    if temporal_sampling is not None:
        kwargs["temporal_sampling"] = temporal_sampling
    return PreprocessingSpec(
        frames=frames,
        resolution=_RESOLUTION,
        mean=mean,
        std=std,
        model_input_layout=layout,
        **kwargs,
    )


_P16_045_PATHWAY = _preprocessing(
    16, _MEAN_045, _STD_0225, "one-element pathway list containing B,C,T,H,W"
)
_P32_045_PATHWAY = _preprocessing(
    32, _MEAN_045, _STD_0225, "one-element pathway list containing B,C,T,H,W"
)
_P32_045_BCTHW = _preprocessing(32, _MEAN_045, _STD_0225, "B,C,T,H,W")
_P8_045_BCTHW = _preprocessing(8, _MEAN_045, _STD_0225, "B,C,T,H,W")
_P32_IMAGENET_BCTHW = _preprocessing(
    32, _MEAN_IMAGENET, _STD_IMAGENET, "B,C,T,H,W"
)
_P8_IMAGENET_BTCHW = _preprocessing(
    8, _MEAN_IMAGENET, _STD_IMAGENET, "B,T,C,H,W"
)
_P16_IMAGENET_BTCHW = _preprocessing(
    16, _MEAN_IMAGENET, _STD_IMAGENET, "B,T,C,H,W"
)
_P16_IMAGENET_BCTHW = _preprocessing(
    16, _MEAN_IMAGENET, _STD_IMAGENET, "B,C,T,H,W"
)
_P32_VTN = _preprocessing(
    32,
    _MEAN_045,
    _STD_0225,
    "B,C,T,H,W",
    temporal_sampling="use all 32 decoded frames (cap at 250 frames)",
)


def _available(dataset, classes, checkpoint, preprocessing, top1, params_m, gflops):
    return DatasetModelSpec(
        dataset=dataset,
        num_classes=classes,
        available=True,
        checkpoint=checkpoint,
        preprocessing=preprocessing,
        published=PublishedValues(top1, params_m, gflops),
        availability_reason="exact supervised classifier pair is registered",
    )


def _unavailable(dataset, classes, reason, incompatible_artifact=None):
    return DatasetModelSpec(
        dataset=dataset,
        num_classes=classes,
        available=False,
        checkpoint=None,
        preprocessing=None,
        published=None,
        availability_reason=reason,
        incompatible_artifact=incompatible_artifact,
    )


def _datasets(k400, ssv2):
    return {"k400": k400, "ssv2": ssv2}


# Stage 1 screening values remain explicit and separate from the dataset-pair
# values below.  They are not silently reused as exact-checkpoint metadata when
# the published screening row and runnable checkpoint recipe differ.
_REQUESTED_SPECS = (
    RequestedModelSpec(
        "uniformer-b",
        "UniFormer-B",
        _lazy("models.uniformer", "UniFormerModel", "B"),
        PublishedValues(82.9, 50.3, 259.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/uniformer/uniformer_base_k400_32x4.pth",
                _P32_045_PATHWAY, 82.9, 50.3, 259.0,
            ),
            _available(
                "ssv2", 174,
                "checkpoints/legacy_ssv2_released/uniformer_base_sthv2_16_prek400.pth",
                _P16_045_PATHWAY, 70.4, 49.7, 290.0,
            ),
        ),
    ),
    RequestedModelSpec(
        "uniformer-s",
        "UniFormer-S",
        _lazy("models.uniformer", "UniFormerModel", "S"),
        PublishedValues(80.8, 21.5, 41.8),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/uniformer/uniformer_small_k400_16x4.pth",
                _P16_045_PATHWAY, 80.8, 21.5, 41.8,
            ),
            _available(
                "ssv2", 174,
                "checkpoints/legacy_ssv2_released/uniformer_small_sthv2_16_prek400.pth",
                _P16_045_PATHWAY, 67.7, 21.3, 125.0,
            ),
        ),
    ),
    RequestedModelSpec(
        "mvit-b-24-32x3",
        "MViT-B-24, 32x3",
        _lazy("models.mvit_slowfast", "MViTSlowFastModel", "B-24-32x3"),
        PublishedValues(81.2, 52.9, 236.0),
        _datasets(
            _available(
                "k400", 400, "checkpoints/mvit/MVIT_B_24_32x3.pyth",
                _P32_045_BCTHW, 80.4, 52.9, 236.0,
            ),
            _unavailable(
                "ssv2", 174,
                "no exact supervised SSV2 checkpoint matches the selected MViT-B-24 32x3 v1 configuration",
            ),
        ),
    ),
    RequestedModelSpec(
        "mvit-v1-b",
        "MViT-B, 32x3",
        _lazy("models.mvit_slowfast", "MViTSlowFastModel", "B-32x3"),
        PublishedValues(80.2, 36.6, 170.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/mvit/MVIT_B_32x3_f294077834.pyth",
                _P32_045_BCTHW, 80.2, 36.6, 170.0,
            ),
            _unavailable(
                "ssv2", 174,
                "no exact supervised SSV2 checkpoint matches the selected MViT-B 32x3 v1 configuration",
            ),
        ),
    ),
    RequestedModelSpec(
        "videoswin-t",
        "VideoSwin-T",
        _lazy("models.videoswin", "VideoSwinModel", "T"),
        PublishedValues(78.8, 28.2, 88.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/videoswin/swin_tiny_patch244_window877_kinetics400_1k.pth",
                _P32_IMAGENET_BCTHW, 78.8, 28.2, 88.0,
            ),
            _unavailable(
                "ssv2", 174,
                "the integrated Video Swin SSV2 release has Base but no exact Tiny classifier pair",
            ),
        ),
    ),
    RequestedModelSpec(
        "videoswin-s",
        "VideoSwin-S",
        _lazy("models.videoswin", "VideoSwinModel", "S"),
        PublishedValues(80.6, 49.8, 166.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/videoswin/swin_small_patch244_window877_kinetics400_1k.pth",
                _P32_IMAGENET_BCTHW, 80.6, 49.8, 166.0,
            ),
            _unavailable(
                "ssv2", 174,
                "the integrated Video Swin SSV2 release has Base but no exact Small classifier pair",
            ),
        ),
    ),
    RequestedModelSpec(
        "videoswin-b",
        "VideoSwin-B",
        _lazy("models.videoswin", "VideoSwinModel", "B"),
        PublishedValues(80.6, 88.0, 282.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/videoswin/swin_base_patch244_window877_kinetics400_1k.pth",
                _P32_IMAGENET_BCTHW, 80.6, 88.0, 282.0,
            ),
            _available(
                "ssv2", 174,
                "checkpoints/legacy_ssv2_released/swin_base_patch244_window1677_sthv2.pth",
                _P32_IMAGENET_BCTHW, 69.6, 88.0, 320.6,
            ),
        ),
    ),
    RequestedModelSpec(
        "vivit-s",
        "ViViT-S",
        _lazy("models.vivit", "ViViTModel", "S"),
        PublishedValues(78.5, 31.8, 55.0),
        _datasets(
            _unavailable(
                "k400", 400,
                "no exact ViViT-S K400 architecture/checkpoint is present; the available ViViT-B weight is not a substitute",
                "checkpoints/vivit/k400-vit-b-16-f8.pt",
            ),
            _unavailable(
                "ssv2", 174,
                "no exact supervised ViViT-S SSV2 architecture/checkpoint pair is registered",
            ),
        ),
    ),
    RequestedModelSpec(
        "timesformer-b",
        "TimeSformer-B",
        _lazy("models.timesformer", "TimeSformerModel", "B"),
        PublishedValues(78.0, 121.4, 196.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/timesformer/TimeSformer_divST_8x32_224_K400.pyth",
                _P8_045_BCTHW, 78.0, 121.4, 196.0,
            ),
            _available(
                "ssv2", 174,
                "checkpoints/legacy_ssv2_released/TimeSformer_divST_8_224_SSv2.pyth",
                _P8_045_BCTHW, 59.1, 121.4, 196.0,
            ),
        ),
    ),
    RequestedModelSpec(
        "video-focalnet-t",
        "Video-FocalNet-T",
        _lazy("models.videofocalnet", "VideoFocalNetModel", "T"),
        PublishedValues(79.8, 28.0, 63.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/focalnet/video-focalnet_tiny_kinetics400.pth",
                _P8_IMAGENET_BTCHW, 79.8, 28.0, 63.0,
            ),
            _unavailable(
                "ssv2", 174,
                "the integrated Video-FocalNets SSV2 release has Base but no exact Tiny classifier pair",
            ),
        ),
    ),
    RequestedModelSpec(
        "video-focalnet-s",
        "Video-FocalNet-S",
        _lazy("models.videofocalnet", "VideoFocalNetModel", "S"),
        PublishedValues(81.4, 49.0, 124.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/focalnet/video-focalnet_small_kinetics400.pth",
                _P8_IMAGENET_BTCHW, 81.4, 49.0, 124.0,
            ),
            _unavailable(
                "ssv2", 174,
                "the integrated Video-FocalNets SSV2 release has Base but no exact Small classifier pair",
            ),
        ),
    ),
    RequestedModelSpec(
        "video-focalnet-b",
        "Video-FocalNet-B",
        _lazy("models.videofocalnet", "VideoFocalNetModel", "B"),
        PublishedValues(83.6, 88.0, 149.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/focalnet/video-focalnet_base_kinetics400.pth",
                _P8_IMAGENET_BTCHW, 83.6, 88.0, 149.0,
            ),
            _available(
                "ssv2", 174,
                "checkpoints/legacy_ssv2_released/video-focalnet_base_ssv2.pth",
                _P8_IMAGENET_BTCHW, 71.1, 88.0, 149.0,
            ),
        ),
    ),
    RequestedModelSpec(
        "dualformer-t",
        "DualFormer-T",
        _lazy("models.dualformer", "DualFormerModel", "T"),
        PublishedValues(79.5, 21.8, 240.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/dualformer/dualformer_tiny_patch244_window877.pth",
                _P32_IMAGENET_BCTHW, 79.5, 21.8, 240.0,
            ),
            _unavailable(
                "ssv2", 174,
                "no exact released and registered supervised SSV2 classifier exists for DualFormer-T",
            ),
        ),
    ),
    RequestedModelSpec(
        "omnivore-b",
        "Omnivore-B (IN21K)",
        _lazy("models.omnivore", "OmnivoreModel", "B-IN21K"),
        PublishedValues(84.0, 88.0, 282.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/omnivore/swinB_In21k_checkpoint.torch",
                _P32_IMAGENET_BCTHW, 84.0, 88.0, 282.0,
            ),
            _unavailable(
                "ssv2", 174,
                "no official registered 174-class SSV2 head exists for Omnivore-B",
            ),
        ),
    ),
    RequestedModelSpec(
        "videomae-b",
        "VideoMAE-B",
        _lazy("models.videomae", "VideoMAEModel", "B1600"),
        PublishedValues(81.5, 87.0, 180.0),
        _datasets(
            _available(
                "k400", 400,
                "checkpoints/videomae/videomae_vit_b_k400_1600e_ft.pth",
                _P16_IMAGENET_BTCHW, 81.5, 87.0, 180.0,
            ),
            _available(
                "ssv2", 174,
                "checkpoints/legacy_ssv2_released/videomae_vit_b_ssv2_2400e.pth",
                _P16_IMAGENET_BCTHW, 70.8, 86.2, 180.0,
            ),
        ),
    ),
    RequestedModelSpec(
        "svt-b",
        "SVT-B",
        _lazy("models.svt", "SVTModel", "B"),
        PublishedValues(78.1, 86.0, 180.0),
        _datasets(
            _unavailable(
                "k400", 400,
                "the local SVT artifact is a self-supervised backbone/projection checkpoint without a trained 400-class head",
                "checkpoints/svt/kinetics400_vitb_ssl.pth",
            ),
            _unavailable(
                "ssv2", 174,
                "no exact supervised SVT-B SSV2 classifier pair is registered",
            ),
        ),
    ),
    RequestedModelSpec(
        "vtn-b",
        "VTN-B",
        _lazy("models.vtn", "VTNModel", "B"),
        PublishedValues(78.6, 114.0, 218.0),
        _datasets(
            _available(
                "k400", 400, "checkpoints/vtn/VTN_VIT_B_KINETICS.pyth",
                _P32_VTN, 77.72, 114.0, 4218.0,
            ),
            _unavailable(
                "ssv2", 174,
                "no exact released and registered supervised SSV2 classifier exists for VTN-B",
            ),
        ),
    ),
)


REQUESTED_MODEL_SPECS = {spec.key: spec for spec in _REQUESTED_SPECS}
REQUESTED_MODEL_KEYS = tuple(REQUESTED_MODEL_SPECS)

# These compatibility views are derived from the authoritative dataset specs.
MODEL_DATASET_AVAILABILITY = {
    key: {
        dataset: spec.datasets[dataset].available
        for dataset in ("k400", "ssv2")
    }
    for key, spec in REQUESTED_MODEL_SPECS.items()
}
DATASET_MODEL_KEYS = {
    dataset: tuple(
        key
        for key in REQUESTED_MODEL_KEYS
        if REQUESTED_MODEL_SPECS[key].datasets[dataset].available
    )
    for dataset in ("k400", "ssv2")
}


# Keep this as a literal dictionary: the read-only asset inspector parses its
# keys with ``ast`` so it can verify registration without importing models.
# Values for every requested key come from REQUESTED_MODEL_SPECS; additional
# legacy/experimental keys remain available for backwards compatibility.
MODEL_REGISTRY = {
    # Existing, independently useful adapters.
    "timesformer-b": REQUESTED_MODEL_SPECS["timesformer-b"].builder,
    "timesformer-hr": _lazy("models.timesformer", "TimeSformerModel", "HR"),
    "videomae-b1600": _lazy("models.videomae", "VideoMAEModel", "B1600"),
    "videomae-b": REQUESTED_MODEL_SPECS["videomae-b"].builder,
    "videomae-l": _lazy("models.videomae", "VideoMAEModel", "L"),
    "videoswin-t": REQUESTED_MODEL_SPECS["videoswin-t"].builder,
    "videoswin-s": REQUESTED_MODEL_SPECS["videoswin-s"].builder,
    "videoswin-b": REQUESTED_MODEL_SPECS["videoswin-b"].builder,
    "mvit-v2-s": _lazy("models.mvit", "MViTModel", "V2-S"),
    "vivit-b": _lazy("models.vivit", "ViViTModel", "B"),
    "actionclip-b8": _lazy("models.actionclip", "ActionCLIPModel", "B8"),
    "actionclip-b16": _lazy("models.actionclip", "ActionCLIPModel", "B16"),
    "zeroi2v-b": _lazy("models.zeroi2v", "ZeroI2VModel", "B"),

    # Requested rows. These point at exact/local adapters, never proxy models.
    "uniformer-s": REQUESTED_MODEL_SPECS["uniformer-s"].builder,
    "uniformer-b": REQUESTED_MODEL_SPECS["uniformer-b"].builder,
    "mvit-b-24-32x3": REQUESTED_MODEL_SPECS["mvit-b-24-32x3"].builder,
    "mvit-v1-b": REQUESTED_MODEL_SPECS["mvit-v1-b"].builder,
    "vivit-s": REQUESTED_MODEL_SPECS["vivit-s"].builder,
    "video-focalnet-t": REQUESTED_MODEL_SPECS["video-focalnet-t"].builder,
    "video-focalnet-s": REQUESTED_MODEL_SPECS["video-focalnet-s"].builder,
    "video-focalnet-b": REQUESTED_MODEL_SPECS["video-focalnet-b"].builder,
    "dualformer-t": REQUESTED_MODEL_SPECS["dualformer-t"].builder,
    "omnivore-b": REQUESTED_MODEL_SPECS["omnivore-b"].builder,
    "svt-b": REQUESTED_MODEL_SPECS["svt-b"].builder,
    "vtn-b": REQUESTED_MODEL_SPECS["vtn-b"].builder,

    # Preserve access to old recipes under honest keys. The torchvision port
    # of MViT-B 16x4 is NOT the frozen17 exact checkpoint; it keeps a
    # distinct key so it can never contaminate a frozen17 run.
    "mvit-v1-b-16x4-torchvision": _lazy("models.mvit", "MViTModel", "V1-B"),
    "omnivore-b-standard": _lazy("models.omnivore", "OmnivoreModel", "B-STANDARD"),

    # Frozen K400 -> SSV2 canonical names. Legacy names above remain aliases.
    "mvit-v1-b-16x4": _lazy("models.mvit_slowfast", "MViTSlowFastModel", "B-16x4"),
    "mvit-v1-b-32x3": REQUESTED_MODEL_SPECS["mvit-v1-b"].builder,
    "zeroi2v-b16-8f": _lazy("models.zeroi2v", "ZeroI2VModel", "B"),
    "omnivore-b-in21k": REQUESTED_MODEL_SPECS["omnivore-b"].builder,
    "dualformer-s": _lazy("models.dualformer", "DualFormerModel", "S"),
    "dualformer-b-in21k": _lazy("models.dualformer", "DualFormerModel", "B-IN21K"),
}

# The frozen 17 canonical benchmark keys (single source of numeric truth:
# configs/frozen17.yaml). This tuple is the ACTIVE run set; every other
# MODEL_REGISTRY key is a legacy/compatibility alias and must never be part
# of a frozen17 run.
FROZEN17_MODEL_KEYS = (
    "uniformer-s",
    "dualformer-t",
    "video-focalnet-t",
    "videoswin-t",
    "mvit-v1-b-16x4",
    "video-focalnet-s",
    "mvit-v1-b-32x3",
    "dualformer-s",
    "videoswin-s",
    "zeroi2v-b16-8f",
    "video-focalnet-b",
    "uniformer-b",
    "videomae-b",
    "dualformer-b-in21k",
    "omnivore-b-in21k",
    "videoswin-b",
    "timesformer-b",
)


__all__ = [
    "MODEL_REGISTRY",
    "FROZEN17_MODEL_KEYS",
    "REQUESTED_MODEL_SPECS",
    "REQUESTED_MODEL_KEYS",
    "MODEL_DATASET_AVAILABILITY",
    "DATASET_MODEL_KEYS",
    "LazyFactory",
    "PublishedValues",
    "PreprocessingSpec",
    "DatasetModelSpec",
    "RequestedModelSpec",
]
