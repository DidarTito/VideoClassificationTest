"""Lazy model registry used by the benchmark commands.

Factories import their implementation only when that model is selected.  This
keeps a missing optional dependency from making every benchmark unavailable.
"""
from importlib import import_module


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


# The seventeen rows explicitly requested for the Pareto/device run.
REQUESTED_MODEL_KEYS = (
    "uniformer-b",
    "uniformer-s",
    "mvit-b-24-32x3",
    "mvit-v1-b",
    "videoswin-t",
    "videoswin-s",
    "videoswin-b",
    "vivit-s",
    "timesformer-b",
    "video-focalnet-t",
    "video-focalnet-s",
    "video-focalnet-b",
    "dualformer-t",
    "omnivore-b",
    "videomae-b",
    "svt-b",
    "vtn-b",
)


# Dataset availability means that an official supervised classifier exists
# and this repository has an exact adapter for it.  It deliberately does not
# mean that the checkpoint is already present on this machine; the dataset
# preflight reports that separately.  Keeping this matrix explicit prevents a
# K400 head, random head, or similarly named proxy architecture from entering
# an SSV2 run.
MODEL_DATASET_AVAILABILITY = {
    "uniformer-b": {"k400": True, "ssv2": True},
    "uniformer-s": {"k400": True, "ssv2": True},
    "mvit-b-24-32x3": {"k400": True, "ssv2": False},
    "mvit-v1-b": {"k400": True, "ssv2": False},
    "videoswin-t": {"k400": True, "ssv2": False},
    "videoswin-s": {"k400": True, "ssv2": False},
    "videoswin-b": {"k400": True, "ssv2": True},
    "vivit-s": {"k400": False, "ssv2": False},
    "timesformer-b": {"k400": True, "ssv2": True},
    "video-focalnet-t": {"k400": True, "ssv2": False},
    "video-focalnet-s": {"k400": True, "ssv2": False},
    "video-focalnet-b": {"k400": True, "ssv2": True},
    "dualformer-t": {"k400": True, "ssv2": False},
    "omnivore-b": {"k400": True, "ssv2": False},
    "videomae-b": {"k400": True, "ssv2": True},
    "svt-b": {"k400": False, "ssv2": False},
    "vtn-b": {"k400": True, "ssv2": False},
}

DATASET_MODEL_KEYS = {
    dataset: tuple(
        key
        for key in REQUESTED_MODEL_KEYS
        if MODEL_DATASET_AVAILABILITY[key][dataset]
    )
    for dataset in ("k400", "ssv2")
}


MODEL_REGISTRY = {
    # Existing, independently useful adapters.
    "timesformer-b": _lazy("models.timesformer", "TimeSformerModel", "B"),
    "timesformer-hr": _lazy("models.timesformer", "TimeSformerModel", "HR"),
    "videomae-b1600": _lazy("models.videomae", "VideoMAEModel", "B1600"),
    "videomae-b": _lazy("models.videomae", "VideoMAEModel", "B1600"),
    "videomae-l": _lazy("models.videomae", "VideoMAEModel", "L"),
    "videoswin-t": _lazy("models.videoswin", "VideoSwinModel", "T"),
    "videoswin-s": _lazy("models.videoswin", "VideoSwinModel", "S"),
    "videoswin-b": _lazy("models.videoswin", "VideoSwinModel", "B"),
    "mvit-v2-s": _lazy("models.mvit", "MViTModel", "V2-S"),
    "vivit-b": _lazy("models.vivit", "ViViTModel", "B"),
    "actionclip-b8": _lazy("models.actionclip", "ActionCLIPModel", "B8"),
    "actionclip-b16": _lazy("models.actionclip", "ActionCLIPModel", "B16"),
    "zeroi2v-b": _lazy("models.zeroi2v", "ZeroI2VModel", "B"),

    # Requested rows.  These point at exact/local adapters, never proxy models.
    "uniformer-s": _lazy("models.uniformer", "UniFormerModel", "S"),
    "uniformer-b": _lazy("models.uniformer", "UniFormerModel", "B"),
    "mvit-b-24-32x3": _lazy("models.mvit_slowfast", "MViTSlowFastModel", "B-24-32x3"),
    "mvit-v1-b": _lazy("models.mvit_slowfast", "MViTSlowFastModel", "B-32x3"),
    "vivit-s": _lazy("models.vivit", "ViViTModel", "S"),
    "video-focalnet-t": _lazy("models.videofocalnet", "VideoFocalNetModel", "T"),
    "video-focalnet-s": _lazy("models.videofocalnet", "VideoFocalNetModel", "S"),
    "video-focalnet-b": _lazy("models.videofocalnet", "VideoFocalNetModel", "B"),
    "dualformer-t": _lazy("models.dualformer", "DualFormerModel", "T"),
    "omnivore-b": _lazy("models.omnivore", "OmnivoreModel", "B-IN21K"),
    "svt-b": _lazy("models.svt", "SVTModel", "B"),
    "vtn-b": _lazy("models.vtn", "VTNModel", "B"),

    # Preserve access to the old torchvision 16x4 recipe under an honest key.
    "mvit-v1-b-16x4": _lazy("models.mvit", "MViTModel", "V1-B"),
    "omnivore-b-standard": _lazy("models.omnivore", "OmnivoreModel", "B-STANDARD"),
}


__all__ = [
    "MODEL_REGISTRY",
    "REQUESTED_MODEL_KEYS",
    "MODEL_DATASET_AVAILABILITY",
    "DATASET_MODEL_KEYS",
    "LazyFactory",
]
