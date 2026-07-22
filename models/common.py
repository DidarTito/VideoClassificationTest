"""Shared helpers for benchmark model adapters."""
from contextlib import contextmanager
from pathlib import Path
import importlib
import json
import sys

import torch

from .kinetics_labels import kinetics400_classes

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "k400": {"num_classes": 400, "directory": "kinetics400"},
    "ssv2": {"num_classes": 174, "directory": "ssv2"},
}
DATASET_ALIASES = {
    "k400": "k400",
    "kinetics400": "k400",
    "kinetics-400": "k400",
    "ssv2": "ssv2",
    "sth-v2": "ssv2",
    "sthv2": "ssv2",
    "something-something-v2": "ssv2",
}


class ModelUnavailableError(RuntimeError):
    """The exact requested architecture/checkpoint is not available locally."""


def canonical_dataset(dataset):
    key = str(dataset or "k400").strip().lower().replace("_", "-")
    try:
        return DATASET_ALIASES[key]
    except KeyError as exc:
        choices = ", ".join(DATASETS)
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of: {choices}") from exc


def dataset_num_classes(dataset):
    return int(DATASETS[canonical_dataset(dataset)]["num_classes"])


def require_dataset(dataset, supported, model_name):
    dataset = canonical_dataset(dataset)
    supported = tuple(canonical_dataset(item) for item in supported)
    if dataset not in supported:
        raise ModelUnavailableError(
            f"{model_name} has no {dataset.upper()} classifier checkpoint in the "
            "checkpoints folder. Available dataset(s): " + ", ".join(supported)
        )
    return dataset


def dataset_class_names(dataset, dataset_root=None):
    """Return folder labels when available, otherwise stable index labels.

    SSV2 label text is supplied by the dataset annotations rather than encoded
    in these checkpoints.  Once the dataset is installed using one directory
    per class, the same measured-accuracy path used for K400 works unchanged.
    """
    dataset = canonical_dataset(dataset)
    root = Path(dataset_root) if dataset_root else ROOT / "datasets" / DATASETS[dataset]["directory"]
    if root.is_dir():
        names = sorted(path.name for path in root.iterdir() if path.is_dir())
        if len(names) == dataset_num_classes(dataset):
            return names
    if dataset == "k400":
        return kinetics400_classes()
    label_file = root / "labels" / "labels.json"
    if label_file.is_file():
        label_map = json.loads(label_file.read_text(encoding="utf-8"))
        names = [None] * dataset_num_classes(dataset)
        for name, index in label_map.items():
            index = int(index)
            if not 0 <= index < len(names) or names[index] is not None:
                raise RuntimeError(f"Invalid SSV2 class mapping in {label_file}")
            names[index] = str(name)
        if all(name is not None for name in names):
            return names
    return [f"class {index}" for index in range(dataset_num_classes(dataset))]


def require_file(path, message=None):
    path = Path(path)
    if not path.is_file():
        raise ModelUnavailableError(message or f"Required file is missing: {path}")
    return path


def sample_frames(clip, frames):
    """Resample a pipeline tensor (B,T,C,H,W) to a fixed frame count."""
    if clip.ndim != 5 or clip.shape[2] != 3:
        raise ValueError(f"Expected (B,T,3,H,W), received {tuple(clip.shape)}")
    idx = torch.linspace(0, clip.shape[1] - 1, frames, device=clip.device).long()
    return clip.index_select(1, idx)


def normalize(clip, mean, std):
    mean = clip.new_tensor(mean).view(1, 1, 3, 1, 1)
    std = clip.new_tensor(std).view(1, 1, 3, 1, 1)
    return (clip - mean) / std


def validate_logits(logits, batch_size=1, classes=400):
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"Model returned {type(logits).__name__}, expected torch.Tensor")
    if logits.shape != (batch_size, classes):
        raise ValueError(
            f"Model returned logits shaped {tuple(logits.shape)}; "
            f"expected ({batch_size}, {classes})"
        )
    if not torch.isfinite(logits).all():
        raise ValueError("Model returned non-finite logits")
    return logits


def load_exact_state_dict(model, state_dict, checkpoint):
    """Strict loading with a checkpoint-specific diagnostic."""
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Checkpoint is incompatible with {checkpoint}: {exc}") from exc


@contextmanager
def isolated_vendor_package(package, import_root):
    """Temporarily import a vendored package without poisoning global modules.

    UniFormer and SlowFast both expose a top-level package named ``slowfast``.
    Their model instances remain usable after construction, while this context
    restores any package modules that were already loaded by another adapter.
    """
    prefix = package + "."
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name == package or name.startswith(prefix)
    }
    for name in saved:
        sys.modules.pop(name, None)

    import_root = str(Path(import_root))
    sys.path.insert(0, import_root)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name == package or name.startswith(prefix):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        try:
            sys.path.remove(import_root)
        except ValueError:
            pass
        importlib.invalidate_caches()


class AdapterMetadata:
    """Small mixin implementing the benchmark metadata contract."""

    info = None
    variant = None

    @property
    def accuracy(self):
        return float(self.info["accuracy"])

    @property
    def gflops(self):
        return float(self.info["gflops"])

    @property
    def parameters(self):
        return sum(p.numel() for p in self.model.parameters())

    @property
    def class_names(self):
        return dataset_class_names(getattr(self, "dataset", "k400"))

    @property
    def num_classes(self):
        return dataset_num_classes(getattr(self, "dataset", "k400"))
