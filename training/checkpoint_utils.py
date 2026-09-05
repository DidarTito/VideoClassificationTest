"""Strict transfer loading: only an explicitly named classifier may differ."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch
from torch import nn


@dataclass
class LoadReport:
    checkpoint: str
    classifier_path: str
    removed_prefix: str | None
    loaded_keys: list[str]
    classifier_keys: list[str]
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: dict

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def unwrap_state_dict(payload):
    if not isinstance(payload, dict):
        raise RuntimeError("Checkpoint payload is not a mapping")
    for key in ("model_state", "state_dict", "model", "module"):
        value = payload.get(key)
        if isinstance(value, dict) and value:
            return value
    if payload and all(torch.is_tensor(v) for v in payload.values()):
        return payload
    raise RuntimeError("No tensor state_dict found in checkpoint")


def normalize_prefix(state, model_keys):
    keys = list(state)
    for prefix in ("module.", "model."):
        if keys and all(k.startswith(prefix) for k in keys):
            stripped = {k[len(prefix):]: v for k, v in state.items()}
            if set(stripped) & set(model_keys):
                return stripped, prefix
    return dict(state), None


def _classifier_key(key, classifier_path):
    return key == classifier_path or key.startswith(classifier_path + ".")


_QUARANTINED_TOKENS = ("legacy_ssv2_released", "ssv2", "sthv2")


def assert_k400_initialization(checkpoint, allow_legacy_ssv2=False):
    """Refuse quarantined SSV2 classifiers as fine-tuning initialization.

    The released SSV2 checkpoints under checkpoints/legacy_ssv2_released/
    belong to the preliminary legacy experiment only. Fine-tuning must start
    from the exact K400 checkpoints in configs/frozen17.yaml.
    """
    if allow_legacy_ssv2:
        return
    lowered = str(Path(checkpoint).as_posix()).lower()
    hits = [token for token in _QUARANTINED_TOKENS if token in lowered]
    if hits:
        raise RuntimeError(
            f"Refusing to initialize fine-tuning from {checkpoint}: the path "
            f"matches quarantined SSV2 tokens {hits}. Legacy released SSV2 "
            "checkpoints must never seed the K400 -> SSV2 experiment. Pass "
            "allow_legacy_ssv2=True only to reproduce the legacy inference "
            "experiment."
        )


def load_k400_backbone_strict(
    model, checkpoint, classifier_path, metadata_path=None, allow_legacy_ssv2=False
):
    """Load all backbone tensors exactly, excluding only the declared K400 head."""
    checkpoint = Path(checkpoint)
    assert_k400_initialization(checkpoint, allow_legacy_ssv2)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = unwrap_state_dict(payload)
    model_state = model.state_dict()
    state, prefix = normalize_prefix(state, model_state)
    head = {k: v for k, v in state.items() if _classifier_key(k, classifier_path)}
    if not head:
        raise RuntimeError(f"Declared classifier {classifier_path!r} is absent from checkpoint")
    output_dims = {int(v.shape[0]) for v in head.values() if torch.is_tensor(v) and v.ndim in (1, 2)}
    if 400 not in output_dims:
        raise RuntimeError(f"Checkpoint classifier is not 400-way: observed {sorted(output_dims)}")
    backbone = {k: v for k, v in state.items() if k not in head}
    missing = [k for k in model_state if k not in backbone and not _classifier_key(k, classifier_path)]
    unexpected = [k for k in backbone if k not in model_state]
    mismatches = {
        k: {"checkpoint": list(v.shape), "model": list(model_state[k].shape)}
        for k, v in backbone.items()
        if k in model_state and tuple(v.shape) != tuple(model_state[k].shape)
    }
    report = LoadReport(str(checkpoint), classifier_path, prefix, sorted(backbone), sorted(head), missing, unexpected, mismatches)
    if metadata_path:
        report.write(metadata_path)
    if missing or unexpected or mismatches:
        raise RuntimeError(
            "Backbone mismatch is forbidden: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}, shapes={list(mismatches)[:10]}"
        )
    # Merge the validated backbone with the model's current head and perform a
    # genuinely strict load. No permissive state-dict load is used.
    merged = dict(model_state)
    merged.update(backbone)
    model.load_state_dict(merged, strict=True)
    return report


def resolve_module(root, path):
    parts = path.split(".")
    parent = root
    for part in parts[:-1]:
        if isinstance(parent, (dict, nn.ModuleDict)):
            parent = parent[part]
        elif part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def replace_classifier(model, classifier_path, target_classes=174):
    parent, name = resolve_module(model, classifier_path)
    old = parent[int(name)] if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else (parent[name] if isinstance(parent, (dict, nn.ModuleDict)) else getattr(parent, name))
    if not isinstance(old, nn.Linear) or old.out_features != 400:
        raise RuntimeError(f"Expected a 400-way Linear at {classifier_path}, found {old!r}")
    new = nn.Linear(old.in_features, target_classes, bias=old.bias is not None)
    nn.init.trunc_normal_(new.weight, std=0.02)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = new
    elif isinstance(parent, (dict, nn.ModuleDict)):
        parent[name] = new
    else:
        setattr(parent, name, new)
    return new
