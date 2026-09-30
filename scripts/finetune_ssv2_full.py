#!/usr/bin/env python3
"""
True end-to-end K400 -> Something-Something-v2 Full Fine-Tuning
for the two pilot models:

  1) dualformer-s
  2) zeroi2v-b16-8f

Target machine:
  RTX 5090 lab machine
  outer repository: /home/esai01/ESAI_Didar/VideoClassificationTest

This is FULL FT:
  - exact K400 checkpoint
  - replace 400-class head with 174-class SSV2 head
  - explicitly unfreeze every learnable parameter
  - forward/backward through the full model
  - AdamW updates all model parameters
  - best epoch chosen only on SSV2 validation excluding the fixed 1000 holdout

Storage policy:
  - latest.pt       = one resumable checkpoint, overwritten every epoch
  - best_weights.pt = one model-only best checkpoint
  - interrupted.pt  = only if interrupted/crashed
  - NO per-epoch checkpoint archive

The script intentionally does NOT use the old linear-probe feature cache.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The lab audit installed ZeroI2V dependencies here.
EXTRA_DEPS = Path("/tmp/ssv2-full-ft-deps")
if EXTRA_DEPS.exists() and str(EXTRA_DEPS) not in sys.path:
    sys.path.insert(0, str(EXTRA_DEPS))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    import pynvml
except Exception:
    pynvml = None

from data import load_clip
from models import MODEL_REGISTRY

try:
    from models.common import sample_frames as repo_sample_frames
except Exception:
    repo_sample_frames = None

SUPPORTED = ("dualformer-s", "zeroi2v-b16-8f")
NUM_CLASSES = 174
DECODE_FRAMES = 32
RESOLUTION = 224

EXPECTED = {
    "dualformer-s": {
        "checkpoint": ROOT / "checkpoints/dualformer/dualformer_small_patch244_window877.pth",
        "sha256": "1a377476e1244c86fdf0f2bb639ac88906e85ad328d951fc648567ce0ba17cce",
        "model_frames": 32,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "zeroi2v-b16-8f": {
        "checkpoint": ROOT / "checkpoints/ZeroI2V/CLIP-B_k400_8x16x1.pth",
        "sha256": "f62f2e10aa0ae0cb755046f9d84b5ecffd58d4a5f1ae043ace88c3b27621929f",
        "model_frames": 8,
        "mean": [122.769 / 255.0, 116.74 / 255.0, 104.04 / 255.0],
        "std": [68.493 / 255.0, 66.63 / 255.0, 70.321 / 255.0],
    },
}

TRAIN_JSON = ROOT / "datasets/ssv2/annotations/labels/train.json"
VAL_JSON = ROOT / "datasets/ssv2/annotations/labels/validation.json"
LABELS_JSON = ROOT / "datasets/ssv2/annotations/labels/labels.json"
VIDEO_DIR = ROOT / "datasets/ssv2/videos"
HOLDOUT_MANIFEST = ROOT / "manifests/ssv2_1000_seed0.csv"
EXPECTED_HOLDOUT_SHA = "c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"


# ---------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def human_time(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating..."
    sec = int(seconds + 0.5)
    d, r = divmod(sec, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d {h:02d}:{m:02d}:{s:02d}"
    return f"{h:02d}:{m:02d}:{s:02d}"


def eta_clock(seconds: Optional[float]) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "estimating..."
    return (dt.datetime.now() + dt.timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S")


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def free_gib(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def git_snapshot() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).splitlines()
        return {"commit": commit, "status_porcelain": status}
    except Exception:
        return {"commit": None, "status_porcelain": []}


# ---------------------------------------------------------------------
# GPU telemetry
# ---------------------------------------------------------------------

class NVML:
    def __init__(self, index: int = 0):
        self.ok = False
        self.handle = None
        if pynvml is None:
            return
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self.ok = True
        except Exception:
            pass

    def read(self) -> Dict[str, float]:
        if not self.ok:
            return {}
        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            out = {
                "nvml_vram_used_mib": mem.used / (1024 ** 2),
                "gpu_util_pct": float(util.gpu),
                "gpu_temp_c": float(
                    pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                ),
                "gpu_power_w": pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0,
            }
            try:
                out["gpu_power_limit_w"] = (
                    pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0
                )
            except Exception:
                pass
            return out
        except Exception:
            return {}


# ---------------------------------------------------------------------
# SSV2 dataset
# ---------------------------------------------------------------------

def load_labels() -> Dict[str, int]:
    raw = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    mapping = {str(k): int(v) for k, v in raw.items()}
    if sorted(set(mapping.values())) != list(range(NUM_CLASSES)):
        raise RuntimeError("SSV2 labels.json is not exactly 174 classes indexed 0..173")
    return mapping


def _normalize_ssv2_label(text: str) -> str:
    """
    Canonicalize SSV2 action-template text.

    Official annotation templates may contain arbitrary bracketed phrases,
    e.g.:
        Taking [one of many similar things on the table]

    while labels.json may use the same phrase without brackets.
    """
    text = str(text).strip().lower()

    # Brackets are placeholder notation, not class identity.
    text = text.replace("[", "").replace("]", "")

    # Normalize whitespace.
    text = " ".join(text.split())

    return text


def official_rows(path: Path, label_map: Dict[str, int]) -> List[Tuple[str, int]]:
    items = json.loads(path.read_text(encoding="utf-8"))

    normalized_map: Dict[str, int] = {}

    for name, class_id in label_map.items():
        key = _normalize_ssv2_label(name)
        class_id = int(class_id)

        if key in normalized_map and normalized_map[key] != class_id:
            raise RuntimeError(
                f"Normalized SSV2 label collision: "
                f"{name!r} -> {key!r}, "
                f"{normalized_map[key]} vs {class_id}"
            )

        normalized_map[key] = class_id

    rows: List[Tuple[str, int]] = []
    unknown = []

    for item in items:
        vid = str(item["id"])

        # The official action template defines the 174-way class.
        candidates = [
            item.get("template"),
            item.get("label"),
        ]

        class_id = None
        tried = []

        for candidate in candidates:
            if candidate is None:
                continue

            candidate = str(candidate)
            tried.append(candidate)

            # Exact labels.json lookup first.
            if candidate in label_map:
                class_id = int(label_map[candidate])
                break

            # Then normalized template lookup.
            key = _normalize_ssv2_label(candidate)

            if key in normalized_map:
                class_id = normalized_map[key]
                break

        if class_id is None:
            unknown.append(
                {
                    "id": vid,
                    "template": item.get("template"),
                    "label": item.get("label"),
                }
            )
            continue

        video_path = VIDEO_DIR / f"{vid}.webm"

        if video_path.exists():
            rows.append((vid, class_id))

    if unknown:
        examples = unknown[:20]
        raise KeyError(
            f"{len(unknown)} SSV2 annotations could not be mapped "
            f"from {path}. First examples: {examples!r}"
        )

    labels_seen = sorted({label for _, label in rows})

    if labels_seen != list(range(NUM_CLASSES)):
        raise RuntimeError(
            f"Mapped split does not contain all {NUM_CLASSES} classes. "
            f"Found {len(labels_seen)} classes; "
            f"range={labels_seen[:5]}...{labels_seen[-5:]}"
        )

    return rows


def holdout_ids(
    path: Path,
    allowed_ids: set[str],
) -> set[str]:
    """
    Parse the fixed SSV2 holdout manifest.

    Manifest schema:
      ClipIndex
      RelativePath
      Label
      Dataset
      Seed
      DecodedFrames
      Resolution

    RelativePath is the authoritative video identifier.
    Example:
      211653.webm -> video id "211653"
    """

    result: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if not reader.fieldnames:
            raise RuntimeError(
                f"Holdout manifest has no header: {path}"
            )

        print("Holdout manifest columns:", reader.fieldnames)

        if "RelativePath" not in reader.fieldnames:
            raise RuntimeError(
                "Expected RelativePath column in SSV2 holdout manifest"
            )

        for row_num, row in enumerate(reader, start=2):
            rel = str(row.get("RelativePath", "")).strip()

            if not rel:
                raise RuntimeError(
                    f"Missing RelativePath at row {row_num}"
                )

            vid = Path(rel.replace("\\", "/")).stem

            if vid not in allowed_ids:
                raise RuntimeError(
                    f"Holdout video {vid!r} from row {row_num} "
                    f"is not in official SSV2 validation"
                )

            result.add(vid)

    if len(result) != 1000:
        raise RuntimeError(
            f"Expected exactly 1000 unique holdout IDs, "
            f"got {len(result)}"
        )

    return result


def build_splits(common_output: Path):
    required = [TRAIN_JSON, VAL_JSON, LABELS_JSON, VIDEO_DIR, HOLDOUT_MANIFEST]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)

    actual_holdout_sha = sha256(HOLDOUT_MANIFEST)
    if actual_holdout_sha != EXPECTED_HOLDOUT_SHA:
        raise RuntimeError(
            "Fixed 1000-clip holdout changed.\n"
            f"expected={EXPECTED_HOLDOUT_SHA}\nactual={actual_holdout_sha}"
        )

    label_map = load_labels()
    train = official_rows(TRAIN_JSON, label_map)
    off_val = official_rows(VAL_JSON, label_map)
    official_val_ids = {vid for vid, _ in off_val}
    hold = holdout_ids(
        HOLDOUT_MANIFEST,
        allowed_ids=official_val_ids,
    )
    val = [(vid, y) for vid, y in off_val if vid not in hold]

    train_ids = {v for v, _ in train}
    off_val_ids = {v for v, _ in off_val}
    val_ids = {v for v, _ in val}

    assert not (train_ids & off_val_ids)
    assert not (train_ids & hold)
    assert not (val_ids & hold)
    assert hold.issubset(off_val_ids)
    assert len(hold) == 1000

    manifests = common_output / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)

    for name, rows in (("train_full.csv", train), ("val_selection.csv", val)):
        with (manifests / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["video_id", "label", "path"])
            for vid, y in rows:
                w.writerow([vid, y, str(VIDEO_DIR / f"{vid}.webm")])

    summary = {
        "train": len(train),
        "official_validation": len(off_val),
        "selection_validation": len(val),
        "fixed_holdout": len(hold),
        "num_classes": NUM_CLASSES,
        "holdout_sha256": actual_holdout_sha,
        "train_val_disjoint": True,
        "train_holdout_disjoint": True,
        "selection_val_holdout_disjoint": True,
    }
    write_json(common_output / "dataset_split_audit.json", summary)
    return train, val, hold, summary


class FullFTDataset(Dataset):
    """
    Uses the repo's validated data.load_clip() decoder.

    It returns raw uint8 T,C,H,W. We normalize exactly once in prepare_input().
    No horizontal flip is used.
    """
    def __init__(self, rows: Sequence[Tuple[str, int]], train: bool):
        self.rows = list(rows)
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        vid, label = self.rows[index]
        path = VIDEO_DIR / f"{vid}.webm"
        try:
            # Keep the validated benchmark spatial pipeline. This is conservative:
            # center-crop preprocessing, no left/right-breaking horizontal flip.
            clip = load_clip(path, frames=DECODE_FRAMES, size=RESOLUTION)
            if not torch.is_tensor(clip):
                clip = torch.as_tensor(clip)
            if clip.ndim != 4:
                raise RuntimeError(f"unexpected clip shape {tuple(clip.shape)}")
            return clip.to(torch.uint8), int(label), vid, ""
        except Exception as exc:
            return None, int(label), vid, f"{type(exc).__name__}: {exc}"


def collate_skip_bad(batch):
    good = []
    errors = []
    for clip, label, vid, err in batch:
        if clip is None:
            errors.append((vid, err))
        else:
            good.append((clip, label, vid))
    if not good:
        return None
    clips = torch.stack([x[0] for x in good], dim=0)
    labels = torch.tensor([x[1] for x in good], dtype=torch.long)
    vids = [x[2] for x in good]
    return clips, labels, vids, errors


# ---------------------------------------------------------------------
# Model loading / differentiable forward
# ---------------------------------------------------------------------

def get_dotted(obj: Any, path: str):
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def set_dotted(obj: Any, path: str, value: Any):
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


def instantiate_wrapper(model_key: str, device: torch.device):
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"{model_key} missing from MODEL_REGISTRY")
    factory = MODEL_REGISTRY[model_key]
    attempts = [
        {"device": str(device), "dataset": "k400"},
        {"device": device, "dataset": "k400"},
        {"device": str(device)},
        {"device": device},
        {},
    ]
    errors = []
    for kwargs in attempts:
        try:
            return factory(**kwargs)
        except TypeError as exc:
            errors.append(f"{kwargs}: {exc}")
    raise RuntimeError(
        f"Unable to instantiate {model_key}. Factory={factory}, signature={inspect.signature(factory)}\n"
        + "\n".join(errors)
    )


def find_raw_model(wrapper: Any) -> nn.Module:
    # Some wrappers may themselves be nn.Module.
    if isinstance(wrapper, nn.Module):
        try:
            get_dotted(wrapper, "cls_head.fc_cls")
            return wrapper
        except Exception:
            pass

    for name in ("model", "raw_model", "net", "network", "module", "_model"):
        obj = getattr(wrapper, name, None)
        if isinstance(obj, nn.Module):
            try:
                get_dotted(obj, "cls_head.fc_cls")
                return obj
            except Exception:
                pass

    # Last resort: inspect wrapper attributes.
    if hasattr(wrapper, "__dict__"):
        for name, obj in vars(wrapper).items():
            if isinstance(obj, nn.Module):
                try:
                    get_dotted(obj, "cls_head.fc_cls")
                    print(f"[info] raw model found via wrapper.{name}")
                    return obj
                except Exception:
                    pass
    raise RuntimeError(
        f"Could not locate differentiable raw model with cls_head.fc_cls inside {type(wrapper)}"
    )


def verify_source_checkpoint(model_key: str) -> Dict[str, str]:
    cfg = EXPECTED[model_key]
    p: Path = cfg["checkpoint"]
    if not p.exists():
        raise FileNotFoundError(p)
    actual = sha256(p)
    if actual != cfg["sha256"]:
        raise RuntimeError(
            f"{model_key} checkpoint SHA mismatch\nexpected={cfg['sha256']}\nactual={actual}"
        )
    return {"path": str(p), "sha256": actual}


def replace_k400_head(raw: nn.Module) -> nn.Linear:
    old = get_dotted(raw, "cls_head.fc_cls")
    if not isinstance(old, nn.Linear):
        raise TypeError(f"cls_head.fc_cls is {type(old)}, expected nn.Linear")
    if old.out_features != 400:
        raise RuntimeError(f"Expected 400-class K400 head, got {old.out_features}")
    new = nn.Linear(old.in_features, NUM_CLASSES, bias=old.bias is not None)
    nn.init.normal_(new.weight, std=0.02)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    set_dotted(raw, "cls_head.fc_cls", new)
    return new


def build_fresh_model(model_key: str, device: torch.device):
    source = verify_source_checkpoint(model_key)
    wrapper = instantiate_wrapper(model_key, device)
    raw = find_raw_model(wrapper).to(device)

    old_head = repr(get_dotted(raw, "cls_head.fc_cls"))
    new_head = replace_k400_head(raw)

    # CRITICAL: vendor ZeroI2V and old wrappers may freeze params.
    # Unfreeze AFTER loading and AFTER replacing the classifier.
    for p in raw.parameters():
        p.requires_grad_(True)

    raw.train()
    return wrapper, raw, new_head, source, old_head


def sample_frames_exact(x_btchw: torch.Tensor, n: int) -> torch.Tensor:
    if repo_sample_frames is not None:
        return repo_sample_frames(x_btchw, n)
    if x_btchw.shape[1] == n:
        return x_btchw
    idx = torch.linspace(0, x_btchw.shape[1] - 1, n, device=x_btchw.device).long()
    return x_btchw.index_select(1, idx)


def prepare_input(model_key: str, clips_btchw_u8: torch.Tensor, device: torch.device):
    cfg = EXPECTED[model_key]
    x = clips_btchw_u8.to(device=device, dtype=torch.float32, non_blocking=True) / 255.0
    x = sample_frames_exact(x, int(cfg["model_frames"]))

    mean = torch.tensor(cfg["mean"], dtype=x.dtype, device=device).view(1, 1, 3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=x.dtype, device=device).view(1, 1, 3, 1, 1)
    x = (x - mean) / std

    # B,T,C,H,W -> B,C,T,H,W
    return x.permute(0, 2, 1, 3, 4).contiguous()


def extract_logits(out: Any) -> torch.Tensor:
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        for k in ("logits", "cls_score", "pred", "output"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
    if isinstance(out, (tuple, list)):
        for item in out:
            try:
                return extract_logits(item)
            except Exception:
                pass
    for attr in ("logits", "pred_score", "cls_score"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
    raise TypeError(f"Cannot extract logits from {type(out)}")


def forward_logits(raw: nn.Module, x_bcthw: torch.Tensor) -> torch.Tensor:
    """
    Tries differentiable raw paths only. Never call the inference wrapper __call__.
    """
    errors = []

    # Preferred for MMAction-like models.
    if hasattr(raw, "extract_feat") and hasattr(raw, "cls_head"):
        try:
            feat = raw.extract_feat(x_bcthw)
            if isinstance(feat, tuple) and len(feat) == 2 and isinstance(feat[1], dict):
                feat = feat[0]
            logits = extract_logits(raw.cls_head(feat))
            return _validate_logits(logits)
        except Exception as exc:
            errors.append(f"extract_feat->cls_head: {type(exc).__name__}: {exc}")

    # Direct backbone/head path.
    if hasattr(raw, "backbone") and hasattr(raw, "cls_head"):
        try:
            feat = raw.backbone(x_bcthw)
            logits = extract_logits(raw.cls_head(feat))
            return _validate_logits(logits)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            errors.append(f"backbone->cls_head: {type(exc).__name__}: {exc}")

    # MMEngine recognizer tensor mode.
    try:
        logits = extract_logits(raw(x_bcthw, mode="tensor"))
        return _validate_logits(logits)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        errors.append(f"raw(..., mode='tensor'): {type(exc).__name__}: {exc}")

    # Plain nn.Module.
    try:
        logits = extract_logits(raw(x_bcthw))
        return _validate_logits(logits)
    except torch.cuda.OutOfMemoryError:
        raise
    except Exception as exc:
        errors.append(f"raw(x): {type(exc).__name__}: {exc}")

    raise RuntimeError("No differentiable forward path succeeded:\n" + "\n".join(errors))


def _validate_logits(logits: torch.Tensor) -> torch.Tensor:
    while logits.ndim > 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    if logits.ndim != 2 or logits.shape[-1] != NUM_CLASSES:
        raise RuntimeError(f"Expected [B,{NUM_CLASSES}] logits, got {tuple(logits.shape)}")
    return logits


# ---------------------------------------------------------------------
# Optimizer / schedule / audits
# ---------------------------------------------------------------------

def parameter_groups(raw: nn.Module, head: nn.Module):
    head_ids = {id(p) for p in head.parameters()}
    backbone, head_params = [], []
    for p in raw.parameters():
        (head_params if id(p) in head_ids else backbone).append(p)
    return backbone, head_params


def build_optimizer(raw, head, backbone_lr, head_lr, weight_decay):
    backbone, head_params = parameter_groups(raw, head)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr, "name": "backbone"},
            {"params": head_params, "lr": head_lr, "name": "head"},
        ],
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )


def audit_trainability(raw: nn.Module):
    total = sum(p.numel() for p in raw.parameters())
    trainable = sum(p.numel() for p in raw.parameters() if p.requires_grad)
    frozen = [n for n, p in raw.named_parameters() if not p.requires_grad]
    return {
        "total_params": total,
        "trainable_params": trainable,
        "trainable_pct": 100.0 * trainable / max(total, 1),
        "frozen_names": frozen,
        "pass": total == trainable and not frozen,
    }


def audit_optimizer(raw: nn.Module, optimizer):
    model = [p for p in raw.parameters() if p.requires_grad]
    opt = [p for g in optimizer.param_groups for p in g["params"]]
    counts = Counter(id(p) for p in opt)
    missing = [n for n, p in raw.named_parameters() if p.requires_grad and id(p) not in counts]
    duplicated = [n for n, p in raw.named_parameters() if p.requires_grad and counts.get(id(p), 0) != 1]
    return {
        "model_trainable_tensors": len(model),
        "optimizer_tensors": len(opt),
        "missing": missing,
        "duplicated": duplicated,
        "pass": not missing and not duplicated and len(model) == len(opt),
    }


def build_scheduler(optimizer, total_updates: int, warmup_updates: int):
    def lr_lambda(step):
        if warmup_updates > 0 and step < warmup_updates:
            return max(1e-8, (step + 1) / warmup_updates)
        if total_updates <= warmup_updates:
            return 1.0
        p = (step - warmup_updates) / max(1, total_updates - warmup_updates)
        p = min(max(p, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def first_batch(dataset: Dataset, batch_size: int, workers: int):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_skip_bad,
    )
    for batch in loader:
        if batch is not None:
            return batch
    raise RuntimeError("No decodable batch found")


def readiness_audit(
    model_key: str,
    train_ds: Dataset,
    device: torch.device,
    out_dir: Path,
    *,
    batch_size: int,
    workers: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    label_smoothing: float,
):
    print("\n" + "=" * 76)
    print(f"FULL FT READINESS AUDIT — {model_key}")
    print("=" * 76)

    wrapper, raw, head, source, old_head = build_fresh_model(model_key, device)
    trainability = audit_trainability(raw)
    optimizer = build_optimizer(raw, head, backbone_lr, head_lr, weight_decay)
    opt_audit = audit_optimizer(raw, optimizer)

    print(f"Old classifier:      {old_head}")
    print(f"New classifier:      {get_dotted(raw, 'cls_head.fc_cls')}")
    print(f"Total params:        {trainability['total_params']:,}")
    print(f"Trainable params:    {trainability['trainable_params']:,}")
    print(f"Trainable percentage:{trainability['trainable_pct']:.6f}%")
    print(f"Frozen learnable:    {len(trainability['frozen_names'])}")
    print(f"Optimizer missing:   {len(opt_audit['missing'])}")
    print(f"Optimizer duplicated:{len(opt_audit['duplicated'])}")

    if not trainability["pass"]:
        raise RuntimeError("FULL FT FAIL: not every learnable parameter is trainable")
    if not opt_audit["pass"]:
        raise RuntimeError("FULL FT FAIL: optimizer does not cover all params exactly once")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 not supported on this CUDA device")

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    batch = first_batch(train_ds, min(batch_size, 4), workers)
    clips, labels, vids, errors = batch
    labels = labels.to(device, non_blocking=True)
    x = prepare_input(model_key, clips, device)

    head_ids = {id(p) for p in head.parameters()}
    tracked = []
    for n, p in raw.named_parameters():
        if id(p) not in head_ids and p.requires_grad and p.numel() > 0:
            tracked.append((n, p))
        if len(tracked) >= 8:
            break
    before = {n: p.detach().clone() for n, p in tracked}

    raw.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = forward_logits(raw, x)
        loss = criterion(logits, labels)

    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise RuntimeError("FULL FT FAIL: non-finite logits/loss")

    loss.backward()

    grad_examples = []
    nonfinite_grad = []
    for n, p in raw.named_parameters():
        if id(p) in head_ids or not p.requires_grad or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            nonfinite_grad.append(n)
        elif float(p.grad.detach().abs().sum()) > 0:
            grad_examples.append(n)

    if nonfinite_grad:
        raise RuntimeError(f"FULL FT FAIL: non-finite grads: {nonfinite_grad[:10]}")
    if len(grad_examples) < 2:
        raise RuntimeError("FULL FT FAIL: backbone is not receiving non-zero gradients")

    torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    optimizer.step()

    changed = [n for n, p in tracked if not torch.equal(before[n], p.detach())]
    if not changed:
        raise RuntimeError("FULL FT FAIL: backbone weights did not update")

    audit = {
        "model": model_key,
        "source_checkpoint": source,
        "old_classifier": old_head,
        "new_classifier": repr(get_dotted(raw, "cls_head.fc_cls")),
        "trainability": trainability,
        "optimizer": opt_audit,
        "bf16_supported": True,
        "bf16_forward_backward": True,
        "backbone_gradient_examples": grad_examples[:20],
        "backbone_updated_examples": changed,
        "full_ft_ready": True,
    }
    write_json(out_dir / "readiness_audit.json", audit)

    print(f"Backbone gradients:  PASS ({len(grad_examples)} tensors with non-zero grad)")
    print(f"Backbone update:     PASS ({len(changed)} tracked tensors changed)")
    print("BF16 fwd/bwd:        PASS")
    print("FULL FT READY:       YES")
    print("=" * 76)

    del optimizer, raw, wrapper, head, x, logits, loss
    torch.cuda.empty_cache()
    return audit


# ---------------------------------------------------------------------
# Micro-batch probe
# ---------------------------------------------------------------------

def probe_batch_size(
    model_key, train_ds, device, workers,
    backbone_lr, head_lr, weight_decay, label_smoothing,
):
    candidates = (16, 8, 4, 2, 1)
    print(f"\n{model_key}: micro-batch probe {candidates}")
    for bs in candidates:
        wrapper = raw = head = optimizer = None
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            wrapper, raw, head, _, _ = build_fresh_model(model_key, device)
            optimizer = build_optimizer(raw, head, backbone_lr, head_lr, weight_decay)
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            clips, labels, _, _ = first_batch(train_ds, bs, min(workers, 2))
            labels = labels.to(device, non_blocking=True)
            x = prepare_input(model_key, clips, device)

            optimizer.zero_grad(set_to_none=True)
            raw.train()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = forward_logits(raw, x)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Important: AdamW allocates optimizer states on the first step.
            # Include this in the probe so peak VRAM reflects real training.
            optimizer.step()
            torch.cuda.synchronize()

            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(
                f"  batch {bs}: PASS including AdamW step, "
                f"peak allocated {peak:.2f} GiB"
            )
            return bs
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs}: OOM")
        finally:
            try:
                del optimizer, raw, wrapper, head
            except Exception:
                pass
            torch.cuda.empty_cache()
    raise RuntimeError("Even micro-batch 1 OOMed")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

class Running:
    def __init__(self):
        self.loss_sum = 0.0
        self.samples = 0
        self.top1 = 0
        self.top5 = 0

    def update(self, loss_value: float, logits: torch.Tensor, labels: torch.Tensor):
        n = labels.numel()
        self.loss_sum += float(loss_value) * n
        self.samples += n
        self.top1 += int((logits.argmax(1) == labels).sum())
        pred5 = logits.topk(min(5, logits.shape[1]), 1).indices
        self.top5 += int(pred5.eq(labels[:, None]).any(1).sum())

    def dict(self):
        n = max(1, self.samples)
        return {
            "loss": self.loss_sum / n,
            "top1": 100.0 * self.top1 / n,
            "top5": 100.0 * self.top5 / n,
            "samples": self.samples,
        }


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------

def cpu_state_dict(model: nn.Module):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def resume_checkpoint(
    model_key, raw, optimizer, scheduler, epoch, global_update,
    best_top1, best_epoch, elapsed, config, source
):
    return {
        "format": "ssv2_full_ft_resume_v1",
        "model_key": model_key,
        "dataset": "ssv2",
        "num_classes": NUM_CLASSES,
        "epoch": epoch,
        "global_update": global_update,
        "best_val_top1": best_top1,
        "best_epoch": best_epoch,
        "elapsed_seconds": elapsed,
        "model_state_dict": cpu_state_dict(raw),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "source_checkpoint": source,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
    }


def best_weights_checkpoint(model_key, raw, epoch, val_top1, config, source):
    return {
        "format": "ssv2_full_ft_weights_v1",
        "model_key": model_key,
        "dataset": "ssv2",
        "num_classes": NUM_CLASSES,
        "epoch": epoch,
        "val_top1": val_top1,
        "source_checkpoint": source,
        "config": config,
        "model_state_dict": cpu_state_dict(raw),
    }


def load_resume(path, raw, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("format") != "ssv2_full_ft_resume_v1":
        raise RuntimeError(f"Bad resume format: {ckpt.get('format')}")
    raw.load_state_dict(ckpt["model_state_dict"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)
    return ckpt


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

@torch.no_grad()
def validate(model_key, raw, loader, device, criterion, epoch):
    raw.eval()
    run = Running()
    decode_errors = 0
    started = time.perf_counter()

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(
            loader,
            desc=f"{model_key} VAL e{epoch}",
            dynamic_ncols=True,
            leave=False,
        )

    for batch in iterator:
        if batch is None:
            continue
        clips, labels, vids, errors = batch
        decode_errors += len(errors)
        labels = labels.to(device, non_blocking=True)
        x = prepare_input(model_key, clips, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_logits(raw, x)
            loss = criterion(logits, labels)
        run.update(float(loss.item()), logits, labels)

        if tqdm is not None:
            m = run.dict()
            iterator.set_postfix(loss=f"{m['loss']:.3f}", top1=f"{m['top1']:.2f}%")

    return run.dict(), time.perf_counter() - started, decode_errors


# ---------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------

def make_loader(ds, batch_size, shuffle, workers):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_skip_bad,
        drop_last=False,
    )
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def group_lrs(optimizer):
    vals = {g.get("name", str(i)): float(g["lr"]) for i, g in enumerate(optimizer.param_groups)}
    return vals.get("backbone", float(optimizer.param_groups[0]["lr"])), vals.get(
        "head", float(optimizer.param_groups[-1]["lr"])
    )


def save_history(path: Path, rows: List[Dict[str, Any]]):
    if not rows:
        return
    tmp = path.with_suffix(".csv.tmp")
    fields = list(rows[0].keys())
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def train_one(args, model_key, train_rows, val_rows):
    out = Path(args.output_dir) / model_key
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FullFTDataset(train_rows, train=True)
    val_ds = FullFTDataset(val_rows, train=False)
    device = torch.device("cuda:0")
    nvml = NVML(0)

    seed_all(args.seed)

    if args.batch_size == 0:
        micro_batch = probe_batch_size(
            model_key, train_ds, device, args.num_workers,
            args.backbone_lr, args.head_lr, args.weight_decay, args.label_smoothing
        )
    else:
        micro_batch = args.batch_size

    if args.effective_batch % micro_batch != 0:
        # keep effective batch >= requested and use integer accumulation
        grad_accum = math.ceil(args.effective_batch / micro_batch)
    else:
        grad_accum = args.effective_batch // micro_batch
    effective_batch = micro_batch * grad_accum

    readiness_audit(
        model_key, train_ds, device, out,
        batch_size=min(micro_batch, 4),
        workers=min(args.num_workers, 2),
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    # Fresh exact K400 reload after audit/probe.
    wrapper, raw, head, source, old_head = build_fresh_model(model_key, device)

    train_loader = make_loader(train_ds, micro_batch, True, args.num_workers)
    val_loader = make_loader(val_ds, micro_batch, False, args.num_workers)

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = updates_per_epoch * args.warmup_epochs

    optimizer = build_optimizer(
        raw, head, args.backbone_lr, args.head_lr, args.weight_decay
    )
    scheduler = build_scheduler(optimizer, total_updates, warmup_updates)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    config = {
        "model": model_key,
        "epochs": args.epochs,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "effective_batch": effective_batch,
        "backbone_lr": args.backbone_lr,
        "head_lr": args.head_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "warmup_epochs": args.warmup_epochs,
        "gradient_clip": args.grad_clip,
        "precision": "bf16",
        "seed": args.seed,
        "num_workers": args.num_workers,
        "train_videos": len(train_rows),
        "selection_val_videos": len(val_rows),
        "fixed_holdout_videos": 1000,
        "decode_frames": DECODE_FRAMES,
        "model_frames": EXPECTED[model_key]["model_frames"],
        "resolution": RESOLUTION,
        "horizontal_flip": False,
        "source_checkpoint": source,
    }
    write_json(out / "config.json", config)

    write_json(out / "environment.json", {
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "git": git_snapshot(),
        "repo_root": str(ROOT),
        "output_dir": str(out),
        "holdout_manifest": str(HOLDOUT_MANIFEST),
        "holdout_sha256": sha256(HOLDOUT_MANIFEST),
        "source_checkpoint": source,
    })

    # Space guard: we only keep latest + best (+ interrupted only when needed).
    remaining = free_gib(out)
    print(f"\nStorage free at output location: {remaining:.2f} GiB")
    if remaining < 4.0:
        raise RuntimeError(
            "Less than 4 GiB free at output path. Use the mounted large disk; "
            "do not start Full FT here."
        )

    latest = ckpt_dir / "latest.pt"
    best = ckpt_dir / "best_weights.pt"
    interrupted = ckpt_dir / "interrupted.pt"

    start_epoch = 1
    global_update = 0
    best_top1 = -1.0
    best_epoch = 0
    prior_elapsed = 0.0

    if args.resume and latest.exists():
        print(f"Resuming {model_key} from {latest}")
        ck = load_resume(latest, raw, optimizer, scheduler, device)
        if ck["model_key"] != model_key:
            raise RuntimeError("Resume model key mismatch")
        start_epoch = int(ck["epoch"]) + 1
        global_update = int(ck["global_update"])
        best_top1 = float(ck["best_val_top1"])
        best_epoch = int(ck["best_epoch"])
        prior_elapsed = float(ck.get("elapsed_seconds", 0.0))

    history_path = out / "training_history.csv"
    log_path = out / "training_log.jsonl"
    history: List[Dict[str, Any]] = []
    if history_path.exists():
        with history_path.open("r", encoding="utf-8", newline="") as f:
            history = list(csv.DictReader(f))

    print("\n" + "=" * 78)
    print("TRUE FULL FINE-TUNING VERIFIED")
    print("=" * 78)
    print(f"Model:                 {model_key}")
    print(f"Source K400 checkpoint:{source['path']}")
    print(f"Source SHA256:         {source['sha256']}")
    print(f"Target:                SSV2 ({NUM_CLASSES} classes)")
    print(f"Train videos:          {len(train_rows):,}")
    print(f"Selection validation:  {len(val_rows):,}")
    print(f"Final fixed holdout:    1,000 (NOT used for selection)")
    print(f"Epochs:                {args.epochs}")
    print(f"Precision:             BF16 AMP")
    print(f"Micro-batch:           {micro_batch}")
    print(f"Grad accumulation:     {grad_accum}")
    print(f"Effective batch:       {effective_batch}")
    print(f"Backbone LR:           {args.backbone_lr:g}")
    print(f"Head LR:               {args.head_lr:g}")
    print(f"Output:                {out}")
    print("STATUS: TRUE FULL FT READY — STARTING")
    print("=" * 78)

    run_start = time.perf_counter()
    completed_epoch_durations: List[float] = []
    stop_requested = {"value": False}

    def _sigint(signum, frame):
        stop_requested["value"] = True
        print("\nCtrl+C received: will save interrupted checkpoint at next safe point...")

    old_sigint = signal.signal(signal.SIGINT, _sigint)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            raw.train()
            torch.cuda.reset_peak_memory_stats()
            metrics = Running()
            decode_errors = 0
            optimizer.zero_grad(set_to_none=True)

            epoch_start = time.perf_counter()
            batch_times = deque(maxlen=100)

            iterator = train_loader
            if tqdm is not None:
                iterator = tqdm(
                    train_loader,
                    desc=f"{model_key} TRAIN e{epoch}/{args.epochs}",
                    dynamic_ncols=True,
                    leave=True,
                )

            for batch_idx, batch in enumerate(iterator, start=1):
                if stop_requested["value"]:
                    raise KeyboardInterrupt

                b0 = time.perf_counter()
                if batch is None:
                    continue
                clips, labels, vids, errors = batch
                decode_errors += len(errors)

                labels = labels.to(device, non_blocking=True)
                x = prepare_input(model_key, clips, device)

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = forward_logits(raw, x)
                    loss_full = criterion(logits, labels)
                    loss = loss_full / grad_accum

                if not torch.isfinite(loss_full):
                    raise RuntimeError(f"Non-finite loss at epoch={epoch} batch={batch_idx}")

                loss.backward()
                metrics.update(float(loss_full.item()), logits.detach(), labels)

                do_step = (batch_idx % grad_accum == 0) or (batch_idx == len(train_loader))
                if do_step:
                    torch.nn.utils.clip_grad_norm_(raw.parameters(), args.grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_update += 1

                batch_dt = time.perf_counter() - b0
                batch_times.append(batch_dt)

                # Progress/ETA.
                avg_batch = sum(batch_times) / len(batch_times)
                remaining_this_epoch = max(0, len(train_loader) - batch_idx) * avg_batch

                if completed_epoch_durations:
                    avg_epoch = sum(completed_epoch_durations) / len(completed_epoch_durations)
                else:
                    avg_epoch = len(train_loader) * avg_batch

                future_epochs = args.epochs - epoch
                total_eta = remaining_this_epoch + future_epochs * avg_epoch
                elapsed_total = prior_elapsed + (time.perf_counter() - run_start)
                m = metrics.dict()
                blr, hlr = group_lrs(optimizer)
                alloc = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                max_alloc = torch.cuda.max_memory_allocated() / 1024**3
                tel = nvml.read()

                if tqdm is not None:
                    iterator.set_postfix(
                        loss=f"{m['loss']:.3f}",
                        top1=f"{m['top1']:.1f}%",
                        lr=f"{blr:.1e}",
                        vram=f"{alloc:.1f}G",
                        epETA=human_time(remaining_this_epoch),
                        allETA=human_time(total_eta),
                    )

                if batch_idx == 1 or batch_idx % args.log_every == 0:
                    record = {
                        "time": dt.datetime.now().isoformat(timespec="seconds"),
                        "model": model_key,
                        "epoch": epoch,
                        "epochs_total": args.epochs,
                        "batch": batch_idx,
                        "batches_total": len(train_loader),
                        "loss_running": m["loss"],
                        "train_top1_running": m["top1"],
                        "train_top5_running": m["top5"],
                        "backbone_lr": blr,
                        "head_lr": hlr,
                        "gpu_alloc_gib": alloc,
                        "gpu_reserved_gib": reserved,
                        "gpu_max_alloc_gib": max_alloc,
                        "epoch_elapsed": time.perf_counter() - epoch_start,
                        "epoch_eta": remaining_this_epoch,
                        "full_elapsed": elapsed_total,
                        "full_eta": total_eta,
                        "estimated_finish": eta_clock(total_eta),
                        "decode_errors": decode_errors,
                        **tel,
                    }
                    append_jsonl(log_path, record)

                    if tqdm is None:
                        print(
                            f"[{model_key}] epoch {epoch}/{args.epochs} "
                            f"batch {batch_idx}/{len(train_loader)} "
                            f"loss={m['loss']:.3f} top1={m['top1']:.1f}% "
                            f"LR={blr:.2e}/{hlr:.2e} VRAM={alloc:.1f}GiB "
                            f"epochETA={human_time(remaining_this_epoch)} "
                            f"fullETA={human_time(total_eta)} finish={eta_clock(total_eta)}"
                        )

            train_stats = metrics.dict()
            epoch_train_seconds = time.perf_counter() - epoch_start

            val_stats, val_seconds, val_decode_errors = validate(
                model_key, raw, val_loader, device, criterion, epoch
            )

            epoch_total_seconds = time.perf_counter() - epoch_start
            completed_epoch_durations.append(epoch_total_seconds)

            is_best = val_stats["top1"] > best_top1
            if is_best:
                best_top1 = float(val_stats["top1"])
                best_epoch = epoch
                atomic_torch_save(
                    best_weights_checkpoint(
                        model_key, raw, epoch, best_top1, config, source
                    ),
                    best,
                )
                best_sha = sha256(best)
            else:
                best_sha = sha256(best) if best.exists() else ""

            elapsed_total = prior_elapsed + (time.perf_counter() - run_start)
            remaining_epochs = args.epochs - epoch
            avg_epoch = sum(completed_epoch_durations) / len(completed_epoch_durations)
            total_eta = remaining_epochs * avg_epoch

            # Save one resumable latest checkpoint, overwriting previous.
            atomic_torch_save(
                resume_checkpoint(
                    model_key, raw, optimizer, scheduler,
                    epoch, global_update, best_top1, best_epoch,
                    elapsed_total, config, source,
                ),
                latest,
            )

            # Remove stale interrupted checkpoint after a clean epoch save.
            interrupted.unlink(missing_ok=True)

            blr, hlr = group_lrs(optimizer)
            peak_mib = torch.cuda.max_memory_allocated() / 1024**2

            row = {
                "model": model_key,
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_top1": train_stats["top1"],
                "train_top5": train_stats["top5"],
                "val_loss": val_stats["loss"],
                "val_top1": val_stats["top1"],
                "val_top5": val_stats["top5"],
                "backbone_lr": blr,
                "head_lr": hlr,
                "epoch_train_seconds": epoch_train_seconds,
                "validation_seconds": val_seconds,
                "epoch_total_seconds": epoch_total_seconds,
                "total_elapsed_seconds": elapsed_total,
                "estimated_remaining_seconds": total_eta,
                "estimated_finish": eta_clock(total_eta),
                "peak_train_vram_mib": peak_mib,
                "train_decode_errors": decode_errors,
                "val_decode_errors": val_decode_errors,
                "best_val_top1": best_top1,
                "best_epoch": best_epoch,
                "is_best": int(is_best),
                "best_checkpoint": str(best),
                "best_checkpoint_sha256": best_sha,
            }
            history.append(row)
            save_history(history_path, history)

            print("\n" + "=" * 78)
            print(f"MODEL: {model_key}")
            print(f"EPOCH: {epoch} / {args.epochs}")
            print(f"Train loss:   {train_stats['loss']:.4f}")
            print(f"Train Top-1:  {train_stats['top1']:.2f}%")
            print(f"Train Top-5:  {train_stats['top5']:.2f}%")
            print(f"Val loss:     {val_stats['loss']:.4f}")
            print(f"Val Top-1:    {val_stats['top1']:.2f}%")
            print(f"Val Top-5:    {val_stats['top5']:.2f}%")
            print(f"Best Val Top1:{best_top1:.2f}% @ epoch {best_epoch}")
            print(f"Epoch time:   {human_time(epoch_total_seconds)}")
            print(f"Total elapsed:{human_time(elapsed_total)}")
            print(f"Full FT ETA:  {human_time(total_eta)}")
            print(f"Finish around:{eta_clock(total_eta)}")
            print(f"Peak VRAM:    {peak_mib:.0f} MiB")
            print(f"Free disk:    {free_gib(out):.2f} GiB")
            print("=" * 78)

            if free_gib(out) < 2.5:
                raise RuntimeError(
                    "Disk free space fell below 2.5 GiB after checkpoint save. "
                    "Move output to the large mounted disk before continuing."
                )

        print(f"\n{model_key} COMPLETE")
        print(f"Best validation Top-1: {best_top1:.2f}% at epoch {best_epoch}")
        print(f"Best weights: {best}")
        print(f"SHA256: {sha256(best)}")

        # Training finished: latest.pt is only needed for resume.
        # Keep it by default for audit/reproducibility; user can remove it later
        # after copying best_weights.pt and verifying the SHA.
        return {
            "model": model_key,
            "best_val_top1": best_top1,
            "best_epoch": best_epoch,
            "best_checkpoint": str(best),
            "best_sha256": sha256(best),
        }

    except (KeyboardInterrupt, Exception) as exc:
        elapsed_total = prior_elapsed + (time.perf_counter() - run_start)
        try:
            # Best-effort interruption checkpoint. May add temporary disk usage.
            atomic_torch_save(
                resume_checkpoint(
                    model_key, raw, optimizer, scheduler,
                    max(start_epoch - 1, 0), global_update,
                    best_top1, best_epoch, elapsed_total, config, source,
                ),
                interrupted,
            )
            print(f"\nSaved interruption state: {interrupted}")
        except Exception as save_exc:
            print(f"\nCould not save interrupted checkpoint: {save_exc}")
        if isinstance(exc, KeyboardInterrupt):
            raise
        traceback.print_exc()
        raise
    finally:
        signal.signal(signal.SIGINT, old_sigint)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        nargs="+",
        default=["dualformer-s", "zeroi2v-b16-8f"],
        choices=SUPPORTED,
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/media/esai01/10A2D9F6A2D9DFF0/ssv2_full_ft"),
        help="Use the large mounted disk. Do NOT default to /home for this run.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="0 = auto probe 16,8,4,2,1",
    )
    p.add_argument("--effective-batch", type=int, default=16)
    p.add_argument("--backbone-lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build splits and print environment only; do not train.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print("Repository:", ROOT)
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16:", torch.cuda.is_bf16_supported())
    print("Output:", args.output_dir)
    print(f"Output free space: {free_gib(args.output_dir):.2f} GiB")

    # The audit found two repository copies. Refuse to run from the nested copy.
    if ROOT.name == "VideoClassificationTest" and ROOT.parent.name == "VideoClassificationTest":
        raise RuntimeError(
            "You are running from the nested project copy. "
            "Run this script from the OUTER repository that contains real checkpoints and SSV2 videos."
        )

    common = Path(args.output_dir)
    train_rows, val_rows, hold, split_info = build_splits(common)
    print("\nDataset audit:")
    print(json.dumps(split_info, indent=2))

    # Expected current complete split counts from the lab audit.
    if len(train_rows) < 160000:
        raise RuntimeError(
            f"Only {len(train_rows)} train videos found; expected the complete local SSV2 train split."
        )
    if len(val_rows) < 23000:
        raise RuntimeError(
            f"Only {len(val_rows)} selection-val videos found; expected validation minus 1000 holdout."
        )

    if args.dry_run:
        print("DRY RUN COMPLETE")
        return

    results = []
    for model_key in args.models:
        result = train_one(args, model_key, train_rows, val_rows)
        results.append(result)
        write_json(Path(args.output_dir) / "pilot_results.json", results)

    print("\n" + "=" * 78)
    print("TWO-MODEL FULL-FT PILOT COMPLETE")
    print("=" * 78)
    for r in results:
        print(
            f"{r['model']}: best val Top-1={r['best_val_top1']:.2f}% "
            f"epoch={r['best_epoch']} checkpoint={r['best_checkpoint']}"
        )


if __name__ == "__main__":
    main()
