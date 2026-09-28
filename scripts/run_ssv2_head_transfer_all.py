#!/usr/bin/env python3
"""Frozen K400 backbone, SSV2 linear-probe / head-transfer for remaining 11.

Extracts features once, trains only nn.Linear(D, 174), evaluates the fixed
1000-clip SSV2 manifest. Not full-network fine-tuning and not random-head eval.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import clip_to_input, load_clip, load_ssv2_annotation_map
from models import MODEL_REGISTRY
from training.checkpoint_utils import resolve_module
from training.frozen_preflight import validate_frozen_model
from training.registry import model_spec

KEYS = [
    "videoswin-t", "videoswin-s", "video-focalnet-t", "video-focalnet-s",
    "dualformer-t", "dualformer-s", "dualformer-b-in21k",
    "mvit-v1-b-16x4", "mvit-v1-b-32x3", "omnivore-b-in21k", "zeroi2v-b16-8f",
]
VIDEOS = ROOT / "datasets/ssv2/videos"
TRAIN_ANN = ROOT / "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_train_split.txt"
VAL_ANN = ROOT / "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt"
EVAL_MANIFEST = ROOT / "manifests/ssv2_1000_seed0.csv"
EVAL_SHA = "c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"
OUT = ROOT / "results/ssv2_head_transfer"
MANIFESTS = OUT / "manifests"
CSV_PATH = ROOT / "results/ssv2_head_transfer_remaining11.csv"
REPORT = ROOT / "reports/SSV2_HEAD_TRANSFER_REMAINING11.md"
LRS = (1e-2, 3e-3, 1e-3, 3e-4)
FIELDS = [
    "model", "checkpoint", "pretraining_dataset", "feature_dim", "train_clips",
    "validation_clips", "eval_clips", "head_trainable_params", "backbone_trainable_params",
    "best_lr", "best_epoch", "validation_top1", "top1", "top5", "correct_top1",
    "total_eval", "backbone_unchanged", "checkpoint_load_status", "status",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def save_json(path: Path, obj) -> None:
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: list[dict], fields=None) -> None:
    fields = fields or list(rows[0])
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def tensor_hash(state) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def classifier_linear(raw: nn.Module, path: str) -> nn.Linear:
    try:
        module = raw.get_submodule(path)
    except Exception:
        parent, name = resolve_module(raw, path)
        if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
            module = parent[int(name)]
        elif isinstance(parent, (dict, nn.ModuleDict)):
            module = parent[name]
        else:
            module = getattr(parent, name)
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"{path} is {type(module).__name__}, not Linear")
    if module.out_features != 400:
        raise RuntimeError(f"{path} is {module.out_features}-way, expected 400")
    return module


def flatten_feature(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim > 2:
        value = value.flatten(1)
    if value.ndim != 2:
        raise RuntimeError(f"feature rank {tuple(value.shape)}")
    return value


def present_stems() -> set[str]:
    return {p.stem for p in VIDEOS.glob("*.webm")}


def eval_rows() -> list[dict]:
    if sha256_file(EVAL_MANIFEST) != EVAL_SHA:
        raise RuntimeError("fixed SSV2 eval manifest checksum mismatch")
    with EVAL_MANIFEST.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1000:
        raise RuntimeError(f"eval manifest has {len(rows)} rows")
    return rows


def build_split_pools(seed: int, train_per_class: int, val_per_class: int):
    train_map = load_ssv2_annotation_map(TRAIN_ANN)
    val_map = load_ssv2_annotation_map(VAL_ANN)
    if set(train_map) & set(val_map):
        raise RuntimeError("official train/val annotation overlap")
    available = present_stems()
    eval_ids = [Path(r["RelativePath"]).stem for r in eval_rows()]
    if len(set(eval_ids)) != 1000:
        raise RuntimeError("eval IDs are not unique")
    if not set(eval_ids) <= available:
        raise RuntimeError("eval videos missing from disk")
    if not set(eval_ids) <= set(val_map):
        raise RuntimeError("eval IDs are not in the official validation list")
    rng = random.Random(seed)
    train_by = defaultdict(list)
    val_by = defaultdict(list)
    for vid, label in train_map.items():
        if vid in available and vid not in eval_ids:
            train_by[int(label)].append(vid)
    for vid, label in val_map.items():
        if vid in available and vid not in eval_ids:
            val_by[int(label)].append(vid)
    train_samples, val_samples = [], []
    train_backups, val_backups = {}, {}
    for label in range(174):
        tpool = list(train_by[label])
        vpool = list(val_by[label])
        rng.shuffle(tpool)
        rng.shuffle(vpool)
        if not tpool:
            raise RuntimeError(f"no present train videos for class {label}")
        if not vpool:
            raise RuntimeError(f"no present val videos for class {label} after removing eval")
        train_samples.extend((vid, label) for vid in tpool[:train_per_class])
        val_samples.extend((vid, label) for vid in vpool[:val_per_class])
        train_backups[str(label)] = tpool[train_per_class:train_per_class * 3]
        val_backups[str(label)] = vpool[val_per_class:val_per_class * 3]
    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    eval_samples = [(vid, int(val_map[vid])) for vid in eval_ids]
    train_ids = {v for v, _ in train_samples}
    val_ids = {v for v, _ in val_samples}
    eval_set = set(eval_ids)
    if train_ids & val_ids or train_ids & eval_set or val_ids & eval_set:
        raise RuntimeError("train/val/eval leakage")
    for name, samples in [("train_50_per_class", train_samples),
                          ("val_10_per_class", val_samples),
                          ("eval_1000", eval_samples)]:
        rows = [dict(video_id=vid, label=label, relative_path=f"{vid}.webm")
                for vid, label in samples]
        path = MANIFESTS / f"{name}.csv"
        if path.is_file():
            old = [(r["video_id"], int(r["label"])) for r in csv.DictReader(path.open(encoding="utf-8", newline=""))]
            if old != [(r["video_id"], r["label"]) for r in rows]:
                raise RuntimeError(f"saved {path.name} changed; pass a new output dir or --force")
        else:
            save_csv(path, rows)
    save_json(MANIFESTS / "split_identity.json", dict(
        seed=seed, train_per_class=train_per_class, val_per_class=val_per_class,
        train_clips=len(train_samples), val_clips=len(val_samples), eval_clips=1000,
        available_train=sum(v in available for v in train_map),
        available_val=sum(v in available for v in val_map),
        official_train=len(train_map), official_val=len(val_map),
        eval_sha256=EVAL_SHA, train_backups=train_backups, val_backups=val_backups,
    ))
    print(
        f"manifests: train={len(train_samples)} val={len(val_samples)} eval=1000 "
        f"disjoint=True",
        flush=True,
    )
    return dict(train=train_samples, val=val_samples, eval=eval_samples)


def load_manifest_split(name: str) -> list[tuple[str, int]]:
    path = MANIFESTS / name
    rows = list(csv.DictReader(path.open(encoding="utf-8", newline="")))
    return [(r["video_id"], int(r["label"])) for r in rows]


def decode_clip(video_id: str):
    path = VIDEOS / f"{video_id}.webm"
    return load_clip(path, frames=32, size=224)


def freeze_backbone(raw: nn.Module) -> None:
    raw.eval()
    for parameter in raw.parameters():
        parameter.requires_grad = False
    if not all(not p.requires_grad for p in raw.parameters()):
        raise RuntimeError("backbone freeze failed")


def extract_split(wrapper, head: nn.Linear, samples, cache_path: Path, failed_path: Path, backups=None):
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (len(payload["video_ids"]) == len(samples)
                and payload["labels"].tolist() == [y for _, y in samples]):
            print(f"reuse {cache_path.name} N={len(samples)} D={payload['features'].shape[1]}", flush=True)
            return payload
        print(f"replacing incomplete cache {cache_path.name}", flush=True)
    leftover = {k: list(v) for k, v in (backups or {}).items()}
    captured = []
    handle = head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    ids, feats, labels, failed = [], [], [], []
    try:
        with torch.inference_mode():
            for index, (video_id, label) in enumerate(samples):
                candidates = [video_id] + leftover.get(str(label), [])
                chosen = None
                last_error = None
                used_backup = False
                for candidate in candidates:
                    try:
                        clip = decode_clip(candidate)
                        captured.clear()
                        logits = wrapper(clip_to_input(clip))
                        if logits.shape[-1] != 400:
                            raise RuntimeError(f"expected 400-way logits, got {tuple(logits.shape)}")
                        if len(captured) != 1:
                            raise RuntimeError("classifier hook did not fire once")
                        feature = flatten_feature(captured[0]).float().cpu()
                        if feature.shape[0] != 1 or not torch.isfinite(feature).all():
                            raise RuntimeError(f"bad feature {tuple(feature.shape)}")
                        chosen = (candidate, feature[0].clone())
                        if candidate != video_id:
                            used_backup = True
                            leftover[str(label)] = [v for v in leftover.get(str(label), []) if v != candidate]
                        break
                    except Exception as exc:
                        last_error = exc
                        failed.append(dict(video_id=candidate, error=str(exc)))
                        print(f"decode/extract fail {candidate}: {exc}", flush=True)
                if chosen is None:
                    raise RuntimeError(
                        f"no decodable replacement for class {label} after {video_id}: {last_error}"
                    )
                ids.append(chosen[0])
                feats.append(chosen[1])
                labels.append(label)
                if (index + 1) % 25 == 0 or index + 1 == len(samples):
                    print(f"  extract {index+1}/{len(samples)} ok={len(ids)} fail={len(failed)} backups={used_backup}", flush=True)
    finally:
        handle.remove()
    if failed:
        save_json(failed_path, failed)
    stacked = torch.stack(feats)
    payload = dict(
        video_ids=ids,
        features=stacked.half() if stacked.abs().max() < 65500 else stacked,
        labels=torch.tensor(labels, dtype=torch.long),
    )
    torch.save(payload, cache_path)
    return payload


def score_logits(logits: torch.Tensor, labels: torch.Tensor):
    top = logits.topk(5, dim=-1).indices
    n1 = int((top[:, 0] == labels).sum())
    n5 = int((top == labels[:, None]).any(1).sum())
    n = len(labels)
    return n1, n5, 100.0 * n1 / n, 100.0 * n5 / n


def train_head(train, val, out_dir: Path, force: bool):
    history_path = out_dir / "training_history.csv"
    best_path = out_dir / "head_best.pt"
    x = train["features"].float()
    y = train["labels"].long()
    vx = val["features"].float()
    vy = val["labels"].long()
    dim = int(x.shape[1])
    if y.min() < 0 or y.max() > 173:
        raise RuntimeError("train labels outside 0..173")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    best = None
    init_checksum = None
    for lr in LRS:
        torch.manual_seed(0)
        head = nn.Linear(dim, 174).to(device)
        if init_checksum is None:
            init_checksum = tensor_hash(head.state_dict())
        opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        named = [name for name, p in head.named_parameters() if p.requires_grad]
        print(f"trainable parameter names: {named}", flush=True)
        print(f"trainable parameter count: {sum(p.numel() for p in head.parameters())}", flush=True)
        print("frozen parameter count: 0 (backbone not in graph)", flush=True)
        patience, wait = 8, 0
        for epoch in range(50):
            head.train()
            perm = torch.randperm(len(x))
            total_loss, seen = 0.0, 0
            for start in range(0, len(x), 256):
                index = perm[start:start + 256]
                logits = head(x[index].to(device))
                if logits.shape != (len(index), 174) or not torch.isfinite(logits).all():
                    raise RuntimeError("invalid train logits")
                loss = criterion(logits, y[index].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * len(index)
                seen += len(index)
            head.eval()
            with torch.inference_mode():
                vlogits = head(vx.to(device)).cpu()
            n1, n5, a1, a5 = score_logits(vlogits, vy)
            vloss = float(nn.functional.cross_entropy(vlogits, vy))
            row = dict(lr=lr, epoch=epoch, train_loss=total_loss / seen,
                       val_loss=vloss, val_top1=a1, val_top5=a5)
            records.append(row)
            rank = (a1, -vloss, -epoch)
            if best is None or rank > best[0]:
                best = (rank, lr, epoch, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}, a1)
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
            print(f"  lr={lr:g} epoch={epoch} val_top1={a1:.3f}", flush=True)
    if best is None:
        raise RuntimeError("head training produced no checkpoint")
    trained_hash = tensor_hash(best[3])
    if trained_hash == init_checksum:
        raise RuntimeError("classifier weights did not change")
    torch.save(dict(state_dict=best[3], lr=best[1], epoch=best[2],
                    val_top1=best[4], feature_dim=dim, classes=174), best_path)
    save_csv(history_path, records)
    return best[1], best[2], best[4], dim, sum(v.numel() for v in best[3].values())


def evaluate_head(eval_payload, head_path: Path, pred_path: Path):
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    dim = int(eval_payload["features"].shape[1])
    head = nn.Linear(dim, 174)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.inference_mode():
        logits = head(eval_payload["features"].float())
    if logits.shape != (1000, 174) or not torch.isfinite(logits).all():
        raise RuntimeError("invalid eval logits")
    labels = eval_payload["labels"].long()
    n1, n5, a1, a5 = score_logits(logits, labels)
    pred = logits.argmax(-1)
    hist = torch.bincount(pred, minlength=174)
    if int((hist > 0).sum()) == 1:
        raise RuntimeError("all predictions collapsed to one class")
    rows = [dict(video_id=eval_payload["video_ids"][i], true_label=int(labels[i]),
                 predicted_label=int(pred[i]), correct=int(pred[i] == labels[i]))
            for i in range(1000)]
    save_csv(pred_path, rows)
    return n1, n5, a1, a5, hist.tolist()


def write_master(rows: list[dict]) -> None:
    save_csv(CSV_PATH, rows, FIELDS)
    lines = [
        "# SSV2 head-transfer / linear-probe remaining 11",
        "",
        "For models without an exact released SSV2 checkpoint, we retained the pretrained "
        "Kinetics-400/video backbone, froze all pretrained backbone parameters, extracted "
        "fixed video representations from the locally available SSV2 training subset, and "
        "trained only a newly initialized 174-class linear classification head. Therefore "
        "these measurements represent SSV2 linear-probe/head-transfer accuracy rather than "
        "full-network fine-tuning.",
        "",
        "86,680 official training videos are currently present locally rather than the "
        "complete SSV2 train split (168,913). Head training used up to 50 present train "
        "videos per class and up to 10 present validation videos per class, seed 20260923, "
        "disjoint from the fixed 1000-clip evaluation manifest.",
        "",
        f"Completed {sum(r['status']=='COMPLETE_HEAD_TRAINING' for r in rows)}/11.",
        "",
        "| Model | Top-1 | Top-5 | Val Top-1 | D | Train N | LR | Backbone unchanged | Status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['top1']} | {row['top5']} | {row['validation_top1']} | "
            f"{row['feature_dim']} | {row['train_clips']} | {row['best_lr']} | "
            f"{row['backbone_unchanged']} | {row['status']} |"
        )
    atomic_text(REPORT, "\n".join(lines) + "\n")


def load_master() -> dict[str, dict]:
    if not CSV_PATH.is_file():
        return {}
    return {r["model"]: r for r in csv.DictReader(CSV_PATH.open(encoding="utf-8", newline=""))}


def run_model(key: str, args) -> dict:
    spec = model_spec(key)
    out_dir = OUT / key
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = load_master().get(key)
    if (
        existing
        and existing.get("status") == "COMPLETE_HEAD_TRAINING"
        and not args.force
        and (out_dir / "head_best.pt").is_file()
        and (out_dir / "predictions_1000.csv").is_file()
        and not args.extract_only
        and not args.train_head_only
        and not args.eval_only
    ):
        print(f"skip completed {key}", flush=True)
        return existing

    train_samples = load_manifest_split("train_50_per_class.csv")
    val_samples = load_manifest_split("val_10_per_class.csv")
    eval_samples = load_manifest_split("eval_1000.csv")
    caches = dict(
        train=out_dir / "features_train.pt",
        val=out_dir / "features_val.pt",
        eval=out_dir / "features_eval.pt",
    )
    need_extract = args.extract_only or not all(p.is_file() for p in caches.values()) or args.force
    backbone_hash = None
    load_status = "reused_cache"
    if need_extract and not args.train_head_only and not args.eval_only:
        print(f"Loading exact K400 checkpoint: {key}", flush=True)
        wrapper = MODEL_REGISTRY[key](device="cuda", dataset="k400")
        validate_frozen_model(key, wrapper, dataset="k400",
                              checkpoint=str((ROOT / spec["k400_checkpoint"]).resolve()))
        freeze_backbone(wrapper.model)
        head = classifier_linear(wrapper.model, spec["classifier"]["path"])
        print(f"{key} feature dim={head.in_features} classifier={spec['classifier']['path']}", flush=True)
        backbone_hash = tensor_hash({k: v for k, v in wrapper.model.state_dict().items()
                                     if spec["classifier"]["path"] not in k})
        save_json(out_dir / "backbone_hash_before.json", dict(sha256=backbone_hash))
        trainable = [n for n, p in wrapper.model.named_parameters() if p.requires_grad]
        frozen = sum(p.numel() for p in wrapper.model.parameters() if not p.requires_grad)
        print(f"trainable parameter names: {trainable}", flush=True)
        print(f"trainable parameter count: {sum(p.numel() for n,p in wrapper.model.named_parameters() if p.requires_grad)}", flush=True)
        print(f"frozen parameter count: {frozen}", flush=True)
        if trainable:
            raise RuntimeError(f"backbone still trainable: {trainable}")
        identity = json.loads((MANIFESTS / "split_identity.json").read_text(encoding="utf-8")) if (MANIFESTS / "split_identity.json").is_file() else {}
        extract_split(wrapper, head, train_samples, caches["train"], out_dir / "decode_fail_train.json", identity.get("train_backups"))
        extract_split(wrapper, head, val_samples, caches["val"], out_dir / "decode_fail_val.json", identity.get("val_backups"))
        extract_split(wrapper, head, eval_samples, caches["eval"], out_dir / "decode_fail_eval.json", None)
        after = tensor_hash({k: v for k, v in wrapper.model.state_dict().items()
                             if spec["classifier"]["path"] not in k})
        save_json(out_dir / "backbone_hash_after.json", dict(sha256=after))
        if after != backbone_hash:
            raise RuntimeError("BACKBONE CHANGED during feature extraction")
        load_status = "strict_k400_loaded"
        del wrapper, head
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if args.extract_only:
        return dict(model=key, status="FEATURES_CACHED", checkpoint_load_status=load_status,
                    **{k: "N/A" for k in FIELDS if k not in {"model", "status", "checkpoint_load_status"}})

    train = torch.load(caches["train"], map_location="cpu", weights_only=True)
    val = torch.load(caches["val"], map_location="cpu", weights_only=True)
    ev = torch.load(caches["eval"], map_location="cpu", weights_only=True)
    if len(ev["video_ids"]) != 1000:
        raise RuntimeError(f"eval cache has {len(ev['video_ids'])} clips")
    if args.train_head_only or args.force or not (out_dir / "head_best.pt").is_file():
        best_lr, best_epoch, val_top1, dim, head_n = train_head(train, val, out_dir, args.force)
    else:
        ckpt = torch.load(out_dir / "head_best.pt", map_location="cpu", weights_only=True)
        best_lr, best_epoch, val_top1 = ckpt["lr"], ckpt["epoch"], ckpt["val_top1"]
        dim, head_n = int(ev["features"].shape[1]), 174 * (dim := int(ev["features"].shape[1])) + 174
    if args.train_head_only:
        return dict(model=key, status="HEAD_TRAINED", best_lr=best_lr, best_epoch=best_epoch,
                    validation_top1=val_top1, feature_dim=dim)

    n1, n5, a1, a5, hist = evaluate_head(ev, out_dir / "head_best.pt", out_dir / "predictions_1000.csv")
    if a1 < 2.0:
        print(f"WARNING {key} top1={a1:.3f}% near chance; kept measured value", flush=True)
    unchanged = True
    before = out_dir / "backbone_hash_before.json"
    after = out_dir / "backbone_hash_after.json"
    if before.is_file() and after.is_file():
        unchanged = json.loads(before.read_text())["sha256"] == json.loads(after.read_text())["sha256"]
    if not unchanged:
        raise RuntimeError("BACKBONE_UNCHANGED is false")
    save_json(out_dir / "eval_histogram.json", hist)
    row = dict(
        model=key,
        checkpoint=spec["k400_checkpoint"],
        pretraining_dataset="k400",
        feature_dim=int(ev["features"].shape[1]),
        train_clips=len(train["video_ids"]),
        validation_clips=len(val["video_ids"]),
        eval_clips=1000,
        head_trainable_params=int(ev["features"].shape[1]) * 174 + 174,
        backbone_trainable_params=0,
        best_lr=best_lr,
        best_epoch=best_epoch,
        validation_top1=round(float(val_top1), 4),
        top1=round(a1, 4),
        top5=round(a5, 4),
        correct_top1=n1,
        total_eval=1000,
        backbone_unchanged="true",
        checkpoint_load_status=load_status if need_extract else "reused_cache",
        status="COMPLETE_HEAD_TRAINING",
    )
    save_json(out_dir / "result.json", row)
    print(json.dumps(row), flush=True)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--train-per-class", type=int, default=50)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--train-head-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    keys = KEYS if args.models in {"all", "remaining11"} else [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in KEYS]
    if unknown:
        raise SystemExit(f"not remaining-11 keys: {unknown}")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    if args.force and len(keys) == 1:
        for name in ("train_50_per_class.csv", "val_10_per_class.csv", "eval_1000.csv"):
            path = MANIFESTS / name
            if path.is_file():
                path.unlink()
    if not (MANIFESTS / "train_50_per_class.csv").is_file():
        build_split_pools(args.seed, args.train_per_class, args.val_per_class)
    else:
        train_ids = {v for v, _ in load_manifest_split("train_50_per_class.csv")}
        val_ids = {v for v, _ in load_manifest_split("val_10_per_class.csv")}
        eval_ids = {v for v, _ in load_manifest_split("eval_1000.csv")}
        assert not (train_ids & val_ids or train_ids & eval_ids or val_ids & eval_ids)
        print("reusing saved disjoint manifests", flush=True)
    if len(keys) > 1:
        failed = []
        for key in keys:
            log = OUT / key / "run.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--models", key, "--resume",
                       "--train-per-class", str(args.train_per_class),
                       "--val-per-class", str(args.val_per_class),
                       "--seed", str(args.seed)]
            if args.extract_only:
                command.append("--extract-only")
            if args.train_head_only:
                command.append("--train-head-only")
            if args.eval_only:
                command.append("--eval-only")
            print(f"==== {key} ====", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                child = subprocess.Popen(
                    command, cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True,
                    encoding="utf-8", errors="replace",
                )
                for line in child.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = child.wait()
            if code:
                failed.append(key)
                print(f"FAILED {key}; see {log}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        rows = load_master()
        done = [rows[k] for k in KEYS if k in rows]
        if done:
            write_master(done)
        if failed:
            print(f"incomplete: {failed}", flush=True)
            return 1
        return 0
    rows = load_master()
    for key in keys:
        try:
            row = run_model(key, args)
            if row.get("status") == "COMPLETE_HEAD_TRAINING":
                rows[key] = {k: row[k] for k in FIELDS}
                ordered = [rows[k] for k in KEYS if k in rows]
                write_master(ordered)
        except Exception as exc:
            print(f"FAILED {key}: {type(exc).__name__}: {exc}", flush=True)
            fail = dict.fromkeys(FIELDS, "N/A")
            fail.update(model=key, status=f"FAILED_{type(exc).__name__}",
                        checkpoint=model_spec(key)["k400_checkpoint"],
                        backbone_unchanged="true", backbone_trainable_params=0)
            save_json(OUT / key / "failure.json", dict(error=str(exc), type=type(exc).__name__))
            if key not in rows:
                rows[key] = fail
                write_master([rows[k] for k in KEYS if k in rows])
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    done = [rows[k] for k in KEYS if k in rows]
    if done:
        write_master(done)
    missing = [k for k in keys if rows.get(k, {}).get("status") != "COMPLETE_HEAD_TRAINING"]
    if missing:
        print(f"incomplete: {missing}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
