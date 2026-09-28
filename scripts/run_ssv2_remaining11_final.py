#!/usr/bin/env python3
"""K400 -> SSV2 fine-tune then 1000-clip eval for the remaining 11 frozen models.

Refuses to train or claim valid SSV2 accuracy until the official train and val
video pools are complete. Historical remaining-11 K400-on-SSV2 scores are not
copied into the final CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.datasets.ssv2 import audit_dataset
from training.registry import model_spec

KEYS = [
    "dualformer-t", "video-focalnet-t", "videoswin-t", "mvit-v1-b-16x4",
    "video-focalnet-s", "mvit-v1-b-32x3", "dualformer-s", "videoswin-s",
    "zeroi2v-b16-8f", "dualformer-b-in21k", "omnivore-b-in21k",
]
FORBIDDEN = {
    "uniformer-s", "uniformer-b", "videomae-b", "videoswin-b",
    "timesformer-b", "video-focalnet-b",
}
assert not FORBIDDEN.intersection(KEYS)

MANIFEST = ROOT / "manifests/ssv2_1000_seed0.csv"
EXPECTED_MANIFEST = "c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"
VIDEOS = ROOT / "datasets/ssv2/videos"
LABELS = ROOT / "datasets/ssv2/labels/labels.json"
TRAIN_ANN = ROOT / "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_train_split.txt"
VAL_ANN = ROOT / "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt"
OUT_CSV = ROOT / "results/ssv2_remaining11_final.csv"
OUT_DIR = ROOT / "results/ssv2_remaining11_final"
REPORT = ROOT / "reports/SSV2_REMAINING11_FINAL.md"
CKPT_ROOT = ROOT / "checkpoints/ssv2_finetuned"
FIELDS = [
    "Model", "CheckpointSource", "Pretraining", "SSV2Method", "PublishedTop1",
    "MeasuredTop1", "MeasuredTop5", "GFLOPs", "ParamsM", "LatencyMs",
    "LatencyStdMs", "FPS", "AveragePowerW", "EnergyPerClipJ", "TotalEnergyJ",
    "PeakVRAMMiB", "NS-E", "NS-M", "NS#", "CheckpointSHA256", "Status",
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


def blank_row(key: str, status: str, notes: str) -> dict:
    spec = model_spec(key)
    k400 = spec["k400_checkpoint"]
    digest = sha256_file(ROOT / k400) if (ROOT / k400).is_file() else "N/A"
    return {
        "Model": key,
        "CheckpointSource": k400,
        "Pretraining": spec.get("pretraining", "K400"),
        "SSV2Method": "K400_PRETRAINED_THEN_SSV2_FINETUNED",
        "PublishedTop1": spec["stage1"]["paper_top1"],
        "MeasuredTop1": "N/A",
        "MeasuredTop5": "N/A",
        "GFLOPs": "N/A",
        "ParamsM": "N/A",
        "LatencyMs": "N/A",
        "LatencyStdMs": "N/A",
        "FPS": "N/A",
        "AveragePowerW": "N/A",
        "EnergyPerClipJ": "N/A",
        "TotalEnergyJ": "N/A",
        "PeakVRAMMiB": "N/A",
        "NS-E": "N/A",
        "NS-M": "N/A",
        "NS#": "N/A",
        "CheckpointSHA256": digest,
        "Status": status,
        "_notes": notes,
    }


def write_csv(rows: list[dict]) -> None:
    buf_rows = [{k: row[k] for k in FIELDS} for row in rows]
    text_lines = []
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(buf_rows)
    atomic_text(OUT_CSV, stream.getvalue())


def write_report(rows: list[dict], audit: dict, blocked: bool) -> None:
    missing_train = len(audit["missing"]["train"])
    missing_val = len(audit["missing"]["val"])
    present_train = audit["train_count"] - missing_train
    present_val = audit["val_count"] - missing_val
    lines = [
        "# SSV2 remaining 11 final results",
        "",
        "Valid SSV2 accuracy requires either an exact 174-class released checkpoint "
        "or K400 initialization, a replaced 174-class head, and complete-split fine-tuning. "
        "K400-only scores on SSV2 labels are not used here.",
        "",
        "## Audit",
        "",
        "- Genuine exact SSV2 checkpoints for these 11 variants: **0/11**",
        "- Require K400 -> SSV2 fine-tuning: **11/11**",
        f"- Official train videos present: **{present_train}/{audit['train_count']}** "
        f"(missing {missing_train})",
        f"- Official val videos present: **{present_val}/{audit['val_count']}** "
        f"(missing {missing_val})",
        f"- Train/val ID overlap: {len(audit['overlap'])}",
        f"- Final eval manifest SHA256: `{EXPECTED_MANIFEST}`",
        "- This machine GPU: NVIDIA GeForce RTX 3050 Laptop GPU (4 GiB). Fine-tuning "
        "was requested on RTX 5090 when data is complete.",
        "",
        "Missing official `.webm` IDs:",
        f"- `{OUT_DIR.relative_to(ROOT).as_posix()}/missing_train_ids.txt`",
        f"- `{OUT_DIR.relative_to(ROOT).as_posix()}/missing_val_ids.txt`",
        "",
        "Place the missing files in `datasets/ssv2/videos/<id>.webm` from the official "
        "Something-Something V2 release, then rerun:",
        "",
        "```powershell",
        r".\venv\Scripts\python.exe -u scripts\run_ssv2_remaining11_final.py --run",
        "```",
        "",
        "## Per-model status",
        "",
        "| Model | SSV2Method | Status | MeasuredTop1 | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        notes = str(row.get("_notes", "")).replace("|", "/")
        lines.append(
            f"| {row['Model']} | {row['SSV2Method']} | {row['Status']} | "
            f"{row['MeasuredTop1']} | {notes} |"
        )
    lines += [
        "",
        "## Why training did not start" if blocked else "",
        "",
        "The project fine-tune runner refuses `--allow-partial-data` except as a "
        "non-final smoke. Training on 86,680/168,913 train videos would leak no "
        "eval clips, but it is not a valid complete-split SSV2 result.",
        "",
    ]
    atomic_text(REPORT, "\n".join(line for line in lines if line is not None) + "\n")


def dump_missing(audit: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_text(
        OUT_DIR / "missing_train_ids.txt",
        "\n".join(audit["missing"]["train"]) + ("\n" if audit["missing"]["train"] else ""),
    )
    atomic_text(
        OUT_DIR / "missing_val_ids.txt",
        "\n".join(audit["missing"]["val"]) + ("\n" if audit["missing"]["val"] else ""),
    )
    save_json(OUT_DIR / "dataset_audit.json", {
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "classes": audit["classes"],
        "train_listed": audit["train_count"],
        "val_listed": audit["val_count"],
        "missing_train": len(audit["missing"]["train"]),
        "missing_val": len(audit["missing"]["val"]),
        "overlap": len(audit["overlap"]),
        "annotation_hash": audit["annotation_hash"],
        "videos": str(VIDEOS),
        "complete": not audit["missing"]["train"] and not audit["missing"]["val"] and not audit["overlap"],
    })


def copy_finetuned(key: str, run_dir: Path) -> Path:
    dest = CKPT_ROOT / key
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("best.pth", "last.pth"):
        src = run_dir / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    spec = model_spec(key)
    k400 = ROOT / spec["k400_checkpoint"]
    best = dest / "best.pth"
    info = {
        "model": key,
        "architecture": spec["display_name"],
        "source_k400_checkpoint": spec["k400_checkpoint"],
        "source_k400_sha256": spec["checkpoint_sha256"],
        "ssv2_checkpoint": str(best.relative_to(ROOT)) if best.is_file() else None,
        "ssv2_checkpoint_sha256": sha256_file(best) if best.is_file() else None,
        "training_seed": 0,
        "run_dir": str(run_dir.relative_to(ROOT)),
    }
    if (run_dir / "best_metadata.json").is_file():
        info.update(json.loads((run_dir / "best_metadata.json").read_text(encoding="utf-8")))
    save_json(dest / "checkpoint_info.json", info)
    cfg = yaml.safe_load((ROOT / spec["finetune"]["config"]).read_text(encoding="utf-8"))
    save_json(dest / "training_config.json", {
        "model": key,
        "finetune_yaml": spec["finetune"],
        "resolved_partial": cfg,
        "k400_checkpoint": spec["k400_checkpoint"],
    })
    if (run_dir / "resolved_config.yaml").is_file():
        shutil.copy2(run_dir / "resolved_config.yaml", dest / "resolved_config.yaml")
    return best


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="fine-tune then evaluate when the train split is complete")
    args = parser.parse_args()
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST:
        raise SystemExit("wrong SSV2 eval manifest")
    audit = audit_dataset(VIDEOS, LABELS, TRAIN_ANN, VAL_ANN)
    dump_missing(audit)
    complete = (
        not audit["missing"]["train"]
        and not audit["missing"]["val"]
        and not audit["overlap"]
        and audit["classes"] == 174
    )
    notes = (
        "No exact released 174-class SSV2 checkpoint for this frozen variant. "
        f"Blocked: missing {len(audit['missing']['train'])} train and "
        f"{len(audit['missing']['val'])} val official .webm files under datasets/ssv2/videos/."
    )
    rows = [blank_row(key, "BLOCKED_INCOMPLETE_SSV2_TRAIN" if not complete else "PENDING_FINETUNE", notes) for key in KEYS]
    write_csv(rows)
    write_report(rows, audit, blocked=not complete)
    save_json(OUT_DIR / "audit_summary.json", {
        "genuine_ssv2_checkpoints": 0,
        "need_k400_to_ssv2_finetune": 11,
        "train_complete": not audit["missing"]["train"],
        "val_complete": not audit["missing"]["val"],
        "gpu": "NVIDIA GeForce RTX 3050 Laptop GPU",
    })
    print(f"genuine SSV2 checkpoints: 0/11")
    print(f"need K400->SSV2 fine-tune: 11/11")
    print(
        f"train videos: {audit['train_count']-len(audit['missing']['train'])}/{audit['train_count']} "
        f"missing={len(audit['missing']['train'])}"
    )
    print(
        f"val videos: {audit['val_count']-len(audit['missing']['val'])}/{audit['val_count']} "
        f"missing={len(audit['missing']['val'])}"
    )
    print(f"wrote {OUT_CSV} and {REPORT}")
    if not complete:
        print("STOP: complete official SSV2 train/val videos are required before valid fine-tuning.")
        return 2
    if not args.run:
        print("Dataset is complete. Pass --run to start fine-tuning.")
        return 0
    python = sys.executable
    for key in KEYS:
        dest_best = CKPT_ROOT / key / "best.pth"
        if dest_best.is_file():
            print(f"skip completed {key}")
            continue
        cmd = [python, "-u", str(ROOT / "scripts/finetune_ssv2.py"), "--model", key, "--device", "cuda"]
        print(" ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"{key} fine-tune failed with {result.returncode}", flush=True)
            continue
        runs = sorted((ROOT / "runs/ssv2" / key).glob("*/best.pth"))
        if not runs:
            print(f"{key}: no best.pth after training")
            continue
        copy_finetuned(key, runs[-1].parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
