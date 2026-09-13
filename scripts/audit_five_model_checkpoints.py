#!/usr/bin/env python3
"""Inventory local checkpoints for the five-model SSV2 availability audit.

This reads metadata and streams SHA256; it never downloads or rewrites weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("uniformer", "focalnet", "videomae", "dualformer")
SUFFIXES = {".pth", ".pt", ".pyth", ".bin", ".safetensors"}
HEAD_KEYS = ("head.weight", "cls_head.fc_cls.weight", "classifier.weight")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_shapes(path: Path) -> tuple[str, dict[str, tuple[int, ...]]]:
    if path.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as source:
            return "safetensors", {key: tuple(source.get_slice(key).get_shape()) for key in source.keys()}
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    wrapper = "raw"
    for key in ("model_state", "state_dict", "model", "module"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            payload = payload[key]
            wrapper = key
            break
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a state dictionary")
    return wrapper, {
        key: tuple(value.shape)
        for key, value in payload.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }


def summarize(path: Path) -> dict[str, object]:
    row: dict[str, object] = {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "wrapper": "",
        "tensor_count": "",
        "classifier_key": "",
        "classifier_shape": "",
        "decoder_head_shape": "",
        "inspect_error": "",
    }
    try:
        wrapper, shapes = tensor_shapes(path)
        row["wrapper"] = wrapper
        row["tensor_count"] = len(shapes)
        for suffix in HEAD_KEYS:
            matches = [
                key for key in shapes
                if (key == suffix or key.endswith("." + suffix))
                and "decoder" not in key.split(".")
            ]
            if matches:
                key = min(matches, key=len)
                row["classifier_key"] = key
                row["classifier_shape"] = str(list(shapes[key]))
                break
        for key, shape in shapes.items():
            if key == "decoder.head.weight" or key.endswith(".decoder.head.weight"):
                row["decoder_head_shape"] = str(list(shape))
                break
    except Exception as exc:
        row["inspect_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/five_model_checkpoint_inventory.csv")
    args = parser.parse_args()
    checkpoints = ROOT / "checkpoints"
    paths = sorted(
        path for path in checkpoints.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUFFIXES
        and any(family in path.relative_to(checkpoints).as_posix().lower() for family in FAMILIES)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "path", "bytes", "sha256", "wrapper", "tensor_count",
            "classifier_key", "classifier_shape", "decoder_head_shape", "inspect_error",
        ])
        writer.writeheader()
        for path in paths:
            writer.writerow(summarize(path))
    print(f"Wrote {len(paths)} checkpoint records to {args.output}")


if __name__ == "__main__":
    main()
