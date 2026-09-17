#!/usr/bin/env python3
"""Audit the exact five K400 benchmark checkpoints without allocating a GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.registry import ROOT, model_spec


MODEL_KEYS = (
    "uniformer-s",
    "video-focalnet-t",
    "uniformer-b",
    "videomae-b",
    "dualformer-t",
)

RUNTIME_LOADERS = {
    "uniformer-s": "models.uniformer.UniFormerModel(S)",
    "video-focalnet-t": "models.videofocalnet.VideoFocalNetModel(T)",
    "uniformer-b": "models.uniformer.UniFormerModel(B)",
    "videomae-b": "models.videomae.VideoMAEModel(B1600): official module state",
    "dualformer-t": "models.dualformer.DualFormerModel(T)",
}

FIELDS = (
    "model",
    "checkpoint_path",
    "file_size_bytes",
    "sha256",
    "expected_sha256",
    "sha_match",
    "checkpoint_format",
    "tensor_count",
    "state_tensor_elements",
    "classifier_key",
    "classifier_shape",
    "classifier_output_classes",
    "expected_classes",
    "decoded_frames",
    "model_frames",
    "sampling_rate",
    "patch_embedding_shape",
    "positional_embedding_shape",
    "decoder_tensor_count",
    "runtime_loader",
    "status",
    "inspect_error",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict(path: Path) -> tuple[str, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    wrapper = "raw state_dict"
    for key in ("model_state", "state_dict", "model", "module"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            payload = payload[key]
            wrapper = f"dict:{key}"
            break
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a state dictionary")
    tensors = {
        key: value
        for key, value in payload.items()
        if isinstance(key, str) and torch.is_tensor(value)
    }
    if not tensors:
        raise ValueError("checkpoint state dictionary has no tensors")
    return wrapper, tensors


def _shape(tensors: dict[str, torch.Tensor], *keys: str) -> str:
    for key in keys:
        value = tensors.get(key)
        if value is not None:
            return str(list(value.shape))
    return ""


def inspect(key: str) -> dict[str, object]:
    spec = model_spec(key)
    relative = Path(spec["k400_checkpoint"])
    path = ROOT / relative
    expected_sha = str(spec.get("checkpoint_sha256") or "").lower()
    classifier_key = f"{spec['classifier']['path']}.weight"
    row: dict[str, object] = {
        "model": key,
        "checkpoint_path": relative.as_posix(),
        "file_size_bytes": path.stat().st_size if path.is_file() else "",
        "sha256": "",
        "expected_sha256": expected_sha,
        "sha_match": False,
        "checkpoint_format": "",
        "tensor_count": "",
        "state_tensor_elements": "",
        "classifier_key": classifier_key,
        "classifier_shape": "",
        "classifier_output_classes": "",
        "expected_classes": int(spec["k400_classes"]),
        "decoded_frames": 32,
        "model_frames": int(spec["input"]["num_frames"]),
        "sampling_rate": int(spec["input"]["sampling_rate"]),
        "patch_embedding_shape": "",
        "positional_embedding_shape": "",
        "decoder_tensor_count": "",
        "runtime_loader": RUNTIME_LOADERS[key],
        "status": "BLOCKED",
        "inspect_error": "",
    }
    if not path.is_file():
        row["inspect_error"] = "checkpoint missing"
        return row
    try:
        actual_sha = sha256_file(path).lower()
        wrapper, tensors = state_dict(path)
        classifier = tensors.get(classifier_key)
        row.update({
            "sha256": actual_sha,
            "sha_match": actual_sha == expected_sha,
            "checkpoint_format": wrapper,
            "tensor_count": len(tensors),
            "state_tensor_elements": sum(tensor.numel() for tensor in tensors.values()),
            "classifier_shape": str(list(classifier.shape)) if classifier is not None else "",
            "classifier_output_classes": (
                int(classifier.shape[0]) if classifier is not None and classifier.ndim == 2 else ""
            ),
            "patch_embedding_shape": _shape(
                tensors, "patch_embed.proj.weight", "patch_embed1.proj.weight",
                "backbone.patch_embeds.0.proj.weight",
            ),
            "positional_embedding_shape": _shape(tensors, "pos_embed", "backbone.pos_embed"),
            "decoder_tensor_count": sum("decoder" in name.split(".") for name in tensors),
        })
        classes_ok = row["classifier_output_classes"] == row["expected_classes"]
        row["status"] = (
            "READY_EXACT"
            if row["sha_match"] and classes_ok and spec.get("checkpoint_status") == "READY_EXACT"
            else "BLOCKED"
        )
        if not classes_ok:
            row["inspect_error"] = "expected exact 400-class classifier weight"
    except Exception as exc:
        row["inspect_error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/five_model_checkpoint_inventory.csv",
    )
    args = parser.parse_args()
    rows = [inspect(key) for key in MODEL_KEYS]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    ready = sum(row["status"] == "READY_EXACT" for row in rows)
    print(f"Wrote {len(rows)} checkpoint records to {args.output}")
    print(f"READY_EXACT: {ready}/{len(rows)}")
    return 0 if ready == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
