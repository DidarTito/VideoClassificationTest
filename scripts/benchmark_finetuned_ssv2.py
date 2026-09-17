#!/usr/bin/env python3
"""Measure final SSV2 checkpoints on the fixed validation sample (batch size = 1)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

import run_benchmark
from data import load_annotation_ids, load_clip_set, load_ground_truth, manifest_sha256, read_sample_manifest
from training.adapters.runtime import build_trainable
from training.registry import ROOT, load_manifest, model_spec


FIVE_MODELS = ("uniformer-s", "video-focalnet-t", "uniformer-b", "videomae-b", "dualformer-t")
MANIFEST = ROOT / "manifests/ssv2_1000_seed0.csv"
ANNOTATIONS = ROOT / "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt"
DEFAULT_OUTPUT = ROOT / "results/finetuned/ssv2_five_model_1000_seed0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rooted_path(value) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def checkpoint_selection(args) -> dict[str, Path]:
    if args.checkpoint_map:
        if args.model or args.checkpoint:
            raise ValueError("--checkpoint-map cannot be combined with --model/--checkpoint")
        payload = json.loads(args.checkpoint_map.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != set(FIVE_MODELS):
            raise ValueError("checkpoint map must contain exactly the five requested model keys")
        return {key: _rooted_path(payload[key]) for key in FIVE_MODELS}
    if not (args.model and args.checkpoint):
        raise ValueError("provide --checkpoint-map, or both --model and --checkpoint")
    return {args.model: _rooted_path(args.checkpoint)}


def validate_final_checkpoint(key: str, checkpoint: Path) -> str:
    source = model_spec(key)
    if source.get("checkpoint_status") != "READY_EXACT" or not source.get("checkpoint_sha256"):
        raise ValueError(f"{key}: exact K400 source checkpoint is not ready")
    expected_root = (ROOT / "runs/ssv2" / key).resolve()
    if checkpoint.name != "best.pth" or checkpoint.parent.parent != expected_root:
        raise ValueError(f"{key}: expected runs/ssv2/{key}/<run-id>/best.pth")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{key}: final checkpoint missing: {checkpoint}")
    run_dir = checkpoint.parent
    status = json.loads((run_dir / "protocol_status.json").read_text(encoding="utf-8"))
    complete = json.loads((run_dir / "complete.json").read_text(encoding="utf-8"))
    best = json.loads((run_dir / "best_metadata.json").read_text(encoding="utf-8"))
    if status.get("final_result") is not True or status.get("protocol_deviation") is not False:
        raise ValueError(f"{key}: checkpoint is from a smoke or partial-data run")
    if complete.get("status") != "completed":
        raise ValueError(f"{key}: fine-tuning run is not complete")
    if best.get("selection_rule") != "highest_val_top1_then_earlier_epoch":
        raise ValueError(f"{key}: unexpected best-checkpoint selection rule")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict) or state.get("model_key") != key or not isinstance(state.get("model"), dict):
        raise ValueError(f"{key}: best.pth has the wrong model key or no model state")
    head_key = "raw." + source["classifier"]["path"] + ".weight"
    head = state["model"].get(head_key)
    if not torch.is_tensor(head) or head.ndim != 2 or head.shape[0] != 174:
        raise ValueError(f"{key}: best.pth lacks the expected 174-class head {head_key}")
    if state.get("epoch") != best.get("epoch") or state.get("best_top1") != best.get("top1"):
        raise ValueError(f"{key}: best.pth does not match best_metadata.json")
    del state
    return sha256_file(checkpoint)


def validate_sample(videos: Path) -> None:
    expected = load_manifest()["benchmark_manifests"]["ssv2_1000_seed0"]["sha256"]
    actual = manifest_sha256(MANIFEST)
    if actual != expected:
        raise ValueError(f"fixed SSV2 manifest SHA256 mismatch: {actual} != {expected}")
    sources = read_sample_manifest(
        MANIFEST, videos, max_clips=1000, expected_dataset="ssv2",
        expected_seed=0, expected_frames=32, expected_size=224,
    )
    allowed = load_annotation_ids(ANNOTATIONS)
    outside = [source for source in sources if Path(source).stem not in allowed]
    missing = [source for source in sources if not (videos / source).is_file()]
    if outside or missing:
        raise ValueError(f"fixed SSV2 sample has {len(outside)} IDs outside validation and {len(missing)} missing videos")
    labels = load_ground_truth([videos / source for source in sources], "ssv2", ANNOTATIONS)
    if len(labels) != 1000 or not all(isinstance(label, int) and 0 <= label < 174 for label in labels):
        raise ValueError("fixed SSV2 sample does not have 1000 valid class labels")


class DeploymentAdapter:
    """Strict 174-way checkpoint on the training graph and preprocessing path."""

    num_classes = 174
    accuracy = float("nan")

    def __init__(self, key: str, checkpoint: Path, device: str):
        self.spec = model_spec(key)
        self.model, self.owner = build_trainable(key, device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or state.get("model_key") != key or not isinstance(state.get("model"), dict):
            raise ValueError(f"{checkpoint}: wrong model key or missing model state")
        self.model.load_state_dict(state["model"], strict=True)
        self.model.eval()
        self.device = torch.device(device)
        self.name = self.spec["display_name"]
        self.gflops = float(self.spec["stage1"]["gflops_per_view"])
        self.frames = int(self.spec["input"]["num_frames"])
        self.info = {"checkpoint": str(checkpoint), "frames": self.frames}

    @property
    def parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, clip):
        return self.model(clip.to(self.device, non_blocking=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-map", type=Path, help="JSON mapping of all five model keys to final best.pth paths")
    parser.add_argument("--model", choices=FIVE_MODELS, help="single-model final run")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--videos-dir", type=Path, default=ROOT / "datasets/ssv2/videos")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--power", choices=("auto", "nvml", "nvidia-smi", "tegrastats", "none"), default="nvml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="final-result directory")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="validate sample and final checkpoint provenance without allocating models")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = checkpoint_selection(args)
        videos = args.videos_dir.resolve()
        validate_sample(videos)
        hashes = {key: validate_final_checkpoint(key, path) for key, path in selection.items()}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f"PRECHECK FAILED: {exc}") from exc
    for key, path in selection.items():
        print(f"[OK] {key}: {path} SHA256={hashes[key]}")
    print(f"[OK] SSV2 validation: 1000 clips, manifest SHA256={manifest_sha256(MANIFEST)}")
    if args.preflight:
        return 0

    output_dir = args.output.resolve()
    safe_root = (ROOT / "results/finetuned").resolve()
    if output_dir != safe_root and safe_root not in output_dir.parents:
        raise SystemExit(f"final measurements must stay under {safe_root}")
    run_benchmark._seed_everything(0)
    device_info = run_benchmark.resolve_device(args.device, precision="fp32", validate=True)
    clips, paths = load_clip_set(
        videos, max_clips=1000, frames=32, size=224, seed=0, use_cache=True,
        allowed_stems=load_annotation_ids(ANNOTATIONS), manifest=MANIFEST,
    )
    if len(clips) != 1000:
        raise SystemExit(f"expected 1000 decoded SSV2 clips, found {len(clips)}")
    labels = load_ground_truth(paths, "ssv2", ANNOTATIONS)
    output_csv = output_dir / "device_metrics.csv"
    config = run_benchmark.ExperimentConfig(
        dataset="ssv2", dataset_root=videos, checkpoint_dir=ROOT / "checkpoints",
        manifest=MANIFEST, output_csv=output_csv, num_clips=1000, seed=0,
        batch_size=1, precision="fp32", decoded_frames=32, resolution=224,
        warmup=2, power_kind=args.power, power_sample_ms=20,
        split_name="validation", annotations=ANNOTATIONS, use_cache=True,
        mode="finetuned-ssv2", cooldown_temp=None, cooldown_timeout=0,
    )
    expected = {}
    for key, path in selection.items():
        row = run_benchmark._base_result_row(key, device_info, config)
        row["Checkpoint"] = str(path)
        row["CheckpointSHA256"] = hashes[key]
        row["ModelFrames"] = int(model_spec(key)["input"]["num_frames"])
        row["FramesUsed"] = row["ModelFrames"]
        expected[key] = row
    store = run_benchmark.ResultStore(output_csv, resume=args.resume, force=args.force)
    if args.resume:
        store.validate_resume(expected)
        for row in store.rows:
            key = str(row["ModelKey"])
            if row.get("CheckpointSHA256") != hashes[key]:
                raise SystemExit(f"Cannot resume: {key} final checkpoint contents changed")
    remaining = [key for key in selection if key not in store.successful_keys]
    factories = {
        key: (lambda key=key, path=path: DeploymentAdapter(key, path, str(device_info.torch_device)))
        for key, path in selection.items()
    }

    def save(row):
        row["CheckpointSHA256"] = hashes[row["ModelKey"]]
        store.upsert(row)

    if remaining:
        run_benchmark.benchmark_models(
            remaining, clips, args.power, labels=labels, dataset="ssv2",
            device=str(device_info.torch_device), split_name="validation",
            annotation_file=str(ANNOTATIONS), device_info=device_info,
            config=config, result_callback=save,
            final_ssv2_factories=factories, final_ssv2_checkpoints=selection,
        )
    frame = store.frame()
    run_benchmark._write_rankings(frame, output_dir)
    print(frame[["ModelKey", "Status", "MeasuredTop1", "LatencyMean", "AvgPower", "PeakVRAM", "Error"]].to_string(index=False))
    print(f"Saved: {output_csv}")
    return 0 if len(frame) == len(selection) and all(frame["Status"] == "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
