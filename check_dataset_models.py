"""Read-only supervised-checkpoint preflight for one benchmark dataset.

The Stage-1 candidate list is shared across datasets, but not every paper
released a supervised classifier for every dataset.  This command reports the
explicit official-classifier intersection and checks local assets without
importing or allocating any model.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from check_models import inspect_requested_models
from models import (
    DATASET_MODEL_KEYS,
    MODEL_DATASET_AVAILABILITY,
    MODEL_REGISTRY,
    REQUESTED_MODEL_KEYS,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class LocalAssetSpec:
    name: str
    checkpoint: str
    expected_bytes: int
    sources: tuple[str, ...]


SSV2_ASSETS = {
    "uniformer-b": LocalAssetSpec(
        "UniFormer-B",
        "checkpoints/uniformer/uniformer_base_sthv2_16_prek400.pth",
        199_061_829,
        (
            "third_party/UniFormer/video_classification/slowfast/models/uniformer.py",
            "third_party/UniFormer/video_classification/exp/uniformer_b16_sthv2_prek400/config.yaml",
        ),
    ),
    "uniformer-s": LocalAssetSpec(
        "UniFormer-S",
        "checkpoints/uniformer/uniformer_small_sthv2_16_prek400.pth",
        85_271_581,
        (
            "third_party/UniFormer/video_classification/slowfast/models/uniformer.py",
            "third_party/UniFormer/video_classification/exp/uniformer_s16_sthv2_prek400/config.yaml",
        ),
    ),
    "videoswin-b": LocalAssetSpec(
        "VideoSwin-B",
        "checkpoints/videoswin/swin_base_patch244_window1677_sthv2.pth",
        473_410_081,
        (
            "third_party/Video-Swin-Transformer/configs/recognition/swin/swin_base_patch244_window1677_sthv2.py",
            "models/videoswin.py",
        ),
    ),
    "timesformer-b": LocalAssetSpec(
        "TimeSformer-B",
        "checkpoints/timesformer/TimeSformer_divST_8_224_SSv2.pyth",
        971_325_489,
        (
            "third_party/TimeSformer/timesformer/models/vit.py",
            "third_party/TimeSformer/configs/SSv2/TimeSformer_divST_8_224.yaml",
        ),
    ),
    "video-focalnet-b": LocalAssetSpec(
        "Video-FocalNet-B",
        "checkpoints/focalnet/video-focalnet_base_ssv2.pth",
        628_860_159,
        (
            "third_party/Video-FocalNets/classification/videofocalnet.py",
            "third_party/Video-FocalNets/configs/ssv2/video-focalnet_base.yaml",
        ),
    ),
    "videomae-b": LocalAssetSpec(
        "VideoMAE-B",
        "checkpoints/videomae/videomae_vit_b_ssv2_2400e.pth",
        172_756_423,
        (
            "third_party/VideoMAE/modeling_finetune.py",
            "third_party/VideoMAE/scripts/ssv2/videomae_vit_base_patch16_224_tubemasking_ratio_0.9_epoch_2400/finetune.sh",
        ),
    ),
}


def _inspect_k400(root: Path) -> list[dict[str, object]]:
    _, inspected = inspect_requested_models(root)
    by_key = {result.spec.key: result for result in inspected}
    rows = []
    for key in DATASET_MODEL_KEYS["k400"]:
        result = by_key[key]
        rows.append(
            {
                "key": key,
                "name": result.spec.name,
                "ready": result.ready,
                "checkpoint": [
                    str(asset.path)
                    for asset in result.assets
                    if asset.kind == "checkpoint" and asset.compatibility == "exact"
                ],
                "blockers": list(result.blockers),
            }
        )
    return rows


def _inspect_ssv2(root: Path) -> list[dict[str, object]]:
    rows = []
    for key in DATASET_MODEL_KEYS["ssv2"]:
        spec = SSV2_ASSETS[key]
        checkpoint = root / spec.checkpoint
        sources = [root / relative for relative in spec.sources]
        blockers = []
        actual_bytes = checkpoint.stat().st_size if checkpoint.is_file() else None
        if actual_bytes is None:
            blockers.append(f"missing checkpoint: {checkpoint}")
        elif actual_bytes != spec.expected_bytes:
            blockers.append(
                f"checkpoint size mismatch: expected {spec.expected_bytes}, "
                f"found {actual_bytes} bytes"
            )
        for source in sources:
            if not source.is_file():
                blockers.append(f"missing source/config: {source}")
        if key not in MODEL_REGISTRY:
            blockers.append(f"missing registry adapter: {key}")
        rows.append(
            {
                "key": key,
                "name": spec.name,
                "ready": not blockers,
                "checkpoint": str(checkpoint),
                "expected_bytes": spec.expected_bytes,
                "actual_bytes": actual_bytes,
                "sources": [str(path) for path in sources],
                "blockers": blockers,
            }
        )
    return rows


def inspect_dataset_models(dataset: str, root: Path = ROOT) -> dict[str, object]:
    dataset = dataset.lower()
    if dataset not in DATASET_MODEL_KEYS:
        raise ValueError(f"unsupported dataset: {dataset}")
    root = root.resolve()
    rows = _inspect_k400(root) if dataset == "k400" else _inspect_ssv2(root)
    unavailable = [
        key
        for key in REQUESTED_MODEL_KEYS
        if not MODEL_DATASET_AVAILABILITY[key][dataset]
    ]
    ready_count = sum(bool(row["ready"]) for row in rows)
    return {
        "schema_version": 1,
        "mode": "read-only dataset classifier preflight",
        "workspace": str(root),
        "dataset": dataset,
        "official_classifier_count": len(rows),
        "ready_count": ready_count,
        "blocked_count": len(rows) - ready_count,
        "all_ready": ready_count == len(rows),
        "models": rows,
        "no_official_classifier_or_adapter": unavailable,
    }


def _print_human(report: dict[str, object]) -> None:
    dataset = str(report["dataset"]).upper()
    print(f"{dataset} exact supervised-classifier preflight (read-only)")
    print(
        f"Result: {report['ready_count']}/{report['official_classifier_count']} "
        "official model/dataset pairs asset-ready"
    )
    for row in report["models"]:
        status = "READY" if row["ready"] else "BLOCKED"
        print(f"[{status:7}] {row['key']:22} {row['name']}")
        checkpoint = row["checkpoint"]
        if isinstance(checkpoint, list):
            for path in checkpoint:
                print(f"          checkpoint: {path}")
        else:
            print(f"          checkpoint: {checkpoint}")
        for blocker in row["blockers"]:
            print(f"          blocker: {blocker}")
    unavailable = report["no_official_classifier_or_adapter"]
    if unavailable:
        print("No exact supervised pair in this project: " + " ".join(unavailable))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(DATASET_MODEL_KEYS), required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.list_models:
        print(" ".join(DATASET_MODEL_KEYS[args.dataset]))
        return 0
    report = inspect_dataset_models(args.dataset, args.root)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        _print_human(report)
    return 0 if report["all_ready"] or args.allow_missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
