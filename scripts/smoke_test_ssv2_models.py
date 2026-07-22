"""Sequential forward-pass smoke test for exact official SSV2 classifiers.

The test uses one synthetic clip so it never changes datasets or benchmark
results.  Models are destroyed between passes, which keeps the check usable
on the 4-GiB RTX 3050 used by this project.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from check_dataset_models import inspect_dataset_models  # noqa: E402
from models import DATASET_MODEL_KEYS, MODEL_REGISTRY  # noqa: E402


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
    return value


def main(argv=None) -> int:
    available = DATASET_MODEL_KEYS["ssv2"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=available, default=list(available)
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args(argv)
    if args.frames < 1 or args.size < 1:
        parser.error("--frames and --size must be positive")

    report = inspect_dataset_models("ssv2", ROOT)
    rows = {row["key"]: row for row in report["models"]}
    blocked = {
        key: rows[key]["blockers"] for key in args.models if not rows[key]["ready"]
    }
    if blocked:
        details = "; ".join(
            f"{key}: {', '.join(reasons)}" for key, reasons in blocked.items()
        )
        raise SystemExit(f"SSV2 asset preflight failed: {details}")

    device = _device(args.device)
    clip = torch.zeros((1, args.frames, 3, args.size, args.size))
    print(f"SSV2 smoke test: {len(args.models)} model(s), device={device}")
    for key in args.models:
        model = None
        try:
            model = MODEL_REGISTRY[key](dataset="ssv2", device=device)
            logits = model(clip)
            if logits.shape != (1, 174):
                raise RuntimeError(
                    f"{key} returned {tuple(logits.shape)}, expected (1, 174)"
                )
            if not torch.isfinite(logits).all():
                raise RuntimeError(f"{key} returned non-finite logits")
            print(
                f"PASS {key:22} logits={tuple(logits.shape)} "
                f"params={model.parameters:,} published_top1={model.accuracy:.1f}%"
            )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
