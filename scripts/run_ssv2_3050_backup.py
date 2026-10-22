#!/usr/bin/env python3
"""Independent RTX 3050 VideoSwin-T/S SSV2 full fine-tuning."""

import argparse
import hashlib
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = ("videoswin-t", "videoswin-s")

def file_sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    import torch
    import scripts.finetune_ssv2 as ft
    from training.registry import model_spec

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    spec = model_spec(args.model)
    checkpoint = ROOT / spec["k400_checkpoint"]

    if not checkpoint.is_file():
        raise SystemExit(
            f"ERROR: K400 checkpoint missing: {checkpoint}"
        )

    if file_sha256(checkpoint) != spec["checkpoint_sha256"]:
        raise SystemExit("ERROR: K400 checkpoint SHA256 mismatch.")

    print("K400 checkpoint verified.", flush=True)

    # Keep the official 32-frame, 224-resolution model configuration.
    original_config = ft.config_for

    def laptop_config(key):
        cfg = dict(original_config(key))
        cfg.update({
            "epochs": 10,
            "warmup_epochs": 3,
            "microbatch": 1,
            "effective_batch": 16,
            "num_workers": 0,
            "persistent_workers": False,
        })
        return cfg

    ft.config_for = laptop_config

    # Enable the official VideoSwin block checkpointing mechanism.
    original_build = ft.build_trainable

    def build_with_checkpointing(key, device):
        model, owner = original_build(key, device)
        count = 0

        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = True
                count += 1

        if count == 0:
            raise RuntimeError(
                "VideoSwin gradient checkpointing was not enabled."
            )

        print(
            f"Gradient checkpointing enabled: {count} blocks",
            flush=True
        )
        return model, owner

    ft.build_trainable = build_with_checkpointing

    run_id = (
        "rtx3050_smoke_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        if args.smoke
        else "rtx3050_full10_backup"
    )

    run_dir = ROOT / "runs" / "ssv2" / args.model / run_id

    if not args.smoke:
        if (run_dir / "complete.json").exists():
            print("Already complete:", run_dir)
            return

        if (run_dir / "best.pth").exists() and not (
            run_dir / "last.pth"
        ).exists():
            raise SystemExit(
                "ERROR: Best checkpoint exists but resume state "
                "is missing. Refusing to overwrite."
            )

    argv = [
        "scripts/finetune_ssv2.py",
        "--model", args.model,
        "--device", "cuda",
        "--run-id", run_id,
    ]

    if args.smoke:
        argv.extend(["--smoke", "--allow-partial-data"])
    elif (run_dir / "last.pth").exists():
        argv.append("--force")
        print("RESUMING from existing last.pth", flush=True)

    sys.argv = argv

    print("=" * 65)
    print("MODEL:", args.model)
    print("MODE:", "NONFINAL SMOKE" if args.smoke else "FULL FT")
    print("EPOCHS:", 1 if args.smoke else 10)
    print("MICROBATCH: 1")
    print("GRADIENT ACCUMULATION: 16")
    print("OUTPUT:", run_dir)
    print("=" * 65, flush=True)

    ft.main()

    print("SUCCESS:", run_dir, flush=True)

if __name__ == "__main__":
    main()
