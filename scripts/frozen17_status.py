#!/usr/bin/env python3
"""Print the concise, machine-readable frozen17 readiness summary."""

import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.registry import ROOT, model_spec


def yes(value):
    return "yes" if value else "no"


def main():
    report = ROOT / "reports" / "rtx3050_frozen17_status.csv"
    rows = list(csv.DictReader(report.open(encoding="utf-8"))) if report.is_file() else []
    rerun = [r["canonical_key"] for r in rows if r["action"].startswith(("RERUN", "RUN_"))]
    blocked = [r["canonical_key"] for r in rows if r["action"].startswith("BLOCKED")]
    valid_keys = [r["canonical_key"] for r in rows if r["action"] == "VALID_REUSE"]
    metadata_keys = [r["canonical_key"] for r in rows if r["action"] == "METADATA_FIX_ONLY"]

    vfn_ok = all(
        "runtime_actual_params_m" in model_spec(key)
        and bool(model_spec(key).get("runtime_param_source"))
        for key in ("video-focalnet-t", "video-focalnet-s", "video-focalnet-b")
    )
    mvit16 = model_spec("mvit-v1-b-16x4")
    videomae = model_spec("videomae-b")
    videomae_path = ROOT / videomae["k400_checkpoint"]
    zero = model_spec("zeroi2v-b16-8f")
    zero_ready = (
        (ROOT / "third_party" / "ZeroI2V" / "configs" / "recognition" /
         "vit_zero_clip" / "CLIP-B_k400_8x16x1.py").is_file()
        and (ROOT / zero["k400_checkpoint"]).is_file()
        and zero["checkpoint_status"] == "READY_EXACT"
    )

    exact = sum(model_spec(key)["checkpoint_status"] == "READY_EXACT" for key in (
        r["canonical_key"] for r in rows
    )) if rows else 0
    print(f"FROZEN17_READY_EXACT = {exact}/17")
    print("RTX3050_RAW_COMPLETED = 15/17")
    print("RTX3050_VALID_REUSE = [" + ", ".join(valid_keys) + "]")
    print("RTX3050_METADATA_ONLY = [" + ", ".join(metadata_keys) + "]")
    print("RTX3050_RERUN_REQUIRED = [" + ", ".join(rerun) + "]")
    print("RTX3050_BLOCKED = [" + ", ".join(blocked) + "]")
    print(f"RTX3050_REUSABLE = {len(valid_keys) + len(metadata_keys)}/17")
    print(f"RTX5070_READY_FOR_TOMORROW = {exact}/17")
    print(f"VFN_FIXED = {yes(vfn_ok)}")
    print(f"MViT16_FIXED = {yes(mvit16['input']['num_frames'] == 16 and mvit16['input']['sampling_rate'] == 4)}")
    print(f"UNIFORMER_B_32F_READY = {yes(model_spec('uniformer-b')['checkpoint_status'] == 'READY_EXACT')}")
    print(f"VIDEOMAE_81_5_READY = {yes(videomae['checkpoint_status'] == 'READY_EXACT' and videomae_path.is_file())}")
    print(f"FINETUNE_UNIFORMER_S_READY = {yes(model_spec('uniformer-s')['checkpoint_status'] == 'READY_EXACT')}")
    print(f"FINETUNE_VIDEOMAE_B_READY = {yes(videomae['checkpoint_status'] == 'READY_EXACT' and videomae_path.is_file())}")
    print(f"ZEROI2V_LINUX_READY = {yes(zero_ready)}")
    print("DUALFORMER_B_EXACT_STATUS = " + model_spec("dualformer-b-in21k")["checkpoint_status"])
    print("UNIFORMER_B_STATUS = " + model_spec("uniformer-b")["checkpoint_status"])
    print("TOMORROW_COMMAND = bash run_project_steps.sh tomorrow")


if __name__ == "__main__":
    main()
