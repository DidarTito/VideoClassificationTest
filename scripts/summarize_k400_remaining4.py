"""Validate the four fresh K400 runs and append them to the user's 13 rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metrics import netscore_e, netscore_m, netscore_hash
from training.registry import model_spec

FRESH_KEYS = ("mvit-v1-b-16x4", "dualformer-s", "zeroi2v-b16-8f", "dualformer-b-in21k")
NUMERIC = ("PublishedTop1", "MeasuredTop1", "LatencyMean", "AvgPower", "EnergyPerClip",
           "TotalEnergy", "TotalTime", "PeakVRAM", "NS-E", "NS-M", "NS#")
FIELDS = ("Rank", "ModelKey", "Model", *NUMERIC, "Dataset", "ClipCount", "Evidence",
          "SourceCSV", "RuntimeModelKey", "Checkpoint", "CheckpointSHA256", "Manifest",
          "ManifestSHA256", "DeviceName", "GPUUUID", "Precision", "BatchSize", "ModelFrames")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_fresh(rows):
    require(len(rows) == 4 and {r["ModelKey"] for r in rows} == set(FRESH_KEYS),
            "Expected exactly the four requested model rows")
    manifest = ROOT / "manifests/k400_1000_seed0.csv"
    manifest_hash = digest(manifest)
    require(len(read_csv(manifest)) == 1000, "Manifest must have exactly 1000 clips")
    for row in rows:
        key = row["ModelKey"]
        spec = model_spec(key)
        require(row["Status"] == "completed", f"{key}: incomplete run")
        require(row["Dataset"] == "k400" and int(row["ClipCount"]) == 1000,
                f"{key}: wrong dataset/count")
        require(row["AccuracyValid"].lower() == "true" and int(row["ClassifierClasses"]) == 400,
                f"{key}: invalid accuracy/classifier")
        require(row["Precision"] == "fp32" and int(row["BatchSize"]) == 1,
                f"{key}: wrong precision/batch")
        require(int(row["WarmupRuns"]) == 2 and int(row["PowerSampleMs"]) == 20,
                f"{key}: wrong warmup/power sampling")
        require(row["ManifestSHA256"] == manifest_hash and Path(row["Manifest"]).resolve() == manifest.resolve(),
                f"{key}: manifest mismatch")
        checkpoint = ROOT / spec["k400_checkpoint"]
        require(Path(row["Checkpoint"]).resolve() == checkpoint.resolve(), f"{key}: wrong checkpoint")
        require(digest(checkpoint) == row["CheckpointSHA256"] == spec["checkpoint_sha256"],
                f"{key}: checkpoint hash mismatch")
        require(int(row["ModelFrames"]) == spec["input"]["num_frames"], f"{key}: wrong frame count")
        for field in NUMERIC:
            require(math.isfinite(float(row[field])) and float(row[field]) > 0, f"{key}: invalid {field}")
        values = {field: float(row[field]) for field in NUMERIC}
        require(0 < values["MeasuredTop1"] <= 100, f"{key}: accuracy out of range")
        require(int(row["PowerSamples"]) > 0 and "NVML" in row["MeasurementBackend"],
                f"{key}: missing power measurements")
        expected = {
            "TotalTime": values["LatencyMean"] * 1000 / 1000,
            "EnergyPerClip": values["AvgPower"] * values["LatencyMean"] / 1000,
            "TotalEnergy": values["AvgPower"] * values["TotalTime"],
            "NS-E": netscore_e(values["MeasuredTop1"], values["TotalTime"], values["AvgPower"]),
            "NS-M": netscore_m(values["MeasuredTop1"], values["PeakVRAM"]),
            "NS#": netscore_hash(values["MeasuredTop1"], values["PeakVRAM"], values["TotalTime"], values["AvgPower"]),
        }
        for field, value in expected.items():
            require(math.isclose(values[field], value, rel_tol=1e-9), f"{key}: inconsistent {field}")
        paper_top1 = spec.get("checkpoint_published", spec["stage1"])["paper_top1"]
        require(values["PublishedTop1"] == paper_top1, f"{key}: wrong checkpoint paper accuracy")
    require(len({r["GPUUUID"] for r in rows}) == 1, "Fresh runs used different GPUs")
    return manifest_hash


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-metrics", type=Path, default=ROOT / "results/k400_remaining4_20260925/NVIDIA_GeForce_RTX_3050_Laptop_GPU/k400/full_1000_seed0/device_metrics.csv")
    args = parser.parse_args()
    baseline = ROOT / "data/k400_user13_20260925.csv"
    fresh = read_csv(args.fresh_metrics)
    manifest_hash = validate_fresh(fresh)
    rows = read_csv(baseline)
    require(len(rows) == 13, "Expected the user's 13 baseline rows")
    for row in rows:
        row.update(Dataset="k400", ClipCount=1000, Evidence="user-supplied prior measurement; not rerun",
                   SourceCSV=str(baseline.relative_to(ROOT)))
    for row in fresh:
        row.update(Evidence="fresh measured 2026-09-25; validated", RuntimeModelKey=row["ModelKey"],
                   SourceCSV=str(args.fresh_metrics.resolve().relative_to(ROOT)))
        if row["ModelKey"] == "dualformer-b-in21k":
            row["ModelKey"] = "dualformer-b-in1k"
            row["Model"] = "DualFormer-B (IN1K)"
        elif row["ModelKey"] == "zeroi2v-b16-8f":
            row["Model"] = "ZeroI2V ViT-B/16 (8f, original graph)"
    rows.extend(fresh)
    require(len(rows) == len({r["ModelKey"] for r in rows}) == 17, "Expected 17 unique model variants")
    rows.sort(key=lambda row: float(row["NS#"]), reverse=True)
    for rank, row in enumerate(rows, 1):
        row["Rank"] = rank
    output = ROOT / "results/k400_full17_20260925.csv"
    write_csv(output, rows)
    write_csv(ROOT / "results/k400_remaining4_20260925.csv", sorted(fresh, key=lambda row: row["Rank"]))

    report = ROOT / "reports/K400_FULL17_20260925.md"
    lines = ["# K400: 13 existing rows plus four completed measurements", "",
             "All four requested models completed 1,000 labeled K400 clips on the RTX 3050 Laptop GPU. "
             "The combined table preserves the 13 values supplied by the user and adds four fresh runs; "
             "the original 13 were not rerun or replaced by conflicting older CSVs.", "",
             "Fresh-run protocol: FP32, batch 1, two untimed warmups, 32 decoded frames at 224 x 224, "
             "one uniformly sampled full-video view with model-specific frame resampling, NVML power at 20 ms. "
             "Timing covers the adapter (including preprocessing and transfer); decoding is excluded. "
             "Peak memory is CUDA allocated memory in MiB. Energy equals sampled mean whole-GPU power times timed inference. "
             "The reported total time is inference-only, not wall-clock duration.", "",
             f"Fixed manifest SHA-256: `{manifest_hash}`.", "",
             "DualFormer-B uses the official IN1K-pretrained K400 release (paper Top-1 81.1%). "
             "The legacy runtime key ends in `in21k`; the combined CSV uses `dualformer-b-in1k` and preserves "
             "the runtime key separately. ZeroI2V uses the original unreparameterized checkpoint graph. "
             "Paper K400 accuracy is supervised accuracy, not zero-shot accuracy.", "",
             "The original 13 rows retain their supplied rounding and NetScores; their checkpoint provenance "
             "is not established by the pasted table. Fresh rows have validated checkpoint hashes, strict 400-class "
             "loaders, the fixed manifest, and checked energy/score arithmetic. This is a local subset comparison "
             "across separate runs, not a fresh controlled 17-model sweep or full official validation. "
             "Power and thermal conditions can differ between runs.", "",
             "Ranked by NS#; **new** marks the four fresh measurements.", "",
             "| Rank | Model | Paper Top-1 (%) | Measured Top-1 (%) | ms/clip | Power (W) | J/clip | Total J | Total s | Peak MiB | NS-E | NS-M | NS# |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        name = row["Model"] + (" **new**" if row in fresh else "")
        values = [str(row["Rank"]), name] + [f"{float(row[field]):.{places}f}" for field, places in (
            ("PublishedTop1", 1), ("MeasuredTop1", 1), ("LatencyMean", 1), ("AvgPower", 2),
            ("EnergyPerClip", 3), ("TotalEnergy", 2), ("TotalTime", 3), ("PeakVRAM", 1),
            ("NS-E", 3), ("NS-M", 3), ("NS#", 3))]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Artifacts", "", "- [Full 17-model CSV](../results/k400_full17_20260925.csv)",
              "- [Four new rows](../results/k400_remaining4_20260925.csv)",
              f"- [Raw benchmark CSV](../{args.fresh_metrics.resolve().relative_to(ROOT).as_posix()})",
              "- [Run log](../logs/k400_remaining4_20260925.log)",
              "- [Preserved user baseline](../data/k400_user13_20260925.csv)", "",
              "Reproduce the fresh measurements:", "", "```powershell",
              "venv/Scripts/python.exe -u run_benchmark.py --device cuda:0 --dataset k400 "
              "--models mvit-v1-b-16x4 dualformer-s zeroi2v-b16-8f dualformer-b-in21k "
              "--num-clips 1000 --manifest manifests/k400_1000_seed0.csv "
              "--output-dir results/k400_remaining4_20260925 --precision fp32 --warmup 2 --power-sample-ms 20 --resume",
              "venv/Scripts/python.exe scripts/summarize_k400_remaining4.py", "```", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    audit = {"status": "passed", "fresh_models": 4, "fresh_clip_inferences": 4000,
             "combined_models": 17, "manifest_sha256": manifest_hash,
             "raw_csv_sha256": digest(args.fresh_metrics), "baseline_csv_sha256": digest(baseline),
             "checks": ["completed", "dataset and count", "400-class accuracy validity", "input protocol",
                        "checkpoint paths and hashes", "manifest identity", "power samples",
                        "energy and NetScore arithmetic", "checkpoint paper accuracy", "same GPU", "17 unique models"]}
    (ROOT / "results/k400_remaining4_20260925/validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Validated four fresh measurements; wrote {output} and {report}")


if __name__ == "__main__":
    main()
