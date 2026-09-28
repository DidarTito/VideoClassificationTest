"""Merge preserved SSV2 classifications with explicitly unscored device runs."""
from __future__ import annotations

import csv
import hashlib
from functools import lru_cache
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.registry import load_manifest
from models import REQUESTED_MODEL_SPECS

HISTORICAL = ROOT / "results/rtx3050/ssv2_1000_seed0_20260722/device_metrics.csv"
RAW = ROOT / "results/ssv2_device_only_raw.csv"
CSV = ROOT / "results/ssv2_full17_device_metrics.csv"
REPORT = ROOT / "reports/SSV2_FULL17_DEVICE_TABLE.md"
GENUINE = {"uniformer-s", "uniformer-b", "videomae-b", "videoswin-b", "timesformer-b", "video-focalnet-b"}
NOTICE = ("Models without released exact SSV2 checkpoints were evaluated using their exact "
          "K400 checkpoints on the same SSV2 video inputs for device-performance metrics only. "
          "Their classification accuracy is not reported because the K400 classifier predicts "
          "400 Kinetics classes rather than the 174 Something-Something V2 classes.")
COLUMNS = ["Model", "CheckpointProtocol", "Checkpoint", "CheckpointSHA256", "PublishedSSV2Top1",
           "MeasuredSSV2Top1", "LatencyMs", "PowerW", "EnergyPerClipJ", "TotalEnergyJ", "PeakVRAMMiB",
           "FPS", "GFLOPs", "RuntimeParamsM", "ModelFrames", "AccuracyValid", "Status"]
EXTRA = ["ModelKey", "LatencyMeanMs", "LatencyStdMs", "AveragePowerW", "DecodedFrames",
         "RuntimeActualParamsM", "GFLOPsPerView", "GFLOPsSource", "MeasuredTop1", "Top5", "NS-E", "NS-M", "NS#",
         "Manifest", "ManifestSHA256", "ClipCount", "BatchSize", "Precision", "Seed", "WarmupRuns",
         "Repeats", "PowerMode", "PowerSampleMs", "PowerSamples", "MeasurementBackend", "SourceCSV", "Error"]

def read(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

@lru_cache(None)
def sha(path):
    if not Path(path).is_file():
        return "N/A"
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def pick(row, *names):
    return next((row[n] for n in names if row.get(n) not in (None, "", "nan", "N/A")), "N/A")

def generate():
    historical = {r["ModelKey"]: r for r in read(HISTORICAL)}
    current = {r["ModelKey"]: r for r in read(RAW)}
    retry_path = ROOT / "results/ssv2_device_only_videoswin_retry.csv"
    for r in read(retry_path):
        if r.get("Status") == "completed":
            current[r["ModelKey"]] = r
    rows = []
    manifest = "manifests/ssv2_1000_seed0.csv"
    for spec in load_manifest()["models"]:
        key = spec["key"]
        genuine = key in GENUINE
        source = historical.get(key, {}) if genuine else current.get(key, {})
        checkpoint = (REQUESTED_MODEL_SPECS[key].for_dataset("ssv2").checkpoint if genuine
                      else spec["k400_checkpoint"])
        status = "PRESERVED_GENUINE_SSV2" if genuine else "PENDING"
        if not genuine and source:
            status = "COMPLETED_DEVICE_ONLY" if source.get("Status") == "completed" else "SKIPPED_FAILED"
        row = {c: "N/A" for c in COLUMNS + EXTRA}
        row.update(Model=key, ModelKey=key, CheckpointProtocol="GENUINE_SSV2" if genuine else "K400_ON_SSV2_INPUT",
                   Checkpoint=checkpoint, CheckpointSHA256=sha(ROOT / checkpoint),
                   AccuracyValid="TRUE" if genuine else "FALSE", Status=status,
                   Manifest=manifest, ManifestSHA256=sha(ROOT / manifest), ClipCount="1000", BatchSize="1",
                   Precision="fp32", Seed="0", WarmupRuns="2", Repeats="1", PowerMode="nvml", PowerSampleMs="20",
                   DecodedFrames="32", SourceCSV=str((HISTORICAL if genuine else RAW).relative_to(ROOT)),
                   Error=source.get("Error", ""))
        if not genuine and source.get("ModelKey") in {"videoswin-t", "videoswin-s"} and source.get("Status") == "completed":
            row["SourceCSV"] = retry_path.relative_to(ROOT).as_posix()
        mappings = {"LatencyMs": ("LatencyMean", "Latency(ms)"), "LatencyStdMs": ("LatencyStd", "LatencyStd(ms)"),
                    "PowerW": ("AvgPower", "AvgPower(W)"), "EnergyPerClipJ": ("EnergyPerClip", "Energy(J)"),
                    "TotalEnergyJ": ("TotalEnergy", "TotalEnergy(J)"), "PeakVRAMMiB": ("PeakVRAM", "PeakVRAM(MiB)"),
                    "FPS": ("FPS",), "GFLOPs": ("GFLOPs",), "RuntimeParamsM": ("RuntimeActualParamsM", "Params(M)"),
                    "ModelFrames": ("ModelFrames", "FramesUsed"), "PowerSamples": ("PowerSamples",),
                    "MeasurementBackend": ("MeasurementBackend", "PowerBackend")}
        if genuine or status == "COMPLETED_DEVICE_ONLY":
            for target, names in mappings.items():
                row[target] = pick(source, *names)
            row["GFLOPsSource"] = "preserved historical metadata" if genuine else pick(source, "GFLOPsSource")
        if genuine:
            row["PublishedSSV2Top1"] = pick(source, "Top1Published(%)")
            row["MeasuredSSV2Top1"] = pick(source, "Top1Measured(%)")
            row["MeasuredTop1"] = row["MeasuredSSV2Top1"]
            for metric in ("NS-E", "NS-M", "NS#"):
                row[metric] = pick(source, metric + "(1/8)")
        for target, original in (("LatencyMeanMs", "LatencyMs"), ("AveragePowerW", "PowerW"),
                                 ("RuntimeActualParamsM", "RuntimeParamsM"), ("GFLOPsPerView", "GFLOPs")):
            row[target] = row[original]
        rows.append(row)
    temporary = CSV.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS + EXTRA)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(CSV)
    lines = [NOTICE, "", "# SSV2 full17 device performance", "",
             "Fixed input: `manifests/ssv2_1000_seed0.csv`; 1,000 clips; batch 1; FP32; seed 0; warmup 2; repeat 1; NVML at 20 ms. "
             "The six genuine SSV2 classification rows are preserved from the historical CSV without rerunning. "
             "`N/A` means unavailable or inapplicable; pending/failed rows have no invented device values.", "",
             "FPS is clips/second (1,000 / latency in milliseconds). 32 frames are decoded; ModelFrames is the temporal input to each model. "
             "Energy is average sampled GPU power multiplied by measured inference time. Peak VRAM is peak allocated CUDA memory. "
             "New GFLOPs values use an fvcore trace of the loaded inference graph (one multiply-add counted as one FLOP); "
             "unsupported operators are listed per row in the CSV and are excluded from the count. Historical GFLOPs remain unchanged source metadata.", "",
             "The `dualformer-b-in21k` key uses the existing official IN1K K400 Base release specified in frozen17; "
             "its checkpoint is not relabeled as IN21K. The checkpoint path and actual SHA256 identify the artifact.", "",
             "| " + " | ".join(COLUMNS) + " |", "| " + " | ".join("---" for _ in COLUMNS) + " |"]
    lines.extend("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in COLUMNS) + " |" for r in rows)
    lines += ["", "## Execution status", ""]
    lines.extend(f"- `{r['Model']}`: {r['Status']}" + (f" — {r['Error']}" if r['Error'] else "") for r in rows)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Updated 17 rows: {sum(r['Status'] == 'COMPLETED_DEVICE_ONLY' for r in rows)} new device runs, 6 preserved classifications.")

if __name__ == "__main__":
    generate()
    if "--watch" in sys.argv:
        import time
        last = RAW.stat().st_mtime_ns if RAW.exists() else None
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            if len(read(RAW)) == 11:
                break
            time.sleep(10)
            stamp = RAW.stat().st_mtime_ns if RAW.exists() else None
            if stamp != last:
                generate()
                last = stamp
