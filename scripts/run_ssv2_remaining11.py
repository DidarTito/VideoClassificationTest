"""Resume TASK_SSV2_11 using audited historical rows and existing model loaders.

Each accepted/completed model is atomically saved before moving to the next.
The six excluded models can never be dispatched by this script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/ssv2_remaining11"
REPORT = ROOT / "reports/SSV2_REMAINING11_RESULTS.md"
MANIFEST = ROOT / "manifests/ssv2_1000_seed0.csv"
EXPECTED = "c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"
KEYS = ["dualformer-t", "video-focalnet-t", "videoswin-t", "mvit-v1-b-16x4",
        "video-focalnet-s", "mvit-v1-b-32x3", "dualformer-s", "videoswin-s",
        "zeroi2v-b16-8f", "dualformer-b-in21k", "omnivore-b-in21k"]
FORBIDDEN = {"uniformer-s", "uniformer-b", "videomae-b", "videoswin-b",
             "timesformer-b", "video-focalnet-b"}
assert not FORBIDDEN.intersection(KEYS)
FIELDS = ["Model", "Protocol", "Checkpoint", "CheckpointSHA256", "AccuracyValid",
          "MeasuredTop1", "MeasuredTop5", "GFLOPs", "ParamsM", "ModelFrames",
          "DecodedFrames", "LatencyMeanMs", "LatencyStdMs", "FPS", "AveragePowerW",
          "EnergyPerClipJ", "TotalEnergyJ", "PeakVRAMMiB", "NS-E", "NS-M", "NS#",
          "Status", "Notes", "Dataset", "EvaluatedClips", "Top1Correct", "Top5Correct",
          "TotalTimedSeconds", "InputProtocol", "SourceCSV", "Reused"]
TABLE = {"Model": "Model", "Protocol": "Protocol", "AccuracyValid": "AccuracyValid",
         "Top1": "MeasuredTop1", "Top5": "MeasuredTop5", "GFLOPs": "GFLOPs",
         "Latency": "LatencyMeanMs", "FPS": "FPS", "Power": "AveragePowerW",
         "Energy/Clip": "EnergyPerClipJ", "Total Energy": "TotalEnergyJ",
         "VRAM": "PeakVRAMMiB", "NS#": "NS#", "Status": "Status"}
HISTORICAL = ["results/ssv2_device_only_raw.csv", "results/ssv2_device_only_videoswin_retry.csv"]
AUDIT_ROOT = ROOT / "results/ssv2_11_exact_accuracy"
SOURCES = {
    "dualformer": ("dual_zoo.txt", "https://github.com/sail-sg/dualformer",
                   "No exact SSV2 release found in the official model zoo or prior source audit."),
    "video-focalnet": ("focal_zoo.txt", "https://github.com/TalalWasim/Video-FocalNets",
                       "Official SSV2 release is Base, not Tiny or Small."),
    "videoswin": ("swin_zoo.txt", "https://github.com/SwinTransformer/Video-Swin-Transformer",
                  "Official SSV2 release is Base, not Tiny or Small."),
    "mvit": ("mvit_zoo.txt", "https://github.com/facebookresearch/SlowFast/blob/main/MODEL_ZOO.md",
             "No exact MViTv1 B 16x4/32x3 SSV2 release found; MViTv2 and B-24 are different variants."),
    "zeroi2v": ("zero_zoo.txt", "https://github.com/MCG-NJU/ZeroI2V",
                "Official SSV2 checkpoint is L/14 with 32 frames, not B/16 with 8 frames."),
    "omnivore": ("omnivore_zoo.txt", "https://github.com/facebookresearch/omnivore/tree/main/omnivore",
                 "Released IN21K Omnivore-B video head has 400 classes; local OmniMAE is a different model."),
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def save_json(path, obj):
    atomic_text(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def summaries(rows):
    good = [r for r in rows if r["Status"] in {"COMPLETED", "REUSED_VERIFIED"}]
    return [f"SSV2 checkpoints used: {sum(r['AccuracyValid'] == 'TRUE' for r in good)}/11",
            f"K400 pretrained checkpoints used: {sum(r['Protocol'] == 'PRETRAIN_K400_ON_SSV2' for r in good)}/11",
            f"Successful device benchmarks: {len(good)}/11",
            f"Valid SSV2 accuracy models: {sum(r['AccuracyValid'] == 'TRUE' for r in good)}/11",
            f"Failed models: {sum(r['Status'] in {'FAILED_RUNTIME', 'FAILED_CHECKPOINT', 'NO_IMPLEMENTATION', 'NO_CHECKPOINT'} for r in rows)}/11"]


def table(rows):
    lines = ["| " + " | ".join(TABLE) + " |", "| " + " | ".join(["---"] * len(TABLE)) + " |"]
    for row in rows:
        values = []
        for key in TABLE.values():
            value = row[key]
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def publish(records):
    rows = [records[key]["result"] for key in KEYS if key in records]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(OUT / "device_metrics.csv", buf.getvalue())
    lines = ["# SSV2 remaining 11 results", "",
             f"Saved {len(rows)}/11 task rows. Each model has an independently saved JSON record.", "",
             "Manifest: `manifests/ssv2_1000_seed0.csv`; SHA256: `" + EXPECTED + "`.",
             "Batch 1; FP32; seed 0; warmup 2; one repeat; requested NVML interval 20 ms.",
             "Historical measurements are reused only after checksum, manifest, settings, head-shape, and metric consistency checks.",
             "The six excluded models were not rerun. No training or Git push was performed.", "",
             "The existing SSV2 protocol decodes 32 uniformly spaced full-video frames, resizes the short side to 224, "
             "and center crops 224x224. Adapters retain their own 8/16/32-frame selection and normalization. "
             "K400 training strides in frozen17 are provenance metadata, not the temporal sampling used by these historical SSV2 measurements. "
             "These are single-view device measurements, not paper multi-view evaluations.", "",
             "Latency and its standard deviation are milliseconds; FPS is clips/second; power is W; "
             "energy is J; VRAM is peak allocated CUDA MiB. Energy = mean sampled power times timed inference duration. "
             "GFLOPs use the existing fvcore trace (one MAC = one FLOP); unsupported operators are omitted and preserved in each JSON record.", "",
             "K400 classifiers are scored against official SSV2 integer labels. AccuracyValid is TRUE only for a 174-class SSV2 head.", "",
             *table(rows), "", *summaries(rows), "",
             "## Checkpoint audit", "",
             "No exact SSV2 checkpoint for these 11 variants was identified in the local inventory and official source audit. "
             "The previous audit and source snapshots are retained in `results/ssv2_11_exact_accuracy/`.", ""]
    for _, (file, url, reason) in SOURCES.items():
        lines.append(f"- [{file}]({url}): {reason}")
    lines += ["", "## Model notes", ""]
    lines += [f"- `{r['Model']}`: {r['Notes']}" for r in rows]
    lines += ["", "## Resume", "", "```powershell",
              r".\venv\Scripts\python.exe -u scripts\run_ssv2_remaining11.py --run-missing", "```", "",
              "Successful raw results are verified and reused on resume. Failed models remain retryable. "
              "Checkpoint mismatches are recorded instead of substituting another variant."]
    atomic_text(REPORT, "\n".join(lines) + "\n")


def validate_raw(row, spec, digest):
    wanted = {"Status": "completed", "Dataset": "ssv2",
              "ManifestSHA256": EXPECTED, "CheckpointSHA256": digest,
              "ClipCount": "1000", "BatchSize": "1", "Precision": "fp32", "Seed": "0",
              "WarmupRuns": "2", "PowerSampleMs": "20", "PowerMode": "nvml", "Repeats": "1",
              "Resolution": "224x224", "DecodedFrames": "32",
              "ModelFrames": str(spec["input"]["num_frames"]),
              "InputProtocol": "single uniformly sampled full-video view"}
    for key, value in wanted.items():
        if str(row.get(key, "")).lower() != value.lower():
            raise ValueError(f"{key}: {row.get(key)!r} != {value!r}")
    if Path(row["Checkpoint"]).resolve() != (ROOT / spec["k400_checkpoint"]).resolve():
        raise ValueError("wrong checkpoint path")
    if "NVML" not in row["MeasurementBackend"] or int(row["PowerSamples"]) <= 0:
        raise ValueError("missing NVML power samples")
    for key in ("MeasuredTop1", "Top5"):
        try:
            value = float(row[key])
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"missing accuracy in {key}") from exc
        if not math.isfinite(value) or value < 0 or value > 100:
            raise ValueError(f"invalid accuracy in {key}: {row.get(key)!r}")
    nums = {key: float(row[key]) for key in ("GFLOPs", "ParamsM", "LatencyMean", "LatencyStd", "FPS",
            "AvgPower", "EnergyPerClip", "TotalEnergy", "PeakVRAM", "TotalTime")}
    if any(not math.isfinite(v) or v < 0 or (v == 0 and k != "LatencyStd") for k, v in nums.items()):
        raise ValueError("missing/non-finite/non-positive measurement")
    checks = [(nums["FPS"], 1000 / nums["LatencyMean"]),
              (nums["TotalTime"], nums["LatencyMean"]),
              (nums["EnergyPerClip"], nums["AvgPower"] * nums["LatencyMean"] / 1000),
              (nums["TotalEnergy"], nums["EnergyPerClip"] * 1000)]
    if not all(math.isclose(a, b, rel_tol=1e-8) for a, b in checks):
        raise ValueError("inconsistent timing/energy arithmetic")


def base_row(key, spec, digest):
    row = dict.fromkeys(FIELDS, "N/A")
    row.update(Model=key, Protocol="PRETRAIN_K400_ON_SSV2", Checkpoint=spec["k400_checkpoint"],
               CheckpointSHA256=digest, AccuracyValid="FALSE", ModelFrames=spec["input"]["num_frames"],
               DecodedFrames=32, Dataset="ssv2", EvaluatedClips=0, Reused="FALSE",
               InputProtocol="single uniformly sampled full-video view", Notes="", Status="PENDING")
    return row


def run_one(key):
    assert key in KEYS and key not in FORBIDDEN
    raw = OUT / "raw" / f"{key}.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "run_benchmark.py"),
           "--dataset", "ssv2", "--models", key, "--manifest", str(MANIFEST),
           "--annotations", str(ROOT / "third_party/UniFormer/video_classification/data_list/sthv2/somesomev2_rgb_validation_split.txt"),
           "--num-clips", "1000", "--batch-size", "1", "--precision", "fp32", "--seed", "0",
           "--warmup", "2", "--repeats", "1", "--power", "nvml", "--power-sample-ms", "20",
           "--device", "cuda:0", "--allow-skips", "--resume", "--output", str(raw)]
    save_json(OUT / "commands" / f"{key}.json", cmd)
    with (OUT / f"{key}.log").open("a", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    print(f"{key}: process exit {result.returncode}", flush=True)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-missing", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    specs = {s["key"]: s for s in yaml.safe_load((ROOT / "configs/frozen17.yaml").read_text())["models"]}
    manifest_rows = read_csv(MANIFEST)
    assert sha(MANIFEST) == EXPECTED, "wrong SSV2 manifest"
    assert len(manifest_rows) == 1000 and len({r["RelativePath"] for r in manifest_rows}) == 1000
    assert all((ROOT / "datasets/ssv2/videos" / r["RelativePath"]).is_file() for r in manifest_rows)
    cache = None
    for candidate in (ROOT / "datasets/ssv2/videos").glob(".clip_cache_v4_32f_224px_1000c*/manifest.json"):
        cache_payload = json.loads(candidate.read_text())
        if [Path(e["source"]).name for e in cache_payload["entries"]] == [r["RelativePath"] for r in manifest_rows]:
            cache = candidate
            break
    assert cache is not None, "no ordered cache matches the fixed manifest"
    assert all((cache.parent / e["cache"]).is_file() for e in cache_payload["entries"])
    save_json(OUT / "dataset_validation.json", dict(manifest=str(MANIFEST), sha256=EXPECTED,
              clips=1000, unique_clips=1000, missing_videos=0, cache=str(cache), cache_sha256=sha(cache),
              ordered_cache_matches=True, missing_cache_tensors=0))
    inventory = {r["path"]: r for r in json.loads((AUDIT_ROOT / "local_checkpoint_inventory.json").read_text())}
    records = {}
    # Existing per-model records survive a crash between the JSON and master saves.
    for key in KEYS:
        path = OUT / "models" / f"{key}.json"
        if path.is_file():
            records[key] = json.loads(path.read_text(encoding="utf-8"))
    for key in KEYS:
        spec = specs[key]
        checkpoint = ROOT / spec["k400_checkpoint"]
        digest = sha(checkpoint) if checkpoint.is_file() else "N/A"
        row = base_row(key, spec, digest)
        evidence = dict(checked_utc=datetime.now(timezone.utc).isoformat(), spec=spec,
                        manifest_sha256=EXPECTED, checkpoint_sha256=digest,
                        previous_checkpoint_inventory=inventory.get(spec["k400_checkpoint"]), reuse_rejections=[])
        if not checkpoint.is_file():
            row.update(Status="NO_CHECKPOINT", Notes="Configured exact K400 checkpoint is missing.")
        elif digest != spec["checkpoint_sha256"]:
            row.update(Status="FAILED_CHECKPOINT", Notes="Local checkpoint SHA256 differs from frozen17.")
        else:
            candidates = [(ROOT / p, r) for p in HISTORICAL for r in read_csv(ROOT / p)
                          if r.get("ModelKey") == key]
            fresh = OUT / "raw" / f"{key}.csv"
            candidates.extend((fresh, r) for r in read_csv(fresh) if r.get("ModelKey") == key)
            accepted = None
            for source, raw in reversed(candidates):
                try:
                    validate_raw(raw, spec, digest)
                    old = inventory.get(spec["k400_checkpoint"], {})
                    if source != fresh:
                        assert old.get("sha256") == digest and not old.get("error")
                        assert any(s[0] == 400 for s in old.get("head_shapes", {}).values())
                    accepted = (source, raw)
                    break
                except (ValueError, KeyError, AssertionError) as exc:
                    evidence["reuse_rejections"].append(dict(source=str(source), reason=str(exc)))
            if accepted is None and args.run_missing:
                print(f"Running {key}; previous records are not reusable", flush=True)
                if fresh.is_file():
                    fresh.unlink()
                run_one(key)
                for raw in reversed(read_csv(fresh)):
                    if raw.get("ModelKey") == key:
                        evidence["attempt_raw"] = raw
                        try:
                            validate_raw(raw, spec, digest)
                            accepted = (fresh, raw)
                        except (ValueError, KeyError) as exc:
                            row.update(Status="FAILED_RUNTIME", Notes=raw.get("Error") or str(exc))
                        break
                if accepted is None and row["Status"] == "PENDING":
                    row.update(Status="FAILED_RUNTIME", Notes=f"Benchmark produced no completed row; see {key}.log")
            if accepted:
                source, raw = accepted
                reused = source != fresh
                mapping = {"GFLOPs": "GFLOPs", "ParamsM": "RuntimeActualParamsM", "LatencyMeanMs": "LatencyMean",
                           "LatencyStdMs": "LatencyStd", "FPS": "FPS", "AveragePowerW": "AvgPower",
                           "EnergyPerClipJ": "EnergyPerClip", "TotalEnergyJ": "TotalEnergy", "PeakVRAMMiB": "PeakVRAM",
                           "TotalTimedSeconds": "TotalTime", "MeasuredTop1": "MeasuredTop1", "MeasuredTop5": "Top5",
                           "NS-E": "NS-E", "NS-M": "NS-M", "NS#": "NS#"}
                row.update({a: float(raw[b]) for a, b in mapping.items() if raw.get(b) not in (None, "", "nan", "N/A")})
                classes = int(float(raw.get("ClassifierClasses") or 400))
                valid = str(raw.get("AccuracyValid", "")).lower() in {"true", "1"} or classes == 174
                row.update(EvaluatedClips=1000, AccuracyValid="TRUE" if valid else "FALSE",
                           Protocol="SSV2_CHECKPOINT" if valid else "PRETRAIN_K400_ON_SSV2",
                           Status="REUSED_VERIFIED" if reused else "COMPLETED",
                           Reused="TRUE" if reused else "FALSE", SourceCSV=source.relative_to(ROOT).as_posix(),
                           Top1Correct=raw.get("Top1Correct", "N/A"), Top5Correct=raw.get("Top5Correct", "N/A"),
                           Notes=("Verified historical scored measurement; no rerun." if reused else
                                  "Completed exact K400 model on 1000 SSV2 clips with Top-1/Top-5 scoring."))
                if key == "dualformer-b-in21k":
                    row["Notes"] += " Local checkpoint is the IN1K K400 Base release, not an IN21K SSV2 classifier."
                evidence.update(source_csv_sha256=sha(source), raw=raw, normalization=spec["input"],
                                classifier_classes=classes, accuracy_scoring=True)
            elif row["Status"] == "PENDING":
                row["Notes"] = "Awaiting runtime benchmark; run this script with --run-missing."
        record = dict(result=row, evidence=evidence)
        save_json(OUT / "models" / f"{key}.json", record)
        records[key] = record
        publish(records)
        print(f"Saved {key}: {row['Status']}", flush=True)
    rows = [records[key]["result"] for key in KEYS]
    output = "\n".join(table(rows) + [""] + summaries(rows)) + "\n"
    atomic_text(OUT / "final_summary.txt", output)
    print(output, flush=True)


if __name__ == "__main__":
    main()
