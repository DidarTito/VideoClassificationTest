"""Reproducible, provenance-aware final K400/SSV2 tables and interpretation."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.registry import load_manifest
from scripts.generate_ssv2_full17_device import sha, read, pick
from scripts.ssv2_accuracy_evidence import EVIDENCE
from metrics import netscore_e, netscore_m, netscore_hash

RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
FRESH = RESULTS / "ssv2_accuracy_verification_20260918/device_metrics.csv"
K400 = RESULTS / "NVIDIA_GeForce_RTX_3050_Laptop_GPU/k400/full_1000_seed0/device_metrics.csv"
K400_EXTRA = RESULTS / "missing_k400_1h/NVIDIA_GeForce_RTX_3050_Laptop_GPU/k400/full_1000_seed0/device_metrics.csv"
EXPECTED = {k.split("_")[0]: v["sha256"] for k, v in load_manifest()["benchmark_manifests"].items()}
SPECS = {s["key"]: s for s in load_manifest()["models"]}
MViT_PAPER = "https://openaccess.thecvf.com/content/ICCV2021/papers/Fan_Multiscale_Vision_Transformers_ICCV_2021_paper.pdf"
OMNIVORE_PAPER = "https://openaccess.thecvf.com/content/CVPR2022/papers/Girdhar_Omnivore_A_Single_Model_for_Many_Visual_Modalities_CVPR_2022_paper.pdf"
PUBLISHED_ONLY = {
    "mvit-v1-b-16x4": (64.7, 89.2, "K400 -> SSV2; Table 6, 16x4", MViT_PAPER),
    "mvit-v1-b-32x3": (67.1, 90.8, "K400 -> SSV2; Table 6, 32x3", MViT_PAPER),
    "omnivore-b-in21k": (71.4, 93.5, "Omnivore joint pretraining then SSV2; Table 6, Swin-B (not OmniMAE)", OMNIVORE_PAPER),
}
SOURCES = {
    "uniformer-s": "https://github.com/Sense-X/UniFormer/tree/main/video_classification#something-something-v2",
    "uniformer-b": "https://github.com/Sense-X/UniFormer/tree/main/video_classification#something-something-v2",
    "videomae-b": "https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md#something-something-v2",
    "videoswin-b": "https://github.com/SwinTransformer/Video-Swin-Transformer#something-something-v2",
    "timesformer-b": "https://github.com/facebookresearch/TimeSformer#model-zoo",
    "video-focalnet-b": "https://github.com/TalalWasim/Video-FocalNets#model-zoo",
}
MISMATCHES = {
    "uniformer-s": "historical 16x8 checkpoint and preprocessing differ from frozen 16x4",
    "uniformer-b": "historical 16-frame K400 checkpoint differs from frozen 32x4",
    "videomae-b": "historical Hugging Face checkpoint/preprocessing differ from frozen official 1600e checkpoint",
    "mvit-v1-b-16x4": "historical input had 32 model frames; frozen recipe requires 16",
}
FIELDS = ["Model", "Dataset", "CheckpointProtocol", "PublishedTop1", "MeasuredTop1", "MeasuredTop5",
          "LatencyMs", "PowerW", "EnergyPerClipJ", "TotalEnergyJ", "PeakVRAMMiB", "FPS", "GFLOPs", "RuntimeParamsM",
          "NS-E", "NS-M", "NS#", "Checkpoint", "CheckpointSHA256", "ManifestSHA256", "Device", "AccuracyStatus",
          "PublishedTop5", "ModelKey", "AccuracyValid", "ExactFrozenRecipe", "ClassifierClasses", "ModelFrames",
          "DecodedFrames", "ClipCount", "BatchSize", "Precision", "Seed", "WarmupRuns", "Repeats", "PowerSampleMs",
          "MeasurementBackend", "TotalTimeS", "LatencyStdMs", "GFLOPsSource", "SourceCSV", "SourceCSVSHA256",
          "EvidenceURL", "EvidenceDetail", "GPUUUID", "DriverVersion", "TorchVersion", "CUDAVersion", "Rank"]

def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None

def fmt(value, digits=3):
    n = num(value)
    return f"{n:.{digits}f}".rstrip("0").rstrip(".") if n is not None else str(value or "N/A")

def write_csv(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def table(rows, columns):
    # columns are (field, displayed title) pairs
    lines = ["| " + " | ".join(title for _, title in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(field, "N/A")).replace("|", "\\|") for field, _ in columns) + " |")
    return "\n".join(lines)

def source_inventory():
    target = RESULTS / "final_package_source_inventory.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    files = []
    for parent in ("results", "reports", "checkpoints", "configs", "runs"):
        root = ROOT / parent
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if "ssv2_accuracy_verification_20260918" in str(path):
                continue
            files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    data = {"GitHEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "OriginMain": subprocess.check_output(["git", "rev-parse", "origin/main"], cwd=ROOT, text=True).strip(),
            "GitStatus": subprocess.check_output(["git", "status", "-sb"], cwd=ROOT, text=True), "files": files}
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data

def measured_row(key, source, path, dataset):
    row = {field: "N/A" for field in FIELDS}
    row.update(Model=key, ModelKey=key, Dataset=dataset,
               CheckpointProtocol="GENUINE_SSV2" if dataset == "ssv2" else "GENUINE_K400",
               AccuracyStatus="MEASURED_GENUINE_SSV2" if dataset == "ssv2" else "MEASURED_EXACT_K400",
               AccuracyValid="TRUE", ExactFrozenRecipe="TRUE", ClassifierClasses=174 if dataset == "ssv2" else 400,
               SourceCSV=path.relative_to(ROOT).as_posix(), SourceCSVSHA256=sha(path), EvidenceDetail="")
    mappings = {"PublishedTop1": ("PublishedTop1", "Top1Published(%)"), "PublishedTop5": ("PublishedTop5",),
                "MeasuredTop1": ("MeasuredTop1", "Top1Measured(%)"), "MeasuredTop5": ("MeasuredTop5", "Top5"),
                "LatencyMs": ("LatencyMean", "Latency(ms)"), "LatencyStdMs": ("LatencyStd", "LatencyStd(ms)"),
                "PowerW": ("AvgPower", "AvgPower(W)"), "EnergyPerClipJ": ("EnergyPerClip", "Energy(J)"),
                "TotalEnergyJ": ("TotalEnergy", "TotalEnergy(J)"), "PeakVRAMMiB": ("PeakVRAM", "PeakVRAM(MiB)"),
                "TotalTimeS": ("TotalTime", "TotalTime(s)"), "RuntimeParamsM": ("RuntimeActualParamsM", "ParamsM", "Params(M)"),
                "Device": ("DeviceName", "Device"), "ModelFrames": ("ModelFrames", "FramesUsed"),
                "DecodedFrames": ("DecodedFrames", "Frames"), "MeasurementBackend": ("MeasurementBackend", "PowerBackend")}
    for field in FIELDS:
        if field in mappings:
            row[field] = pick(source, *mappings[field])
        elif field in source and field not in {"Model", "ModelKey", "Dataset", "CheckpointProtocol", "AccuracyValid", "ClassifierClasses"}:
            row[field] = pick(source, field)
    checkpoint = Path(source["Checkpoint"])
    row["Checkpoint"] = checkpoint.relative_to(ROOT).as_posix()
    row["CheckpointSHA256"] = sha(checkpoint)
    row["GFLOPsSource"] = pick(source, "GFLOPsSource")
    if dataset == "ssv2":
        row["EvidenceURL"] = SOURCES[key]
    else:
        if key in MISMATCHES:
            row.update(ExactFrozenRecipe="FALSE", AccuracyStatus="MEASURED_K400_RECIPE_MISMATCH", EvidenceDetail=MISMATCHES[key])
        row["GFLOPsSource"] = "historical CSV metadata; view convention may differ; excluded from runtime FLOP comparisons"
        for profile_path in (RESULTS / "ssv2_device_only_raw.csv", RESULTS / "ssv2_device_only_videoswin_retry.csv"):
            profile = next((p for p in read(profile_path) if p["ModelKey"] == key and p.get("Status") == "completed"), None)
            if (profile and row["ExactFrozenRecipe"] == "TRUE" and profile.get("CheckpointSHA256") == row["CheckpointSHA256"]
                    and str(profile.get("ModelFrames")) == str(row["ModelFrames"]) and num(profile.get("GFLOPs")) is not None):
                row["GFLOPs"] = profile["GFLOPs"]
                row["GFLOPsSource"] = "same exact K400 checkpoint graph and input shape, traced separately on SSV2 inputs: " + profile_path.relative_to(ROOT).as_posix() + "; " + profile.get("GFLOPsSource", "")
    a, t, p, v = (num(row[f]) for f in ("MeasuredTop1", "TotalTimeS", "PowerW", "PeakVRAMMiB"))
    if all(x is not None and x > 0 for x in (a, t, p, v)):
        row.update({"NS-E": netscore_e(a, t, p), "NS-M": netscore_m(a, v), "NS#": netscore_hash(a, v, t, p)})
    return row

def build_rows():
    k400 = {r["ModelKey"]: (r, K400) for r in read(K400)}
    for r in read(K400_EXTRA):
        key = "mvit-v1-b-32x3" if r["ModelKey"] == "mvit-v1-b" else r["ModelKey"]
        if r.get("Status") == "completed":
            k400[key] = (r, K400_EXTRA)
    fresh = {r["ModelKey"]: r for r in read(FRESH)}
    rows = []
    for dataset in ("k400", "ssv2"):
        for key, spec in SPECS.items():
            src, path = (k400.get(key, ({}, K400)) if dataset == "k400" else (fresh.get(key, {}), FRESH))
            if src.get("Status") == "completed" and not (dataset == "k400" and key == "dualformer-b-in21k"):
                rows.append(measured_row(key, src, path, dataset))
                continue
            row = {field: "N/A" for field in FIELDS}
            row.update(Model=key, ModelKey=key, Dataset=dataset, AccuracyValid="FALSE", ExactFrozenRecipe="N/A",
                       AccuracyStatus="NO_EXACT_SSV2_ACCURACY_EVIDENCE_FOR_FROZEN_VARIANT" if dataset == "ssv2" else "NO_EXACT_K400_MEASUREMENT",
                       CheckpointProtocol="NO_EXACT_SSV2_CHECKPOINT" if dataset == "ssv2" else "K400_UNAVAILABLE",
                       EvidenceDetail=src.get("Error", "No compatible completed measurement"))
            if dataset == "k400":
                row["PublishedTop1"] = spec["stage1"]["paper_top1"]
                row["EvidenceURL"] = spec["source_repo"]
            if dataset == "ssv2" and key in PUBLISHED_ONLY:
                top1, top5, detail, url = PUBLISHED_ONLY[key]
                row.update(PublishedTop1=top1, PublishedTop5=top5, AccuracyStatus="PUBLISHED_ONLY_EXACT_VARIANT",
                           CheckpointProtocol="PUBLISHED_ONLY", EvidenceURL=url,
                           EvidenceDetail=detail + "; exact SSV2 checkpoint absent locally / not found in prior official release audit")
            elif dataset == "ssv2" and key in EVIDENCE:
                row.update(PublishedTop1=EVIDENCE[key][1], PublishedTop5=EVIDENCE[key][2] or "N/A",
                           AccuracyStatus="GENUINE_SSV2_VERIFICATION_PENDING_OR_FAILED", EvidenceURL=SOURCES[key])
            elif dataset == "ssv2":
                row["EvidenceDetail"] = "No exact frozen-variant SSV2 result established by the local official-release audit; this is not proof of universal nonexistence."
            elif key == "dualformer-b-in21k":
                row["EvidenceDetail"] = "Local Base artifact is IN1K, not the frozen IN21K identity. Prior 93.1% K400 measurement is preserved in its source CSV and excluded here."
            rows.append(row)
    for dataset in ("k400", "ssv2"):
        ranked = sorted([r for r in rows if r["Dataset"] == dataset and r["ExactFrozenRecipe"] == "TRUE" and num(r["NS#"]) is not None], key=lambda r: -float(r["NS#"]))
        for index, row in enumerate(ranked, 1):
            row["Rank"] = index
    return rows

def comparisons(rows):
    by = {(r["Dataset"], r["Model"]): r for r in rows}
    cross = []
    for key in EVIDENCE:
        k, s = by[("k400", key)], by[("ssv2", key)]
        if not all(num(r["MeasuredTop1"]) is not None for r in (k, s)):
            continue
        cross.append({"Model": key, "K400Top1": k["MeasuredTop1"], "SSV2Top1": s["MeasuredTop1"],
                      "SSV2MinusK400PP": float(s["MeasuredTop1"]) - float(k["MeasuredTop1"]),
                      "K400LatencyMs": k["LatencyMs"], "SSV2LatencyMs": s["LatencyMs"],
                      "SSV2/K400Latency": float(s["LatencyMs"]) / float(k["LatencyMs"]),
                      "K400EnergyJ": k["EnergyPerClipJ"], "SSV2EnergyJ": s["EnergyPerClipJ"],
                      "SSV2/K400Energy": float(s["EnergyPerClipJ"]) / float(k["EnergyPerClipJ"]),
                      "K400VRAMMiB": k["PeakVRAMMiB"], "SSV2VRAMMiB": s["PeakVRAMMiB"],
                      "SSV2/K400VRAM": float(s["PeakVRAMMiB"]) / float(k["PeakVRAMMiB"]),
                      "Comparability": "dataset-specific recipes; historical K400 recipe mismatch" if k["ExactFrozenRecipe"] != "TRUE" else "dataset-specific recipes; frozen identities verified",
                      "K400Frames": k["ModelFrames"], "SSV2Frames": s["ModelFrames"]})
    write_csv(RESULTS / "final_cross_dataset_comparison.csv", cross)
    gaps = [{"Model": r["Model"], "Dataset": r["Dataset"], "PublishedTop1": r["PublishedTop1"],
             "MeasuredTop1": r["MeasuredTop1"], "MeasuredMinusPublishedPP": float(r["MeasuredTop1"]) - float(r["PublishedTop1"]),
             "ExactFrozenRecipe": r["ExactFrozenRecipe"]} for r in rows if all(num(r[f]) is not None for f in ("MeasuredTop1", "PublishedTop1"))]
    write_csv(RESULTS / "final_published_measured_gaps.csv", gaps)
    families = {"UniFormer": ["uniformer-s", "uniformer-b"], "DualFormer": ["dualformer-t", "dualformer-s", "dualformer-b-in21k"],
                "Video-FocalNet": ["video-focalnet-t", "video-focalnet-s", "video-focalnet-b"],
                "MViT": ["mvit-v1-b-16x4", "mvit-v1-b-32x3"], "VideoSwin": ["videoswin-t", "videoswin-s", "videoswin-b"]}
    scaling = []
    device_rows = {r["ModelKey"]: r for r in read(RESULTS / "ssv2_full17_device_metrics.csv")}
    for family, keys in families.items():
        for dataset in ("k400", "ssv2", "ssv2_device_only"):
            for key in keys[1:]:
                if dataset == "ssv2_device_only":
                    a, b = device_rows.get(keys[0], {}), device_rows.get(key, {})
                    if any(r.get("CheckpointProtocol") != "K400_ON_SSV2_INPUT" or r.get("Status") != "COMPLETED_DEVICE_ONLY" for r in (a, b)):
                        continue
                    amap = {"LatencyMs": "LatencyMs", "EnergyPerClipJ": "EnergyPerClipJ", "PeakVRAMMiB": "PeakVRAMMiB", "RuntimeParamsM": "RuntimeParamsM"}
                else:
                    a, b = by[(dataset, keys[0])], by[(dataset, key)]
                    if any(r["ExactFrozenRecipe"] != "TRUE" for r in (a, b)):
                        continue
                    amap = {f: f for f in ("LatencyMs", "EnergyPerClipJ", "PeakVRAMMiB", "RuntimeParamsM")}
                row = {"Family": family, "Dataset": dataset, "Baseline": keys[0], "Compared": key}
                for field, source in amap.items():
                    va, vb = num(a.get(source)), num(b.get(source))
                    row[field + "Baseline"] = va or "N/A"
                    row[field + "Compared"] = vb or "N/A"
                    row[field + "Multiplier"] = vb / va if va and vb else "N/A"
                row["AccuracyChangePP"] = float(b["MeasuredTop1"]) - float(a["MeasuredTop1"]) if dataset != "ssv2_device_only" else "N/A"
                row["Limitation"] = "IN1K Base artifact; not exact frozen IN21K" if key == "dualformer-b-in21k" else "single runs; recipe/pretraining and run conditions may differ"
                scaling.append(row)
    write_csv(RESULTS / "final_family_scaling.csv", scaling)
    return cross, gaps, scaling

def winner_text(rows):
    if not rows:
        return "No eligible completed rows yet."
    choices = [("highest measured Top-1", "MeasuredTop1", max), ("lowest latency", "LatencyMs", min),
               ("lowest average power", "PowerW", min), ("lowest energy/clip", "EnergyPerClipJ", min),
               ("lowest VRAM", "PeakVRAMMiB", min), ("highest NS-E", "NS-E", max), ("highest NS-M", "NS-M", max), ("highest NS#", "NS#", max)]
    lines = []
    for title, field, op in choices:
        eligible = [r for r in rows if num(r[field]) is not None]
        win = op(eligible, key=lambda r: float(r[field]))
        lines.append(f"- {title}: **{win['Model']}**, {fmt(win[field])} ({field}).")
    best = max(rows, key=lambda r: float(r["MeasuredTop1"]))
    fast = min(rows, key=lambda r: float(r["LatencyMs"]))
    energy = min(rows, key=lambda r: float(r["EnergyPerClipJ"]))
    lines += ["", f"The lowest-latency model, {fast['Model']}, has {float(best['LatencyMs']) / float(fast['LatencyMs']):.3f}x "
              f"the throughput of the highest-accuracy model, {best['Model']}; their measured Top-1 difference is "
              f"{float(best['MeasuredTop1']) - float(fast['MeasuredTop1']):.3f} percentage points. "
              f"{energy['Model']} uses {float(best['EnergyPerClipJ']) / float(energy['EnergyPerClipJ']):.3f}x less energy per clip than {best['Model']}."]
    return "\n".join(lines)

LIMITATIONS = ("Results describe fixed local 1,000-clip subsets and one uniformly sampled full-video view, not full official validation. "
              "Paper protocols can use multiple views, different temporal sampling, resolution, preprocessing or checkpoint recipes; no specific discrepancy cause is proven. "
              "K400 and SSV2 are different tasks, label spaces, checkpoints and training recipes: accuracy differences are not pure transfer degradation. "
              "Measurements were taken in separate runs with one repeat and no confidence intervals; thermal conditions and other GPU activity may differ. "
              "FPS means clips/s, not decoded frames/s. Power is whole-GPU NVML power; energy is power times timed adapter inference duration. "
              "Runtime GFLOPs use fvcore with one multiply-add counted as one FLOP; unsupported operators are omitted and listed in GFLOPsSource. "
              "Where an exact K400 graph and frame count have a separate runtime trace, its input-shape-invariant FLOP count is reused with provenance; "
              "otherwise historical K400 FLOPs have inconsistent view conventions and are retained as metadata, not compared as uniform runtime profiles. "
              "VideoMAE's SSV2 2400-epoch model zoo recipe uses no external pretraining data; it is not asserted to be K400-to-SSV2 transfer.")
AVAILABILITY = ("The frozen17 shortlist was selected before considering SSV2 checkpoint availability. Six exact variants have genuine local released SSV2 classifiers; "
                "three have verified published SSV2 values but no exact local checkpoint; eight lack exact SSV2 accuracy evidence in the audited releases. "
                "No proxy size or architecture supplies any SSV2 accuracy row. K400_ON_SSV2_INPUT rows are excluded from SSV2 accuracy rankings and all NS scores. "
                "This is a methodological availability limitation; missing values remain N/A.")

def reports(rows, cross, gaps, scaling):
    groups = {d: [r for r in rows if r["Dataset"] == d] for d in ("k400", "ssv2")}
    eligible = {d: [r for r in groups[d] if r["ExactFrozenRecipe"] == "TRUE"] for d in groups}
    metrics = [("GFLOPs", "GFLOPs"), ("LatencyMs", "Latency (ms)"), ("PowerW", "Power (W)"),
               ("EnergyPerClipJ", "Energy/Clip (J)"), ("TotalEnergyJ", "Total Energy (J)"), ("PeakVRAMMiB", "Peak VRAM (MiB)"),
               ("NS-E", "NS-E"), ("NS-M", "NS-M"), ("NS#", "NS#")]
    kc = [("Rank", "Rank by NS#"), ("Model", "Model"), ("PublishedTop1", "Published K400 Top-1"), ("MeasuredTop1", "Measured K400 Top-1")] + metrics + [("AccuracyStatus", "Evidence status")]
    sc = [("Rank", "Rank by NS#"), ("AccuracyStatus", "Evidence Status"), ("Model", "Model"), ("CheckpointProtocol", "Checkpoint Protocol"),
          ("PublishedTop1", "Published SSV2 Top-1"), ("PublishedTop5", "Published SSV2 Top-5"),
          ("MeasuredTop1", "Measured SSV2 Top-1"), ("MeasuredTop5", "Measured SSV2 Top-5")] + metrics + [("EvidenceURL", "Accuracy Evidence")]
    (REPORTS / "FINAL_K400_TABLE.md").write_text("# Final K400 table\n\nMeasured Top-1 on fixed local K400 1000-clip subset. Published K400 Top-1 is not zero-shot accuracy. "
        "Ranks include only exact frozen recipes; four historical recipe mismatches retain their measurements but have no rank. "
        "The IN1K DualFormer Base release is not substituted for IN21K. Source paths, hashes and hardware are in the master CSV.\n\n" + table(groups["k400"], kc) + "\n\n" + LIMITATIONS + "\n", encoding="utf-8")
    (REPORTS / "FINAL_SSV2_TABLE.md").write_text("# Final SSV2 accuracy table\n\n" + AVAILABILITY + "\n\n" + table(groups["ssv2"], sc) + "\n\n" + LIMITATIONS + "\n", encoding="utf-8")
    (REPORTS / "K400_SSV2_COMPARISON.md").write_text("# K400 versus genuine SSV2\n\n" + LIMITATIONS + "\n\n" +
        table(cross, [(k, k) for k in (list(cross[0]) if cross else ["Model"])]) + "\n\n" +
        "Measured accuracy differs by the listed percentage points between the two dataset-specific inference settings. "
        "Rows marked K400 recipe mismatch are descriptive architecture-level comparisons, not exact frozen-protocol comparisons.\n", encoding="utf-8")
    s_gaps = [g for g in gaps if g["Dataset"] == "ssv2"]
    closest = min(s_gaps, key=lambda g: abs(g["MeasuredMinusPublishedPP"])) if s_gaps else None
    gap_note = f"The smallest absolute SSV2 paper-to-measured difference is {closest['Model']}: {fmt(closest['MeasuredMinusPublishedPP'])} percentage points (measured minus published)." if closest else "Fresh SSV2 values pending."
    largest_gaps = "Largest absolute paper-to-measured discrepancies in the recorded settings: " + "; ".join(
        f"{g['Dataset']}/{g['Model']} {g['MeasuredMinusPublishedPP']:+.3f} percentage points"
        for g in sorted(gaps, key=lambda g: -abs(g["MeasuredMinusPublishedPP"]))[:4]) + ". These are protocol-specific discrepancies, not proof of a particular cause."
    scaling_table = table(scaling, [(k, k) for k in ("Family", "Dataset", "Baseline", "Compared", "LatencyMsBaseline", "LatencyMsCompared", "LatencyMsMultiplier", "EnergyPerClipJMultiplier", "PeakVRAMMiBMultiplier", "RuntimeParamsMMultiplier", "AccuracyChangePP", "Limitation")])
    sections = ["# Intermediate Conclusions", "## 1. K400 Accuracy–Efficiency Trade-off", winner_text(eligible["k400"]),
                "Ranked conclusions use exact compatible K400 rows. Historical mismatched rows remain visible and unranked in FINAL_K400_TABLE.md.",
                "## 2. SSV2 Genuine-Checkpoint Accuracy–Efficiency Trade-off", winner_text(eligible["ssv2"]), gap_note,
                "## 3. Model-Family Scaling", scaling_table,
                "MViT 16-to-32-frame scaling uses the K400-on-SSV2-input device runs because the historical K400 16x4 row had the wrong frame recipe. "
                "UniFormer scaling uses genuine SSV2 results. DualFormer IN1K Base device ratios are labeled and are not exact IN21K evidence. "
                "No accuracy change is calculated from device-only measurements. All raw baseline/compared values are in final_family_scaling.csv.",
                "## 4. Cross-Dataset K400 vs SSV2", table(cross, [(k, k) for k in ("Model", "K400Top1", "SSV2Top1", "SSV2MinusK400PP", "SSV2/K400Latency", "SSV2/K400Energy", "SSV2/K400VRAM", "Comparability")]),
                "Measured accuracy differs by these percentage points between the two dataset-specific inference settings; this is not a measure of pure transfer degradation.",
                "## 5. Published vs Measured Accuracy", largest_gaps, table(gaps, [(k, k) for k in ("Model", "Dataset", "PublishedTop1", "MeasuredTop1", "MeasuredMinusPublishedPP", "ExactFrozenRecipe")]), LIMITATIONS,
                "## 6. Device-Aware Selection", "K400 deployment profiles:\n\n" + winner_text(eligible["k400"]), "Genuine SSV2 deployment profiles:\n\n" + winner_text(eligible["ssv2"]),
                "## 7. SSV2 Checkpoint Availability Limitation", AVAILABILITY]
    (REPORTS / "INTERMEDIATE_CONCLUSIONS_K400_SSV2.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    draft = ["# Draft Results and Discussion", "## Experimental Setup",
             "We evaluated frozen checkpoints on an RTX 3050 Laptop GPU using deterministic 1,000-video manifests, batch size 1, FP32, seed 0, "
             "224-pixel inputs, two warmups and one repeat. Each common input contained 32 decoded frames, temporally resampled by the model adapter. "
             "NVML power sampling used a 20 ms interval. Host conversion and decoding were excluded from latency; adapter normalization, transfer and forward were included. "
             "Classification used the original dataset-specific heads. NS-E, NS-M and NS# used measured accuracy in percentage points, total timed inference seconds, watts and MiB.",
             "## K400 Results", winner_text(eligible["k400"]), "## SSV2 Accuracy Results", winner_text(eligible["ssv2"]), gap_note,
             "## Accuracy–Efficiency Trade-offs", "The measured deployment profiles above separate maximum accuracy from minimum latency, energy and memory. "
             "NS# provides a specified combined score rather than a universal best model. Full measured values and equations are in the master table and validation report.",
             "## Model Family Scaling", scaling_table, "## Cross-Dataset Observations",
             table(cross, [(k, k) for k in ("Model", "K400Top1", "SSV2Top1", "SSV2MinusK400PP", "SSV2/K400Latency", "SSV2/K400Energy", "SSV2/K400VRAM", "Comparability")]),
             "## Checkpoint Availability", AVAILABILITY, "## Limitations", largest_gaps, LIMITATIONS,
             "## Intermediate Conclusions", f"The current package contains {len(eligible['ssv2'])}/17 fresh genuine SSV2 measurements and "
             f"{len(eligible['k400'])}/17 exact-compatible historical K400 rows. Three additional SSV2 rows are published-only evidence; "
             "unavailable accuracy and incompatible frozen recipes are not substituted. Quantified selections are conditional on this hardware and protocol."]
    (REPORTS / "DRAFT_RESULTS_AND_DISCUSSION.md").write_text("\n\n".join(draft) + "\n", encoding="utf-8")

def validate(rows, inventory):
    checks, warnings = [], []
    def check(name, passed, detail=""):
        checks.append({"Check": name, "Status": "PASS" if passed else "FAIL", "Detail": detail})
    for dataset, expected in EXPECTED.items():
        check(dataset + " manifest SHA256", sha(ROOT / f"manifests/{dataset}_1000_seed0.csv") == expected)
    check("34 unique dataset/model pairs", len(rows) == 34 and len({(r["Dataset"], r["Model"]) for r in rows}) == 34)
    check("All six fresh SSV2 runs complete", sum(r["AccuracyStatus"] == "MEASURED_GENUINE_SSV2" for r in rows) == 6)
    head_path = RESULTS / "final_checkpoint_head_audit.json"
    heads = {r["Checkpoint"]: r for r in json.loads(head_path.read_text(encoding="utf-8"))} if head_path.exists() else {}
    current_uuid = next((r["GPUUUID"] for r in rows if r["Dataset"] == "ssv2" and r["AccuracyValid"] == "TRUE"), None)
    seen_hashes = {}
    for r in rows:
        tag = r["Dataset"] + "/" + r["Model"]
        if r["AccuracyValid"] != "TRUE":
            check(tag + " unmeasured accuracy/NS empty", all(num(r[f]) is None for f in ("MeasuredTop1", "MeasuredTop5", "NS-E", "NS-M", "NS#")))
            continue
        check(tag + " manifest/clip/protocol", r["ManifestSHA256"] == EXPECTED[r["Dataset"]] and str(r["ClipCount"]) == "1000" and str(r["BatchSize"]) == "1" and r["Precision"] == "fp32")
        check(tag + " checkpoint file hash", sha(ROOT / r["Checkpoint"]) == r["CheckpointSHA256"] and r["CheckpointSHA256"] != "N/A")
        head = heads.get(r["Checkpoint"], {})
        check(tag + " checkpoint classifier tensor", head.get("Valid") is True and head.get("ExpectedClasses") == int(r["ClassifierClasses"]) and head.get("SHA256") == r["CheckpointSHA256"])
        if r["ExactFrozenRecipe"] == "TRUE":
            expected_hash = EVIDENCE[r["Model"]][0] if r["Dataset"] == "ssv2" else SPECS[r["Model"]]["checkpoint_sha256"]
            check(tag + " exact variant checkpoint hash", r["CheckpointSHA256"] == expected_hash)
        else:
            warnings.append(tag + ": " + r["EvidenceDetail"] + "; preserved, excluded from exact rankings")
        if current_uuid:
            check(tag + " same physical GPU", r["GPUUUID"] == current_uuid, str(r["GPUUUID"]))
        seen_hashes.setdefault(r["CheckpointSHA256"], []).append(tag)
        a, t, p, v = (float(r[f]) for f in ("MeasuredTop1", "TotalTimeS", "PowerW", "PeakVRAMMiB"))
        scores = {"NS-E": 20 * math.log10(a*a / ((t*p)**0.125)), "NS-M": 20 * math.log10(a*a / (v**0.125)), "NS#": 20 * math.log10(a*a / ((v*t*p)**0.125))}
        check(tag + " NS equations", all(math.isclose(float(r[f]), value, abs_tol=1e-9) for f, value in scores.items()))
        check(tag + " energy consistency", math.isclose(float(r["TotalEnergyJ"]), t*p, rel_tol=0.001) and math.isclose(float(r["EnergyPerClipJ"])*1000, float(r["TotalEnergyJ"]), rel_tol=0.001))
        if r["Dataset"] == "ssv2":
            side = json.loads((FRESH.parent / (r["Model"] + ".json")).read_text(encoding="utf-8"))
            check(tag + " runtime 174-class output", side.get("ClassifierClasses") == 174 and side.get("Status") == "completed")
            check(tag + " Top5 >= Top1", float(r["MeasuredTop5"]) >= a)
        else:
            # Pre-existing checkpoint inventory records head shape independently of CSV names.
            audit = {x["model"]: x for x in read(REPORTS / "frozen17_checkpoint_audit.csv")}
            if r["ExactFrozenRecipe"] == "TRUE":
                check(tag + " audited 400-class checkpoint", audit.get(r["Model"], {}).get("classes") == "400" and audit[r["Model"]]["sha256"] == r["CheckpointSHA256"])
    check("No duplicate measured checkpoint hashes", all(len(tags) == 1 for tags in seen_hashes.values()), str({h: t for h, t in seen_hashes.items() if len(t) > 1}))
    for r in read(RESULTS / "ssv2_full17_device_metrics.csv"):
        if r["CheckpointProtocol"] == "K400_ON_SSV2_INPUT":
            check(r["Model"] + " device-only accuracy and NS absent", r["AccuracyValid"] == "FALSE" and all(num(r[f]) is None for f in ("MeasuredSSV2Top1", "MeasuredTop1", "Top5", "NS-E", "NS-M", "NS#")))
    for f in inventory["files"]:
        # The initial device-only table is an explicitly live report, not a historical source.
        if f["path"] in {"results/ssv2_full17_device_metrics.csv"}:
            continue
        check("Preserved " + f["path"], sha(ROOT / f["path"]) == f["sha256"])
    warnings += ["Historical K400 measurements were not rerun; their 400-class identity is supported by fresh checkpoint tensor inspection, not a new forward pass.",
                 "Fresh and historical sessions share a GPU UUID, but temperatures and run dates differ; small performance differences are not controlled causal effects.",
                 "No exact IN21K DualFormer Base accuracy is substituted from the local IN1K Base release.",
                 "Unsupported FLOP operators are disclosed per runtime profile; totals are counts of supported operations."]
    if len([r for r in rows if r["AccuracyStatus"] == "MEASURED_GENUINE_SSV2"]) != 6:
        warnings.append("Fresh SSV2 verification is incomplete; regenerate once all six checkpoints finish.")
    text = "# Result Validation Report\n\n" + f"{sum(c['Status']=='PASS' for c in checks)} checks passed; {sum(c['Status']=='FAIL' for c in checks)} failed.\n\n" + table(checks, [(k, k) for k in ("Check", "Status", "Detail")]) + "\n\n## Explicit limitations and exclusions\n\n" + "\n".join("- " + w for w in warnings) + "\n"
    (REPORTS / "RESULT_VALIDATION_REPORT.md").write_text(text, encoding="utf-8")
    write_csv(RESULTS / "final_validation_checks.csv", checks)
    return [c for c in checks if c["Status"] == "FAIL"]

def main():
    inventory = source_inventory()
    audit_lines = ["# Final source audit", "", f"Git HEAD: `{inventory['GitHEAD']}`; origin/main: `{inventory['OriginMain']}`.", "",
                   "The working tree was already dirty. No commit or push was made. The source inventory records CSV hashes before final reporting.", "",
                   "Controlled primary K400 source: `" + K400.relative_to(ROOT).as_posix() + "`. "
                   "The two completed rows in `" + K400_EXTRA.relative_to(ROOT).as_posix() + "` supply the later DualFormer-S measurement and the canonical MViT-B 32x3 alias. "
                   "All imported rows explicitly record 1,000 clips, the controlled K400 manifest hash and the same RTX 3050 GPU UUID. "
                   "The 10,777-clip run, smoke tests, and unrelated architectures are excluded.", "",
                   "The older `results/rtx3050/k400_1000_seed0_20260722/device_metrics.csv` is preserved but not silently mixed with the controlled primary source.", "",
                   "## Historical frozen-recipe mismatches", ""]
    audit_lines += [f"- `{key}`: {detail}. Retained as measured historical evidence, excluded from exact rankings." for key, detail in MISMATCHES.items()]
    audit_lines += ["- `dualformer-b-in21k`: the local IN1K Base checkpoint and its prior measured CSV are preserved; they do not establish an exact IN21K accuracy row.",
                    "- `zeroi2v-b16-8f`: no completed compatible loader measurement; no proxy architecture used.", "",
                    "## Primary SSV2 evidence", ""]
    audit_lines += [f"- `{key}`: [official model zoo]({url})." for key, url in SOURCES.items()]
    audit_lines += [f"- `{key}`: Top-1 {record[0]}, Top-5 {record[1]}; [original paper, Table 6]({record[3]})." for key, record in PUBLISHED_ONLY.items()]
    (REPORTS / "FINAL_SOURCE_AUDIT.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    rows = build_rows()
    write_csv(RESULTS / "final_k400_ssv2_master.csv", rows, FIELDS)
    cross, gaps, scaling = comparisons(rows)
    reports(rows, cross, gaps, scaling)
    failures = validate(rows, inventory)
    evidence = [{"Model": r["Model"], "AccuracyStatus": r["AccuracyStatus"], "PublishedTop1": r["PublishedTop1"], "PublishedTop5": r["PublishedTop5"],
                 "EvidenceURL": r["EvidenceURL"], "EvidenceDetail": r["EvidenceDetail"]} for r in rows if r["Dataset"] == "ssv2"]
    write_csv(RESULTS / "final_ssv2_published_evidence.csv", evidence)
    lines = [f"Genuine SSV2 measured accuracy: {sum(r['AccuracyStatus']=='MEASURED_GENUINE_SSV2' for r in rows)}/17",
             f"Published-only exact SSV2 accuracy: {sum(r['AccuracyStatus']=='PUBLISHED_ONLY_EXACT_VARIANT' for r in rows)}/17",
             f"No exact SSV2 accuracy evidence: {sum(r['AccuracyStatus']=='NO_EXACT_SSV2_ACCURACY_EVIDENCE_FOR_FROZEN_VARIANT' for r in rows)}/17",
             f"K400 measured rows: {sum(r['Dataset']=='k400' and r['AccuracyValid']=='TRUE' for r in rows)}/17 (four explicitly mismatched historical recipes are unranked)"]
    for dataset in ("k400", "ssv2"):
        selected = [r for r in rows if r["Dataset"] == dataset and r["ExactFrozenRecipe"] == "TRUE"]
        lines += ["", dataset + " exact-compatible selection:", winner_text(selected)]
    lines += ["", f"Validation failures: {len(failures)}", "", "Generated artifacts:"]
    lines += [p.relative_to(ROOT).as_posix() for p in sorted(RESULTS.glob("final_*"))]
    lines += [p.relative_to(ROOT).as_posix() for p in sorted(FRESH.parent.glob("*"))]
    lines += ["reports/" + name for name in ("FINAL_K400_TABLE.md", "FINAL_SSV2_TABLE.md", "K400_SSV2_COMPARISON.md", "INTERMEDIATE_CONCLUSIONS_K400_SSV2.md", "DRAFT_RESULTS_AND_DISCUSSION.md", "RESULT_VALIDATION_REPORT.md")]
    lines += ["reports/FINAL_SOURCE_AUDIT.md", "reports/FINAL_TERMINAL_SUMMARY.txt", "reports/SSV2_FULL17_DEVICE_TABLE.md", "results/ssv2_full17_device_metrics.csv"]
    lines += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "figures").glob("final_accuracy_device_tradeoffs.*"))]
    summary = "\n".join(lines) + "\n"
    (REPORTS / "FINAL_TERMINAL_SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()
