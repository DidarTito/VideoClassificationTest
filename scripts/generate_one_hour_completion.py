"""Generate the bounded one-hour completion report from benchmark CSVs."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_K400 = ROOT / "results" / "rtx3050" / "k400_1000_seed0_20260722" / "device_metrics.csv"
HISTORICAL_SSV2 = ROOT / "results" / "rtx3050" / "ssv2_1000_seed0_20260722" / "device_metrics.csv"
NEW_K400 = ROOT / "results" / "missing_k400_1h" / "NVIDIA_GeForce_RTX_3050_Laptop_GPU" / "k400" / "full_1000_seed0" / "device_metrics.csv"
REPORT = ROOT / "reports" / "ONE_HOUR_COMPLETION.md"
MASTER = ROOT / "results" / "master_benchmark_results.csv"
CHECKPOINTS = {
    ("k400", "mvit-v1-b"): ROOT / "checkpoints" / "mvit" / "MVIT_B_32x3_f294077834.pyth",
    ("ssv2", "uniformer-s"): ROOT / "checkpoints" / "legacy_ssv2_released" / "uniformer_small_sthv2_16_prek400.pth",
    ("ssv2", "uniformer-b"): ROOT / "checkpoints" / "legacy_ssv2_released" / "uniformer_base_sthv2_16_prek400.pth",
    ("ssv2", "videomae-b"): ROOT / "checkpoints" / "legacy_ssv2_released" / "videomae_vit_b_ssv2_2400e.pth",
    ("ssv2", "videoswin-b"): ROOT / "checkpoints" / "legacy_ssv2_released" / "swin_base_patch244_window1677_sthv2.pth",
    ("ssv2", "timesformer-b"): ROOT / "checkpoints" / "legacy_ssv2_released" / "TimeSformer_divST_8_224_SSv2.pyth",
    ("ssv2", "video-focalnet-b"): ROOT / "checkpoints" / "legacy_ssv2_released" / "video-focalnet_base_ssv2.pth",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, str], *names: str) -> str:
    for name in names:
        if row.get(name) not in (None, ""):
            return row[name]
    return ""


def sha256(path: str) -> str:
    candidate = Path(path)
    if not candidate.is_file():
        return ""
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measured_row(row: dict[str, str], status: str = "COMPLETED") -> dict[str, str]:
    dataset = value(row, "Dataset", "dataset")
    key = value(row, "ModelKey", "model_key")
    checkpoint = value(row, "Checkpoint", "checkpoint") or str(CHECKPOINTS.get((dataset, key), ""))
    return {
        "Dataset": dataset,
        "Model": value(row, "Model", "model"),
        "ModelKey": key,
        "Status": status,
        "MeasuredTop1(%)": value(row, "Top1Measured(%)", "MeasuredTop1"),
        "PublishedTop1(%)": value(row, "Top1Published(%)", "PublishedTop1"),
        "Latency(ms)": value(row, "Latency(ms)", "LatencyMean"),
        "AvgPower(W)": value(row, "AvgPower(W)", "AvgPower"),
        "EnergyPerClip(J)": value(row, "Energy(J)", "EnergyPerClip"),
        "TotalEnergy(J)": value(row, "TotalEnergy(J)", "TotalEnergy"),
        "PeakVRAM(MiB)": value(row, "PeakVRAM(MiB)", "PeakVRAM"),
        "NS-E": value(row, "NS-E(1/8)", "NS-E"),
        "NS-M": value(row, "NS-M(1/8)", "NS-M"),
        "NS#": value(row, "NS#(1/8)", "NS#"),
        "Checkpoint": checkpoint,
        "SHA256": sha256(checkpoint),
    }


def blocked(dataset: str, key: str, model: str, reason: str, checkpoint: str = "") -> dict[str, str]:
    return {
        "Dataset": dataset,
        "Model": model,
        "ModelKey": key,
        "Status": reason,
        "MeasuredTop1(%)": "",
        "PublishedTop1(%)": "",
        "Latency(ms)": "",
        "AvgPower(W)": "",
        "EnergyPerClip(J)": "",
        "TotalEnergy(J)": "",
        "PeakVRAM(MiB)": "",
        "NS-E": "",
        "NS-M": "",
        "NS#": "",
        "Checkpoint": checkpoint,
        "SHA256": sha256(checkpoint),
    }


def render_table(rows: list[dict[str, str]]) -> list[str]:
    columns = [
        "Model", "ModelKey", "Status", "MeasuredTop1(%)", "PublishedTop1(%)",
        "Latency(ms)", "AvgPower(W)", "EnergyPerClip(J)", "TotalEnergy(J)",
        "PeakVRAM(MiB)", "NS-E", "NS-M", "NS#", "Checkpoint", "SHA256",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "").replace("|", "\\|") for column in columns) + " |")
    return lines


def main() -> None:
    historical_k400 = {row.get("ModelKey", ""): row for row in read_rows(HISTORICAL_K400)}
    historical_ssv2 = {row.get("ModelKey", ""): row for row in read_rows(HISTORICAL_SSV2)}
    new_k400 = {row.get("ModelKey", ""): row for row in read_rows(NEW_K400)}

    k400 = []
    if "mvit-v1-b" in historical_k400:
        k400.append(measured_row(historical_k400["mvit-v1-b"]))
    if "dualformer-s" in new_k400:
        k400.append(measured_row(new_k400["dualformer-s"]))
    k400.append(blocked("k400", "zeroi2v-b16-8f", "ZeroI2V ViT-B/16, 8f", "BLOCKED_LOADER", str(ROOT / "checkpoints" / "ZeroI2V" / "CLIP-B_k400_8x16x1.pth")))
    k400.append(blocked("k400", "dualformer-b-in21k", "DualFormer-B (IN21K)", "BLOCKED_PROVENANCE", str(ROOT / "checkpoints" / "dualformer" / "dualformer_base_patch244_window877.pth")))

    ssv2 = []
    for key in ("uniformer-s", "uniformer-b", "videomae-b", "videoswin-b", "timesformer-b", "video-focalnet-b"):
        if key in historical_ssv2:
            ssv2.append(measured_row(historical_ssv2[key]))
    blocked_ssv2 = {
        "dualformer-t": ("DualFormer-T", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "video-focalnet-t": ("Video-FocalNet-T", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "videoswin-t": ("VideoSwin-T", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "mvit-v1-b-16x4": ("MViT-B, 16x4", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "video-focalnet-s": ("Video-FocalNet-S", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "mvit-v1-b-32x3": ("MViT-B, 32x3", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "dualformer-s": ("DualFormer-S", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "videoswin-s": ("VideoSwin-S", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "zeroi2v-b16-8f": ("ZeroI2V ViT-B/16, 8f", "BLOCKED_LOADER"),
        "dualformer-b-in21k": ("DualFormer-B (IN21K)", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
        "omnivore-b-in21k": ("Omnivore-B (IN21K)", "NO_RELEASED_EXACT_SSV2_CHECKPOINT"),
    }
    for key, (model, status) in blocked_ssv2.items():
        ssv2.append(blocked("ssv2", key, model, status))

    all_rows = k400 + ssv2
    MASTER.parent.mkdir(exist_ok=True)
    with MASTER.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    lines = [
        "# One-Hour Benchmark Completion",
        "",
        "Generated from the historical RTX 3050 CSVs and the new `missing_k400_1h` CSV; blocked rows contain no invented measurements.",
        "",
        "## Environment and Dataset Checks",
        "",
        "- Device: NVIDIA GeForce RTX 3050 Laptop GPU; PyTorch 2.6.0+cu124; CUDA 12.4; driver 595.97; 4,095.5 MiB VRAM; compute capability 8.6.",
        "- K400 manifest: SHA256 `b8ec0d92f50b4ae5dff46754f62c2ef1b2b8467332f88bb0cc29c8e6172ff749`; 1,000 rows; 400 canonical class directories (the extra directory is the ignored clip cache).",
        "- SSV2 manifest: SHA256 `c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02`; 1,000 rows; existing validation annotations were used for the historical six rows.",
        "",
        "## K400",
        "",
        *render_table(k400),
        "",
        "New K400 rows completed: 1/4 requested priority rows. `mvit-v1-b` already existed; DualFormer-S is newly completed; ZeroI2V and DualFormer-B-IN21K remain blocked.",
        "",
        "## SSV2",
        "",
        *render_table(ssv2),
        "",
        "New SSV2 rows completed: 0/11. The six existing measured rows were preserved and not rerun.",
        "",
        "## Explicit Blockers",
        "",
        "- `NO_RELEASED_EXACT_SSV2_CHECKPOINT`: exact Tiny/Small variants for DualFormer, Video-FocalNet, and VideoSwin are not in the official model tables; MViT v1 SSV2 classifier releases were not found; Omnivore has no exact frozen SSV2 classifier release; DualFormer-B-IN21K is not the published K400 Base checkpoint.",
        "- `BLOCKED_LOADER`: ZeroI2V has a local K400 checkpoint, but its exact adapter requires the official ZeroI2V MMAction2 plugin/runtime, which is not available in this Windows environment. The official project documents SSV2 ViT-L/14 32-frame, not the frozen ViT-B/16 8-frame key.",
        "- Existing SSV2 rows are released external baselines, not controlled fine-tuned final checkpoints.",
        "",
        "The merged machine-readable table is `results/master_benchmark_results.csv`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT}")
    print(f"Wrote {MASTER}")


if __name__ == "__main__":
    main()