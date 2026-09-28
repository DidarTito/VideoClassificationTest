"""Pinned evidence and per-model provenance for the fresh SSV2 verification."""
import json
import math
from pathlib import Path
import subprocess

from training.frozen_preflight import sha256_file
from training.registry import ROOT

EXPECTED_MANIFEST = "c0b8d5fc57ced12f57fb50468e281cae9d98b5a16e15e79be37de8de2c0c8c02"
EVIDENCE = {
    "uniformer-s": ("4f5df1bf569e7e0ead9ae9ea43ecef5a852ef14dcf67575e36aa3ce8edd0ed9f", 67.7, None),
    "uniformer-b": ("b3091fbfd2b47845988831599a0325c426b6ec84aee6a545c152fafe84adeffe", 70.4, None),
    "videomae-b": ("25f4060e564ca716ff1f1804f5aba92f9240da723e22b383e306273ca91de377", 70.8, 92.4),
    "videoswin-b": ("f7fe417a0659680ab9db4f5e02c39b3f2825b58f0e244d4fbe3de6658fe135fc", 69.6, 92.7),
    "timesformer-b": ("7a606efdda9345f1e2d83e10cc2b03196edf157ff7e9a07a19368352614109c4", 59.1, 85.6),
    "video-focalnet-b": ("187e4e9469de75ae36252d07418d9802dde16fc3a7beb9815ec949eff25dd9d6", 71.1, None),
}

def verify_before_load(key, row, config):
    if key not in EVIDENCE:
        raise RuntimeError("Only the six pinned genuine SSV2 checkpoints may be accuracy-verified")
    required = dict(dataset="ssv2", num_clips=1000, batch_size=1, precision="fp32", seed=0,
                    warmup=2, repeats=1, resolution=224, decoded_frames=32, power_kind="nvml", power_sample_ms=20)
    for name, expected in required.items():
        if getattr(config, name) != expected:
            raise RuntimeError(f"Verification protocol mismatch: {name}")
    if Path(config.manifest).resolve() != (ROOT / "manifests/ssv2_1000_seed0.csv").resolve():
        raise RuntimeError("Wrong SSV2 manifest path")
    if sha256_file(str(Path(config.manifest).resolve())) != EXPECTED_MANIFEST:
        raise RuntimeError("Wrong SSV2 manifest SHA256")
    digest, top1, top5 = EVIDENCE[key]
    actual = sha256_file(str(Path(row["Checkpoint"]).resolve()))
    if actual != digest:
        raise RuntimeError(f"Wrong SSV2 checkpoint SHA256 for {key}: {actual}")
    row.update(CheckpointProtocol="GENUINE_SSV2", CheckpointSHA256=actual,
               PublishedTop1=top1, PublishedTop5=top5, AccuracyValid=True)

def write_sidecar(row, directory):
    aliases = {"LatencyMeanMs": "LatencyMean", "LatencyStdMs": "LatencyStd", "AveragePowerW": "AvgPower",
               "EnergyPerClipJ": "EnergyPerClip", "TotalEnergyJ": "TotalEnergy", "PeakVRAMMiB": "PeakVRAM",
               "GFLOPsPerView": "GFLOPs", "RuntimeParamsM": "RuntimeActualParamsM", "Driver": "DriverVersion",
               "CUDA": "CUDAVersion"}
    data = dict(row)
    data.update({new: row.get(old) for new, old in aliases.items()})
    data["GitHEAD"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    data["WorkingTreeDirty"] = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    for name, value in data.items():
        if isinstance(value, float) and not math.isfinite(value):
            data[name] = None
    target = Path(directory) / (row["ModelKey"] + ".json")
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
