#!/usr/bin/env python3
"""Device benchmark of frozen-K400 + trained 174-class SSV2 linear heads.

Does not train. Timing, NVML power, energy, VRAM, and fvcore GFLOPs follow
run_benchmark.py. Classic NetScore uses linear-probe Top-1, runtime params, and
profiled GFLOPs: 20*log10(A^2 / (sqrt(Params_M)*sqrt(GFLOPs))).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_benchmark
from data import clip_to_input, load_clip
from metrics import netscore, netscore_e, netscore_hash, netscore_m
from models import MODEL_REGISTRY
from power import make_power_logger
from scripts.run_ssv2_head_transfer_all import EVAL_SHA, classifier_linear, sha256_file
from training.checkpoint_utils import resolve_module
from training.registry import model_spec
from utils.device import resolve_device

KEYS = [
    "videoswin-t", "videoswin-s", "video-focalnet-t", "video-focalnet-s",
    "dualformer-t", "dualformer-s", "dualformer-b-in21k",
    "mvit-v1-b-16x4", "mvit-v1-b-32x3", "omnivore-b-in21k", "zeroi2v-b16-8f",
]
EXPECTED_DIM = {
    "videoswin-t": 768, "videoswin-s": 768, "video-focalnet-t": 768,
    "video-focalnet-s": 768, "dualformer-t": 512, "dualformer-s": 768,
    "dualformer-b-in21k": 1024, "mvit-v1-b-16x4": 768, "mvit-v1-b-32x3": 768,
    "omnivore-b-in21k": 1024, "zeroi2v-b16-8f": 768,
}
EXPECTED_CORRECT = {
    "videoswin-t": 106, "videoswin-s": 111, "video-focalnet-t": 95,
    "video-focalnet-s": 111, "dualformer-t": 112, "dualformer-s": 133,
    "dualformer-b-in21k": 121, "mvit-v1-b-16x4": 92, "mvit-v1-b-32x3": 105,
    "omnivore-b-in21k": 147, "zeroi2v-b16-8f": 131,
}
HEAD_ROOT = ROOT / "results/ssv2_head_transfer"
EVAL_PROBE = HEAD_ROOT / "manifests/eval_1000.csv"
OFFICIAL = ROOT / "manifests/ssv2_1000_seed0.csv"
VIDEOS = ROOT / "datasets/ssv2/videos"
OUT = ROOT / "results/ssv2_device_benchmark"
TABLE = ROOT / "results/SSV2_LINEAR_PROBE_DEVICE_BENCHMARK.csv"
COMBINED = ROOT / "results/SSV2_COMBINED_NATIVE_AND_LINEAR_PROBE_DEVICE.csv"
SIX = [
    dict(Model="uniformer-s", Paper=67.70, Measured=65.1, GFLOPs=125, NetScore=38.97,
         Latency=78.8, Power=52.95, Energy=4.173, Total=4173.25, VRAM=215, Protocol="NATIVE_SSV2"),
    dict(Model="uniformer-b", Paper=70.40, Measured=67.1, GFLOPs=290, NetScore=32.32,
         Latency=163.1, Power=52.51, Energy=8.563, Total=8563.34, VRAM=323, Protocol="NATIVE_SSV2"),
    dict(Model="videomae-b", Paper=70.80, Measured=68.6, GFLOPs=180, NetScore=32.09,
         Latency=190.7, Power=56.76, Energy=10.826, Total=10825.76, VRAM=809, Protocol="NATIVE_SSV2"),
    dict(Model="videoswin-b", Paper=69.60, Measured=69.5, GFLOPs=320.6, NetScore=29.16,
         Latency=601.1, Power=58.98, Energy=35.452, Total=35452.26, VRAM=2156, Protocol="NATIVE_SSV2"),
    dict(Model="timesformer-b", Paper=59.50, Measured=57.6, GFLOPs=196, NetScore=27.10,
         Latency=167.1, Power=55.36, Energy=9.251, Total=9250.83, VRAM=750, Protocol="NATIVE_SSV2"),
    dict(Model="video-focalnet-b", Paper=71.10, Measured=61.1, GFLOPs=149, NetScore=30.38,
         Latency=228.1, Power=57.44, Energy=13.103, Total=13102.67, VRAM=1461, Protocol="NATIVE_SSV2"),
]


def atomic_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import io
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(stream.getvalue(), encoding="utf-8", newline="\n")
    temp.replace(path)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def percentile(values, p):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def install_trained_head(raw: nn.Module, path: str, head_path: Path, expected_dim: int) -> nn.Linear:
    old = classifier_linear(raw, path)
    payload = torch.load(head_path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    weight = state["weight"]
    if tuple(weight.shape) != (174, expected_dim):
        raise RuntimeError(f"{head_path}: weight {tuple(weight.shape)} != (174, {expected_dim})")
    new = nn.Linear(expected_dim, 174, bias="bias" in state)
    new.load_state_dict(state)
    new = new.to(device=old.weight.device, dtype=old.weight.dtype)
    parent, name = resolve_module(raw, path)
    if name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(name)] = new
    elif isinstance(parent, (dict, nn.ModuleDict)):
        parent[name] = new
    else:
        setattr(parent, name, new)
    return new


class LinearProbeAdapter:
    num_classes = 174
    accuracy = float("nan")

    def __init__(self, key: str, device: str):
        spec = model_spec(key)
        self.spec = spec
        self.key = key
        self.owner = MODEL_REGISTRY[key](device=device, dataset="k400")
        self.owner.model.eval()
        for parameter in self.owner.model.parameters():
            parameter.requires_grad = False
        head_path = HEAD_ROOT / key / "head_best.pt"
        self.head = install_trained_head(
            self.owner.model, spec["classifier"]["path"], head_path, EXPECTED_DIM[key]
        )
        self.head.eval()
        self.model = self.owner.model
        self.device = torch.device(device)
        self.frames = int(spec["input"]["num_frames"])
        self.name = spec["display_name"] + " SSV2 linear probe"
        self.info = {"checkpoint": str(head_path), "frames": self.frames, "k400": spec["k400_checkpoint"]}
        self.gflops = float("nan")

    @property
    def parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, clip):
        cls = type(self.owner)
        saved = None
        owner = base = None
        for candidate in cls.__mro__:
            if "num_classes" in candidate.__dict__:
                saved = candidate.__dict__["num_classes"]
                base = candidate
                break
        if base is None:
            raise RuntimeError(f"{cls.__name__} has no num_classes")
        setattr(base, "num_classes", property(lambda _self: 174))
        try:
            return self.owner(clip)
        finally:
            setattr(base, "num_classes", saved)


def verify_cached_head(key: str) -> None:
    cache = HEAD_ROOT / key / "features_eval.pt"
    head_path = HEAD_ROOT / key / "head_best.pt"
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    ckpt = torch.load(head_path, map_location="cpu", weights_only=True)
    head = nn.Linear(EXPECTED_DIM[key], 174)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()
    with torch.inference_mode():
        logits = head(payload["features"].float())
    pred = logits.argmax(-1)
    labels = payload["labels"].long()
    correct = int((pred == labels).sum())
    expected = EXPECTED_CORRECT[key]
    if correct != expected or len(labels) != 1000:
        raise RuntimeError(f"{key}: cached-head Top-1 {correct}/1000 != {expected}/1000")
    print(f"{key}: cached-head verified {correct}/1000", flush=True)


def assert_eval_identity() -> tuple[list[str], list[int]]:
    if sha256_file(OFFICIAL) != EVAL_SHA:
        raise RuntimeError("official ssv2_1000_seed0.csv checksum mismatch; manifest was not modified")
    probe = list(csv.DictReader(EVAL_PROBE.open(encoding="utf-8", newline="")))
    official = list(csv.DictReader(OFFICIAL.open(encoding="utf-8", newline="")))
    if len(probe) != 1000 or len(official) != 1000:
        raise RuntimeError("eval manifests are not 1000 rows")
    probe_ids = [r["relative_path"] for r in probe]
    official_ids = [r["RelativePath"] for r in official]
    if probe_ids != official_ids:
        raise RuntimeError("linear-probe eval_1000.csv IDs differ from official ssv2_1000_seed0.csv")
    return [r["video_id"] for r in probe], [int(r["label"]) for r in probe]


def load_eval_clips():
    ids, labels = assert_eval_identity()
    print("decoding 1000 SSV2 eval clips with load_clip(32f, 224) matching head-transfer", flush=True)
    clips = [load_clip(VIDEOS / f"{video_id}.webm", frames=32, size=224) for video_id in ids]
    if len(clips) != 1000:
        raise RuntimeError(f"expected 1000 clips, got {len(clips)}")
    return clips, ids, labels


def write_tables(summaries: dict[str, dict]) -> None:
    rows = []
    for key in KEYS:
        if key not in summaries or summaries[key].get("status") != "COMPLETE":
            continue
        s = summaries[key]
        rows.append({
            "Overall Rank": 0,
            "Model": key,
            "Top-1 Paper": model_spec(key)["stage1"]["paper_top1"],
            "K400→SSV2 Linear-Probe Top-1 Measured (%)": s["top1"],
            "GFLOPs": f"{s['gflops']:.3f}",
            "NetScore": f"{s['netscore']:.2f}",
            "Latency (ms)": f"{s['mean_latency_ms']:.1f}",
            "Average Power (W)": f"{s['average_power_w']:.2f}",
            "Energy per Clip (J)": f"{s['energy_per_clip_j']:.3f}",
            "Total Energy for 1000 Clips (J)": f"{s['total_energy_1000_j']:.2f}",
            "Peak GPU Memory (MiB)": f"{s['peak_gpu_memory_mib']:.1f}",
        })
    rows.sort(key=lambda r: float(r["NetScore"]), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["Overall Rank"] = i
    fields = ["Overall Rank", "Model", "Top-1 Paper",
              "K400→SSV2 Linear-Probe Top-1 Measured (%)", "GFLOPs", "NetScore",
              "Latency (ms)", "Average Power (W)", "Energy per Clip (J)",
              "Total Energy for 1000 Clips (J)", "Peak GPU Memory (MiB)"]
    if rows:
        atomic_csv(TABLE, rows, fields)
    combined = []
    for item in SIX:
        combined.append({
            "Overall Rank": 0, "Model": item["Model"], "Protocol": item["Protocol"],
            "Top-1 Paper": item["Paper"], "Native SSV2 Top-1": item["Measured"],
            "K400→SSV2 Linear-Probe Top-1": "", "GFLOPs": item["GFLOPs"],
            "NetScore": item["NetScore"], "Latency (ms)": item["Latency"],
            "Average Power (W)": item["Power"], "Energy per Clip (J)": item["Energy"],
            "Total Energy for 1000 Clips (J)": item["Total"],
            "Peak GPU Memory (MiB)": item["VRAM"],
        })
    for row in rows:
        combined.append({
            "Overall Rank": 0, "Model": row["Model"],
            "Protocol": "K400_SSV2_LINEAR_PROBE",
            "Top-1 Paper": row["Top-1 Paper"], "Native SSV2 Top-1": "",
            "K400→SSV2 Linear-Probe Top-1": row["K400→SSV2 Linear-Probe Top-1 Measured (%)"],
            "GFLOPs": row["GFLOPs"], "NetScore": row["NetScore"],
            "Latency (ms)": row["Latency (ms)"], "Average Power (W)": row["Average Power (W)"],
            "Energy per Clip (J)": row["Energy per Clip (J)"],
            "Total Energy for 1000 Clips (J)": row["Total Energy for 1000 Clips (J)"],
            "Peak GPU Memory (MiB)": row["Peak GPU Memory (MiB)"],
        })
    combined.sort(key=lambda r: float(r["NetScore"]), reverse=True)
    for i, row in enumerate(combined, start=1):
        row["Overall Rank"] = i
    atomic_csv(COMBINED, combined, list(combined[0]) if combined else [])


def timed_collect(key, clips, ids, labels, device_info):
    """Single timed 1000-clip pass using the project benchmark() implementation."""
    verify_cached_head(key)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_info.torch_device)
    torch.cuda.synchronize(device_info.torch_device)
    adapter = LinearProbeAdapter(key, str(device_info.torch_device))
    gflops, flop_notes = run_benchmark._profile_device_gflops(adapter, clips[0])
    print(f"  GFLOPs/view: {gflops:.3f}", flush=True)
    torch.cuda.synchronize(device_info.torch_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_info.torch_device)
    power_logger = make_power_logger("nvml", interval_ms=20, device_index=device_info.nvml_index)
    times, preds, top5, chunk_power, all_power = [], [], [], [], []
    cuda_device = device_info.torch_device
    with torch.inference_mode():
        warmup_input = clip_to_input(clips[0])
        for _ in range(20):
            run_benchmark._validate_logits(adapter(warmup_input), 174)
        torch.cuda.synchronize(cuda_device)
        torch.cuda.reset_peak_memory_stats(cuda_device)
        for raw_chunk in run_benchmark._progress_clip_chunks(clips, 16):
            prepared = [clip_to_input(clip) for clip in raw_chunk]
            power_logger.start()
            time.sleep(0.1)
            reset = getattr(power_logger, "reset_samples", None)
            if reset is not None:
                reset()
            try:
                for clip in prepared:
                    torch.cuda.synchronize(cuda_device)
                    t0 = time.perf_counter()
                    out = adapter(clip)
                    torch.cuda.synchronize(cuda_device)
                    times.append((time.perf_counter() - t0) * 1000.0)
                    out = run_benchmark._validate_logits(out, 174)
                    preds.append(int(out.argmax(-1).item()))
                    top5.append(out.topk(5, dim=-1).indices[0].tolist())
            finally:
                samples = power_logger.stop() or []
            all_power.extend(samples)
            mean_chunk = statistics.mean(samples) if samples else float("nan")
            chunk_power.extend([mean_chunk] * len(prepared))
    correct = int(sum(p == y for p, y in zip(preds, labels)))
    if correct != EXPECTED_CORRECT[key]:
        raise RuntimeError(
            f"{key}: video Top-1 {correct}/1000 != saved linear-probe {EXPECTED_CORRECT[key]}/1000"
        )
    top5_n = int(sum(y in t for y, t in zip(labels, top5)))
    latency = statistics.mean(times)
    mean_power = statistics.mean(all_power) if all_power else float("nan")
    peak = torch.cuda.max_memory_allocated(cuda_device) / 1024.0 / 1024.0
    params_m = adapter.parameters / 1e6
    energy = mean_power * latency / 1000.0
    total_energy = energy * 1000.0
    total_time = latency * len(clips) / 1000.0
    top1 = 100.0 * correct / 1000.0
    per_clip = []
    for i, video_id in enumerate(ids):
        e = chunk_power[i] * times[i] / 1000.0 if math.isfinite(chunk_power[i]) else energy
        per_clip.append(dict(
            video_id=video_id, correct=int(preds[i] == labels[i]),
            latency_ms=times[i], power_w=chunk_power[i], energy_j=e,
            peak_memory_mib=peak,
        ))
    summary = dict(
        model=key, top1=top1, top5=100.0 * top5_n / 1000.0, gflops=float(gflops),
        netscore=netscore(top1, params_m, gflops),
        netscore_formula="20*log10(A^2 / (sqrt(Params_M)*sqrt(GFLOPs))); A=K400→SSV2 linear-probe Top-1; GFLOPs=fvcore of backbone+174 head",
        ns_e=netscore_e(top1, total_time, mean_power),
        ns_m=netscore_m(top1, peak),
        ns_hash=netscore_hash(top1, peak, total_time, mean_power),
        mean_latency_ms=latency, median_latency_ms=statistics.median(times),
        p50_latency_ms=percentile(times, 50), p95_latency_ms=percentile(times, 95),
        p99_latency_ms=percentile(times, 99), latency_std_ms=statistics.stdev(times),
        average_power_w=mean_power, energy_per_clip_j=energy,
        total_energy_1000_j=total_energy, peak_gpu_memory_mib=peak,
        eval_clips=1000, correct_top1=correct, params_m=params_m,
        flop_notes=flop_notes, power_samples=len(all_power),
        status="COMPLETE", protocol="K400_SSV2_LINEAR_PROBE",
        warmup=20, paper_top1=model_spec(key)["stage1"]["paper_top1"],
    )
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
    return summary, per_clip


def run_model(key: str, clips, ids, labels, device_info) -> dict:
    out_dir = OUT / key
    summary_path = out_dir / "summary.json"
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("status") == "COMPLETE" and existing.get("correct_top1") == EXPECTED_CORRECT[key]:
            print(f"skip completed {key}", flush=True)
            return existing
    print("=" * 70, flush=True)
    print(f"{key} linear-probe device benchmark", flush=True)
    summary, per_clip = timed_collect(key, clips, ids, labels, device_info)
    atomic_csv(out_dir / "per_clip_metrics.csv", per_clip,
               ["video_id", "correct", "latency_ms", "power_w", "energy_j", "peak_memory_mib"])
    save_json(summary_path, summary)
    print(json.dumps({k: summary[k] for k in ("model", "top1", "gflops", "netscore", "mean_latency_ms",
                                              "average_power_w", "energy_per_clip_j", "peak_gpu_memory_mib")}),
          flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    keys = KEYS if args.models in {"all", "remaining11"} else [k.strip() for k in args.models.split(",") if k.strip()]
    unknown = [k for k in keys if k not in KEYS]
    if unknown:
        raise SystemExit(f"unknown keys: {unknown}")
    if len(keys) > 1:
        failed = []
        for key in keys:
            log = OUT / key / "run.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--models", key, "--resume"]
            print(f"==== {key} ====", flush=True)
            with log.open("a", encoding="utf-8") as stream:
                child = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace")
                for line in child.stdout:
                    stream.write(line)
                    stream.flush()
                    print(line, end="", flush=True)
                code = child.wait()
            if code:
                failed.append(key)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summaries = {}
        for key in KEYS:
            path = OUT / key / "summary.json"
            if path.is_file():
                summaries[key] = json.loads(path.read_text(encoding="utf-8"))
        write_tables(summaries)
        if failed:
            print(f"incomplete: {failed}", flush=True)
            return 1
        return 0
    run_benchmark._seed_everything(0)
    device_info = resolve_device("cuda:0", precision="fp32", validate=True)
    clips, ids, labels = load_eval_clips()
    summary = run_model(keys[0], clips, ids, labels, device_info)
    summaries = {}
    for key in KEYS:
        path = OUT / key / "summary.json"
        if path.is_file():
            summaries[key] = json.loads(path.read_text(encoding="utf-8"))
    summaries[keys[0]] = summary
    write_tables(summaries)
    return 0 if summary.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
