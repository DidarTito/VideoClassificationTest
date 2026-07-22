"""Stage 2.2/2.3 - Hardware measurement and deployment NetScore ranking.

Measures per-inference latency, average power and energy for each model on
the current device, then computes the paper's 1/8-weight NS-E, NS-M, and NS#.

Usage:
  # single test clip, 10 timed runs per model
  python run_benchmark.py --video videos/test.mp4 --models videoswin-t mvit-v2-s

  # 1000-clip Kinetics-400 subset (seeded random sample, cached after first decode)
  python run_benchmark.py --videos-dir datasets/kinetics400 --max-clips 1000 --device-name RTX3050
"""
import argparse
import math
import statistics
import time
from pathlib import Path

import pandas as pd
import torch

from data import (
    clip_to_input,
    load_annotation_ids,
    load_clip,
    load_clip_set,
    load_ground_truth,
)
from metrics import netscore, netscore_e, netscore_hash, netscore_m
from models import MODEL_REGISTRY
from power import make_power_logger


INPUT_PROTOCOL = "single uniformly sampled full-video view"


def _validate_logits(output, num_classes=400):
    """Require one finite dataset-sized logit vector per inference."""
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"model output must be a torch.Tensor, got {type(output).__name__}"
        )
    if tuple(output.shape) != (1, num_classes):
        raise ValueError(
            f"model output must have shape (1, {num_classes}), got {tuple(output.shape)}"
        )
    if not torch.isfinite(output).all().item():
        raise ValueError("model output contains non-finite logits")
    return output


def _total_energy_joules(energy_mj_per_clip, clip_count):
    """Convert per-clip millijoules to joules for the complete timed set."""
    return energy_mj_per_clip * clip_count / 1_000.0


def _clip_chunks(clips, chunk_size):
    if hasattr(clips, "iter_chunks"):
        yield from clips.iter_chunks(chunk_size)
    else:
        for start in range(0, len(clips), chunk_size):
            yield clips[start : start + chunk_size]


def _model_cuda_device(model):
    device = getattr(model, "device", None)
    if device is None:
        return torch.device("cuda") if torch.cuda.is_available() else None
    device = torch.device(device)
    return device if device.type == "cuda" else None


def benchmark(model, clips, power_logger, warmup=2, chunk_size=16):
    """One timed inference per clip; power sampled across the whole timed
    window. clips are (T, C, H, W) uint8; the float conversion happens one
    clip at a time (streaming - never materialize the whole float set) and
    outside the timed section. Predictions (argmax) are recorded after the
    timer stops, so accuracy comes for free without affecting latency."""
    if not clips:
        raise ValueError("cannot benchmark an empty clip set")

    times, preds, power_samples = [], [], []
    cuda_device = _model_cuda_device(model)
    with torch.inference_mode():
        warmup_input = clip_to_input(clips[0])
        for _ in range(warmup):
            _validate_logits(
                model(warmup_input), getattr(model, "num_classes", 400)
            )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)

        # A sharded clip set is loaded one bounded-memory chunk at a time.
        # Host uint8->float conversion happens before power sampling and timing.
        for raw_chunk in _clip_chunks(clips, chunk_size):
            prepared_chunk = [clip_to_input(clip_u8) for clip_u8 in raw_chunk]
            logger_started = False
            inference_failed = False
            chunk_samples = []
            try:
                if power_logger:
                    logger_started = True
                    power_logger.start()
                    time.sleep(0.1)
                    reset_samples = getattr(power_logger, "reset_samples", None)
                    if reset_samples is not None:
                        reset_samples()

                for clip in prepared_chunk:
                    start = time.perf_counter()
                    out = model(clip)
                    if cuda_device is not None:
                        torch.cuda.synchronize(cuda_device)
                    times.append((time.perf_counter() - start) * 1000)
                    out = _validate_logits(out, getattr(model, "num_classes", 400))
                    preds.append(int(out.argmax(-1).item()))
            except BaseException:
                inference_failed = True
                raise
            finally:
                if power_logger and logger_started:
                    try:
                        chunk_samples = power_logger.stop() or []
                    except Exception as stop_error:
                        if not inference_failed:
                            raise
                        print(f"  WARNING: power logger cleanup failed: {stop_error}")
                power_samples.extend(chunk_samples)

    mean_power = statistics.mean(power_samples) if power_samples else float("nan")
    latency = statistics.mean(times)
    latency_std = statistics.stdev(times) if len(times) > 1 else 0.0
    return latency, latency_std, mean_power, len(power_samples), preds


def _norm(label):
    return label.strip().lower().replace("_", " ").replace("'", "")


def measured_accuracy(model, preds, labels):
    """Top-1 over the clip set: prediction index -> the model's own class
    names -> compared with the ground-truth folder names."""
    if not labels:
        return None
    if all(isinstance(label, int) for label in labels):
        return 100.0 * sum(pred == truth for pred, truth in zip(preds, labels)) / len(labels)
    try:
        names = [_norm(c) for c in model.class_names]
    except AttributeError:
        return None
    truth = [_norm(l) for l in labels]
    correct = sum(1 for p, t in zip(preds, truth) if p < len(names) and names[p] == t)
    return 100.0 * correct / len(truth)


def _metric_or_nan(function, *args):
    try:
        return function(*args)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def benchmark_models(model_keys, clips, power_kind="auto", device_name=None, labels=None,
                     dataset="k400", device="auto", split_name=None,
                     annotation_file=None):
    """Benchmarks each registered model over the clip set. A model that fails
    to load or run is logged and skipped. If `labels` (ground-truth class name
    per clip) is given, measured top-1 accuracy is computed and the deployment
    NetScore variants use it; otherwise published accuracy is used.
    Returns (results DataFrame, errors dict)."""
    requested_device = (
        "cuda" if device == "auto" and torch.cuda.is_available()
        else "cpu" if device == "auto"
        else device
    )
    torch_device = torch.device(requested_device)
    cuda_index = None
    if torch_device.type == "cuda":
        cuda_index = torch_device.index
        if cuda_index is None:
            cuda_index = torch.cuda.current_device()
    device_name = device_name or (
        torch.cuda.get_device_name(cuda_index) if cuda_index is not None else "CPU")
    power_logger = make_power_logger(power_kind, device_index=cuda_index or 0)
    if power_logger is None:
        print("WARNING: no power tool available; energy will be NaN.")

    results, errors = [], {}
    for name in model_keys:
        print("=" * 70)
        print(f"{name}  ({len(clips)} timed inferences)")
        model = None
        try:
            if cuda_index is not None:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch_device)
            model = MODEL_REGISTRY[name](dataset=dataset, device=requested_device)
            latency, latency_std, mean_power, n_samples, preds = benchmark(
                model, clips, power_logger)
            # Metadata extraction and row construction belong to the same error
            # boundary as loading/inference: malformed adapters are skipped too.
            published_accuracy = model.accuracy
            parameters = model.parameters
            gflops = model.gflops
            model_name = model.name
            energy_mj = mean_power * latency  # W * ms = mJ
            acc_measured = measured_accuracy(model, preds, labels)
            metric_accuracy = (
                acc_measured if acc_measured is not None else published_accuracy
            )
            accuracy_source = "measured" if acc_measured is not None else "published"
            total_time_s = latency * len(clips) / 1000.0
            peak_vram_mib = (
                torch.cuda.max_memory_allocated(torch_device) / 1024 / 1024
                if cuda_index is not None else 0.0
            )
            ns_e = _metric_or_nan(
                netscore_e, metric_accuracy, total_time_s, mean_power
            )
            ns_m = _metric_or_nan(netscore_m, metric_accuracy, peak_vram_mib)
            ns_hash = _metric_or_nan(
                netscore_hash,
                metric_accuracy,
                peak_vram_mib,
                total_time_s,
                mean_power,
            )
            row = {
                "Device": device_name,
                "Dataset": dataset,
                "Split": split_name or "unspecified",
                "AnnotationFile": str(annotation_file or ""),
                "InputProtocol": INPUT_PROTOCOL,
                "ClipCount": len(clips),
                "FramesUsed": int(getattr(model, "frames", clips[0].shape[0])),
                "ModelKey": name,
                "Model": model_name,
                "Top1Published(%)": published_accuracy,
                "Top1Measured(%)": round(acc_measured, 2) if acc_measured is not None else float("nan"),
                "MetricAccuracy(%)": round(metric_accuracy, 2),
                "MetricAccuracySource": accuracy_source,
                "Params(M)": parameters / 1e6,
                "GFLOPs": gflops,
                "NetScore": round(netscore(published_accuracy, parameters / 1e6, gflops), 2),
                "NS-E(1/8)": round(ns_e, 2),
                "NS-M(1/8)": round(ns_m, 2),
                "NS#(1/8)": round(ns_hash, 2),
                "Latency(ms)": round(latency, 1),
                "LatencyStd(ms)": round(latency_std, 1),
                "FPS": round(1000 / latency, 2),
                "AvgPower(W)": round(mean_power, 2),
                "AvgPower(mW)": round(mean_power * 1000, 0),
                "PowerBackend": getattr(power_logger, "backend", "none") if power_logger else "none",
                "PowerSamples": n_samples,
                "TotalTime(s)": round(total_time_s, 3),
                "Energy(J)": round(energy_mj / 1000, 3),
                # mean_power [W] * latency [ms] is millijoules per clip.
                "TotalEnergy(J)": round(_total_energy_joules(energy_mj, len(clips)), 2),
                "PeakVRAM(MiB)": round(peak_vram_mib, 0),
            }
            results.append(row)
            acc_str = f"{acc_measured:.2f}%" if acc_measured is not None else "n/a"
            print(f"  latency {latency:.1f} +/- {latency_std:.1f} ms | "
                  f"power {mean_power:.1f} W ({n_samples} samples) | "
                  f"energy {energy_mj / 1000:.3f} J | measured top-1 {acc_str} | "
                  f"NS-E {row['NS-E(1/8)']} | NS-M {row['NS-M(1/8)']} | "
                  f"NS# {row['NS#(1/8)']}")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            errors[name] = f"{type(e).__name__}: {e}"
        finally:
            if model is not None:
                del model
            if cuda_index is not None:
                torch.cuda.reset_peak_memory_stats(torch_device)
                torch.cuda.empty_cache()
    return pd.DataFrame(results), errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="single video file")
    parser.add_argument("--videos-dir", "--video-dir", dest="videos_dir",
                        help="directory with video clips")
    parser.add_argument("--max-clips", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=1,
                        help="timed passes over the clip set (directory mode)")
    parser.add_argument("--seed", type=int, default=0, help="clip sampling seed")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="decode clips in memory without reading or writing a dataset cache",
    )
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--allow-skips", action="store_true",
                        help=("return success when some explicitly requested models fail; "
                              "at least one model must still produce results"))
    parser.add_argument("--dataset", default="k400", choices=["k400", "ssv2"],
                        help="checkpoint/dataset family (SSV2 has 174 classes)")
    parser.add_argument("--annotations",
                        help="SSV2 split JSON (standard flat-video layout)")
    parser.add_argument("--split-name",
                        help="honest split label recorded in every result row")
    parser.add_argument("--device", default="auto",
                        help="Torch device: auto, cpu, cuda, cuda:1, mps, ...")
    parser.add_argument("--runs", type=int, default=10,
                        help="timed runs per model (single-video mode)")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--power", default="auto",
                        choices=["auto", "nvml", "nvidia-smi", "tegrastats", "none"])
    parser.add_argument("--device-name", default=None,
                        help="label for the CSV (e.g. RTX3050, TX2)")
    parser.add_argument("--output", default="benchmark_results.csv")
    args = parser.parse_args()

    if not args.video and not args.videos_dir:
        parser.error("provide --video or --videos-dir")
    if args.dataset == "ssv2" and args.videos_dir and not args.annotations:
        parser.error(
            "--annotations is required for split-safe SSV2 directory sampling"
        )

    explicit_models = args.models is not None
    model_keys = args.models if explicit_models else list(MODEL_REGISTRY.keys())

    labels = None
    if args.videos_dir:
        allowed_stems = (
            load_annotation_ids(args.annotations)
            if args.dataset == "ssv2" and args.annotations else None
        )
        clips, paths = load_clip_set(
            args.videos_dir,
            args.max_clips,
            args.frames,
            args.size,
            args.seed,
            use_cache=not args.no_cache,
            allowed_stems=allowed_stems,
        )
        labels = load_ground_truth(paths, args.dataset, args.annotations)
        clips = clips * args.repeats
        labels = labels * args.repeats
        print(
            "NOTE: measured top-1 uses a common single uniformly sampled view for "
            "hardware comparison; it is not each paper's canonical multi-view protocol."
        )
    else:
        clips = [load_clip(args.video, args.frames, args.size)] * args.runs

    df, errors = benchmark_models(model_keys, clips, args.power, args.device_name, labels,
                                  dataset=args.dataset, device=args.device,
                                  split_name=args.split_name,
                                  annotation_file=args.annotations)
    if errors:
        print("Skipped models:", errors)
    if df.empty:
        raise SystemExit("No model produced results.")

    pd.set_option("display.width", 200)
    print("=" * 70)
    print(df.to_string(index=False))
    df.to_csv(args.output, index=False)
    print(f"Saved: {args.output}")

    device_name = df["Device"].iloc[0].replace(" ", "_")
    ranking_names = {
        "NS-E(1/8)": "ranking_ns_e.csv",
        "NS-M(1/8)": "ranking_ns_m.csv",
        "NS#(1/8)": "ranking_ns_hash.csv",
    }
    for metric, out in ranking_names.items():
        ranked = df.sort_values(metric, ascending=False)[
            ["Model", "Top1Published(%)", "Top1Measured(%)", "Latency(ms)",
             "AvgPower(W)", "Energy(J)", "PeakVRAM(MiB)", metric]
        ]
        ranked.to_csv(out, index=False)
        print(f"Saved: {out}")

    estimated_all_s = float(df["TotalTime(s)"].sum())
    print(
        f"Estimated inference-only time for all {len(df)} candidates: "
        f"{estimated_all_s:.1f} s ({estimated_all_s / 60:.2f} min)"
    )

    if explicit_models and errors and not args.allow_skips:
        failed = ", ".join(errors)
        raise SystemExit(
            f"{len(errors)} explicitly requested model(s) failed: {failed}. "
            "Re-run with --allow-skips to accept partial results."
        )


if __name__ == "__main__":
    main()
