"""Portable Stage 2 video-classification benchmark.

The scientific protocol remains the RTX 3050 protocol: batch size 1, one
uniformly sampled full-video view decoded to 32 frames at 224x224, exact
model/checkpoint pairs, model-specific temporal resampling, FP32 by default,
two untimed warm-ups, NVML power sampling, and peak allocated CUDA memory.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta
import gc
from importlib import import_module
import math
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Callable, Iterable

import pandas as pd
import torch

from data import (
    VIDEO_EXTS,
    SampleManifestError,
    cache_bytes_estimate,
    clip_to_input,
    load_annotation_ids,
    load_clip,
    load_clip_set,
    load_ground_truth,
    manifest_sha256,
    probe_readable,
    promote_cache_manifest,
    read_sample_manifest,
)
from metrics import netscore, netscore_e, netscore_hash, netscore_m
from models import (
    DATASET_MODEL_KEYS,
    FROZEN17_MODEL_KEYS,
    MODEL_REGISTRY,
    REQUESTED_MODEL_SPECS,
)
from models.common import set_checkpoint_root, set_dataset_root
from training.frozen_preflight import validate_frozen_model
from training.registry import model_spec as frozen_model_spec
from power import make_power_logger, probe_nvml
from utils.device import (
    CudaCompatibilityError,
    DeviceInfo,
    DeviceResolutionError,
    print_device_info,
    query_temperature,
    resolve_device,
    sanitize_device_name,
)


ROOT = Path(__file__).resolve().parent
INPUT_PROTOCOL = "single uniformly sampled full-video view"
DEFAULT_DECODED_FRAMES = 32
DEFAULT_RESOLUTION = 224
DEFAULT_PRECISION = "fp32"
DEFAULT_WARMUP = 2

# Useful results arrive early if a one-hour run is interrupted. Only exact
# dataset pairs survive the dataset availability filter below.
DEFAULT_RUN_ORDER = (
    "uniformer-s",
    "video-focalnet-t",
    "dualformer-t",
    "uniformer-b",
    "videomae-b",
    "timesformer-b",
    "videoswin-t",
    "mvit-v1-b",
    "mvit-b-24-32x3",
    "videoswin-s",
    "video-focalnet-b",
    "videoswin-b",
    "vtn-b",
    "video-focalnet-s",
    "omnivore-b",
)

CANONICAL_RESULT_COLUMNS = (
    "Device",
    "DeviceName",
    "DeviceIndex",
    "NVMLDeviceIndex",
    "GPUUUID",
    "CUDADeviceCount",
    "TotalVRAM",
    "ComputeCapability",
    "TorchVersion",
    "CUDAVersion",
    "DriverVersion",
    "Dataset",
    "DatasetRoot",
    "CheckpointRoot",
    "ModelKey",
    "Model",
    "Checkpoint",
    "InputProtocol",
    "Frames",
    "ModelFrames",
    "Resolution",
    "BatchSize",
    "Precision",
    "Seed",
    "ClipCount",
    "Manifest",
    "ManifestSHA256",
    "WarmupRuns",
    "PowerMode",
    "PowerSampleMs",
    "Repeats",
    "RunMode",
    "CooldownTemp",
    "CooldownTimeout",
    "Stage1PublishedTop1",
    "Stage1PublishedParamsM",
    "Stage1PublishedGFLOPs",
    "RuntimeActualParamsM",
    "RuntimeParamSource",
    "CheckpointPublishedTop1",
    "CheckpointPublishedParamsM",
    "CheckpointPublishedGFLOPs",
    "PublishedTop1",
    "MeasuredTop1",
    "Params",
    "ParamsM",
    "GFLOPs",
    "LatencyMean",
    "LatencyStd",
    "FPS",
    "AvgPower",
    "PowerSamples",
    "MeasurementBackend",
    "EnergyPerClip",
    "TotalTime",
    "TotalEnergy",
    "PeakVRAM",
    "PeakVRAMMethod",
    "GPUStartTemp",
    "GPUEndTemp",
    "NS-E",
    "NS-M",
    "NS#",
    "Status",
    "Error",
)

# Compatibility fields keep the existing reporting scripts usable. New code
# should prefer the unambiguous canonical names above.
LEGACY_RESULT_COLUMNS = (
    "Split",
    "AnnotationFile",
    "FramesUsed",
    "Top1Published(%)",
    "Top1Measured(%)",
    "MetricAccuracy(%)",
    "MetricAccuracySource",
    "Params(M)",
    "NetScore",
    "NS-E(1/8)",
    "NS-M(1/8)",
    "NS#(1/8)",
    "Latency(ms)",
    "LatencyStd(ms)",
    "AvgPower(W)",
    "PowerBackend",
    "TotalTime(s)",
    "Energy(J)",
    "TotalEnergy(J)",
    "PeakVRAM(MiB)",
)
RESULT_COLUMNS = CANONICAL_RESULT_COLUMNS + LEGACY_RESULT_COLUMNS

# A resumed file is one interrupted experiment, not a general results database.
# These fields cover the hardware/software environment, sample identity, model
# recipe, and all measurement settings that can change a reported value.
RESUME_IDENTITY_COLUMNS = (
    "Device",
    "DeviceName",
    "DeviceIndex",
    "NVMLDeviceIndex",
    "GPUUUID",
    "CUDADeviceCount",
    "TotalVRAM",
    "ComputeCapability",
    "TorchVersion",
    "CUDAVersion",
    "DriverVersion",
    "Dataset",
    "DatasetRoot",
    "CheckpointRoot",
    "Checkpoint",
    "InputProtocol",
    "Frames",
    "ModelFrames",
    "Resolution",
    "BatchSize",
    "Precision",
    "Seed",
    "ClipCount",
    "ManifestSHA256",
    "WarmupRuns",
    "PowerMode",
    "PowerSampleMs",
    "Repeats",
    "RunMode",
    "CooldownTemp",
    "CooldownTimeout",
    "Split",
    "AnnotationFile",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """One resolved experiment configuration shared by every model."""

    dataset: str
    dataset_root: Path
    checkpoint_dir: Path
    manifest: Path | None
    output_csv: Path
    num_clips: int
    seed: int
    batch_size: int
    precision: str
    decoded_frames: int
    resolution: int
    warmup: int
    power_kind: str
    power_sample_ms: int
    split_name: str | None
    annotations: Path | None
    use_cache: bool
    mode: str
    cooldown_temp: float | None
    cooldown_timeout: float
    repeats: int = 1


class ResultStore:
    """Atomic, model-granular CSV persistence with resume semantics."""

    def __init__(self, path, resume=False, force=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        if self.path.is_file() and not force:
            if not resume:
                raise FileExistsError(
                    f"Result CSV already exists: {self.path}. Use --resume to keep "
                    "successful rows or --force to replace it."
                )
            frame = pd.read_csv(self.path)
            self.rows = frame.to_dict(orient="records")

    @staticmethod
    def _is_success(row):
        status = row.get("Status")
        if status is None or (isinstance(status, float) and math.isnan(status)):
            # Pre-resume CSVs only contained completed models.
            return True
        return str(status).strip().lower() in {"completed", "success", "ok"}

    @property
    def successful_keys(self):
        return {
            str(row.get("ModelKey"))
            for row in self.rows
            if row.get("ModelKey") and self._is_success(row)
        }

    @staticmethod
    def _identity_equal(actual, expected):
        def missing(value):
            return value is None or (
                isinstance(value, float) and math.isnan(value)
            ) or (
                isinstance(value, str) and not value.strip()
            )

        if missing(actual) or missing(expected):
            return missing(actual) and missing(expected)
        # pandas may infer dotted numeric metadata such as driver ``576.80``
        # as the float 576.8.  Compare numeric-looking values numerically while
        # keeping paths, names, and semantic versions as exact strings.
        try:
            actual_number = float(actual)
            expected_number = float(expected)
        except (TypeError, ValueError):
            pass
        else:
            if math.isfinite(actual_number) and math.isfinite(expected_number):
                return math.isclose(
                    actual_number, expected_number, rel_tol=1e-9, abs_tol=1e-9
                )
        return str(actual) == str(expected)

    def validate_resume(self, expected_rows):
        """Reject a CSV whose rows do not belong to this exact experiment."""
        if not self.rows:
            return
        seen = set()
        problems = []
        for row in self.rows:
            raw_key = row.get("ModelKey")
            key = "" if raw_key is None else str(raw_key)
            if not key or key.lower() == "nan":
                problems.append("row without a ModelKey")
                continue
            if key in seen:
                problems.append(f"duplicate ModelKey {key!r}")
                continue
            seen.add(key)
            expected = expected_rows.get(key)
            if expected is None:
                problems.append(f"unexpected ModelKey {key!r}")
                continue
            for column in RESUME_IDENTITY_COLUMNS:
                if column not in row:
                    problems.append(f"{key}: missing {column}")
                    continue
                if not self._identity_equal(row.get(column), expected.get(column)):
                    problems.append(
                        f"{key}: {column}={row.get(column)!r}, expected "
                        f"{expected.get(column)!r}"
                    )
        if problems:
            preview = "; ".join(problems[:8])
            if len(problems) > 8:
                preview += f"; and {len(problems) - 8} more mismatch(es)"
            raise ValueError(
                "Cannot resume because the existing result CSV belongs to a "
                f"different or legacy run: {preview}. Use --force to start this "
                "configuration again, or choose another --output-dir."
            )

    def upsert(self, row):
        key = str(row["ModelKey"])
        self.rows = [item for item in self.rows if str(item.get("ModelKey")) != key]
        self.rows.append(dict(row))
        self.write()

    def frame(self):
        if not self.rows:
            return pd.DataFrame(columns=RESULT_COLUMNS)
        frame = pd.DataFrame(self.rows)
        for column in RESULT_COLUMNS:
            if column not in frame:
                frame[column] = float("nan")
        remaining = [column for column in frame.columns if column not in RESULT_COLUMNS]
        return frame[list(RESULT_COLUMNS) + remaining]

    def write(self):
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        self.frame().to_csv(temporary, index=False)
        temporary.replace(self.path)


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


def _precision_context(precision, device_type):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device_type, dtype=dtype)


def benchmark(
    model,
    clips,
    power_logger,
    warmup=DEFAULT_WARMUP,
    chunk_size=16,
    precision=DEFAULT_PRECISION,
):
    """Run the unchanged per-clip inference timing scope.

    Host uint8-to-float conversion remains outside the timed section. The
    adapter call (temporal resampling, normalization, transfer, forward, and
    adapter-side validation) remains inside it. CUDA is synchronized directly
    before and after every timed call.
    """
    if not clips:
        raise ValueError("cannot benchmark an empty clip set")

    times, preds, power_samples = [], [], []
    cuda_device = _model_cuda_device(model)
    device_type = "cuda" if cuda_device is not None else "cpu"
    power_failed = False
    with torch.inference_mode():
        warmup_input = clip_to_input(clips[0])
        for _ in range(warmup):
            with _precision_context(precision, device_type):
                _validate_logits(
                    model(warmup_input), getattr(model, "num_classes", 400)
                )
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)

        for raw_chunk in _clip_chunks(clips, chunk_size):
            prepared_chunk = [clip_to_input(clip_u8) for clip_u8 in raw_chunk]
            logger_started = False
            chunk_samples = []
            if power_logger and not power_failed:
                try:
                    power_logger.start()
                    logger_started = True
                    time.sleep(0.1)
                    reset_samples = getattr(power_logger, "reset_samples", None)
                    if reset_samples is not None:
                        reset_samples()
                except Exception as exc:
                    power_failed = True
                    print(
                        "  WARNING: power sampling unavailable; continuing without "
                        f"power metrics ({type(exc).__name__}: {exc})"
                    )

            inference_error = None
            try:
                for clip in prepared_chunk:
                    if cuda_device is not None:
                        torch.cuda.synchronize(cuda_device)
                    start = time.perf_counter()
                    with _precision_context(precision, device_type):
                        out = model(clip)
                    if cuda_device is not None:
                        torch.cuda.synchronize(cuda_device)
                    times.append((time.perf_counter() - start) * 1000.0)
                    out = _validate_logits(out, getattr(model, "num_classes", 400))
                    preds.append(int(out.argmax(-1).item()))
            except BaseException as exc:
                inference_error = exc
                raise
            finally:
                if power_logger and logger_started:
                    try:
                        chunk_samples = power_logger.stop() or []
                    except Exception as exc:
                        power_failed = True
                        print(
                            "  WARNING: power sampling failed; continuing without "
                            f"power metrics ({type(exc).__name__}: {exc})"
                        )
                        if inference_error is not None:
                            print("  WARNING: power logger cleanup also failed")
                power_samples.extend(chunk_samples)

    if power_failed:
        power_samples = []
    mean_power = statistics.mean(power_samples) if power_samples else float("nan")
    latency = statistics.mean(times)
    latency_std = statistics.stdev(times) if len(times) > 1 else 0.0
    return latency, latency_std, mean_power, len(power_samples), preds


def _norm(label):
    return label.strip().lower().replace("_", " ").replace("'", "")


def measured_accuracy(model, preds, labels):
    """Compute the existing single-view top-1 accuracy."""
    if not labels:
        return None
    if all(isinstance(label, int) for label in labels):
        return 100.0 * sum(
            pred == truth for pred, truth in zip(preds, labels)
        ) / len(labels)
    try:
        names = [_norm(value) for value in model.class_names]
    except AttributeError:
        return None
    truth = [_norm(value) for value in labels]
    correct = sum(
        1 for prediction, target in zip(preds, truth)
        if prediction < len(names) and names[prediction] == target
    )
    return 100.0 * correct / len(truth)


def _metric_or_nan(function, *args):
    try:
        return function(*args)
    except (TypeError, ValueError, OverflowError):
        return float("nan")


def _finite_or_nan(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def _checkpoint_for_pair(model_key, dataset, checkpoint_dir):
    if model_key in FROZEN17_MODEL_KEYS and dataset == "k400":
        frozen = frozen_model_spec(model_key)
        relative = Path(frozen["k400_checkpoint"] or "")
        if relative.parts and relative.parts[0].lower() == "checkpoints":
            return str(Path(checkpoint_dir).joinpath(*relative.parts[1:]))
        return str(ROOT / relative) if relative.parts else ""
    spec = REQUESTED_MODEL_SPECS.get(model_key)
    if spec is None:
        return ""
    pair = spec.for_dataset(dataset)
    if not pair.checkpoint:
        return pair.incompatible_artifact or ""
    relative = Path(pair.checkpoint)
    if relative.parts and relative.parts[0].lower() == "checkpoints":
        return str(Path(checkpoint_dir).joinpath(*relative.parts[1:]))
    return str(ROOT / relative)


def _base_result_row(model_key, device_info, config):
    spec = REQUESTED_MODEL_SPECS.get(model_key)
    pair = spec.for_dataset(config.dataset) if spec else None
    model_frames = (
        int(frozen_model_spec(model_key)["input"]["num_frames"])
        if model_key in FROZEN17_MODEL_KEYS and config.dataset == "k400"
        else (
            pair.preprocessing.frames
            if pair is not None and pair.preprocessing is not None
            else config.decoded_frames
        )
    )
    checkpoint = _checkpoint_for_pair(model_key, config.dataset, config.checkpoint_dir)
    manifest_text = str(config.manifest) if config.manifest else ""
    manifest_hash = (
        manifest_sha256(config.manifest)
        if config.manifest and config.manifest.is_file()
        else ""
    )
    frozen = (
        frozen_model_spec(model_key)
        if model_key in FROZEN17_MODEL_KEYS and config.dataset == "k400"
        else None
    )
    frozen_stage1 = frozen["stage1"] if frozen else None
    row = {
        **device_info.as_dict(),
        "Dataset": config.dataset,
        "DatasetRoot": str(config.dataset_root.resolve()),
        "CheckpointRoot": str(config.checkpoint_dir.resolve()),
        "ModelKey": model_key,
        "Model": (
            frozen_model_spec(model_key)["display_name"]
            if model_key in FROZEN17_MODEL_KEYS and config.dataset == "k400"
            else (spec.display_name if spec else model_key)
        ),
        "Checkpoint": checkpoint,
        "InputProtocol": INPUT_PROTOCOL,
        "Frames": config.decoded_frames,
        "ModelFrames": model_frames,
        "Resolution": f"{config.resolution}x{config.resolution}",
        "BatchSize": config.batch_size,
        "Precision": config.precision,
        "Seed": config.seed,
        "ClipCount": config.num_clips,
        "Manifest": manifest_text,
        "ManifestSHA256": manifest_hash,
        "WarmupRuns": config.warmup,
        "PowerMode": config.power_kind,
        "PowerSampleMs": config.power_sample_ms,
        "Repeats": config.repeats,
        "RunMode": config.mode,
        "CooldownTemp": config.cooldown_temp,
        "CooldownTimeout": config.cooldown_timeout,
        "Stage1PublishedTop1": (
            frozen_stage1["paper_top1"] if frozen else
            (spec.screening.top1_percent if spec else float("nan"))
        ),
        "Stage1PublishedParamsM": (
            frozen_stage1["params_m"] if frozen else
            (spec.screening.params_m if spec else float("nan"))
        ),
        "Stage1PublishedGFLOPs": (
            frozen_stage1["gflops_per_view"] if frozen else
            (spec.screening.gflops if spec else float("nan"))
        ),
        "RuntimeActualParamsM": (
            frozen.get("runtime_actual_params_m", float("nan"))
            if frozen else float("nan")
        ),
        "RuntimeParamSource": (
            frozen.get("runtime_param_source", "measured after strict load")
            if frozen else ""
        ),
        "CheckpointPublishedTop1": (
            frozen.get("checkpoint_published", frozen_stage1)["paper_top1"] if frozen else
            (pair.published.top1_percent if pair and pair.published else float("nan"))
        ),
        "CheckpointPublishedParamsM": (
            frozen.get("checkpoint_published", frozen_stage1)["params_m"] if frozen else
            (pair.published.params_m if pair and pair.published else float("nan"))
        ),
        "CheckpointPublishedGFLOPs": (
            frozen.get("checkpoint_published", frozen_stage1)["gflops_per_view"] if frozen else
            (pair.published.gflops if pair and pair.published else float("nan"))
        ),
        "PublishedTop1": (
            frozen_stage1["paper_top1"] if frozen else
            (pair.published.top1_percent if pair and pair.published else float("nan"))
        ),
        "MeasuredTop1": float("nan"),
        "Params": float("nan"),
        "ParamsM": (
            pair.published.params_m if pair and pair.published else float("nan")
        ),
        "GFLOPs": (
            frozen_stage1["gflops_per_view"] if frozen else
            (pair.published.gflops if pair and pair.published else float("nan"))
        ),
        "LatencyMean": float("nan"),
        "LatencyStd": float("nan"),
        "FPS": float("nan"),
        "AvgPower": float("nan"),
        "PowerSamples": 0,
        "MeasurementBackend": "none",
        "EnergyPerClip": float("nan"),
        "TotalTime": float("nan"),
        "TotalEnergy": float("nan"),
        "PeakVRAM": float("nan"),
        "PeakVRAMMethod": "torch.cuda.max_memory_allocated (MiB)" if device_info.is_cuda else "unavailable on CPU",
        "GPUStartTemp": float("nan"),
        "GPUEndTemp": float("nan"),
        "NS-E": float("nan"),
        "NS-M": float("nan"),
        "NS#": float("nan"),
        "Status": "failed",
        "Error": "",
        "Split": config.split_name or "unspecified",
        "AnnotationFile": str(config.annotations or ""),
        "FramesUsed": model_frames,
    }
    return row


def _add_legacy_fields(row, measured, metric_accuracy, accuracy_source):
    row.update(
        {
            "Top1Published(%)": row["PublishedTop1"],
            "Top1Measured(%)": measured,
            "MetricAccuracy(%)": metric_accuracy,
            "MetricAccuracySource": accuracy_source,
            "Params(M)": row["ParamsM"],
            "NetScore": _metric_or_nan(
                netscore, row["PublishedTop1"], row["ParamsM"], row["GFLOPs"]
            ),
            "NS-E(1/8)": row["NS-E"],
            "NS-M(1/8)": row["NS-M"],
            "NS#(1/8)": row["NS#"],
            "Latency(ms)": row["LatencyMean"],
            "LatencyStd(ms)": row["LatencyStd"],
            "AvgPower(W)": row["AvgPower"],
            "PowerBackend": row["MeasurementBackend"],
            "TotalTime(s)": row["TotalTime"],
            "Energy(J)": row["EnergyPerClip"],
            "TotalEnergy(J)": row["TotalEnergy"],
            "PeakVRAM(MiB)": row["PeakVRAM"],
        }
    )


def _wait_for_cooldown(device_index, threshold, timeout):
    if threshold is None:
        return
    deadline = time.monotonic() + timeout
    while True:
        temperature = query_temperature(device_index)
        if temperature is None:
            print("  WARNING: GPU temperature unavailable; cooldown skipped")
            return
        if temperature <= threshold:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(
                f"  WARNING: cooldown timeout at {temperature:.0f} C; continuing"
            )
            return
        print(
            f"  Cooling GPU: {temperature:.0f} C > {threshold:.0f} C "
            f"({remaining:.0f}s remaining)"
        )
        time.sleep(min(5.0, remaining))


def benchmark_models(
    model_keys,
    clips,
    power_kind="auto",
    device_name=None,
    labels=None,
    dataset="k400",
    device="auto",
    split_name=None,
    annotation_file=None,
    *,
    precision=DEFAULT_PRECISION,
    warmup=DEFAULT_WARMUP,
    power_sample_ms=20,
    device_info=None,
    config=None,
    result_callback: Callable[[dict], None] | None = None,
):
    """Benchmark every model independently and persist through a callback."""
    del device_name  # Runtime identity always comes from the detected device.
    if device_info is None:
        device_info = resolve_device(device, precision=precision)
    if config is None:
        config = ExperimentConfig(
            dataset=dataset,
            dataset_root=ROOT / "datasets" / ("kinetics400" if dataset == "k400" else "ssv2"),
            checkpoint_dir=ROOT / "checkpoints",
            manifest=None,
            output_csv=ROOT / "benchmark_results.csv",
            num_clips=len(clips),
            seed=0,
            batch_size=1,
            precision=precision,
            decoded_frames=int(clips[0].shape[0]),
            resolution=int(clips[0].shape[-1]),
            warmup=warmup,
            power_kind=power_kind,
            power_sample_ms=power_sample_ms,
            split_name=split_name,
            annotations=Path(annotation_file) if annotation_file else None,
            use_cache=True,
            mode="legacy",
            cooldown_temp=None,
            cooldown_timeout=0,
            repeats=1,
        )

    results, errors = [], {}
    torch_device = device_info.torch_device
    cuda_index = device_info.index if device_info.is_cuda else None
    nvml_index = device_info.nvml_index if device_info.is_cuda else None
    if device_info.is_cuda and nvml_index is None:
        nvml_index = cuda_index
    for model_key in model_keys:
        print("=" * 70)
        print(f"{model_key}  ({len(clips)} timed inferences, {config.precision})")
        row = _base_result_row(model_key, device_info, config)
        model = None
        try:
            spec = REQUESTED_MODEL_SPECS.get(model_key)
            if spec is not None and not spec.for_dataset(config.dataset).available:
                raise RuntimeError(spec.for_dataset(config.dataset).availability_reason)
            if cuda_index is not None:
                _wait_for_cooldown(
                    nvml_index, config.cooldown_temp, config.cooldown_timeout
                )
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(torch_device)
            row["GPUStartTemp"] = (
                query_temperature(nvml_index) if nvml_index is not None else float("nan")
            )

            model = MODEL_REGISTRY[model_key](
                dataset=config.dataset, device=str(torch_device)
            )
            if model_key in FROZEN17_MODEL_KEYS and config.num_clips >= 1000:
                validate_frozen_model(
                    model_key,
                    model,
                    dataset=config.dataset,
                    checkpoint=row["Checkpoint"],
                    manifest=config.manifest,
                    num_clips=config.num_clips,
                    seed=config.seed,
                )
                # A wrong output graph fails before power logging or any timed
                # 1000-clip loop. Adapter constructors have already performed
                # their strict (non-head included) checkpoint load.
                preflight_clip = clip_to_input(clips[0])
                with torch.inference_mode(), _precision_context(
                    config.precision, "cuda" if cuda_index is not None else "cpu"
                ):
                    _validate_logits(model(preflight_clip), 400)
            power_logger = (
                make_power_logger(
                    power_kind,
                    interval_ms=power_sample_ms,
                    device_index=nvml_index,
                )
                if cuda_index is not None and power_kind != "none"
                else None
            )
            if cuda_index is not None and power_kind != "none" and power_logger is None:
                print(
                    "  WARNING: no GPU power backend is available; power, energy, "
                    "NS-E, and NS# will be unavailable"
                )
            latency, latency_std, mean_power, samples, preds = benchmark(
                model,
                clips,
                power_logger,
                warmup=warmup,
                precision=precision,
            )
            measured = measured_accuracy(model, preds, labels)
            frozen = (
                frozen_model_spec(model_key)
                if model_key in FROZEN17_MODEL_KEYS and config.dataset == "k400"
                else None
            )
            published = (
                float(frozen["stage1"]["paper_top1"])
                if frozen else float(model.accuracy)
            )
            params = int(model.parameters)
            params_m = params / 1e6
            gflops = (
                float(frozen["stage1"]["gflops_per_view"])
                if frozen else float(model.gflops)
            )
            total_time = latency * len(clips) / 1000.0
            peak_vram = (
                torch.cuda.max_memory_allocated(torch_device) / 1024.0 / 1024.0
                if cuda_index is not None
                else float("nan")
            )
            energy_per_clip = (
                mean_power * latency / 1000.0
                if math.isfinite(mean_power)
                else float("nan")
            )
            total_energy = (
                mean_power * total_time
                if math.isfinite(mean_power)
                else float("nan")
            )
            # The deployment variants require measured accuracy. Full dataset,
            # quick, and smoke modes all carry labels; no published value is
            # silently substituted when measured accuracy is unavailable.
            metric_accuracy = measured
            ns_e = _metric_or_nan(netscore_e, measured, total_time, mean_power)
            ns_m = _metric_or_nan(netscore_m, measured, peak_vram)
            ns_hash = _metric_or_nan(
                netscore_hash, measured, peak_vram, total_time, mean_power
            )
            model_info = getattr(model, "info", {})
            checkpoint = (
                model_info.get("checkpoint")
                if isinstance(model_info, dict)
                else None
            )
            # Requested pairs keep the authoritative resolved registry path.
            # Adapter metadata is only a fallback for older unregistered keys;
            # some adapters expose a bare filename that would otherwise make
            # a successfully written CSV reject its own --resume operation.
            if checkpoint and not row["Checkpoint"]:
                row["Checkpoint"] = str(checkpoint)
            row.update(
                {
                    "Model": str(model.name),
                    "PublishedTop1": published,
                    "MeasuredTop1": round(measured, 4) if measured is not None else float("nan"),
                    "Params": params,
                    "ParamsM": params_m,
                    "RuntimeActualParamsM": params_m,
                    "GFLOPs": gflops,
                    "LatencyMean": latency,
                    "LatencyStd": latency_std,
                    "FPS": 1000.0 / latency,
                    "AvgPower": mean_power,
                    "PowerSamples": samples,
                    "MeasurementBackend": (
                        getattr(power_logger, "backend", "unavailable")
                        if samples else "unavailable"
                    ),
                    "EnergyPerClip": energy_per_clip,
                    "TotalTime": total_time,
                    "TotalEnergy": total_energy,
                    "PeakVRAM": peak_vram,
                    "GPUEndTemp": (
                        query_temperature(nvml_index)
                        if nvml_index is not None else float("nan")
                    ),
                    "NS-E": ns_e,
                    "NS-M": ns_m,
                    "NS#": ns_hash,
                    "Status": "completed",
                    "Error": "",
                }
            )
            # Authoritative registry metadata supplies actual model frames even
            # when an older adapter lacks a public .frames attribute.
            pair = spec.for_dataset(config.dataset) if spec else None
            if pair and pair.preprocessing:
                row["ModelFrames"] = pair.preprocessing.frames
                row["FramesUsed"] = pair.preprocessing.frames
            elif frozen:
                row["ModelFrames"] = int(frozen["input"]["num_frames"])
                row["FramesUsed"] = int(frozen["input"]["num_frames"])
            _add_legacy_fields(row, measured, metric_accuracy, "measured" if measured is not None else "unavailable")
            print(
                f"  latency {latency:.1f} +/- {latency_std:.1f} ms | "
                f"power {mean_power:.1f} W ({samples} samples) | "
                f"top-1 {measured:.2f}% | peak {peak_vram:.0f} MiB | "
                f"NS# {ns_hash:.2f}"
                if measured is not None
                else f"  latency {latency:.1f} ms | measured top-1 unavailable"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}".replace("\r", " ").replace("\n", " ")
            row["Status"] = "failed"
            row["Error"] = message[:1000]
            _add_legacy_fields(row, float("nan"), float("nan"), "unavailable")
            errors[model_key] = row["Error"]
            print(f"  FAILED: {row['Error']}")
        finally:
            if model is not None:
                del model
            gc.collect()
            if cuda_index is not None:
                if not math.isfinite(_finite_or_nan(row.get("GPUEndTemp"))):
                    row["GPUEndTemp"] = query_temperature(nvml_index)
                torch.cuda.empty_cache()
        results.append(row)
        if result_callback is not None:
            result_callback(row)
    return pd.DataFrame(results), errors


def _default_dataset_root(dataset):
    return (
        ROOT / "datasets" / "kinetics400"
        if dataset == "k400"
        else ROOT / "datasets" / "ssv2" / "videos"
    )


def _default_annotations(dataset):
    if dataset != "ssv2":
        return None
    candidate = (
        ROOT
        / "third_party"
        / "UniFormer"
        / "video_classification"
        / "data_list"
        / "sthv2"
        / "somesomev2_rgb_validation_split.txt"
    )
    return candidate if candidate.is_file() else None


def _default_order(dataset):
    available = set(DATASET_MODEL_KEYS[dataset])
    ordered = [key for key in DEFAULT_RUN_ORDER if key in available]
    ordered.extend(key for key in DATASET_MODEL_KEYS[dataset] if key not in ordered)
    return ordered


def parse_model_keys(values, dataset, smoke_test=False):
    if values is None:
        return ["uniformer-s"] if smoke_test else _default_order(dataset)
    tokens = []
    for value in values:
        tokens.extend(part.strip() for part in str(value).split(",") if part.strip())
    if any(token.lower() == "all" for token in tokens):
        if len(tokens) != 1:
            raise ValueError("--models all cannot be combined with explicit model keys")
        return _default_order(dataset)
    if any(token.lower() == "frozen17" for token in tokens):
        if len(tokens) != 1:
            raise ValueError(
                "--models frozen17 cannot be combined with explicit model keys"
            )
        return list(FROZEN17_MODEL_KEYS)
    unknown = [key for key in tokens if key not in MODEL_REGISTRY]
    if unknown:
        raise ValueError("Unknown model key(s): " + ", ".join(unknown))
    return list(dict.fromkeys(tokens))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--dataset", default="k400", choices=("k400", "ssv2"))
    parser.add_argument(
        "--num-clips", "--max-clips", dest="num_clips",
        help="clip count, or 'all' for every readable dataset video",
    )
    parser.add_argument(
        "--streaming", action="store_true",
        help="decode clips on demand instead of using the sharded disk cache "
        "(automatic for full-dataset runs when the cache would not fit)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default=DEFAULT_PRECISION)
    parser.add_argument("--models", nargs="+", help="all, comma-separated, or space-separated keys")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--dataset-root", "--videos-dir", "--video-dir", dest="dataset_root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--split-name")
    parser.add_argument("--power-sample-ms", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cooldown-temp", type=float)
    parser.add_argument("--cooldown-timeout", type=float, default=300.0)
    parser.add_argument("--frames", type=int, default=DEFAULT_DECODED_FRAMES)
    parser.add_argument("--size", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--power",
        default="auto",
        choices=("auto", "nvml", "nvidia-smi", "tegrastats", "none"),
    )
    parser.add_argument("--video", type=Path, help="legacy single-video diagnostic mode")
    parser.add_argument("--runs", type=int, default=10, help="single-video repetitions")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-skips", action="store_true")
    parser.add_argument("--create-manifest", action="store_true", help="create a new sample only when no stable manifest/cache exists")
    # Legacy compatibility: accepted but never used to identify hardware.
    parser.add_argument("--device-name", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.smoke_test and args.quick:
        parser.error("--smoke-test and --quick are mutually exclusive")
    if args.batch_size != 1:
        parser.error("the comparable Stage 2 protocol requires --batch-size 1")
    args.all_clips = False
    if args.num_clips is not None:
        if str(args.num_clips).strip().lower() == "all":
            args.all_clips = True
            args.num_clips = None
        else:
            try:
                args.num_clips = int(args.num_clips)
            except ValueError:
                parser.error("--num-clips must be a positive integer or 'all'")
            if args.num_clips <= 0:
                parser.error("--num-clips must be positive")
    if args.all_clips and args.video:
        parser.error("--num-clips all cannot be combined with --video")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if args.power_sample_ms <= 0:
        parser.error("--power-sample-ms must be positive")
    if args.repeats <= 0 or args.runs <= 0:
        parser.error("--repeats and --runs must be positive")
    if args.dataset == "ssv2" and not args.video and args.annotations is None:
        parser.error("--annotations is required for split-safe SSV2 directory sampling")
    if args.device_name:
        print("WARNING: --device-name is ignored; hardware identity is detected at runtime.")
    if args.num_clips is None and not args.all_clips:
        args.num_clips = (
            args.runs
            if args.video
            else 5 if args.smoke_test else 100 if args.quick else 1000
        )
    args.dataset_root = (args.dataset_root or _default_dataset_root(args.dataset)).resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    try:
        args.model_keys = parse_model_keys(args.models, args.dataset, args.smoke_test)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def _manifest_destination(dataset, clip_count, seed):
    return ROOT / "manifests" / f"{dataset}_{clip_count}_seed{seed}.csv"


def _ensure_full_manifest(args, allowed_stems):
    """Create or reuse the manifest that freezes the full readable dataset."""
    destination = _manifest_destination(args.dataset, "all", args.seed)
    if destination.is_file():
        return destination
    if not args.create_manifest:
        raise SystemExit(
            f"Full-dataset manifest is missing: {destination}. Run once with "
            "--create-manifest to freeze the complete readable video list."
        )
    paths = sorted(
        path for path in args.dataset_root.rglob("*")
        if path.suffix.lower() in VIDEO_EXTS
    )
    if allowed_stems is not None:
        paths = [path for path in paths if path.stem in allowed_stems]
    if not paths:
        raise SystemExit(f"No videos found under {args.dataset_root}")
    # Identical ordering rule to the sampled subsets: sorted, then one seeded
    # shuffle. The 1000-clip manifest is therefore a prefix-compatible sample.
    random.Random(args.seed).shuffle(paths)
    from tqdm import tqdm

    readable, unreadable = [], 0
    for path in tqdm(paths, desc="Probing videos", unit="video"):
        if probe_readable(path):
            readable.append(path)
        else:
            unreadable += 1
    if unreadable:
        print(f"Excluded {unreadable} unreadable videos from the full manifest.")
    if not readable:
        raise SystemExit("No readable videos were found for the full manifest")
    from data import write_sample_manifest

    write_sample_manifest(
        destination,
        [path.relative_to(args.dataset_root).as_posix() for path in readable],
        args.dataset_root,
        args.dataset,
        args.seed,
        args.frames,
        args.size,
    )
    print(f"Created full-dataset manifest ({len(readable)} clips): {destination}")
    return destination


def _find_or_promote_manifest(args):
    if args.video:
        return None
    if args.manifest:
        return args.manifest.resolve()
    if args.all_clips:
        return _manifest_destination(args.dataset, "all", args.seed)
    exact = _manifest_destination(args.dataset, args.num_clips, args.seed)
    if exact.is_file():
        return exact
    # Smoke/quick runs deliberately take a prefix of the exact full sample.
    full = _manifest_destination(args.dataset, 1000, args.seed)
    if args.num_clips <= 1000 and full.is_file():
        return full

    # Preserve the historical sample by promoting its decoded-cache manifest.
    pattern = (
        f".clip_cache_v4_{args.frames}f_{args.size}px_*c_seed{args.seed}_*"
    )
    candidates = sorted(args.dataset_root.glob(pattern))
    for cache_dir in candidates:
        cache_manifest = cache_dir / "manifest.json"
        if not cache_manifest.is_file():
            continue
        try:
            sources = read_sample_manifest(cache_manifest, args.dataset_root)
        except (OSError, ValueError):
            continue
        if len(sources) < args.num_clips:
            continue
        destination = _manifest_destination(args.dataset, len(sources), args.seed)
        promote_cache_manifest(
            cache_manifest,
            destination,
            args.dataset_root,
            args.dataset,
            seed=args.seed,
        )
        print(f"Promoted stable dataset manifest: {destination}")
        return destination
    return exact


def _output_csv(args, device_info, mode):
    if args.output:
        return args.output.resolve()
    clip_label = "all" if getattr(args, "all_clips", False) else args.num_clips
    run_dir = (
        args.output_dir.resolve()
        / device_info.output_device_name
        / args.dataset
        / f"{mode}_{clip_label}_seed{args.seed}"
    )
    return run_dir / "device_metrics.csv"


def _seed_everything(seed):
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _recommended_torch_command():
    return (
        "python -m pip install --upgrade torch==2.7.1 torchvision==0.22.1 "
        "torchaudio==2.7.1 "
        "--index-url https://download.pytorch.org/whl/cu128"
    )


def _resolve_device_or_exit(args):
    try:
        info = resolve_device(args.device, precision=args.precision, validate=True)
        if info.type == "cpu" and args.precision == "fp16":
            raise SystemExit(
                "CPU fp16 is not supported by this benchmark; use fp32 or bf16."
            )
        return info
    except CudaCompatibilityError as exc:
        print(f"CUDA COMPATIBILITY ERROR: {exc}", file=sys.stderr)
        print("Recommended environment command (run separately):", file=sys.stderr)
        print(_recommended_torch_command(), file=sys.stderr)
        raise SystemExit(2) from exc
    except DeviceResolutionError as exc:
        raise SystemExit(f"DEVICE ERROR: {exc}") from exc


def _make_config(args, manifest_path, output_csv, mode):
    return ExperimentConfig(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        checkpoint_dir=args.checkpoint_dir,
        manifest=manifest_path if manifest_path and manifest_path.is_file() else None,
        output_csv=output_csv,
        num_clips=args.num_clips,
        seed=args.seed,
        batch_size=args.batch_size,
        precision=args.precision,
        decoded_frames=args.frames,
        resolution=args.size,
        warmup=args.warmup,
        power_kind=args.power,
        power_sample_ms=args.power_sample_ms,
        split_name=args.split_name,
        annotations=args.annotations.resolve() if args.annotations else None,
        use_cache=not args.no_cache,
        mode=mode,
        cooldown_temp=args.cooldown_temp,
        cooldown_timeout=args.cooldown_timeout,
        repeats=args.repeats,
    )


def run_preflight(args, device_info, manifest_path):
    """Fast read-only environment, data, and exact-pair validation."""
    fatal = []
    print("DEVICE")
    print_device_info(device_info)
    print(f"[OK] {device_info.name}")

    print("\nCUDA")
    if device_info.is_cuda:
        print("[OK] available; startup kernel executed successfully")
    else:
        print("[OK] CPU debug mode (CUDA measurements will be unavailable)")

    print("\nNVML")
    if device_info.is_cuda:
        nvml_index = (
            device_info.nvml_index
            if device_info.nvml_index is not None
            else device_info.index
        )
        nvml = probe_nvml(nvml_index)
        if nvml.available and nvml.power_supported and nvml.temperature_c is not None:
            print("[OK] NVML power and temperature probe succeeded")
        elif nvml.available and nvml.power_supported:
            print("[WARN] NVML power works but temperature is unavailable")
        elif nvml.available:
            print("[WARN] NVML available but board power is unsupported")
        else:
            print(f"[WARN] unavailable: {nvml.error}")
    else:
        print("[SKIP] disabled for CPU")

    print("\nDATASET")
    if args.dataset_root.is_dir():
        print(f"[OK] {args.dataset.upper()} root: {args.dataset_root}")
    else:
        message = f"dataset root not found: {args.dataset_root}"
        print(f"[FAIL] {message}")
        fatal.append(message)
    if manifest_path and manifest_path.is_file() and args.dataset_root.is_dir():
        try:
            sources = read_sample_manifest(
                manifest_path,
                args.dataset_root,
                max_clips=args.num_clips,
                expected_dataset=args.dataset,
                expected_seed=args.seed,
                expected_frames=args.frames,
                expected_size=args.size,
            )
            print(f"[OK] {len(sources)}-clip manifest: {manifest_path}")
            print(f"Manifest hash: {manifest_sha256(manifest_path)}")
            missing = sum(
                not (args.dataset_root / Path(source)).is_file()
                for source in sources
            )
            if missing:
                message = f"{missing} manifest videos are missing from the dataset root"
                print(f"[FAIL] {message}")
                fatal.append(message)
        except Exception as exc:
            message = f"manifest validation failed: {exc}"
            print(f"[FAIL] {message}")
            fatal.append(message)
    else:
        message = f"stable dataset manifest not found: {manifest_path}"
        print(f"[FAIL] {message}")
        fatal.append(message)

    print("\nIMPORTS")
    for module_name in ("torch", "pandas", "decord", "pynvml"):
        try:
            import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:
            label = "WARN" if module_name == "pynvml" else "FAIL"
            print(f"[{label}] {module_name}: {type(exc).__name__}: {exc}")
            if label == "FAIL":
                fatal.append(f"required import failed: {module_name}")

    print("\nMODELS")
    exact_assets = {}
    default_checkpoint_dir = (ROOT / "checkpoints").resolve()
    if args.checkpoint_dir == default_checkpoint_dir:
        try:
            from check_dataset_models import inspect_dataset_models

            asset_report = inspect_dataset_models(args.dataset, ROOT)
            exact_assets = {
                row["key"]: row for row in asset_report.get("models", [])
            }
            print(
                f"Exact asset validation: {asset_report['ready_count']}/"
                f"{asset_report['official_classifier_count']} pairs ready"
            )
        except Exception as exc:
            message = f"exact checkpoint inspection failed: {type(exc).__name__}: {exc}"
            print(f"[FAIL] {message}")
            fatal.append(message)
    else:
        print(
            "[WARN] custom --checkpoint-dir uses exact registry paths and file "
            "checks, but repository checksum inspection is unavailable"
        )
    ready_count = 0
    for model_key in args.model_keys:
        spec = REQUESTED_MODEL_SPECS.get(model_key)
        if spec is None:
            try:
                import_module(MODEL_REGISTRY[model_key].module)
                print(f"[OK] {model_key} - legacy registry adapter importable")
                ready_count += 1
            except Exception as exc:
                print(f"[SKIP] {model_key} - import failed: {exc}")
            continue
        pair = spec.for_dataset(args.dataset)
        if not pair.available:
            print(f"[SKIP] {model_key} - {pair.availability_reason}")
            continue
        asset_row = exact_assets.get(model_key)
        if asset_row is not None and not asset_row.get("ready"):
            blockers = "; ".join(asset_row.get("blockers", []))
            print(f"[SKIP] {model_key} - exact asset check failed: {blockers}")
            continue
        checkpoint = Path(
            _checkpoint_for_pair(model_key, args.dataset, args.checkpoint_dir)
        )
        if not checkpoint.is_file():
            print(f"[SKIP] {model_key} - checkpoint missing: {checkpoint}")
            continue
        try:
            import_module(spec.builder.module)
        except Exception as exc:
            print(f"[SKIP] {model_key} - adapter import failed: {exc}")
            continue
        print(f"[OK] {model_key}")
        ready_count += 1

    print(
        "\nModel allocation is intentionally deferred to smoke-test mode so "
        "preflight remains a 1-2 minute check."
    )
    print(f"READY: {ready_count} models")
    if ready_count == 0:
        fatal.append("no requested exact model/checkpoint pair is ready")
    if fatal:
        print("\nNOT READY")
        for problem in fatal:
            print(f"- {problem}")
        return 1
    print("\nREADY FOR BENCHMARK")
    return 0


def _load_dataset(args, manifest_path):
    if args.video:
        clips = [load_clip(args.video, args.frames, args.size)] * (
            args.runs * args.repeats
        )
        return clips, None, None
    allowed_stems = (
        load_annotation_ids(args.annotations)
        if args.dataset == "ssv2" and args.annotations
        else None
    )
    if args.all_clips:
        manifest_path = _ensure_full_manifest(args, allowed_stems)
    use_manifest = manifest_path if manifest_path and manifest_path.is_file() else None
    if use_manifest is None and not args.create_manifest:
        raise SystemExit(
            f"Stable sample manifest is missing: {manifest_path}. Refusing to "
            "resample silently; restore the prior manifest/cache or explicitly "
            "use --create-manifest once."
        )
    if use_manifest is not None:
        # Validate scientific identity before opening cached tensors or videos.
        read_sample_manifest(
            use_manifest,
            args.dataset_root,
            max_clips=args.num_clips,
            expected_dataset=args.dataset,
            expected_seed=args.seed,
            expected_frames=args.frames,
            expected_size=args.size,
        )
    streaming = args.streaming
    if args.all_clips and not streaming and use_manifest is not None:
        import shutil as _shutil

        clip_total = len(read_sample_manifest(use_manifest, args.dataset_root))
        if args.no_cache:
            streaming = True
            print(
                "Full-dataset run with --no-cache would materialize every clip "
                "in RAM; switching to streaming decode."
            )
        else:
            needed = cache_bytes_estimate(clip_total, args.frames, args.size)
            free = _shutil.disk_usage(args.dataset_root).free
            headroom = 10 * 1024**3
            if needed + headroom > free:
                streaming = True
                print(
                    f"Sharded cache for {clip_total} clips needs about "
                    f"{needed / 1024**3:.0f} GiB but only "
                    f"{free / 1024**3:.0f} GiB is free; using streaming decode "
                    "(decoding repeats per model, methodology unchanged)."
                )
    clips, paths = load_clip_set(
        args.dataset_root,
        args.num_clips,
        args.frames,
        args.size,
        args.seed,
        use_cache=not args.no_cache,
        allowed_stems=allowed_stems,
        manifest=use_manifest,
        streaming=streaming,
    )
    if use_manifest is None:
        from data import write_sample_manifest

        destination = _manifest_destination(args.dataset, len(paths), args.seed)
        relative_sources = [
            Path(path).resolve().relative_to(args.dataset_root).as_posix()
            for path in paths
        ]
        write_sample_manifest(
            destination,
            relative_sources,
            args.dataset_root,
            args.dataset,
            args.seed,
            args.frames,
            args.size,
        )
        manifest_path = destination
        print(f"Created stable dataset manifest: {destination}")
    labels = load_ground_truth(paths, args.dataset, args.annotations)
    if args.all_clips:
        # The resolved full-dataset size becomes the experiment's clip count.
        args.num_clips = len(clips)
    if args.repeats > 1:
        clips = clips * args.repeats
        labels = labels * args.repeats
    return clips, labels, manifest_path


def _write_rankings(frame, output_dir):
    completed = frame[
        frame["Status"].astype(str).str.lower().isin({"completed", "success", "ok"})
    ].copy()
    if completed.empty:
        return
    outputs = {
        "NS-E": "ranking_ns_e.csv",
        "NS-M": "ranking_ns_m.csv",
        "NS#": "ranking_ns_hash.csv",
    }
    for metric, filename in outputs.items():
        completed.sort_values(metric, ascending=False, na_position="last").to_csv(
            output_dir / filename, index=False
        )


def print_runtime_estimate(frame, target_clips=1000):
    completed = frame[
        frame["Status"].astype(str).str.lower().isin({"completed", "success", "ok"})
    ]
    if completed.empty:
        return
    print(f"\nEstimated {target_clips}-clip inference time:")
    total_seconds = 0.0
    for _, row in completed.iterrows():
        latency = _finite_or_nan(row.get("LatencyMean"))
        if not math.isfinite(latency):
            continue
        seconds = latency * target_clips / 1000.0
        total_seconds += seconds
        print(f"{row['Model']}: {seconds:.1f} s ({seconds / 60:.2f} min)")
    print(f"Total: {total_seconds / 60:.2f} min")
    finish = datetime.now() + timedelta(seconds=total_seconds)
    print(f"Estimated finish time: {finish:%Y-%m-%d %H:%M:%S}")
    print("Estimate is a linear extrapolation of measured inference-only latency.")


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)
    device_info = _resolve_device_or_exit(args)
    set_checkpoint_root(args.checkpoint_dir)
    set_dataset_root(args.dataset, args.dataset_root)

    mode = "smoke" if args.smoke_test else "quick" if args.quick else "full"
    manifest_path = _find_or_promote_manifest(args)
    if args.preflight:
        return run_preflight(args, device_info, manifest_path)
    print_device_info(device_info)

    clips, labels, actual_manifest = _load_dataset(args, manifest_path)
    if actual_manifest is not None:
        manifest_path = actual_manifest.resolve()
        print(f"Dataset manifest: {manifest_path}")
        print(f"Number of clips: {len(clips)}")
        print(f"Manifest hash: {manifest_sha256(manifest_path)}")
    output_csv = _output_csv(args, device_info, mode)
    config = _make_config(args, manifest_path, output_csv, mode)
    if len(clips) != config.num_clips * args.repeats:
        raise SystemExit(
            f"Loaded {len(clips)} clips; expected {config.num_clips * args.repeats}"
        )

    store = ResultStore(output_csv, resume=args.resume, force=args.force)
    if args.resume and store.rows:
        existing_keys = {
            str(row.get("ModelKey"))
            for row in store.rows
            if row.get("ModelKey") is not None
        }
        identity_keys = existing_keys | set(args.model_keys)
        expected_rows = {
            key: _base_result_row(key, device_info, config)
            for key in identity_keys
            if key in REQUESTED_MODEL_SPECS
        }
        try:
            store.validate_resume(expected_rows)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    completed = store.successful_keys
    remaining = [key for key in args.model_keys if key not in completed]
    for key in args.model_keys:
        if key in completed:
            print(f"[RESUME] skipping successful model: {key}")
    if remaining:
        benchmark_models(
            remaining,
            clips,
            args.power,
            labels=labels,
            dataset=args.dataset,
            device=str(device_info.torch_device),
            split_name=args.split_name,
            annotation_file=args.annotations,
            precision=args.precision,
            warmup=args.warmup,
            power_sample_ms=args.power_sample_ms,
            device_info=device_info,
            config=config,
            result_callback=store.upsert,
        )
    else:
        print("All requested models already completed successfully.")

    frame = store.frame()
    _write_rankings(frame, output_csv.parent)
    print_runtime_estimate(frame, target_clips=1000)
    print(f"Saved: {output_csv}")
    failures = frame[
        ~frame["Status"].astype(str).str.lower().isin({"completed", "success", "ok"})
    ]
    successes = len(frame) - len(failures)
    if successes == 0:
        print("No model completed successfully.", file=sys.stderr)
        return 1
    if not failures.empty:
        print(f"Completed {successes}; failed {len(failures)}. Failed models remain retryable with --resume.")
        if not args.allow_skips and args.models:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
