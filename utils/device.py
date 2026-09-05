"""Central device resolution, validation, and metadata collection.

The benchmark supports CPU for debugging and one explicitly selected CUDA
device for measurement.  An explicit CUDA request never silently falls back
to CPU: doing so would make a run look successful while measuring different
hardware than requested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import unicodedata
from typing import Any, TextIO

import torch

from power import (
    NvmlProbeResult,
    find_nvml_device_index,
    probe_nvidia_smi_name,
    probe_nvml,
)


PRECISIONS = ("fp32", "fp16", "bf16")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


class DeviceResolutionError(RuntimeError):
    """The requested runtime device cannot be selected safely."""


class CudaCompatibilityError(DeviceResolutionError):
    """The installed PyTorch/CUDA build cannot execute on the selected GPU."""


@dataclass(frozen=True)
class DeviceInfo:
    """Resolved execution device plus reproducibility metadata."""

    torch_device: torch.device
    type: str
    index: int | None
    name: str
    total_vram_mib: float | None
    compute_capability: str | None
    torch_version: str
    cuda_version: str | None
    driver_version: str | None
    precision: str
    nvml_index: int | None = None
    gpu_uuid: str | None = None
    cuda_device_count: int | None = None
    nvml_available: bool = False
    temperature_c: float | None = None

    @property
    def device_id(self) -> str:
        return str(self.torch_device)

    @property
    def sanitized_name(self) -> str:
        return sanitize_device_name(self.name)

    @property
    def output_device_name(self) -> str:
        name = self.sanitized_name
        if self.is_cuda and (self.cuda_device_count or 0) > 1:
            return f"{name}_cuda{self.index}"
        return name

    @property
    def is_cuda(self) -> bool:
        return self.type == "cuda"

    def as_dict(self) -> dict[str, Any]:
        """Return CSV-friendly names used by benchmark result rows."""

        return {
            "Device": self.device_id,
            "DeviceName": self.name,
            "DeviceIndex": self.index,
            "NVMLDeviceIndex": self.nvml_index,
            "GPUUUID": self.gpu_uuid,
            "CUDADeviceCount": self.cuda_device_count,
            "TotalVRAM": self.total_vram_mib,
            "ComputeCapability": self.compute_capability,
            "TorchVersion": self.torch_version,
            "CUDAVersion": self.cuda_version,
            "DriverVersion": self.driver_version,
            "Precision": self.precision,
        }

    def debug_dict(self) -> dict[str, Any]:
        """Return every dataclass field, useful for structured preflight logs."""

        values = asdict(self)
        values["torch_device"] = self.device_id
        return values


def sanitize_device_name(name: str) -> str:
    """Make a stable, Windows-safe directory component from a device name."""

    normalized = unicodedata.normalize("NFKD", str(name))
    ascii_name = normalized.encode("ascii", errors="ignore").decode("ascii")
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", ascii_name).strip("_.")
    if not sanitized:
        sanitized = "device"
    if sanitized.upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = f"device_{sanitized}"
    return sanitized[:100].rstrip("_.") or "device"


def _normalize_precision(precision: str) -> str:
    value = str(precision).strip().lower()
    if value not in PRECISIONS:
        choices = ", ".join(PRECISIONS)
        raise ValueError(f"Unsupported precision {precision!r}; choose one of: {choices}")
    return value


def _cuda_index(device: torch.device) -> int:
    if device.index is not None:
        index = int(device.index)
    else:
        try:
            index = int(torch.cuda.current_device())
        except Exception as exc:
            raise DeviceResolutionError(
                f"CUDA is available but the current device could not be selected: {exc}"
            ) from exc

    try:
        count = int(torch.cuda.device_count())
    except Exception as exc:
        raise DeviceResolutionError(f"Could not enumerate CUDA devices: {exc}") from exc
    if index < 0 or index >= count:
        raise DeviceResolutionError(
            f"CUDA device index {index} is out of range; detected {count} device(s)."
        )
    return index


def _cuda_context(device: torch.device) -> str:
    index = device.index
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        name = "unknown NVIDIA GPU"
    try:
        major, minor = torch.cuda.get_device_capability(index)
        capability = f"{major}.{minor}"
    except Exception:
        capability = "unknown"
    arch_list_getter = getattr(torch.cuda, "get_arch_list", None)
    try:
        architectures = ", ".join(arch_list_getter()) if arch_list_getter else "unknown"
    except Exception:
        architectures = "unknown"
    return (
        f"device={device} ({name}), compute capability={capability}, "
        f"PyTorch={torch.__version__}, Torch CUDA build={torch.version.cuda or 'none'}, "
        f"compiled architectures={architectures or 'unknown'}"
    )


def check_cuda_kernel_compatibility(device: str | torch.device) -> None:
    """Execute and synchronize a minimal CUDA kernel, raising a clear error.

    Comparing architecture strings alone is not reliable because a wheel may
    contain PTX that the driver can JIT for a newer GPU.  A real one-element
    operation is both faster and more authoritative for preflight validation.
    """

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        return
    try:
        probe = torch.empty(1, device=torch_device)
        probe.add_(1)
        torch.cuda.synchronize(torch_device)
    except Exception as exc:
        context = _cuda_context(torch_device)
        raise CudaCompatibilityError(
            "CUDA GPU detected but the current PyTorch build could not execute "
            f"a kernel ({context}). Original error: {type(exc).__name__}: {exc}"
        ) from exc


def _nvml_for_index(index: int) -> NvmlProbeResult:
    try:
        return probe_nvml(index)
    except Exception as exc:  # defensive: probes are specified to be non-throwing
        return NvmlProbeResult(
            available=False,
            device_index=index,
            error=f"{type(exc).__name__}: {exc}",
        )


def resolve_device(
    requested: str | torch.device | None = "auto",
    precision: str = "fp32",
    validate: bool = True,
) -> DeviceInfo:
    """Resolve ``auto``, ``cpu``, ``cuda``, or ``cuda:N`` to :class:`DeviceInfo`.

    ``auto`` chooses CUDA when PyTorch reports it available and CPU otherwise.
    An explicit CUDA request fails rather than changing the experiment device.
    """

    precision = _normalize_precision(precision)
    request_text = "auto" if requested is None else str(requested).strip().lower()
    if request_text == "auto":
        if torch.cuda.is_available():
            request_text = "cuda"
        else:
            # A CPU-only host should retain the documented auto fallback.  If
            # NVML can see an NVIDIA GPU, however, silently benchmarking CPU
            # would hide a broken/incompatible Torch CUDA installation.
            nvml = _nvml_for_index(0)
            fallback_name = None if nvml.available else probe_nvidia_smi_name(0)
            if nvml.available or fallback_name:
                raise CudaCompatibilityError(
                    "An NVIDIA GPU was detected by the NVIDIA driver"
                    f" ({nvml.name or fallback_name or 'device 0'}), but "
                    "torch.cuda.is_available() "
                    f"is false. Installed PyTorch: {torch.__version__}; Torch CUDA "
                    f"build: {torch.version.cuda or 'none'}."
                )
            request_text = "cpu"

    try:
        candidate = torch.device(request_text)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise DeviceResolutionError(f"Invalid device {requested!r}: {exc}") from exc
    if candidate.type not in {"cpu", "cuda"}:
        raise DeviceResolutionError(
            f"Unsupported benchmark device {candidate}; choose auto, cpu, cuda, or cuda:N."
        )

    if candidate.type == "cpu":
        return DeviceInfo(
            torch_device=torch.device("cpu"),
            type="cpu",
            index=None,
            name="CPU",
            total_vram_mib=None,
            compute_capability=None,
            torch_version=str(torch.__version__),
            cuda_version=torch.version.cuda,
            driver_version=None,
            precision=precision,
        )

    if not torch.cuda.is_available():
        nvml = _nvml_for_index(0)
        fallback_name = None if nvml.available else probe_nvidia_smi_name(0)
        if nvml.available or fallback_name:
            raise CudaCompatibilityError(
                f"CUDA device {requested!r} was requested and NVIDIA GPU "
                f"{nvml.name or fallback_name or 'device 0'} is visible to the "
                "NVIDIA driver, but "
                "torch.cuda.is_available() is false. "
                f"Installed PyTorch: {torch.__version__}; Torch CUDA build: "
                f"{torch.version.cuda or 'none'}."
            )
        raise DeviceResolutionError(
            f"CUDA device {requested!r} was requested, but torch.cuda.is_available() is false. "
            f"Installed PyTorch: {torch.__version__}; Torch CUDA build: "
            f"{torch.version.cuda or 'none'}."
        )

    index = _cuda_index(candidate)
    torch_device = torch.device(f"cuda:{index}")
    if validate:
        check_cuda_kernel_compatibility(torch_device)

    try:
        name = str(torch.cuda.get_device_name(index))
    except Exception as exc:
        raise DeviceResolutionError(
            f"Could not read metadata for CUDA device {index}: {exc}"
        ) from exc

    try:
        major, minor = torch.cuda.get_device_capability(index)
        compute_capability = f"{int(major)}.{int(minor)}"
    except Exception:
        compute_capability = None

    total_memory_bytes = None
    properties = None
    try:
        properties = torch.cuda.get_device_properties(index)
        total_memory_bytes = getattr(properties, "total_memory", None)
    except Exception:
        pass

    gpu_uuid_value = getattr(properties, "uuid", None) if properties is not None else None
    gpu_uuid = str(gpu_uuid_value) if gpu_uuid_value is not None else None
    matched_nvml_index = find_nvml_device_index(gpu_uuid) if gpu_uuid else None
    nvml_index = index if matched_nvml_index is None else matched_nvml_index
    nvml = _nvml_for_index(nvml_index)
    if total_memory_bytes is None:
        total_memory_bytes = nvml.total_memory_bytes
    total_vram_mib = (
        float(total_memory_bytes) / (1024.0 * 1024.0)
        if total_memory_bytes is not None
        else None
    )

    return DeviceInfo(
        torch_device=torch_device,
        type="cuda",
        index=index,
        name=name,
        total_vram_mib=total_vram_mib,
        compute_capability=compute_capability,
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        driver_version=nvml.driver_version,
        precision=precision,
        nvml_index=nvml_index,
        gpu_uuid=gpu_uuid,
        cuda_device_count=int(torch.cuda.device_count()),
        nvml_available=nvml.available,
        temperature_c=nvml.temperature_c,
    )


def query_temperature(device_index: int = 0) -> float | None:
    """Return the current GPU temperature in Celsius, or ``None`` if unavailable."""

    result = _nvml_for_index(int(device_index))
    return result.temperature_c if result.available else None


def print_device_info(info: DeviceInfo, file: TextIO | None = None) -> None:
    """Print concise, copyable runtime metadata for benchmark logs."""

    def show(value, suffix=""):
        return "N/A" if value is None else f"{value}{suffix}"

    lines = (
        f"Device: {info.name}",
        f"Device ID: {info.device_id}",
        f"Device index: {show(info.index)}",
        f"NVML index: {show(info.nvml_index)}",
        f"GPU UUID: {show(info.gpu_uuid)}",
        f"VRAM: {show(round(info.total_vram_mib, 1) if info.total_vram_mib is not None else None, ' MiB')}",
        f"Compute capability: {show(info.compute_capability)}",
        f"PyTorch: {info.torch_version}",
        f"CUDA: {show(info.cuda_version)}",
        f"Driver: {show(info.driver_version)}",
        f"Precision: {info.precision}",
    )
    print("\n".join(lines), file=file)


__all__ = [
    "CudaCompatibilityError",
    "DeviceInfo",
    "DeviceResolutionError",
    "PRECISIONS",
    "check_cuda_kernel_compatibility",
    "print_device_info",
    "probe_nvml",
    "query_temperature",
    "resolve_device",
    "sanitize_device_name",
]
