"""Background power sampling during inference.

``PynvmlLogger`` is the primary desktop/laptop NVIDIA implementation.  It
polls NVML directly through NVIDIA's ``nvidia-ml-py`` bindings, as required by
the experiment protocol.  The older ``nvidia-smi`` subprocess and Jetson
``tegrastats`` implementations remain available as explicit fallbacks.

Every logger returns samples in watts between :meth:`start` and :meth:`stop`.
"""
import importlib.util
import re
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NvmlProbeResult:
    """Best-effort NVML metadata for one physical NVIDIA device.

    Probing is deliberately non-throwing: optional telemetry such as board
    power and temperature is not available on every NVIDIA device/driver.
    ``available`` means that NVML initialized and returned a handle for the
    requested index; individual fields can still be ``None``.
    """

    available: bool
    device_index: int
    name: str | None = None
    total_memory_bytes: int | None = None
    driver_version: str | None = None
    temperature_c: float | None = None
    power_supported: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nvml_text(value):
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalized_gpu_uuid(value):
    text = (_nvml_text(value) or "").strip().lower()
    return text[4:] if text.startswith("gpu-") else text


def find_nvml_device_index(gpu_uuid):
    """Map a CUDA device UUID to its physical NVML index, if possible.

    PyTorch indices are logical and can be reordered by CUDA_VISIBLE_DEVICES;
    NVML indices remain physical.  UUID matching keeps power and temperature
    attached to the CUDA device that actually runs inference.
    """
    target = _normalized_gpu_uuid(gpu_uuid)
    if not target:
        return None
    initialized = False
    try:
        import pynvml

        pynvml.nvmlInit()
        initialized = True
        count = int(pynvml.nvmlDeviceGetCount())
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            if _normalized_gpu_uuid(pynvml.nvmlDeviceGetUUID(handle)) == target:
                return index
    except Exception:
        return None
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
    return None


def probe_nvidia_smi_name(device_index=0):
    """Return a GPU name through nvidia-smi, or ``None`` when unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(device_index)}",
                "--query-gpu=name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return None
    name = completed.stdout.strip().splitlines()
    return name[0].strip() if name and name[0].strip() else None


def probe_nvml(device_index=0):
    """Return non-throwing NVML metadata for ``device_index``.

    This is intended for startup/preflight checks.  It never fabricates
    readings and always attempts ``nvmlShutdown`` after a successful init.
    Unsupported optional queries simply remain unavailable.
    """

    index = int(device_index)
    initialized = False
    try:
        import pynvml
    except Exception as exc:
        return NvmlProbeResult(
            available=False,
            device_index=index,
            error=f"{type(exc).__name__}: {exc}",
        )

    try:
        pynvml.nvmlInit()
        initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)

        def optional(function_name, *args):
            function = getattr(pynvml, function_name, None)
            if function is None:
                return None
            try:
                return function(*args)
            except Exception:
                return None

        name = _nvml_text(optional("nvmlDeviceGetName", handle))
        driver = _nvml_text(optional("nvmlSystemGetDriverVersion"))
        memory = optional("nvmlDeviceGetMemoryInfo", handle)
        total_memory = getattr(memory, "total", None) if memory is not None else None
        if total_memory is not None:
            total_memory = int(total_memory)

        temperature_kind = getattr(pynvml, "NVML_TEMPERATURE_GPU", 0)
        temperature = optional(
            "nvmlDeviceGetTemperature", handle, temperature_kind
        )
        if temperature is not None:
            temperature = float(temperature)

        power = optional("nvmlDeviceGetPowerUsage", handle)
        return NvmlProbeResult(
            available=True,
            device_index=index,
            name=name,
            total_memory_bytes=total_memory,
            driver_version=driver,
            temperature_c=temperature,
            power_supported=power is not None,
        )
    except Exception as exc:
        return NvmlProbeResult(
            available=False,
            device_index=index,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class _StreamingLogger:
    cmd: list
    pattern: re.Pattern
    scale: float  # multiply parsed value to get watts

    def __init__(self):
        self.samples = []
        self._proc = None
        self._thread = None

    def start(self):
        self.samples = []
        self._proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for line in self._proc.stdout:
            m = self.pattern.search(line)
            if m:
                self.samples.append(float(m.group(1)) * self.scale)

    def reset_samples(self):
        """Discard startup/idle readings immediately before the timed window."""
        self.samples.clear()

    def stop(self):
        if self._proc is not None:
            self._proc.terminate()
            self._proc.wait()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return list(self.samples)


class PynvmlLogger:
    """Poll one NVIDIA GPU's board power directly through NVML."""

    backend = "pynvml/NVML"

    def __init__(self, interval_ms=20, device_index=0):
        if interval_ms <= 0:
            raise ValueError("interval_ms must be positive")
        self.interval_seconds = interval_ms / 1000.0
        self.device_index = int(device_index)
        self.samples = []
        self._thread = None
        self._stop_event = None
        self._pynvml = None
        self._handle = None
        self._error = None

    @staticmethod
    def bindings_available():
        return importlib.util.find_spec("pynvml") is not None

    def start(self):
        if self._thread is not None:
            raise RuntimeError("NVML power logger is already running")
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError(
                "Direct NVML sampling requires nvidia-ml-py; install the "
                "project requirements or choose --power nvidia-smi."
            ) from exc

        self.samples = []
        self._error = None
        self._pynvml = pynvml
        pynvml.nvmlInit()
        try:
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        except Exception:
            pynvml.nvmlShutdown()
            self._pynvml = None
            raise
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        while not self._stop_event.is_set():
            try:
                milliwatts = self._pynvml.nvmlDeviceGetPowerUsage(self._handle)
                self.samples.append(float(milliwatts) / 1000.0)
            except Exception as exc:
                self._error = exc
                self._stop_event.set()
                break
            self._stop_event.wait(self.interval_seconds)

    def reset_samples(self):
        self.samples.clear()

    def stop(self):
        if self._thread is None:
            return list(self.samples)
        self._stop_event.set()
        self._thread.join(timeout=max(2.0, 2 * self.interval_seconds))
        self._thread = None
        try:
            if self._pynvml is not None:
                self._pynvml.nvmlShutdown()
        finally:
            self._pynvml = None
            self._handle = None
            self._stop_event = None
        if self._error is not None:
            error = self._error
            self._error = None
            raise RuntimeError(f"NVML power sampling failed: {error}") from error
        return list(self.samples)


class NvidiaSmiLogger(_StreamingLogger):
    backend = "nvidia-smi (NVML)"

    def __init__(self, interval_ms=20, device_index=0):
        super().__init__()
        self.device_index = int(device_index)
        self.cmd = [
            "nvidia-smi", f"--id={self.device_index}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits", "-lms", str(interval_ms),
        ]
        self.pattern = re.compile(r"([\d.]+)")
        self.scale = 1.0


class TegrastatsLogger(_StreamingLogger):
    backend = "tegrastats"

    def __init__(self, interval_ms=50):
        super().__init__()
        self.cmd = ["tegrastats", "--interval", str(interval_ms)]
        # TX2 reports total board power as "POM_5V_IN cur/avg" in mW;
        # newer JetPacks use "VDD_IN".
        self.pattern = re.compile(r"(?:POM_5V_IN|VDD_IN) (\d+)/")
        self.scale = 1e-3


def make_power_logger(kind="auto", interval_ms=20, device_index=0):
    """Returns a logger instance or None if no power tool is available."""
    if kind == "none":
        return None
    if kind == "nvml":
        return PynvmlLogger(interval_ms, device_index)
    if kind == "nvidia-smi":
        return NvidiaSmiLogger(interval_ms, device_index)
    if kind == "tegrastats":
        return TegrastatsLogger(interval_ms)
    # Prefer the direct NVML API when it can actually query this device.  Merely
    # finding the Python package is insufficient on machines with a stale NVML
    # library; in that case auto mode can still fall back to nvidia-smi.
    if PynvmlLogger.bindings_available():
        nvml = probe_nvml(device_index)
        if nvml.available and nvml.power_supported:
            return PynvmlLogger(interval_ms, device_index)
    # Auto-detect fallbacks.
    if shutil.which("tegrastats"):
        return TegrastatsLogger(max(interval_ms, 50))
    if shutil.which("nvidia-smi"):
        return NvidiaSmiLogger(interval_ms, device_index)
    return None
