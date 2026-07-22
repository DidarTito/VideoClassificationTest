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

    def __init__(self, interval_ms=20):
        super().__init__()
        self.cmd = [
            "nvidia-smi", "--query-gpu=power.draw",
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
        return NvidiaSmiLogger(interval_ms)
    if kind == "tegrastats":
        return TegrastatsLogger(interval_ms)
    # Prefer the direct NVML API for discrete NVIDIA GPUs.
    if PynvmlLogger.bindings_available() and shutil.which("nvidia-smi"):
        return PynvmlLogger(interval_ms, device_index)
    # Auto-detect fallbacks.
    if shutil.which("tegrastats"):
        return TegrastatsLogger(max(interval_ms, 50))
    if shutil.which("nvidia-smi"):
        return NvidiaSmiLogger(interval_ms)
    return None
