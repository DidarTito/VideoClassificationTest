"""Unit tests for direct NVML power sampling and backend selection."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from power import (
    NvmlProbeResult,
    NvidiaSmiLogger,
    PynvmlLogger,
    TegrastatsLogger,
    find_nvml_device_index,
    make_power_logger,
    probe_nvml,
)


class _OneSampleEvent:
    """Stop a synchronous fake reader after its first timed wait."""

    def __init__(self):
        self._is_set = False

    def is_set(self):
        return self._is_set

    def set(self):
        self._is_set = True

    def wait(self, _timeout):
        self._is_set = True
        return True


class _ImmediateThread:
    def __init__(self, target, daemon):
        self.target = target
        self.daemon = daemon

    def start(self):
        self.target()

    def join(self, timeout=None):
        self.join_timeout = timeout


def _fake_pynvml(power_milliwatts=52_750):
    module = types.ModuleType("pynvml")
    module.nvmlInit = mock.Mock()
    module.nvmlDeviceGetHandleByIndex = mock.Mock(return_value="gpu-handle")
    module.nvmlDeviceGetPowerUsage = mock.Mock(return_value=power_milliwatts)
    module.nvmlShutdown = mock.Mock()
    return module


class PynvmlLoggerTests(unittest.TestCase):
    def test_samples_milliwatts_as_watts_and_shuts_down_nvml(self):
        pynvml = _fake_pynvml()
        logger = PynvmlLogger(interval_ms=25, device_index=3)

        with mock.patch.dict(sys.modules, {"pynvml": pynvml}):
            with mock.patch("power.threading.Event", _OneSampleEvent):
                with mock.patch("power.threading.Thread", _ImmediateThread):
                    logger.start()
                    samples = logger.stop()

        self.assertEqual(samples, [52.75])
        pynvml.nvmlInit.assert_called_once_with()
        pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)
        pynvml.nvmlDeviceGetPowerUsage.assert_called_once_with("gpu-handle")
        pynvml.nvmlShutdown.assert_called_once_with()

    def test_sampling_error_is_reported_after_nvml_cleanup(self):
        pynvml = _fake_pynvml()
        pynvml.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("GPU lost")
        logger = PynvmlLogger()

        with mock.patch.dict(sys.modules, {"pynvml": pynvml}):
            with mock.patch("power.threading.Event", _OneSampleEvent):
                with mock.patch("power.threading.Thread", _ImmediateThread):
                    logger.start()
                    with self.assertRaisesRegex(
                        RuntimeError, "NVML power sampling failed: GPU lost"
                    ):
                        logger.stop()

        pynvml.nvmlShutdown.assert_called_once_with()

    def test_rejects_nonpositive_sampling_interval(self):
        for interval_ms in (0, -1):
            with self.subTest(interval_ms=interval_ms):
                with self.assertRaisesRegex(ValueError, "positive"):
                    PynvmlLogger(interval_ms=interval_ms)


class NvmlProbeTests(unittest.TestCase):
    def test_uuid_maps_logical_cuda_device_to_physical_nvml_index(self):
        pynvml = _fake_pynvml()
        pynvml.nvmlDeviceGetCount = mock.Mock(return_value=3)
        pynvml.nvmlDeviceGetHandleByIndex = mock.Mock(
            side_effect=lambda index: f"handle-{index}"
        )
        uuids = {
            "handle-0": b"GPU-00000000-0000-0000-0000-000000000000",
            "handle-1": b"GPU-11111111-1111-1111-1111-111111111111",
            "handle-2": b"GPU-22222222-2222-2222-2222-222222222222",
        }
        pynvml.nvmlDeviceGetUUID = mock.Mock(side_effect=lambda handle: uuids[handle])

        with mock.patch.dict(sys.modules, {"pynvml": pynvml}):
            index = find_nvml_device_index(
                "22222222-2222-2222-2222-222222222222"
            )

        self.assertEqual(index, 2)
        pynvml.nvmlShutdown.assert_called_once_with()
    def test_probe_returns_metadata_and_always_shuts_down(self):
        pynvml = _fake_pynvml(power_milliwatts=70_000)
        pynvml.NVML_TEMPERATURE_GPU = 0
        pynvml.nvmlDeviceGetName = mock.Mock(return_value=b"NVIDIA Test GPU")
        pynvml.nvmlSystemGetDriverVersion = mock.Mock(return_value=b"600.12")
        pynvml.nvmlDeviceGetMemoryInfo = mock.Mock(
            return_value=types.SimpleNamespace(total=12 * 1024**3)
        )
        pynvml.nvmlDeviceGetTemperature = mock.Mock(return_value=54)

        with mock.patch.dict(sys.modules, {"pynvml": pynvml}):
            result = probe_nvml(2)

        self.assertTrue(result.available)
        self.assertEqual(result.name, "NVIDIA Test GPU")
        self.assertEqual(result.total_memory_bytes, 12 * 1024**3)
        self.assertEqual(result.driver_version, "600.12")
        self.assertEqual(result.temperature_c, 54.0)
        self.assertTrue(result.power_supported)
        pynvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
        pynvml.nvmlShutdown.assert_called_once_with()

    def test_probe_failure_is_reported_without_raising(self):
        pynvml = _fake_pynvml()
        pynvml.nvmlInit.side_effect = RuntimeError("driver unavailable")

        with mock.patch.dict(sys.modules, {"pynvml": pynvml}):
            result = probe_nvml(0)

        self.assertFalse(result.available)
        self.assertIn("driver unavailable", result.error)
        pynvml.nvmlShutdown.assert_not_called()


class MakePowerLoggerTests(unittest.TestCase):
    def test_auto_prefers_direct_nvml_when_bindings_and_gpu_are_available(self):
        probe = NvmlProbeResult(
            available=True, device_index=2, power_supported=True
        )
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=True
        ), mock.patch("power.probe_nvml", return_value=probe):
            with mock.patch("power.shutil.which", return_value="nvidia-smi.exe"):
                logger = make_power_logger("auto", interval_ms=30, device_index=2)

        self.assertIsInstance(logger, PynvmlLogger)
        self.assertEqual(logger.interval_seconds, 0.03)
        self.assertEqual(logger.device_index, 2)

    def test_auto_uses_direct_nvml_without_nvidia_smi_executable(self):
        probe = NvmlProbeResult(
            available=True, device_index=1, power_supported=True
        )
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=True
        ), mock.patch("power.probe_nvml", return_value=probe), mock.patch(
            "power.shutil.which", return_value=None
        ):
            logger = make_power_logger("auto", interval_ms=25, device_index=1)

        self.assertIsInstance(logger, PynvmlLogger)
        self.assertEqual(logger.device_index, 1)

    def test_auto_falls_back_when_direct_nvml_probe_fails(self):
        unavailable = NvmlProbeResult(
            available=False, device_index=3, error="stale NVML library"
        )
        executables = {"tegrastats": None, "nvidia-smi": "nvidia-smi.exe"}
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=True
        ), mock.patch("power.probe_nvml", return_value=unavailable), mock.patch(
            "power.shutil.which", side_effect=lambda name: executables[name]
        ):
            logger = make_power_logger("auto", interval_ms=50, device_index=3)

        self.assertIsInstance(logger, NvidiaSmiLogger)
        self.assertEqual(logger.device_index, 3)

    def test_auto_falls_back_to_nvidia_smi_without_pynvml_bindings(self):
        executables = {"tegrastats": None, "nvidia-smi": "nvidia-smi.exe"}
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=False
        ):
            with mock.patch(
                "power.shutil.which", side_effect=lambda name: executables[name]
            ):
                logger = make_power_logger("auto", interval_ms=40, device_index=4)

        self.assertIsInstance(logger, NvidiaSmiLogger)
        self.assertEqual(logger.device_index, 4)
        self.assertIn("--id=4", logger.cmd)
        self.assertEqual(logger.cmd[-1], "40")

    def test_explicit_and_unavailable_backend_selection(self):
        self.assertIsNone(make_power_logger("none"))
        self.assertIsInstance(make_power_logger("tegrastats"), TegrastatsLogger)
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=False
        ):
            with mock.patch("power.shutil.which", return_value=None):
                self.assertIsNone(make_power_logger("auto"))


if __name__ == "__main__":
    unittest.main()
