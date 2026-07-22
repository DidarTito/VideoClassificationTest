"""Unit tests for direct NVML power sampling and backend selection."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from power import NvidiaSmiLogger, PynvmlLogger, TegrastatsLogger, make_power_logger


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


class MakePowerLoggerTests(unittest.TestCase):
    def test_auto_prefers_direct_nvml_when_bindings_and_gpu_are_available(self):
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=True
        ):
            with mock.patch("power.shutil.which", return_value="nvidia-smi.exe"):
                logger = make_power_logger("auto", interval_ms=30, device_index=2)

        self.assertIsInstance(logger, PynvmlLogger)
        self.assertEqual(logger.interval_seconds, 0.03)
        self.assertEqual(logger.device_index, 2)

    def test_auto_falls_back_to_nvidia_smi_without_pynvml_bindings(self):
        executables = {"tegrastats": None, "nvidia-smi": "nvidia-smi.exe"}
        with mock.patch.object(
            PynvmlLogger, "bindings_available", return_value=False
        ):
            with mock.patch(
                "power.shutil.which", side_effect=lambda name: executables[name]
            ):
                logger = make_power_logger("auto", interval_ms=40)

        self.assertIsInstance(logger, NvidiaSmiLogger)
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
