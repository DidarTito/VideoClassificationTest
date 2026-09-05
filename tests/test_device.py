"""Focused tests for generic CPU/CUDA device handling."""

from __future__ import annotations

from io import StringIO
import types
import unittest
from unittest import mock

import torch

from power import NvmlProbeResult
from utils.device import (
    CudaCompatibilityError,
    DeviceResolutionError,
    check_cuda_kernel_compatibility,
    print_device_info,
    query_temperature,
    resolve_device,
    sanitize_device_name,
)


class DeviceResolutionTests(unittest.TestCase):
    def test_auto_uses_cpu_when_cuda_is_unavailable(self):
        unavailable = NvmlProbeResult(available=False, device_index=0)
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False), mock.patch(
            "utils.device._nvml_for_index", return_value=unavailable
        ), mock.patch("utils.device.probe_nvidia_smi_name", return_value=None):
            info = resolve_device("auto")

        self.assertEqual(info.torch_device, torch.device("cpu"))
        self.assertEqual(info.type, "cpu")
        self.assertIsNone(info.index)
        self.assertIsNone(info.total_vram_mib)
        self.assertEqual(info.precision, "fp32")

    def test_explicit_cuda_never_silently_falls_back(self):
        unavailable = NvmlProbeResult(available=False, device_index=0)
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False), mock.patch(
            "utils.device._nvml_for_index", return_value=unavailable
        ), mock.patch("utils.device.probe_nvidia_smi_name", return_value=None):
            with self.assertRaisesRegex(DeviceResolutionError, "requested"):
                resolve_device("cuda:0", validate=False)

    def test_auto_fails_when_nvml_sees_gpu_but_torch_cuda_is_unavailable(self):
        visible = NvmlProbeResult(
            available=True, device_index=0, name="NVIDIA GeForce RTX 5070"
        )
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False), mock.patch(
            "utils.device._nvml_for_index", return_value=visible
        ):
            with self.assertRaisesRegex(CudaCompatibilityError, "RTX 5070"):
                resolve_device("auto")

    def test_auto_fails_when_nvidia_smi_sees_gpu_without_nvml_bindings(self):
        unavailable = NvmlProbeResult(available=False, device_index=0)
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False), mock.patch(
            "utils.device._nvml_for_index", return_value=unavailable
        ), mock.patch(
            "utils.device.probe_nvidia_smi_name", return_value="NVIDIA RTX 5070"
        ):
            with self.assertRaisesRegex(CudaCompatibilityError, "RTX 5070"):
                resolve_device("auto")

    def test_explicit_cuda_index_populates_metadata(self):
        properties = types.SimpleNamespace(
            total_memory=8 * 1024**3,
            uuid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        )
        nvml = NvmlProbeResult(
            available=True,
            device_index=1,
            driver_version="600.12",
            temperature_c=47.0,
        )
        patches = (
            mock.patch("utils.device.torch.cuda.is_available", return_value=True),
            mock.patch("utils.device.torch.cuda.device_count", return_value=2),
            mock.patch(
                "utils.device.torch.cuda.get_device_name",
                return_value="NVIDIA GeForce RTX 5070",
            ),
            mock.patch(
                "utils.device.torch.cuda.get_device_capability", return_value=(12, 0)
            ),
            mock.patch(
                "utils.device.torch.cuda.get_device_properties",
                return_value=properties,
            ),
            mock.patch("utils.device.probe_nvml", return_value=nvml),
            mock.patch("utils.device.find_nvml_device_index", return_value=4),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            info = resolve_device("cuda:1", precision="bf16", validate=False)

        self.assertEqual(info.torch_device, torch.device("cuda:1"))
        self.assertEqual(info.index, 1)
        self.assertEqual(info.name, "NVIDIA GeForce RTX 5070")
        self.assertEqual(info.total_vram_mib, 8192.0)
        self.assertEqual(info.compute_capability, "12.0")
        self.assertEqual(info.driver_version, "600.12")
        self.assertEqual(info.nvml_index, 4)
        self.assertEqual(info.gpu_uuid, "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
        self.assertEqual(info.cuda_device_count, 2)
        self.assertEqual(info.precision, "bf16")
        self.assertEqual(info.sanitized_name, "NVIDIA_GeForce_RTX_5070")
        self.assertEqual(
            info.output_device_name, "NVIDIA_GeForce_RTX_5070_cuda1"
        )

    def test_cuda_index_is_range_checked(self):
        with mock.patch("utils.device.torch.cuda.is_available", return_value=True):
            with mock.patch("utils.device.torch.cuda.device_count", return_value=1):
                with self.assertRaisesRegex(DeviceResolutionError, "out of range"):
                    resolve_device("cuda:3", validate=False)

    def test_precision_is_validated(self):
        with self.assertRaisesRegex(ValueError, "precision"):
            resolve_device("cpu", precision="automatic")


class CudaCompatibilityTests(unittest.TestCase):
    def test_kernel_probe_executes_and_synchronizes_selected_device(self):
        tensor = mock.Mock()
        with mock.patch("utils.device.torch.empty", return_value=tensor) as empty:
            with mock.patch("utils.device.torch.cuda.synchronize") as synchronize:
                check_cuda_kernel_compatibility("cuda:2")

        empty.assert_called_once_with(1, device=torch.device("cuda:2"))
        tensor.add_.assert_called_once_with(1)
        synchronize.assert_called_once_with(torch.device("cuda:2"))

    def test_kernel_failure_has_actionable_environment_context(self):
        with mock.patch(
            "utils.device.torch.empty", side_effect=RuntimeError("no kernel image")
        ):
            with mock.patch(
                "utils.device.torch.cuda.get_device_name", return_value="RTX 5070"
            ):
                with mock.patch(
                    "utils.device.torch.cuda.get_device_capability",
                    return_value=(12, 0),
                ):
                    with mock.patch(
                        "utils.device.torch.cuda.get_arch_list", return_value=["sm_90"]
                    ):
                        with self.assertRaises(CudaCompatibilityError) as caught:
                            check_cuda_kernel_compatibility("cuda:0")

        message = str(caught.exception)
        self.assertIn("RTX 5070", message)
        self.assertIn("compute capability=12.0", message)
        self.assertIn("no kernel image", message)

    def test_cpu_has_no_cuda_probe(self):
        with mock.patch("utils.device.torch.empty") as empty:
            check_cuda_kernel_compatibility("cpu")
        empty.assert_not_called()


class DeviceMetadataTests(unittest.TestCase):
    def test_sanitize_device_name_is_stable_and_windows_safe(self):
        self.assertEqual(
            sanitize_device_name("NVIDIA GeForce RTX 5070 / Laptop GPU"),
            "NVIDIA_GeForce_RTX_5070_Laptop_GPU",
        )
        self.assertEqual(sanitize_device_name("CON"), "device_CON")
        self.assertEqual(sanitize_device_name("***"), "device")

    def test_temperature_query_returns_none_when_nvml_is_unavailable(self):
        unavailable = NvmlProbeResult(
            available=False, device_index=0, error="NVML unavailable"
        )
        with mock.patch("utils.device.probe_nvml", return_value=unavailable):
            self.assertIsNone(query_temperature(0))

    def test_temperature_query_returns_real_reading(self):
        available = NvmlProbeResult(
            available=True, device_index=3, temperature_c=61.0
        )
        with mock.patch("utils.device.probe_nvml", return_value=available) as probe:
            self.assertEqual(query_temperature(3), 61.0)
        probe.assert_called_once_with(3)

    def test_print_device_info_contains_reproducibility_fields(self):
        with mock.patch("utils.device.torch.cuda.is_available", return_value=False):
            info = resolve_device("cpu", precision="fp32")
        output = StringIO()
        print_device_info(info, file=output)
        rendered = output.getvalue()
        self.assertIn("Device: CPU", rendered)
        self.assertIn("Device ID: cpu", rendered)
        self.assertIn("Precision: fp32", rendered)


if __name__ == "__main__":
    unittest.main()
