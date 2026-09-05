"""Shared runtime utilities for benchmark entry points."""

from .device import (
    CudaCompatibilityError,
    DeviceInfo,
    DeviceResolutionError,
    check_cuda_kernel_compatibility,
    print_device_info,
    query_temperature,
    resolve_device,
    sanitize_device_name,
)

__all__ = [
    "CudaCompatibilityError",
    "DeviceInfo",
    "DeviceResolutionError",
    "check_cuda_kernel_compatibility",
    "print_device_info",
    "query_temperature",
    "resolve_device",
    "sanitize_device_name",
]
