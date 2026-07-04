"""Portable accelerator selection for PyTorch execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import torch
from torch.optim import Optimizer

DeviceName = Literal["auto", "cpu", "cuda", "mps", "tpu"]


@dataclass(frozen=True)
class DeviceSpec:
    """Resolved backend name and concrete PyTorch device."""

    backend: Literal["cpu", "cuda", "mps", "tpu"]
    torch_device: torch.device

    @property
    def description(self) -> str:
        if self.backend == "cuda":
            return f"cuda ({torch.cuda.get_device_name(self.torch_device)})"
        return str(self.backend)


def resolve_device(requested: DeviceName | str = "auto") -> DeviceSpec:
    """Resolve a requested backend or raise a useful availability error."""
    name = requested.lower()
    valid = {"auto", "cpu", "cuda", "mps", "tpu"}
    if name not in valid:
        raise ValueError(
            f"device must be one of {', '.join(sorted(valid))}"
        )

    if name == "auto":
        if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
            return _resolve_tpu()
        if torch.cuda.is_available():
            return DeviceSpec("cuda", torch.device("cuda"))
        if torch.backends.mps.is_available():
            return DeviceSpec("mps", torch.device("mps"))
        return DeviceSpec("cpu", torch.device("cpu"))

    if name == "cpu":
        return DeviceSpec("cpu", torch.device("cpu"))
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this PyTorch "
                "installation"
            )
        return DeviceSpec("cuda", torch.device("cuda"))
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS was requested but is not available; Apple Silicon, "
                "a supported macOS version, and an MPS-enabled PyTorch build "
                "are required"
            )
        return DeviceSpec("mps", torch.device("mps"))
    return _resolve_tpu()


def optimizer_step(optimizer: Optimizer, device: torch.device) -> None:
    """Apply an optimizer update and materialize lazy XLA execution."""
    optimizer.step()
    if device.type == "xla":
        import torch_xla

        torch_xla.sync()


def device_memory_metrics(device: torch.device) -> dict[str, float]:
    """Return backend memory counters in MiB when the backend exposes them."""
    bytes_per_mib = 1024 * 1024
    if device.type == "cuda":
        return {
            "device_memory_allocated_mib": (
                torch.cuda.memory_allocated(device) / bytes_per_mib
            ),
            "device_memory_reserved_mib": (
                torch.cuda.memory_reserved(device) / bytes_per_mib
            ),
        }
    if device.type == "mps":
        return {
            "device_memory_allocated_mib": (
                torch.mps.current_allocated_memory() / bytes_per_mib
            ),
            "device_memory_driver_mib": (
                torch.mps.driver_allocated_memory() / bytes_per_mib
            ),
        }
    return {}


def _resolve_tpu() -> DeviceSpec:
    try:
        import torch_xla
    except ImportError as error:
        raise RuntimeError(
            "TPU was requested but torch_xla is not installed; install a "
            "PyTorch/XLA TPU build in the TPU runtime"
        ) from error
    return DeviceSpec("tpu", torch_xla.device())
