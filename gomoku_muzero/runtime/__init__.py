"""Runtime concerns shared by commands and training backends."""

from gomoku_muzero.runtime.device import (
    DeviceSpec,
    device_memory_metrics,
    optimizer_step,
    resolve_device,
)

__all__ = [
    "DeviceSpec",
    "device_memory_metrics",
    "optimizer_step",
    "resolve_device",
]
