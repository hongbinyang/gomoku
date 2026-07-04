import pytest
import torch

from gomoku_muzero.runtime.device import resolve_device


def test_cpu_device_is_always_available() -> None:
    device = resolve_device("cpu")

    assert device.backend == "cpu"
    assert device.torch_device == torch.device("cpu")


def test_auto_falls_back_to_cpu_without_accelerators(monkeypatch) -> None:
    monkeypatch.delenv("PJRT_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    device = resolve_device("auto")

    assert device.backend == "cpu"


def test_unavailable_cuda_has_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_unavailable_mps_has_clear_error(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="MPS"):
        resolve_device("mps")


def test_unknown_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="device"):
        resolve_device("quantum")
