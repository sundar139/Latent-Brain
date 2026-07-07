from __future__ import annotations

import pytest
import torch

from latentbrain.torch.device import resolve_device


def test_cpu_resolves_to_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"


def test_auto_returns_valid_device() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_cuda_raises_clear_error_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_cuda_resolves_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cuda").type == "cuda"


def test_invalid_device_string_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_device("gpu")
