from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    """Resolve a configured device string."""
    normalized = device.lower().strip()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            msg = (
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build or use a CPU config intentionally."
            )
            raise RuntimeError(msg)
        return torch.device("cuda")
    msg = "device must be one of: cpu, cuda, auto"
    raise ValueError(msg)
