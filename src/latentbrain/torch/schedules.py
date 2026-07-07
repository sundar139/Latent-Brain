from __future__ import annotations


def linear_warmup(epoch: int, warmup_epochs: int, max_value: float = 1.0) -> float:
    """Zero-indexed linear warmup: epoch 0 is 0, epoch >= warmup_epochs is max_value."""
    if warmup_epochs <= 0:
        return float(max_value)
    if epoch <= 0:
        return 0.0
    return float(min(max_value, max_value * (epoch / warmup_epochs)))
