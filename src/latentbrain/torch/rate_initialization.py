from __future__ import annotations

import numpy as np
import torch


def inverse_softplus_rate(rate_hz: torch.Tensor, min_rate_hz: float) -> torch.Tensor:
    """Return x such that softplus(x) approximates clipped positive rates."""
    if min_rate_hz <= 0.0:
        msg = "min_rate_hz must be positive"
        raise ValueError(msg)
    rate = torch.clamp(rate_hz, min=min_rate_hz)
    threshold = rate.new_tensor(20.0)
    return torch.where(rate > threshold, rate, torch.log(torch.expm1(rate)))


def compute_train_mean_rates_hz(
    train_spikes: np.ndarray,
    bin_size_ms: int,
    min_rate_hz: float,
    max_rate_hz: float,
) -> np.ndarray:
    if bin_size_ms <= 0:
        msg = "bin_size_ms must be positive"
        raise ValueError(msg)
    if min_rate_hz <= 0.0 or max_rate_hz <= min_rate_hz:
        msg = "rate bounds must be positive and increasing"
        raise ValueError(msg)
    spikes = np.asarray(train_spikes, dtype=np.float64)
    if spikes.ndim != 3:
        msg = f"train_spikes must have shape [trials, time, neurons], got {spikes.shape}"
        raise ValueError(msg)
    seconds = spikes.shape[0] * spikes.shape[1] * (bin_size_ms / 1000.0)
    if seconds <= 0.0:
        msg = "train_spikes must contain at least one bin"
        raise ValueError(msg)
    rates = spikes.sum(axis=(0, 1)) / seconds
    return np.asarray(np.clip(rates, min_rate_hz, max_rate_hz), dtype=np.float64)


def initialize_linear_readout_bias_from_rates(
    linear: torch.nn.Linear,
    mean_rates_hz: torch.Tensor,
    min_rate_hz: float,
) -> None:
    if linear.bias is None:
        msg = "linear layer must have a bias"
        raise ValueError(msg)
    if mean_rates_hz.numel() != linear.bias.numel():
        msg = (
            f"mean_rates_hz length {mean_rates_hz.numel()} must match "
            f"bias length {linear.bias.numel()}"
        )
        raise ValueError(msg)
    with torch.no_grad():
        target = mean_rates_hz.to(device=linear.bias.device, dtype=linear.bias.dtype).reshape_as(
            linear.bias
        )
        linear.bias.copy_(inverse_softplus_rate(target, min_rate_hz))
