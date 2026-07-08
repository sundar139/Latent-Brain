from __future__ import annotations

import torch


def sample_neuron_dropout_mask(
    batch_size: int,
    n_neurons: int,
    dropout_rate: float,
    device: torch.device,
    generator: torch.Generator | None = None,
    keep_at_least_one: bool = True,
) -> torch.Tensor:
    """Sample a [batch, neurons] keep mask; 1 keeps, 0 drops."""
    if batch_size <= 0 or n_neurons <= 0:
        msg = "batch_size and n_neurons must be positive"
        raise ValueError(msg)
    if dropout_rate < 0.0 or dropout_rate >= 1.0:
        msg = "dropout_rate must be in [0, 1)"
        raise ValueError(msg)
    keep = torch.rand(batch_size, n_neurons, device=device, generator=generator) >= dropout_rate
    if keep_at_least_one:
        empty = keep.sum(dim=1) == 0
        if empty.any():
            indices = torch.randint(
                n_neurons,
                (int(empty.sum().item()),),
                device=device,
                generator=generator,
            )
            keep[empty] = False
            keep[empty, indices] = True
    return keep.to(dtype=torch.float32)


def apply_input_neuron_dropout(
    heldin_spikes: torch.Tensor,
    dropout_mask: torch.Tensor,
) -> torch.Tensor:
    if heldin_spikes.ndim != 3:
        msg = "heldin_spikes must have shape [batch, time, neurons]"
        raise ValueError(msg)
    if dropout_mask.shape != (heldin_spikes.shape[0], heldin_spikes.shape[2]):
        msg = "dropout_mask must have shape [batch, neurons]"
        raise ValueError(msg)
    return heldin_spikes * dropout_mask[:, None, :].to(dtype=heldin_spikes.dtype)


def summarize_dropout_mask(dropout_mask: torch.Tensor) -> dict[str, float]:
    if dropout_mask.ndim != 2:
        msg = "dropout_mask must have shape [batch, neurons]"
        raise ValueError(msg)
    keep_fraction = float(dropout_mask.detach().float().mean().cpu())
    return {
        "keep_fraction": keep_fraction,
        "dropout_fraction": 1.0 - keep_fraction,
        "min_kept_neurons": float(dropout_mask.detach().sum(dim=1).min().cpu()),
        "max_kept_neurons": float(dropout_mask.detach().sum(dim=1).max().cpu()),
    }
