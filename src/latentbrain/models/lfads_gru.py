from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class LFADSGRUConfig:
    input_dim: int
    output_dim: int
    encoder_hidden_dim: int
    generator_hidden_dim: int
    latent_dim: int
    factor_dim: int
    dropout: float
    min_rate_hz: float
    max_rate_hz: float

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "output_dim",
            "encoder_hidden_dim",
            "generator_hidden_dim",
            "latent_dim",
            "factor_dim",
        ):
            if int(getattr(self, name)) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        if self.dropout < 0.0 or self.dropout >= 1.0:
            msg = "dropout must be in [0, 1)"
            raise ValueError(msg)
        if self.min_rate_hz <= 0.0:
            msg = "min_rate_hz must be positive"
            raise ValueError(msg)
        if self.max_rate_hz <= self.min_rate_hz:
            msg = "max_rate_hz must exceed min_rate_hz"
            raise ValueError(msg)


class LFADSGRU(nn.Module):
    """Minimal LFADS-style sequential VAE foundation, not full LFADS."""

    def __init__(self, config: LFADSGRUConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = nn.GRU(
            input_size=config.input_dim,
            hidden_size=config.encoder_hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        encoded_dim = 2 * config.encoder_hidden_dim
        self.dropout = nn.Dropout(config.dropout)
        self.z0_mean = nn.Linear(encoded_dim, config.latent_dim)
        self.z0_logvar = nn.Linear(encoded_dim, config.latent_dim)
        self.generator_initial = nn.Linear(config.latent_dim, config.generator_hidden_dim)
        self.generator = nn.GRU(
            input_size=1,
            hidden_size=config.generator_hidden_dim,
            batch_first=True,
        )
        self.factor_readout = nn.Linear(config.generator_hidden_dim, config.factor_dim)
        self.rate_readout = nn.Linear(config.factor_dim, config.output_dim)

    def _sample_z0(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + torch.randn_like(std) * std

    def forward(self, heldin_spikes: torch.Tensor) -> dict[str, torch.Tensor]:
        if heldin_spikes.ndim != 3:
            msg = "heldin_spikes must have shape [batch, time, input_dim]"
            raise ValueError(msg)
        if heldin_spikes.shape[-1] != self.config.input_dim:
            msg = f"expected input_dim {self.config.input_dim}, got {heldin_spikes.shape[-1]}"
            raise ValueError(msg)
        encoded, _ = self.encoder(heldin_spikes)
        summary = self.dropout(encoded[:, -1, :])
        z0_mean = self.z0_mean(summary)
        z0_logvar = torch.clamp(self.z0_logvar(summary), min=-10.0, max=10.0)
        z0 = self._sample_z0(z0_mean, z0_logvar)
        initial = torch.tanh(self.generator_initial(z0)).unsqueeze(0)
        generator_input = heldin_spikes.new_zeros(heldin_spikes.shape[0], heldin_spikes.shape[1], 1)
        generator_output, _ = self.generator(generator_input, initial)
        factors = self.factor_readout(self.dropout(generator_output))
        rates = F.softplus(self.rate_readout(factors))
        rates_hz = torch.clamp(rates, min=self.config.min_rate_hz, max=self.config.max_rate_hz)
        return {
            "rates_hz": rates_hz,
            "factors": factors,
            "z0_mean": z0_mean,
            "z0_logvar": z0_logvar,
        }
