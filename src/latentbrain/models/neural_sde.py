from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from latentbrain.torch.rate_initialization import initialize_linear_readout_bias_from_rates


@dataclass(slots=True)
class NeuralSDEConfig:
    input_dim: int
    output_dim: int
    encoder_hidden_dim: int
    drift_hidden_dim: int
    diffusion_hidden_dim: int
    latent_dim: int
    factor_dim: int
    dropout: float
    min_rate_hz: float
    max_rate_hz: float
    dt_seconds: float
    diffusion_scale: float

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "output_dim",
            "encoder_hidden_dim",
            "drift_hidden_dim",
            "diffusion_hidden_dim",
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
        if self.dt_seconds <= 0.0:
            msg = "dt_seconds must be positive"
            raise ValueError(msg)
        if self.diffusion_scale < 0.0:
            msg = "diffusion_scale must be non-negative"
            raise ValueError(msg)


class NeuralSDE(nn.Module):
    """Compact Euler/Euler-Maruyama latent generator for local co-smoothing."""

    def __init__(self, config: NeuralSDEConfig) -> None:
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
        latent_time_dim = config.latent_dim + 1
        self.drift_net = nn.Sequential(
            nn.Linear(latent_time_dim, config.drift_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.drift_hidden_dim, config.latent_dim),
        )
        self.diffusion_net = nn.Sequential(
            nn.Linear(latent_time_dim, config.diffusion_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.diffusion_hidden_dim, config.latent_dim),
        )
        self.factor_readout = nn.Linear(config.latent_dim, config.factor_dim)
        self.rate_readout = nn.Linear(config.factor_dim, config.output_dim)

    def initialize_output_bias_from_rates(self, mean_rates_hz: torch.Tensor) -> None:
        initialize_linear_readout_bias_from_rates(
            self.rate_readout, mean_rates_hz, self.config.min_rate_hz
        )

    def _sample_z0(
        self, mean: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        if not self.training:
            return mean
        if self.config.diffusion_scale == 0.0:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + self._noise_like(std, generator) * std

    def _noise_like(self, tensor: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(tensor)
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )

    def _time_feature(self, z: torch.Tensor, time_index: int, time_bins: int) -> torch.Tensor:
        denom = max(time_bins - 1, 1)
        value = float(time_index) / float(denom)
        return z.new_full((z.shape[0], 1), value)

    def forward(
        self,
        heldin_spikes: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        deterministic: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if heldin_spikes.ndim != 3:
            msg = "heldin_spikes must have shape [batch, time, input_dim]"
            raise ValueError(msg)
        if heldin_spikes.shape[-1] != self.config.input_dim:
            msg = f"expected input_dim {self.config.input_dim}, got {heldin_spikes.shape[-1]}"
            raise ValueError(msg)
        if deterministic is None:
            deterministic = not self.training
        encoded, _ = self.encoder(heldin_spikes)
        encoded = self.dropout(encoded)
        summary = encoded[:, -1, :]
        z0_mean = self.z0_mean(summary)
        z0_logvar = torch.clamp(self.z0_logvar(summary), min=-10.0, max=10.0)
        z = z0_mean if deterministic else self._sample_z0(z0_mean, z0_logvar, generator)
        latents: list[torch.Tensor] = []
        drifts: list[torch.Tensor] = []
        diffusions: list[torch.Tensor] = []
        sqrt_dt = self.config.dt_seconds**0.5
        for time_index in range(heldin_spikes.shape[1]):
            time_feature = self._time_feature(z, time_index, heldin_spikes.shape[1])
            latent_time = torch.cat([z, time_feature], dim=-1)
            drift = self.drift_net(latent_time)
            raw_diffusion = F.softplus(self.diffusion_net(latent_time))
            diffusion = torch.clamp(raw_diffusion * self.config.diffusion_scale, max=10.0)
            latents.append(z)
            drifts.append(drift)
            diffusions.append(diffusion)
            if time_index < heldin_spikes.shape[1] - 1:
                z = z + drift * self.config.dt_seconds
                if self.training and not deterministic and self.config.diffusion_scale > 0.0:
                    z = z + diffusion * sqrt_dt * self._noise_like(z, generator)
        latent_tensor = torch.stack(latents, dim=1)
        drift_tensor = torch.stack(drifts, dim=1)
        diffusion_tensor = torch.stack(diffusions, dim=1)
        factor_tensor = self.factor_readout(self.dropout(latent_tensor))
        rates = F.softplus(self.rate_readout(factor_tensor))
        return {
            "rates_hz": torch.clamp(
                rates, min=self.config.min_rate_hz, max=self.config.max_rate_hz
            ),
            "factors": factor_tensor,
            "latents": latent_tensor,
            "z0_mean": z0_mean,
            "z0_logvar": z0_logvar,
            "drift": drift_tensor,
            "diffusion": diffusion_tensor,
        }
