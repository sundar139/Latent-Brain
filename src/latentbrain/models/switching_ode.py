from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from latentbrain.torch.rate_initialization import initialize_linear_readout_bias_from_rates


@dataclass(slots=True)
class SwitchingODEConfig:
    input_dim: int
    output_dim: int
    encoder_hidden_dim: int
    drift_hidden_dim: int
    latent_dim: int
    factor_dim: int
    n_regimes: int
    regime_hidden_dim: int
    regime_temperature: float
    dropout: float
    min_rate_hz: float
    max_rate_hz: float
    dt_seconds: float
    diffusion_scale: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "output_dim",
            "encoder_hidden_dim",
            "drift_hidden_dim",
            "latent_dim",
            "factor_dim",
            "n_regimes",
            "regime_hidden_dim",
        ):
            if int(getattr(self, name)) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)
        if self.n_regimes < 2:
            msg = "n_regimes must be at least 2"
            raise ValueError(msg)
        if self.regime_temperature <= 0.0:
            msg = "regime_temperature must be positive"
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
        if self.diffusion_scale != 0.0:
            msg = "switching neural-ODE-style tuning requires diffusion_scale == 0.0"
            raise ValueError(msg)


def mix_regime_drifts(regime_probs: torch.Tensor, regime_drifts: torch.Tensor) -> torch.Tensor:
    if regime_probs.ndim != 3 or regime_drifts.ndim != 4:
        msg = (
            "expected regime_probs [batch,time,regimes] and "
            "regime_drifts [batch,time,regimes,latent]"
        )
        raise ValueError(msg)
    return (regime_probs.unsqueeze(-1) * regime_drifts).sum(dim=2)


class SwitchingODE(nn.Module):
    """Soft switching neural-ODE-style latent generator for local co-smoothing."""

    def __init__(self, config: SwitchingODEConfig) -> None:
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
        regime_input_dim = config.latent_dim + encoded_dim + 1
        self.regime_net = nn.Sequential(
            nn.Linear(regime_input_dim, config.regime_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.regime_hidden_dim, config.n_regimes),
        )
        latent_time_dim = config.latent_dim + 1
        self.drift_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_time_dim, config.drift_hidden_dim),
                    nn.Tanh(),
                    nn.Linear(config.drift_hidden_dim, config.latent_dim),
                )
                for _ in range(config.n_regimes)
            ]
        )
        self.factor_readout = nn.Linear(config.latent_dim, config.factor_dim)
        self.rate_readout = nn.Linear(config.factor_dim, config.output_dim)

    def initialize_output_bias_from_rates(self, mean_rates_hz: torch.Tensor) -> None:
        initialize_linear_readout_bias_from_rates(
            self.rate_readout, mean_rates_hz, self.config.min_rate_hz
        )

    def _noise_like(self, tensor: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        if generator is None:
            return torch.randn_like(tensor)
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )

    def _sample_z0(
        self, mean: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None
    ) -> torch.Tensor:
        if not self.training:
            return mean
        std = torch.exp(0.5 * logvar)
        return mean + self._noise_like(std, generator) * std

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
        logits_list: list[torch.Tensor] = []
        probs_list: list[torch.Tensor] = []
        regime_drifts_list: list[torch.Tensor] = []
        mixed_drifts: list[torch.Tensor] = []
        for time_index in range(heldin_spikes.shape[1]):
            time_feature = self._time_feature(z, time_index, heldin_spikes.shape[1])
            latent_time = torch.cat([z, time_feature], dim=-1)
            regime_input = torch.cat([z, encoded[:, time_index, :], time_feature], dim=-1)
            logits = self.regime_net(regime_input)
            probs = F.softmax(logits / self.config.regime_temperature, dim=-1)
            regime_drifts = torch.stack([net(latent_time) for net in self.drift_nets], dim=1)
            mixed = (probs.unsqueeze(-1) * regime_drifts).sum(dim=1)
            latents.append(z)
            logits_list.append(logits)
            probs_list.append(probs)
            regime_drifts_list.append(regime_drifts)
            mixed_drifts.append(mixed)
            if time_index < heldin_spikes.shape[1] - 1:
                z = z + mixed * self.config.dt_seconds
        latent_tensor = torch.stack(latents, dim=1)
        factor_tensor = self.factor_readout(self.dropout(latent_tensor))
        rates = F.softplus(self.rate_readout(factor_tensor))
        regime_probs = torch.stack(probs_list, dim=1)
        regime_drifts_tensor = torch.stack(regime_drifts_list, dim=1)
        mixed_drift_tensor = torch.stack(mixed_drifts, dim=1)
        return {
            "rates_hz": torch.clamp(
                rates, min=self.config.min_rate_hz, max=self.config.max_rate_hz
            ),
            "factors": factor_tensor,
            "latents": latent_tensor,
            "z0_mean": z0_mean,
            "z0_logvar": z0_logvar,
            "regime_probs": regime_probs,
            "regime_logits": torch.stack(logits_list, dim=1),
            "regime_drifts": regime_drifts_tensor,
            "mixed_drift": mixed_drift_tensor,
            "drift": mixed_drift_tensor,
            "diffusion": torch.zeros_like(mixed_drift_tensor),
        }
