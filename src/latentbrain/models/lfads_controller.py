from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from latentbrain.torch.rate_initialization import initialize_linear_readout_bias_from_rates


@dataclass(slots=True)
class LFADSControllerConfig:
    input_dim: int
    output_dim: int
    encoder_hidden_dim: int
    controller_hidden_dim: int
    generator_hidden_dim: int
    latent_dim: int
    factor_dim: int
    inferred_input_dim: int
    dropout: float
    min_rate_hz: float
    max_rate_hz: float

    def __post_init__(self) -> None:
        for name in (
            "input_dim",
            "output_dim",
            "encoder_hidden_dim",
            "controller_hidden_dim",
            "generator_hidden_dim",
            "latent_dim",
            "factor_dim",
            "inferred_input_dim",
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


class LFADSController(nn.Module):
    """Controller-style LFADS-family model with inferred inputs, not full LFADS."""

    def __init__(self, config: LFADSControllerConfig) -> None:
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
        self.controller = nn.GRUCell(
            encoded_dim + config.generator_hidden_dim, config.controller_hidden_dim
        )
        self.inferred_input_mean = nn.Linear(
            config.controller_hidden_dim, config.inferred_input_dim
        )
        self.inferred_input_logvar = nn.Linear(
            config.controller_hidden_dim, config.inferred_input_dim
        )
        self.generator = nn.GRUCell(
            config.latent_dim + config.inferred_input_dim, config.generator_hidden_dim
        )
        self.factor_readout = nn.Linear(config.generator_hidden_dim, config.factor_dim)
        self.rate_readout = nn.Linear(config.factor_dim, config.output_dim)

    def initialize_output_bias_from_rates(self, mean_rates_hz: torch.Tensor) -> None:
        initialize_linear_readout_bias_from_rates(
            self.rate_readout, mean_rates_hz, self.config.min_rate_hz
        )

    def _sample(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
        encoded = self.dropout(encoded)
        summary = encoded[:, -1, :]
        z0_mean = self.z0_mean(summary)
        z0_logvar = torch.clamp(self.z0_logvar(summary), min=-10.0, max=10.0)
        z0 = self._sample(z0_mean, z0_logvar)
        generator_state = torch.tanh(self.generator_initial(z0))
        controller_state = heldin_spikes.new_zeros(
            heldin_spikes.shape[0], self.config.controller_hidden_dim
        )
        factors: list[torch.Tensor] = []
        input_means: list[torch.Tensor] = []
        input_logvars: list[torch.Tensor] = []
        inferred_inputs: list[torch.Tensor] = []
        for time_index in range(heldin_spikes.shape[1]):
            controller_input = torch.cat([encoded[:, time_index, :], generator_state], dim=-1)
            controller_state = self.controller(controller_input, controller_state)
            u_mean = self.inferred_input_mean(controller_state)
            u_logvar = torch.clamp(
                self.inferred_input_logvar(controller_state), min=-10.0, max=10.0
            )
            inferred_input = self._sample(u_mean, u_logvar)
            generator_input = torch.cat([z0, inferred_input], dim=-1)
            generator_state = self.generator(generator_input, generator_state)
            factors.append(self.factor_readout(self.dropout(generator_state)))
            input_means.append(u_mean)
            input_logvars.append(u_logvar)
            inferred_inputs.append(inferred_input)
        factor_tensor = torch.stack(factors, dim=1)
        rates = F.softplus(self.rate_readout(factor_tensor))
        return {
            "rates_hz": torch.clamp(
                rates, min=self.config.min_rate_hz, max=self.config.max_rate_hz
            ),
            "factors": factor_tensor,
            "z0_mean": z0_mean,
            "z0_logvar": z0_logvar,
            "inferred_input_mean": torch.stack(input_means, dim=1),
            "inferred_input_logvar": torch.stack(input_logvars, dim=1),
            "inferred_inputs": torch.stack(inferred_inputs, dim=1),
        }
