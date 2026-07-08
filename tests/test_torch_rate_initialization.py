from __future__ import annotations

import numpy as np
import torch

from latentbrain.torch.rate_initialization import (
    compute_train_mean_rates_hz,
    initialize_linear_readout_bias_from_rates,
    inverse_softplus_rate,
)


def test_inverse_softplus_approximates_target_rate() -> None:
    target = torch.tensor([1.0e-4, 0.1, 5.0, 50.0])

    bias = inverse_softplus_rate(target, min_rate_hz=1.0e-4)

    assert torch.allclose(torch.nn.functional.softplus(bias), target, rtol=1.0e-4, atol=1.0e-5)


def test_train_mean_rates_are_computed_from_counts() -> None:
    spikes = np.array([[[1, 0], [0, 2]], [[3, 1], [0, 1]]])

    rates = compute_train_mean_rates_hz(
        spikes, bin_size_ms=20, min_rate_hz=1.0e-4, max_rate_hz=500.0
    )

    assert np.allclose(rates, [50.0, 50.0])


def test_readout_bias_initialization_changes_only_bias() -> None:
    linear = torch.nn.Linear(3, 2)
    old_weight = linear.weight.detach().clone()
    old_bias = linear.bias.detach().clone()

    initialize_linear_readout_bias_from_rates(linear, torch.tensor([2.0, 4.0]), min_rate_hz=1.0e-4)

    assert torch.allclose(linear.weight, old_weight)
    assert not torch.allclose(linear.bias, old_bias)


def test_initialized_softplus_bias_approximates_mean_rates() -> None:
    linear = torch.nn.Linear(3, 2)
    target = torch.tensor([2.0, 4.0])

    initialize_linear_readout_bias_from_rates(linear, target, min_rate_hz=1.0e-4)

    assert torch.allclose(
        torch.nn.functional.softplus(linear.bias), target, rtol=1.0e-4, atol=1.0e-5
    )
