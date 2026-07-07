from __future__ import annotations

import math

import pytest
import torch

from latentbrain.torch.losses import (
    gaussian_kl_standard_normal,
    lfads_elbo_loss,
    poisson_nll_torch,
)


def test_poisson_nll_matches_hand_computed_value() -> None:
    counts = torch.tensor([0.0, 1.0, 2.0])
    rates_hz = torch.tensor([10.0, 20.0, 30.0])
    expected_counts = rates_hz * 0.1
    expected = -torch.sum(
        counts * torch.log(expected_counts) - expected_counts - torch.lgamma(counts + 1.0)
    )

    assert poisson_nll_torch(counts, rates_hz, bin_size_ms=100) == pytest.approx(float(expected))


def test_gaussian_kl_standard_normal_is_zero_for_matching_standard_normal() -> None:
    mean = torch.zeros(3, 4)
    logvar = torch.zeros(3, 4)

    assert gaussian_kl_standard_normal(mean, logvar).item() == pytest.approx(0.0)


def test_gaussian_kl_increases_when_mean_moves_from_zero() -> None:
    zero = gaussian_kl_standard_normal(torch.zeros(2, 3), torch.zeros(2, 3))
    shifted = gaussian_kl_standard_normal(torch.ones(2, 3), torch.zeros(2, 3))

    assert shifted.item() > zero.item()


def test_lfads_elbo_loss_returns_finite_tensors() -> None:
    loss = lfads_elbo_loss(
        heldin_counts=torch.tensor([[[0.0, 1.0], [2.0, 0.0]]]),
        heldin_rates_hz=torch.full((1, 2, 2), 20.0),
        posterior_mean=torch.zeros(1, 3),
        posterior_logvar=torch.zeros(1, 3),
        bin_size_ms=50,
        kl_beta=0.5,
    )

    assert set(loss) == {"loss", "reconstruction_loss", "kl_loss", "kl_beta"}
    assert all(torch.isfinite(value).all().item() for value in loss.values())
    assert math.isclose(loss["kl_beta"].item(), 0.5)
