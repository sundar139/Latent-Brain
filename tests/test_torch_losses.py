from __future__ import annotations

import math

import pytest
import torch

from latentbrain.torch.losses import (
    gaussian_kl_standard_normal,
    lfads_cosmoothing_loss,
    lfads_elbo_loss,
    masked_poisson_loss,
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


def test_masked_poisson_loss_works_with_mean_normalization() -> None:
    counts = torch.tensor([[[0.0, 1.0]]])
    rates = torch.tensor([[[10.0, 20.0]]])

    loss = masked_poisson_loss(counts, rates, bin_size_ms=100, normalization="mean")

    expected = poisson_nll_torch(counts, rates, bin_size_ms=100) / counts.numel()
    assert loss.item() == pytest.approx(float(expected))


def test_cosmoothing_loss_has_heldin_heldout_and_kl_terms() -> None:
    loss = lfads_cosmoothing_loss(
        heldin_counts=torch.tensor([[[0.0, 1.0]]]),
        heldout_counts=torch.tensor([[[2.0]]]),
        all_rates_hz=torch.full((1, 1, 3), 20.0),
        heldin_indices=torch.tensor([0, 1]),
        heldout_indices=torch.tensor([2]),
        posterior_mean=torch.zeros(1, 2),
        posterior_logvar=torch.zeros(1, 2),
        bin_size_ms=50,
        kl_beta=0.25,
        heldin_loss_weight=1.0,
        heldout_loss_weight=1.0,
        normalization="mean",
    )

    assert set(loss) == {
        "loss",
        "heldin_reconstruction_loss",
        "heldout_prediction_loss",
        "kl_loss",
        "kl_beta",
    }
    assert all(torch.isfinite(value).all().item() for value in loss.values())


def test_cosmoothing_loss_rejects_overlapping_indices() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        lfads_cosmoothing_loss(
            heldin_counts=torch.zeros(1, 1, 2),
            heldout_counts=torch.zeros(1, 1, 1),
            all_rates_hz=torch.full((1, 1, 3), 20.0),
            heldin_indices=torch.tensor([0, 1]),
            heldout_indices=torch.tensor([1]),
            posterior_mean=torch.zeros(1, 2),
            posterior_logvar=torch.zeros(1, 2),
            bin_size_ms=50,
            kl_beta=0.0,
            heldin_loss_weight=1.0,
            heldout_loss_weight=1.0,
            normalization="mean",
        )


def test_cosmoothing_heldout_loss_changes_when_heldout_counts_change() -> None:
    kwargs = {
        "heldin_counts": torch.tensor([[[0.0, 1.0]]]),
        "all_rates_hz": torch.full((1, 1, 3), 20.0),
        "heldin_indices": torch.tensor([0, 1]),
        "heldout_indices": torch.tensor([2]),
        "posterior_mean": torch.zeros(1, 2),
        "posterior_logvar": torch.zeros(1, 2),
        "bin_size_ms": 50,
        "kl_beta": 0.0,
        "heldin_loss_weight": 1.0,
        "heldout_loss_weight": 1.0,
        "normalization": "mean",
    }

    low = lfads_cosmoothing_loss(heldout_counts=torch.tensor([[[0.0]]]), **kwargs)
    high = lfads_cosmoothing_loss(heldout_counts=torch.tensor([[[3.0]]]), **kwargs)

    assert low["heldout_prediction_loss"].item() != high["heldout_prediction_loss"].item()


def test_zero_heldout_weight_removes_heldout_term_from_total() -> None:
    loss = lfads_cosmoothing_loss(
        heldin_counts=torch.tensor([[[0.0, 1.0]]]),
        heldout_counts=torch.tensor([[[10.0]]]),
        all_rates_hz=torch.full((1, 1, 3), 20.0),
        heldin_indices=torch.tensor([0, 1]),
        heldout_indices=torch.tensor([2]),
        posterior_mean=torch.zeros(1, 2),
        posterior_logvar=torch.zeros(1, 2),
        bin_size_ms=50,
        kl_beta=0.0,
        heldin_loss_weight=1.0,
        heldout_loss_weight=0.0,
        normalization="mean",
    )

    assert loss["loss"].item() == pytest.approx(float(loss["heldin_reconstruction_loss"]))
