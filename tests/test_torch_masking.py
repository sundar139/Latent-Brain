from __future__ import annotations

import pytest
import torch

from latentbrain.torch.masking import (
    apply_input_neuron_dropout,
    sample_neuron_dropout_mask,
    summarize_dropout_mask,
)


def test_mask_shape_and_approximate_dropout_rate() -> None:
    generator = torch.Generator().manual_seed(7)
    mask = sample_neuron_dropout_mask(200, 20, 0.25, torch.device("cpu"), generator)

    assert mask.shape == (200, 20)
    assert abs(float((1.0 - mask).mean()) - 0.25) < 0.05


def test_apply_mask_zeros_expected_neurons() -> None:
    spikes = torch.ones(2, 3, 4)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=torch.float32)

    dropped = apply_input_neuron_dropout(spikes, mask)

    torch.testing.assert_close(dropped[0, :, 1], torch.zeros(3))
    torch.testing.assert_close(dropped[1, :, 0], torch.zeros(3))
    torch.testing.assert_close(dropped[0, :, 0], torch.ones(3))


def test_keep_at_least_one_behavior_works() -> None:
    mask = sample_neuron_dropout_mask(8, 3, 0.99, torch.device("cpu"), keep_at_least_one=True)

    assert torch.all(mask.sum(dim=1) >= 1)


def test_generator_is_deterministic() -> None:
    first = sample_neuron_dropout_mask(
        4, 5, 0.5, torch.device("cpu"), torch.Generator().manual_seed(11)
    )
    second = sample_neuron_dropout_mask(
        4, 5, 0.5, torch.device("cpu"), torch.Generator().manual_seed(11)
    )

    torch.testing.assert_close(first, second)


def test_invalid_dropout_rate_raises() -> None:
    with pytest.raises(ValueError, match="dropout_rate"):
        sample_neuron_dropout_mask(2, 3, 1.0, torch.device("cpu"))


def test_summary_has_expected_keys() -> None:
    summary = summarize_dropout_mask(torch.tensor([[1.0, 0.0], [1.0, 1.0]]))

    assert set(summary) == {
        "keep_fraction",
        "dropout_fraction",
        "min_kept_neurons",
        "max_kept_neurons",
    }
    assert summary["dropout_fraction"] == 0.25
