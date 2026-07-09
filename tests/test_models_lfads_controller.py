from __future__ import annotations

import torch

from latentbrain.models.lfads_controller import LFADSController, LFADSControllerConfig


def _model() -> LFADSController:
    return LFADSController(
        LFADSControllerConfig(
            input_dim=5,
            output_dim=7,
            encoder_hidden_dim=6,
            controller_hidden_dim=8,
            generator_hidden_dim=9,
            latent_dim=3,
            factor_dim=4,
            inferred_input_dim=2,
            dropout=0.0,
            min_rate_hz=1.0e-4,
            max_rate_hz=500.0,
        )
    )


def test_forward_pass_returns_required_keys_and_shapes() -> None:
    model = _model()
    output = model(torch.zeros(2, 6, 5))

    assert {
        "rates_hz",
        "factors",
        "z0_mean",
        "z0_logvar",
        "inferred_input_mean",
        "inferred_input_logvar",
        "inferred_inputs",
    }.issubset(output)
    assert output["rates_hz"].shape == (2, 6, 7)
    assert output["factors"].shape == (2, 6, 4)
    assert output["inferred_input_mean"].shape == (2, 6, 2)
    assert output["inferred_input_logvar"].shape == (2, 6, 2)


def test_eval_mode_is_deterministic() -> None:
    model = _model().eval()
    spikes = torch.ones(2, 6, 5)

    first = model(spikes)
    second = model(spikes)

    assert torch.allclose(first["rates_hz"], second["rates_hz"])
    assert torch.allclose(first["inferred_inputs"], second["inferred_inputs"])


def test_training_mode_outputs_are_finite_with_positive_rates() -> None:
    model = _model().train()
    output = model(torch.ones(2, 6, 5))

    for key in (
        "rates_hz",
        "factors",
        "z0_mean",
        "z0_logvar",
        "inferred_input_mean",
        "inferred_input_logvar",
    ):
        assert torch.isfinite(output[key]).all()
    assert torch.all(output["rates_hz"] > 0.0)


def test_parameter_count_is_finite_and_nonzero() -> None:
    parameter_count = sum(parameter.numel() for parameter in _model().parameters())

    assert parameter_count > 0
    assert isinstance(parameter_count, int)
