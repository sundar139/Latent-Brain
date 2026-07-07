from __future__ import annotations

import torch

from latentbrain.models.lfads_gru import LFADSGRU, LFADSGRUConfig


def _model() -> LFADSGRU:
    return LFADSGRU(
        LFADSGRUConfig(
            input_dim=5,
            output_dim=5,
            encoder_hidden_dim=7,
            generator_hidden_dim=11,
            latent_dim=3,
            factor_dim=4,
            dropout=0.0,
            min_rate_hz=1.0e-4,
            max_rate_hz=500.0,
        )
    )


def test_forward_pass_shapes_and_finite_positive_rates() -> None:
    model = _model()
    output = model(torch.zeros(2, 6, 5))

    assert output["rates_hz"].shape == (2, 6, 5)
    assert output["factors"].shape == (2, 6, 4)
    assert output["z0_mean"].shape == (2, 3)
    assert output["z0_logvar"].shape == (2, 3)
    assert torch.isfinite(output["rates_hz"]).all()
    assert torch.all(output["rates_hz"] > 0.0)


def test_backward_pass_produces_finite_gradients() -> None:
    model = _model()
    output = model(torch.ones(2, 6, 5))
    loss = output["rates_hz"].mean() + output["z0_mean"].pow(2).mean()

    loss.backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all().item() for gradient in gradients)
